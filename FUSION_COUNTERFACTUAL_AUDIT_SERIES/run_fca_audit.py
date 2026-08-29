"""FCA: frozen F16 counterfactual audit h = h_ind + alpha * delta_social."""
from __future__ import annotations
import argparse,csv,importlib.util,json,os,sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

ROOT=Path(__file__).resolve().parent;WS=ROOT.parent;PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");FRONT=WS/"FRONT_ENCODER_SERIES";DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse";SOURCE=WS/"MODE_TEMPLATE_GROUP_SERIES"/"EXP_146B_ANCHORED_MODE_TEMPLATE"/"results";ADAPTER=ROOT/"fca_current_decoder_adapter.py";OUT=ROOT/"EXP_153_FCA"
ALPHAS=(0.0,.25,.5,.75,1.0,1.25);LAMBDA=.5
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);assert s and s.loader;s.loader.exec_module(m);return m
def split_indices(raw,num,cfg,kind):
 scenes=np.asarray(raw[:,0,0,0],dtype=np.int64);unique=np.unique(scenes);frac=.05 if kind=="general" else .10
 if kind=="general":rng=np.random.default_rng(20260828)
 else:
  pf=int(cfg.DATA.FREQUENCY*cfg.PREDICTION.OBS_TIME);focal=np.asarray(raw[:,0,:pf,-2:],dtype=np.float32)/float(cfg.PREDICTION.TRAJ_SCALE);step=np.diff(focal,axis=1);speed=np.median(np.linalg.norm(step,axis=-1)[:,-min(3,step.shape[1]):],axis=1);acc=np.linalg.norm(np.diff(step,axis=1),axis=-1).mean(1);density=np.minimum(np.asarray(num),int(cfg.DATA.MAX_AGENTS));
  def rank(x):
   order=np.argsort(x,kind="stable");r=np.empty(len(x),np.float32);r[order]=np.arange(len(x));return r/max(len(x)-1,1)
  challenge=rank(speed)+rank(acc)+1.-(density-1.)/max(int(cfg.DATA.MAX_AGENTS)-1,1)
 vals=[]
 for scene in unique:
  ids=np.flatnonzero(scenes==scene);count=max(1,min(len(ids)-1,int(round(len(ids)*frac))))
  vals.append((rng.permutation(ids)[:count] if kind=="general" else ids[np.argsort(challenge[ids],kind="stable")][-count:]))
 return np.sort(np.concatenate(vals))
@contextmanager
def scaled_social(model,alpha):
 base=model.agent_aware_trans_encoder;cache={}
 def before_gate(module,args):
  x=args[0];cache["individual"],cache["social"]=x[...,:base.dim],x[...,base.dim:]
 def after_gate(module,args,result):cache["gate"]=torch.sigmoid(result)
 def before_norm(module,args):
  if not all(k in cache for k in ("individual","social","gate")):raise RuntimeError("FCA did not capture the F16 fusion inputs")
  return (cache["individual"]+float(alpha)*cache["gate"]*cache["social"],)
 h1=base.social_gate.register_forward_pre_hook(before_gate);h2=base.social_gate.register_forward_hook(after_gate);h3=base.social_norm.register_forward_pre_hook(before_norm)
 try:yield
 finally:h1.remove();h2.remove();h3.remove()
def build(split,seed,cfg,FrontEncoderASTRA,device):
 ckpt=SOURCE/split/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"
 if not ckpt.is_file():raise FileNotFoundError(ckpt)
 os.environ["FCA_SOURCE_CHECKPOINT"]=str(ckpt);model=FrontEncoderASTRA(cfg,"f16_temporal_social_cross");adapter=load(f"fca_adapter_{split}",ADAPTER);adapter.replace_decoder(model,cfg,"fca_current_mode_anchor");model.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=True),strict=True);return model.to(device).eval()
@torch.inference_mode()
def collect(model,loader,helper,device,alpha,max_batches=0):
 out={k:[] for k in ("ade","fde","same","topade","topfde","cost","speed","density")}
 with scaled_social(model,alpha):
  for batch_id,(past,future,num,*_) in enumerate(loader):
   past,future,num=past.to(device),future.to(device),num.to(device).int();rawpast=past.clone();past,future=helper.normalize_to_focal_heading(past,future);center=past.mean(-2,keepdim=True);pred=model(past-center,None,None,mode="testing",num_valid=num,graph_loc=past)[2][:,0]+center[:,0,None];err=torch.linalg.vector_norm(pred-future[:,0,None],dim=-1);ade,fde=err.mean(-1),err[...,-1];cost=ade+LAMBDA*fde;row=torch.arange(len(ade),device=device);top=model.last_mode_logits[:,0].argmax(-1);speed=torch.linalg.vector_norm(rawpast[:,0,-1]-rawpast[:,0,-2],dim=-1);density=num.float()
   for k,v in {"ade":ade.min(1).values,"fde":fde.min(1).values,"same":fde[row,ade.argmin(1)],"topade":ade[row,top],"topfde":fde[row,top],"cost":cost.min(1).values,"speed":speed,"density":density}.items():out[k].append(v.cpu().numpy())
   if max_batches and batch_id+1>=max_batches:break
 return {k:np.concatenate(v) for k,v in out.items()}
def social_labels(split):
 path=WS/"COUNTERFACTUAL_SOCIAL_NECESSITY_AUDIT"/"results"/split/"paired_samples.csv"
 if not path.is_file():return None
 with path.open(encoding="utf-8-sig") as h:return {int(r["index"]):float(r["social_gain_oracle_cost"]) for r in csv.DictReader(h)}
def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=1);p.add_argument("--splits",nargs="+",choices=("general","challenge"),default=["general","challenge"]);p.add_argument("--batch-size",type=int,default=128);p.add_argument("--smoke",action="store_true");a=p.parse_args();device=torch.device(a.device if torch.cuda.is_available() else "cpu");sys.path[:0]=[str(PROJECT),str(FRONT)];os.chdir(PROJECT);from experiment_config import load_locked_q20;from front_encoder_models import FrontEncoderASTRA;helper=load("fca_helper",FRONT/"train_front_variant.py");cfg=load_locked_q20(PROJECT/"configs"/"motif_rbf_univ.yaml");cfg.MODEL.USE_PRETRAINED_UNET=False;cfg.MODEL.USE_CACHED_UNET_FEATURES=False;cfg.DATA.MAX_AGENTS=9;cfg.MODEL.GRAPH.TOP_K=4;cfg.MODEL.GRAPH.RADIUS=6.;cfg.device=str(device);cfg.device_list=[device.index or 0] if device.type=="cuda" else ["cpu"]
 raw=np.load(DATA/"eth_data_train.npy",mmap_mode="r");num=np.load(DATA/"eth_num_train.npy",mmap_mode="r");base=helper.TrajectoryDataset(DATA/"eth_data_train.npy",DATA/"eth_num_train.npy",cfg)
 if a.smoke:
  ids=split_indices(raw,num,cfg,"general");model=build("general",a.seed,cfg,FrontEncoderASTRA,device);value=collect(model,DataLoader(Subset(base,ids),batch_size=2,shuffle=False,num_workers=0),helper,device,1.0,max_batches=1);print("[smoke-ok] samples="+str(len(value["ade"])));return
 OUT.mkdir(parents=True,exist_ok=True);report={"experiment":"EXP-153 FCA frozen social-residual scaling","alphas":list(ALPHAS),"seed":a.seed,"splits":{}}
 for split in a.splits:
  ids=split_indices(raw,num,cfg,split);loader=DataLoader(Subset(base,ids),batch_size=a.batch_size,shuffle=False,num_workers=0);model=build(split,a.seed,cfg,FrontEncoderASTRA,device);runs={str(x):collect(model,loader,helper,device,x) for x in ALPHAS};metrics={}
  for alpha,data in runs.items():metrics[alpha]={"minADE20":float(data["ade"].mean()),"minFDE20":float(data["fde"].mean()),"same_path_FDE":float(data["same"].mean()),"top1_ADE":float(data["topade"].mean()),"top1_FDE":float(data["topfde"].mean()),"oracle_cost":float(data["cost"].mean())}
  costs=np.stack([runs[str(x)]["cost"] for x in ALPHAS]);best=costs.argmin(0);labels=social_labels(split);gain=np.asarray([labels.get(int(i),np.nan) for i in ids]) if labels else None;rho=None if gain is None else float(np.corrcoef(np.argsort(np.argsort(gain)),np.argsort(np.argsort(best)))[0,1]);rows=[]
  for i,index in enumerate(ids):rows.append({"index":int(index),"best_alpha":float(ALPHAS[int(best[i])]),"speed":float(runs["1.0"]["speed"][i]),"density":float(runs["1.0"]["density"][i]),"csna_social_gain_cost":None if gain is None or not np.isfinite(gain[i]) else float(gain[i]),**{f"cost_a{x}":float(runs[str(x)]["cost"][i]) for x in ALPHAS}})
  with (OUT/f"{split}_per_sample.csv").open("w",newline="",encoding="utf-8-sig") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  report["splits"][split]={"samples":len(ids),"metrics":metrics,"best_alpha_distribution":{str(x):int((best==j).sum()) for j,x in enumerate(ALPHAS)},"best_alpha_mean":float(np.mean([ALPHAS[int(x)] for x in best])),"best_alpha_vs_csna_social_gain_rank_correlation":rho}
 (OUT/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8");print("[done] "+str(OUT/"summary.json"))
if __name__=="__main__":main()
