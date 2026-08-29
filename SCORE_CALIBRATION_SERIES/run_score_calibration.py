"""EXP-150B: frozen S0 per-mode logit calibration on SDD validation only."""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
ROOT=Path(__file__).resolve().parent;WS=ROOT.parent;PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");FRONT=WS/"FRONT_ENCODER_SERIES";DATA=WS/"SDD_STANDARD_POTENTIAL_SERIES"/"processed_data";ASSET=WS/"SCENE_QUERY_CURRENT_DECODER_SDD_SERIES"/"processed_scene_assets";DEC=WS/"SCENE_QUERY_CURRENT_DECODER_SDD_SERIES"/"current_scene_query_models.py"
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def collect(model,loader,helper,device):
 A=[];F=[];L=[];S=[]
 with torch.inference_mode():
  for past,future,num,scene,_,_ in loader:
   past,future,num=past.to(device),future.to(device),num.to(device).int();past,future=helper.normalize_to_focal_heading(past,future);c=past.mean(-2,keepdim=True);pred=model(past-c,None,None,mode="testing",num_valid=num,graph_loc=past)[2][:,0]+c[:,0,None];e=torch.linalg.vector_norm(pred-future[:,0,None],dim=-1);A.append(e.mean(-1).cpu());F.append(e[...,-1].cpu());L.append(model.last_mode_logits[:,0].cpu());S.append(scene)
 return tuple(torch.cat(x) for x in (A,F,L,S))
def report(ade,fde,q,score,scene,names):
 row=torch.arange(len(q));top=score.argmax(1);best=q.argmin(1);out={"samples":len(q),"top1_ADE_px":float(ade[row,top].mean()*20),"top1_FDE_px":float(fde[row,top].mean()*20),"combined_hit":float((top==best).float().mean()),"score_entropy":float(torch.distributions.Categorical(logits=score).entropy().mean()),"top_mode_usage":torch.bincount(top,minlength=20).tolist(),"oracle_mode_usage":torch.bincount(best,minlength=20).tolist()}
 for label,mask in {"little":torch.tensor([names[int(x)].startswith("little/") for x in scene]),"non_little":torch.tensor([not names[int(x)].startswith("little/") for x in scene])}.items():
  ids=torch.where(mask)[0];t=top[ids];out[label]={"samples":len(ids),"top1_ADE_px":float(ade[ids,t].mean()*20),"top1_FDE_px":float(fde[ids,t].mean()*20),"combined_hit":float((t==best[ids]).float().mean())}
 return out
def fit(logits,q,kind,tau,steps):
 target=torch.softmax(-q/tau,dim=1);bias=torch.nn.Parameter(torch.zeros(20));log_t=torch.nn.Parameter(torch.zeros(()));params=[]
 if kind in ("b2","b3"):params.append(bias)
 if kind in ("b1","b3"):params.append(log_t)
 if not params:return torch.zeros(20),torch.tensor(1.)
 opt=torch.optim.Adam(params,lr=.05)
 for _ in range(steps):
  temp=torch.exp(log_t).clamp(.15,6.);score=(logits+bias[None])/temp;loss=-(target*torch.log_softmax(score,1)).sum(1).mean()+1e-3*bias.square().mean();opt.zero_grad();loss.backward();opt.step()
 return bias.detach(),torch.exp(log_t.detach()).clamp(.15,6.)
def main():
 p=argparse.ArgumentParser();p.add_argument("--seeds",default="1,2,3");p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=128);p.add_argument("--target-temperature",type=float,default=.25);p.add_argument("--steps",type=int,default=500);p.add_argument("--force",action="store_true");a=p.parse_args();outdir=ROOT/"EXP_150B_SCORE_CALIBRATION";out=outdir/"summary.json"
 if out.is_file() and not a.force:print(f"[skip] {out}");return
 os.chdir(PROJECT);sys.path[:0]=[str(FRONT),str(PROJECT)]
 from experiment_config import load_locked_q20
 from front_encoder_models import FrontEncoderASTRA
 helper=mod("b_helper",FRONT/"train_front_variant.py");decoder=mod("b_dec",DEC);cfg=load_locked_q20(PROJECT/"configs"/"motif_rbf_univ.yaml");cfg.MODEL.USE_PRETRAINED_UNET=False;cfg.MODEL.USE_CACHED_UNET_FEATURES=False;cfg.DATA.MAX_AGENTS=16;cfg.MODEL.GRAPH.TOP_K=4;cfg.MODEL.GRAPH.RADIUS=6.;device=torch.device(a.device if torch.cuda.is_available() else "cpu");cfg.device=str(device);cfg.device_list=[device.index or 0] if device.type=="cuda" else ["cpu"]
 raw=np.load(DATA/"sdd_standard_train_val_data.npy",mmap_mode="r");val=np.flatnonzero(np.asarray(raw[:,0,0,0],dtype=np.int64)==1);base=helper.TrajectoryDataset(DATA/"sdd_standard_train_val_data.npy",DATA/"sdd_standard_train_val_num.npy",cfg,ASSET/"sdd_train_val_scene_ids.npy");vloader=DataLoader(Subset(base,val),batch_size=a.batch_size,shuffle=False,num_workers=0);test=helper.TrajectoryDataset(DATA/"sdd_standard_test_data.npy",DATA/"sdd_standard_test_num.npy",cfg,ASSET/"sdd_test_scene_ids.npy");tloader=DataLoader(test,batch_size=a.batch_size,shuffle=False,num_workers=0);meta=json.loads((ASSET/"scene_asset_manifest.json").read_text());names={int(v):k for k,v in meta["video_to_scene_id"].items()};result={"experiment":"EXP-150B frozen score calibration","fit_split":"SDD validation scene 1 only; test never used in fitting","target":"softmax(-(ADE+0.25*FDE)/tau)","tau":a.target_temperature,"seeds":{}}
 for seed in [int(x) for x in a.seeds.split(",")]:
  ckpt=WS/"SCENE_QUERY_CURRENT_DECODER_SDD_SERIES"/"EXP_147B_CURRENT_DECODER_SDD_SCENE_QUERY"/"results"/"s0_current"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_1"/"best_selected.pth";model=FrontEncoderASTRA(cfg,"f16_temporal_social_cross");decoder.replace_decoder(model,cfg,"s0_current");model.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=True),strict=True);model=model.to(device).eval();va,vf,vl,vs=collect(model,vloader,helper,device);ta,tf,tl,ts=collect(model,tloader,helper,device);vq=va+.25*vf;tq=ta+.25*tf;one={}
  for kind in ("b0","b1","b2","b3"):
   bias,temp=fit(vl,vq,kind,a.target_temperature,a.steps);one[kind]={"temperature":float(temp),"bias":bias.tolist(),"test":report(ta,tf,tq,(tl+bias[None])/temp,ts,names)}
  result["seeds"][str(seed)]=one
 outdir.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8");print("[done] "+str(out))
if __name__=="__main__":main()
