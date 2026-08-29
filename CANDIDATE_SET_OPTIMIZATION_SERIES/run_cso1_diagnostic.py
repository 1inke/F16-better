"""Strict paired B0/CSO1 candidate-set diagnostic."""
from __future__ import annotations
import argparse,importlib.util,json,os,sys
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parent;WORKSPACE=ROOT.parent;PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");FRONT=WORKSPACE/"FRONT_ENCODER_SERIES"
CSRS=WORKSPACE/"COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES";IR1=WORKSPACE/"INTERACTION_RESIDUAL_SUPERVISION_SERIES";RESULTS=ROOT/"results";OUTPUT=ROOT/"diagnostic"
SPLITS=("general","challenge");VARIANTS=("baseline","cso1_winner_protected_repulsion","cso1_repulsion_prototype");METRICS=("min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde");STEPS=(4,8,12);KS=(1,2,5,10,20)
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path)
 if s is None or s.loader is None:raise ImportError(path)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def source(split,seed):return IR1/"results"/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"
def run_dir(split,name,seed):
 if name=="baseline":return CSRS/"results"/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
 return RESULTS/split/name/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def prepare(device):
 sys.path[:0]=[str(PROJECT),str(PROJECT/"scripts"),str(FRONT)];os.chdir(PROJECT);import sitecustomize  # noqa
 from experiment_config import load_locked_q20
 from front_encoder_models import FrontEncoderASTRA
 c=load_locked_q20(PROJECT/"configs"/"motif_rbf_univ.yaml");c.MODEL.USE_PRETRAINED_UNET=False;c.MODEL.USE_CACHED_UNET_FEATURES=False;c.DATA.MAX_AGENTS=9;c.MODEL.GRAPH.TOP_K=4;c.MODEL.GRAPH.RADIUS=6.;c.MODEL.DECODER.NUM_MODES=20;c.device=str(device);c.device_list=[device.index or 0] if device.type=="cuda" else ["cpu"]
 return c,FrontEncoderASTRA
def build(split,name,seed,cfg,cls,device):
 model=cls(cfg,"f16_temporal_social_cross");src=source(split,seed)
 if name=="baseline":os.environ["CSRS_SOURCE_CHECKPOINT"]=str(src);module=load(f"cso_diag_b0_{split}",CSRS/"current_baseline_adapter.py");module.replace_decoder(model,cfg,"current_mode_anchor_template")
 else:os.environ["CSO_SOURCE_CHECKPOINT"]=str(src);module=load(f"cso_diag_{name}_{split}",ROOT/"cso_models.py");module.replace_decoder(model,cfg,name)
 model.load_state_dict(torch.load(run_dir(split,name,seed)/"best_selected.pth",map_location="cpu",weights_only=True),strict=True);return model.to(device).eval()
@torch.inference_mode()
def infer(model,loader,device):
 from train_front_variant import normalize_to_focal_heading
 out={k:[] for k in ("prediction","anchor","target","score",*METRICS)}
 for past,future,num,indices,scenes in loader:
  del indices,scenes;past=past.to(device);future=future.to(device);num=num.to(device).int();past,future=normalize_to_focal_heading(past,future);center=past.mean(-2,keepdim=True)
  pred=model(past-center,None,None,mode="testing",num_valid=num,graph_loc=past)[2]+center.unsqueeze(-3);candidate,target=pred[:,0],future[:,0];err=torch.linalg.vector_norm(candidate-target[:,None],dim=-1);ade,fde=err.mean(-1),err[...,-1];row=torch.arange(len(ade),device=device);winner=ade.argmin(-1);top=model.last_mode_logits[:,0].argmax(-1)
  vals={"prediction":candidate.cpu().numpy(),"anchor":model.decoder.last_corrected_anchors[:,0].cpu().numpy(),"target":target.cpu().numpy(),"score":model.last_mode_logits[:,0].cpu().numpy(),"min_ade_20":ade.min(-1).values.cpu().numpy(),"min_fde_20":fde.min(-1).values.cpu().numpy(),"same_path_fde":fde[row,winner].cpu().numpy(),"top1_ade":ade[row,top].cpu().numpy(),"top1_fde":fde[row,top].cpu().numpy()}
  for k,v in vals.items():out[k].append(v)
 return {k:np.concatenate(v) for k,v in out.items()}
def summary(v):return {m:float(v[m].mean()) for m in METRICS}
def winner_usage(v):
 e=np.linalg.norm(v["prediction"]-v["target"][:,None],axis=-1);w=(e.mean(-1)+.25*e[...,-1]).argmin(-1);count=np.bincount(w,minlength=20);p=count/count.sum();h=-(p*np.log(np.maximum(p,1e-12))).sum()
 return {"entropy":float(h),"normalized_entropy":float(h/np.log(20)),"modes_over_1pct":int((p>.01).sum()),"modes_over_5pct":int((p>.05).sum()),"never_win_modes":int((count==0).sum()),"frequency":p.tolist()}
def diversity(v):
 p,a=v["prediction"],v["anchor"];i,j=np.triu_indices(20,1)
 return {"endpoint":float(np.linalg.norm(p[:,i,-1]-p[:,j,-1],axis=-1).mean()),"trajectory":float(np.linalg.norm(p[:,i]-p[:,j],axis=-1).mean()),"anchor":float(np.linalg.norm(a[:,i]-a[:,j],axis=-1).mean())}
def pairwise(v):
 p=v["prediction"][:,:,-1];i,j=np.triu_indices(20,1);d=np.linalg.norm(p[:,i]-p[:,j],axis=-1).reshape(-1)
 return {"mean":float(d.mean()),"median":float(np.median(d)),"p10":float(np.quantile(d,.10)),"p25":float(np.quantile(d,.25)),"p75":float(np.quantile(d,.75)),"below_margin_0_4":float((d<.4).mean())}
def coverage_curve(v):
 err=np.linalg.norm(v["prediction"]-v["target"][:,None],axis=-1);ade,fde=err.mean(-1),err[...,-1];order=np.argsort(-v["score"],axis=1);rows=np.arange(len(order))[:,None];result={}
 for k in KS:
  chosen=order[:,:k];result[str(k)]={"ade":float(ade[rows[:,:k],chosen].min(1).mean()),"fde":float(fde[rows[:,:k],chosen].min(1).mean())}
 return result
def analyse(split,values,out):
 target_anchor=values["baseline"]["target"][:,[3,7,11]];result={"split":split,"samples":len(target_anchor),"models":{}}
 for name,v in values.items():
  ae=np.linalg.norm(v["anchor"]-target_anchor[:,None],axis=-1);coverage={f"step_{s}":float(ae[:,:,i].min(1).mean()) for i,s in enumerate(STEPS)};coverage["endpoint"]=float(np.linalg.norm(v["prediction"][:,:,-1]-v["target"][:,None,-1],axis=-1).min(1).mean())
  result["models"][name]={"metrics":summary(v),"winner_usage":winner_usage(v),"coverage_at_k":coverage_curve(v),"pairwise_endpoint":pairwise(v),"anchor_coverage":coverage,"candidate_diversity":diversity(v)}
 base=result["models"]["baseline"];result["changes_pct"]={}
 for name in VARIANTS[1:]:
  cur=result["models"][name];result["changes_pct"][name]={"metrics":{k:100*(cur["metrics"][k]/base["metrics"][k]-1) for k in METRICS},"coverage_at_k":{k:{m:100*(cur["coverage_at_k"][k][m]/base["coverage_at_k"][k][m]-1) for m in ("ade","fde")} for k in map(str,KS)},"diversity":{k:100*(cur["candidate_diversity"][k]/base["candidate_diversity"][k]-1) for k in base["candidate_diversity"]},"anchor_coverage":{k:100*(cur["anchor_coverage"][k]/base["anchor_coverage"][k]-1) for k in base["anchor_coverage"]}}
 (out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8");return result
def verify(path,v):
 raw=json.loads(path.read_text(encoding="utf-8"))["evaluation_best_selected"];e={"min_ade_20":raw["oracle"]["min_ade_20"],"min_fde_20":raw["oracle"]["min_fde_20"],"same_path_fde":raw["oracle"]["same_path_fde"],"top1_ade":raw["original"]["ade"],"top1_fde":raw["original"]["fde"]}
 for m,x in e.items():
  if abs(float(v[m].mean())-float(x))>2e-5:raise RuntimeError(f"reproduction {m}")
def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=128);p.add_argument("--seed",type=int,default=1);p.add_argument("--max-samples",type=int);a=p.parse_args();device=torch.device(a.device);OUTPUT.mkdir(parents=True,exist_ok=True);cfg,cls=prepare(device);geom=load("cso_geom",WORKSPACE/"INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES"/"run_geometry_diagnostic.py");results=[]
 for split in SPLITS:
  run=run_dir(split,"baseline",a.seed);indices=np.asarray(np.load(run/"split_indices.npz")["validation"],np.int64)
  if a.max_samples is not None:indices=indices[:a.max_samples]
  info=json.loads((run/"split.json").read_text(encoding="utf-8"));ds=geom.IndexedDataset(Path(info["data_path"]),Path(info["num_path"]),indices,cfg);loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda");vals={}
  for name in VARIANTS:
   print(f"[infer] {split}/{name} samples={len(ds)}",flush=True);model=build(split,name,a.seed,cfg,cls,device);vals[name]=infer(model,loader,device);del model
   if device.type=="cuda":torch.cuda.empty_cache()
   if a.max_samples is None:verify(run_dir(split,name,a.seed)/"metrics.json",vals[name])
  out=OUTPUT/split;out.mkdir(parents=True,exist_ok=True);results.append(analyse(split,vals,out))
 (OUTPUT/"combined_summary.json").write_text(json.dumps({"strict_pairing":True,"results":results},indent=2,ensure_ascii=False),encoding="utf-8");print(f"[done] {OUTPUT}")
if __name__=="__main__":main()
