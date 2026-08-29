"""Strict paired CFB1 diagnostic after the audit-gated training stage."""
from __future__ import annotations
import argparse,importlib.util,json,os
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parent;W=ROOT.parent;P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");CSRS=W/"COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES";IR1=W/"INTERACTION_RESIDUAL_SUPERVISION_SERIES";CSO=W/"CANDIDATE_SET_OPTIMIZATION_SERIES";OUT=ROOT/"diagnostic";KS=(1,2,5,10,20);SPLITS=("general","challenge")
def load(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def source(split,seed):return IR1/"results"/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"
def run(split,name,seed):
 if name=="baseline":return CSRS/"results"/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
 return ROOT/"results"/split/"cfb1_cross_scene_three_future"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def build(diag,split,name,seed,cfg,cls,device):
 if name=="baseline":return diag.build(split,"baseline",seed,cfg,cls,device)
 model=cls(cfg,"f16_temporal_social_cross");os.environ["CFB_SOURCE_CHECKPOINT"]=str(source(split,seed));os.environ["CFB_CACHE"]=str(ROOT/"cache"/f"{split}_seed_{seed}.npz");module=load(f"cfb_diag_{split}",ROOT/"cfb_models.py");module.replace_decoder(model,cfg,"cfb1_cross_scene_three_future");model.load_state_dict(torch.load(run(split,name,seed)/"best_selected.pth",map_location="cpu",weights_only=True),strict=True);return model.to(device).eval()
def subset(v,mask):return {k:float(v[k][mask].mean()) for k in ("min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde")}
def curve(v,mask):
 err=np.linalg.norm(v["prediction"]-v["target"][:,None],axis=-1);ade,fde=err.mean(-1),err[...,-1];order=np.argsort(-v["score"],axis=1);rows=np.arange(len(order))[:,None];result={}
 for k in KS:
  chosen=order[:,:k];result[str(k)]={"ade":float(ade[rows,chosen][mask].min(1).mean()),"fde":float(fde[rows,chosen][mask].min(1).mean())}
 return result
def diversity(v):
 i,j=np.triu_indices(20,1);p,a=v["prediction"],v["anchor"];return {"endpoint":float(np.linalg.norm(p[:,i,-1]-p[:,j,-1],axis=-1).mean()),"trajectory":float(np.linalg.norm(p[:,i]-p[:,j],axis=-1).mean()),"anchor":float(np.linalg.norm(a[:,i]-a[:,j],axis=-1).mean())}
def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=128);p.add_argument("--seed",type=int,default=1);a=p.parse_args();device=torch.device(a.device);diag=load("cfb_cso_diag",CSO/"run_cso1_diagnostic.py");cfg,cls=diag.prepare(device);geom=load("cfb_geom_diag",W/"INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES"/"run_geometry_diagnostic.py");results=[];OUT.mkdir(parents=True,exist_ok=True)
 for split in SPLITS:
  base_run=run(split,"baseline",a.seed);indices=np.asarray(np.load(base_run/"split_indices.npz")["validation"],np.int64);info=json.loads((base_run/"split.json").read_text(encoding="utf-8"));ds=geom.IndexedDataset(Path(info["data_path"]),Path(info["num_path"]),indices,cfg);loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda");vals={}
  for name in ("baseline","cfb1"):
   print(f"[infer] {split}/{name} samples={len(ds)}",flush=True);model=build(diag,split,name,a.seed,cfg,cls,device);vals[name]=diag.infer(model,loader,device);state=model.state_dict() if name=="cfb1" else None;del model
   if device.type=="cuda":torch.cuda.empty_cache()
  labels=np.asarray(np.load(CSRS/"teacher_cache"/f"{split}_seed_{a.seed}.npz",allow_pickle=False)["need_label"])[indices].astype(bool);allmask=np.ones(len(indices),bool);models={}
  for name,v in vals.items():models[name]={"overall":subset(v,allmask),"social_needed":subset(v,labels),"coverage_at_k":curve(v,allmask),"social_needed_at_k":curve(v,labels),"diversity":diversity(v)}
  counts=state["decoder.pseudo_assignment_counts"].cpu().numpy();total=int(state["decoder.pseudo_assignment_total"]);assignment={"total":total,"used_modes":int((counts>0).sum()),"never_pseudo_modes":int((counts==0).sum()),"frequency":(counts/max(total,1)).tolist()}
  result={"split":split,"samples":len(indices),"models":models,"pseudo_assignment":assignment};out=OUT/split;out.mkdir(parents=True,exist_ok=True);(out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8");results.append(result)
 (OUT/"combined_summary.json").write_text(json.dumps({"strict_pairing":True,"results":results},indent=2,ensure_ascii=False),encoding="utf-8");print(f"[done] {OUT}")
if __name__=="__main__":main()
