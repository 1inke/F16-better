"""Strict paired B0/MA1 mode-allocation and candidate-quality diagnostic."""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parent; WORKSPACE=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"); FRONT=WORKSPACE/"FRONT_ENCODER_SERIES"
CSRS=WORKSPACE/"COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"; IR1=WORKSPACE/"INTERACTION_RESIDUAL_SUPERVISION_SERIES"
RESULTS=ROOT/"results"; OUTPUT=ROOT/"diagnostic"; SPLITS=("general","challenge")
METRICS=("min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde"); STEPS=(4,8,12)

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None: raise ImportError(path)
 module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
def source(split,seed): return IR1/"results"/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"
def baseline_dir(split,seed): return CSRS/"results"/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def ma_dir(split,seed): return RESULTS/split/"ma1_interaction_aware_top5"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def cache(split,seed): return CSRS/"teacher_cache"/f"{split}_seed_{seed}.npz"

def prepare(device):
 sys.path[:0]=[str(PROJECT),str(PROJECT/"scripts"),str(FRONT)]; os.chdir(PROJECT); import sitecustomize  # noqa
 from experiment_config import load_locked_q20
 from front_encoder_models import FrontEncoderASTRA
 cfg=load_locked_q20(PROJECT/"configs"/"motif_rbf_univ.yaml"); cfg.MODEL.USE_PRETRAINED_UNET=False; cfg.MODEL.USE_CACHED_UNET_FEATURES=False
 cfg.DATA.MAX_AGENTS=9; cfg.MODEL.GRAPH.TOP_K=4; cfg.MODEL.GRAPH.RADIUS=6.; cfg.MODEL.DECODER.NUM_MODES=20
 cfg.device=str(device); cfg.device_list=[device.index or 0] if device.type=="cuda" else ["cpu"]
 return cfg,FrontEncoderASTRA

def build(split,name,seed,cfg,cls,device):
 model=cls(cfg,"f16_temporal_social_cross"); src=source(split,seed)
 if name=="baseline":
  os.environ["CSRS_SOURCE_CHECKPOINT"]=str(src); module=load(f"ma_diag_b0_{split}",CSRS/"current_baseline_adapter.py")
  module.replace_decoder(model,cfg,"current_mode_anchor_template"); checkpoint=baseline_dir(split,seed)/"best_selected.pth"
 else:
  os.environ["IAMA_SOURCE_CHECKPOINT"]=str(src); module=load(f"ma_diag_ma1_{split}",ROOT/"iama_models.py")
  module.replace_decoder(model,cfg,"ma1_interaction_aware_top5"); checkpoint=ma_dir(split,seed)/"best_selected.pth"
 model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
 return model.to(device).eval()

@torch.inference_mode()
def infer(model,loader,device,is_ma):
 from train_front_variant import normalize_to_focal_heading
 keys=("index","scene","prediction","anchor","target","score",*METRICS); out={k:[] for k in keys}; compat=[]
 for past,future,num,indices,scenes in loader:
  past=past.to(device); future=future.to(device); num=num.to(device).int(); past,future=normalize_to_focal_heading(past,future)
  center=past.mean(dim=-2,keepdim=True); pred=model(past-center,None,None,mode="testing",num_valid=num,graph_loc=past)[2]+center.unsqueeze(-3)
  candidate,target=pred[:,0],future[:,0]; error=torch.linalg.vector_norm(candidate-target[:,None],dim=-1); ade=error.mean(-1); fde=error[...,-1]
  row=torch.arange(len(ade),device=device); winner=ade.argmin(-1); top1=model.last_mode_logits[:,0].argmax(-1)
  values={"index":indices.numpy(),"scene":scenes.numpy(),"prediction":candidate.cpu().numpy(),"anchor":model.decoder.last_corrected_anchors[:,0].cpu().numpy(),
          "target":target.cpu().numpy(),"score":model.last_mode_logits[:,0].cpu().numpy(),"min_ade_20":ade.min(-1).values.cpu().numpy(),
          "min_fde_20":fde.min(-1).values.cpu().numpy(),"same_path_fde":fde[row,winner].cpu().numpy(),
          "top1_ade":ade[row,top1].cpu().numpy(),"top1_fde":fde[row,top1].cpu().numpy()}
  for key,value in values.items(): out[key].append(value)
  if is_ma: compat.append(model.decoder.last_compatibility_logits[:,0].cpu().numpy())
 result={key:np.concatenate(value) for key,value in out.items()}
 if is_ma: result["compatibility"]=np.concatenate(compat)
 return result

def softmax(x):
 x=x-x.max(-1,keepdims=True); y=np.exp(x); return y/y.sum(-1,keepdims=True)
def entropy(p): return -(p*np.log(np.maximum(p,1e-12))).sum(-1)
def js(p,q):
 p=p/np.maximum(p.sum(),1e-12); q=q/np.maximum(q.sum(),1e-12); m=.5*(p+q)
 return float(.5*(p*np.log(np.maximum(p,1e-12)/np.maximum(m,1e-12))).sum()+.5*(q*np.log(np.maximum(q,1e-12)/np.maximum(m,1e-12))).sum())
def summary(v,mask): return {"samples":int(mask.sum()),**{m:float(v[m][mask].mean()) for m in METRICS}}
def changes(b,m): return {k:100*(m[k]/b[k]-1) for k in METRICS}
def diversity(p,a):
 i,j=np.triu_indices(p.shape[1],1)
 return {"endpoint":float(np.linalg.norm(p[:,i,-1]-p[:,j,-1],axis=-1).mean()),
         "trajectory":float(np.linalg.norm(p[:,i]-p[:,j],axis=-1).mean()),
         "anchor":float(np.linalg.norm(a[:,i]-a[:,j],axis=-1).mean())}
def winner_usage(v):
 error=np.linalg.norm(v["prediction"]-v["target"][:,None],axis=-1); cost=error.mean(-1)+.25*error[...,-1]; winner=cost.argmin(-1)
 count=np.bincount(winner,minlength=20); p=count/count.sum()
 return winner,{"entropy":float(entropy(p[None])[0]),"normalized_entropy":float(entropy(p[None])[0]/np.log(20)),
                "modes_over_1pct":int((p>.01).sum()),"modes_over_5pct":int((p>.05).sum()),"never_win_modes":int((count==0).sum()),
                "frequency":p.tolist()}

def analyse(split,b0,ma,indices,out,seed):
 labels=np.asarray(np.load(cache(split,seed),allow_pickle=False)["need_label"])[indices].astype(bool); all_mask=np.ones(len(labels),bool)
 b_winner,b_usage=winner_usage(b0); m_winner,m_usage=winner_usage(ma)
 error=np.linalg.norm(ma["prediction"]-ma["target"][:,None],axis=-1); cost=error.mean(-1)+.25*error[...,-1]+.25*error[:,:,[3,7,11]].mean(-1)
 gap=cost-cost.min(-1,keepdims=True); scale=np.maximum(gap.mean(-1,keepdims=True),.05); geometry_logits=-gap/(.5*scale)
 comp=ma["compatibility"]; comp_scaled=(comp-comp.mean(-1,keepdims=True))/np.maximum(comp.std(-1,keepdims=True),1.)
 alloc_logits=geometry_logits+.25*comp_scaled; top=np.argpartition(alloc_logits,-5,axis=1)[:,-5:]
 mask=np.zeros_like(alloc_logits,bool); mask[np.arange(len(mask))[:,None],top]=True; responsibility=softmax(np.where(mask,alloc_logits,-1e4))
 predicted=softmax(ma["score"]); oracle=cost.argmin(-1)
 assignment={"predicted_top1_accuracy":float((predicted.argmax(-1)==oracle).mean()),"mean_top1_mass":float(predicted.max(-1).mean()),
             "mean_top3_mass":float(np.sort(predicted,axis=1)[:,-3:].sum(-1).mean()),"predicted_entropy":float(entropy(predicted).mean()),
             "target_entropy":float(entropy(responsibility).mean()),"target_to_predicted_kl":float((responsibility*(np.log(np.maximum(responsibility,1e-12))-np.log(np.maximum(predicted,1e-12)))).sum(-1).mean())}
 pos_resp=responsibility[labels].mean(0); neg_resp=responsibility[~labels].mean(0)
 social={"needed_samples":int(labels.sum()),"non_needed_samples":int((~labels).sum()),"responsibility_js_divergence":js(pos_resp,neg_resp),
         "winner_js_divergence":js(np.bincount(m_winner[labels],minlength=20),np.bincount(m_winner[~labels],minlength=20)),
         "needed_responsibility":pos_resp.tolist(),"non_needed_responsibility":neg_resp.tolist()}
 target_anchor=ma["target"][:,[3,7,11]]; coverage={}
 for name,v in (("baseline",b0),("ma1",ma)):
  ae=np.linalg.norm(v["anchor"]-target_anchor[:,None],axis=-1); coverage[name]={f"step_{s}":float(ae[:,:,i].min(1).mean()) for i,s in enumerate(STEPS)}
  coverage[name]["endpoint"]=float(np.linalg.norm(v["prediction"][:,:,-1]-v["target"][:,None,-1],axis=-1).min(1).mean())
 coverage["change_pct"]={k:100*(coverage["ma1"][k]/coverage["baseline"][k]-1) for k in coverage["baseline"]}
 db,dm=diversity(b0["prediction"],b0["anchor"]),diversity(ma["prediction"],ma["anchor"])
 div={k:{"baseline":db[k],"ma1":dm[k],"change_pct":100*(dm[k]/db[k]-1)} for k in db}
 overall_b,overall_m=summary(b0,all_mask),summary(ma,all_mask); pos_b,pos_m=summary(b0,labels),summary(ma,labels)
 result={"split":split,"samples":len(labels),"overall":{"baseline":overall_b,"ma1":overall_m,"change_pct":changes(overall_b,overall_m)},
         "actual_social_needed":{"baseline":pos_b,"ma1":pos_m,"change_pct":changes(pos_b,pos_m)},
         "winner_usage":{"baseline":b_usage,"ma1":m_usage},"assignment":assignment,"social_needed_shift":social,
         "anchor_coverage":coverage,"candidate_diversity":div}
 (out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8"); return result

def verify(path,v):
 raw=json.loads(path.read_text(encoding="utf-8"))["evaluation_best_selected"]
 expected={"min_ade_20":raw["oracle"]["min_ade_20"],"min_fde_20":raw["oracle"]["min_fde_20"],"same_path_fde":raw["oracle"]["same_path_fde"],"top1_ade":raw["original"]["ade"],"top1_fde":raw["original"]["fde"]}
 for key,value in expected.items():
  if abs(float(v[key].mean())-float(value))>2e-5: raise RuntimeError(f"reproduction mismatch: {key}")

def main():
 parser=argparse.ArgumentParser(); parser.add_argument("--device",default="cuda:0"); parser.add_argument("--batch-size",type=int,default=128)
 parser.add_argument("--seed",type=int,default=1); parser.add_argument("--splits",nargs="+",choices=("all",*SPLITS),default=["all"]); parser.add_argument("--max-samples",type=int)
 args=parser.parse_args(); device=torch.device(args.device); splits=list(SPLITS) if "all" in args.splits else args.splits
 OUTPUT.mkdir(parents=True,exist_ok=True); cfg,cls=prepare(device); geom=load("ma_geometry",WORKSPACE/"INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES"/"run_geometry_diagnostic.py"); results=[]
 for split in splits:
  run=baseline_dir(split,args.seed); indices=np.asarray(np.load(run/"split_indices.npz")["validation"],np.int64)
  if args.max_samples is not None: indices=indices[:args.max_samples]
  info=json.loads((run/"split.json").read_text(encoding="utf-8")); ds=geom.IndexedDataset(Path(info["data_path"]),Path(info["num_path"]),indices,cfg)
  loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda"); values={}
  for name in ("baseline","ma1"):
   print(f"[infer] {split}/{name} samples={len(ds)}",flush=True); model=build(split,name,args.seed,cfg,cls,device); values[name]=infer(model,loader,device,name=="ma1"); del model
   if device.type=="cuda": torch.cuda.empty_cache()
  if args.max_samples is None: verify(run/"metrics.json",values["baseline"]); verify(ma_dir(split,args.seed)/"metrics.json",values["ma1"])
  out=OUTPUT/split; out.mkdir(parents=True,exist_ok=True); results.append(analyse(split,values["baseline"],values["ma1"],indices,out,args.seed))
 (OUTPUT/"combined_summary.json").write_text(json.dumps({"strict_pairing":True,"results":results},indent=2,ensure_ascii=False),encoding="utf-8"); print(f"[done] {OUTPUT}")

if __name__=="__main__": main()
