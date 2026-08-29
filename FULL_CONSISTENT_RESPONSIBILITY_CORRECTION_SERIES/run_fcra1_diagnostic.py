"""Strict B0/FCRA1 mechanism diagnostic."""
from __future__ import annotations
import argparse,csv,importlib.util,json,os,sys
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parent;WORKSPACE=ROOT.parent;PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");FRONT=WORKSPACE/"FRONT_ENCODER_SERIES"
CSRS=WORKSPACE/"COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES";IR1=WORKSPACE/"INTERACTION_RESIDUAL_SUPERVISION_SERIES";RESULTS=ROOT/"results";OUTPUT=ROOT/"diagnostic"
SPLITS=("general","challenge");METRICS=("min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde");STEPS=(4,8,12);TOP_M=3;TEMP=.10

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path)
 if s is None or s.loader is None:raise ImportError(path)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def source(split,seed):return IR1/"results"/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"
def cache_path(split,seed):return CSRS/"teacher_cache"/f"{split}_seed_{seed}.npz"
def baseline_dir(split,seed):return CSRS/"results"/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def fcra_dir(split,seed):return RESULTS/split/"fcra1_full_consistent_top3"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"

def prepare(device):
 sys.path[:0]=[str(PROJECT),str(PROJECT/"scripts"),str(FRONT)];os.chdir(PROJECT);import sitecustomize  # noqa
 from experiment_config import load_locked_q20
 from front_encoder_models import FrontEncoderASTRA
 c=load_locked_q20(PROJECT/"configs"/"motif_rbf_univ.yaml");c.MODEL.USE_PRETRAINED_UNET=False;c.MODEL.USE_CACHED_UNET_FEATURES=False;c.DATA.MAX_AGENTS=9;c.MODEL.GRAPH.TOP_K=4;c.MODEL.GRAPH.RADIUS=6.;c.MODEL.DECODER.NUM_MODES=20;c.device=str(device);c.device_list=[device.index or 0] if device.type=="cuda" else ["cpu"]
 return c,FrontEncoderASTRA

def build(split,name,seed,cfg,cls,device):
 model=cls(cfg,"f16_temporal_social_cross");src=source(split,seed)
 if name=="baseline":
  os.environ["CSRS_SOURCE_CHECKPOINT"]=str(src);module=load(f"fcra_diag_b0_{split}",CSRS/"current_baseline_adapter.py");module.replace_decoder(model,cfg,"current_mode_anchor_template");checkpoint=baseline_dir(split,seed)/"best_selected.pth"
 else:
  os.environ["FCRA_SOURCE_CHECKPOINT"]=str(src);os.environ["FCRA_NEED_CACHE"]=str(cache_path(split,seed));module=load(f"fcra_diag_model_{split}",ROOT/"fcra_models.py");module.replace_decoder(model,cfg,"fcra1_full_consistent_top3");checkpoint=fcra_dir(split,seed)/"best_selected.pth"
 model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True);return model.to(device).eval()

@torch.inference_mode()
def infer(model,loader,device,is_fcra):
 from train_front_variant import normalize_to_focal_heading
 keys=("index","scene","prediction","anchor","target",*METRICS);out={k:[] for k in keys};mech={k:[] for k in ("need","assign_logits","responsibility","residual","applied","uncorrected_prediction","uncorrected_anchor")}
 for past,future,num,indices,scenes in loader:
  past=past.to(device);future=future.to(device);num=num.to(device).int()
  if is_fcra:model.decoder.set_scene_batch(-(indices.to(device).long()+2),past)
  past,future=normalize_to_focal_heading(past,future);center=past.mean(dim=-2,keepdim=True);pred=model(past-center,None,None,mode="testing",num_valid=num,graph_loc=past)[2]+center.unsqueeze(-3)
  cand,target=pred[:,0],future[:,0];err=torch.linalg.vector_norm(cand-target[:,None],dim=-1);ade,fde=err.mean(-1),err[...,-1];rows=torch.arange(len(ade),device=device);winner=ade.argmin(-1);top1=model.last_mode_logits[:,0].argmax(-1)
  values={"index":indices.numpy(),"scene":scenes.numpy(),"prediction":cand.cpu().numpy(),"anchor":model.decoder.last_corrected_anchors[:,0].cpu().numpy(),"target":target.cpu().numpy(),
          "min_ade_20":ade.min(-1).values.cpu().numpy(),"min_fde_20":fde.min(-1).values.cpu().numpy(),"same_path_fde":fde[rows,winner].cpu().numpy(),"top1_ade":ade[rows,top1].cpu().numpy(),"top1_fde":fde[rows,top1].cpu().numpy()}
  for k,v in values.items():out[k].append(v)
  if is_fcra:
   mm={"need":model.decoder.last_need_probability[:,0].cpu().numpy(),"assign_logits":model.decoder.last_assignment_logits[:,0].cpu().numpy(),"responsibility":model.decoder.last_sparse_responsibility[:,0].cpu().numpy(),
       "residual":model.decoder.last_predicted_residual[:,0].cpu().numpy(),"applied":model.decoder.last_applied_anchor_delta[:,0].cpu().numpy(),
       "uncorrected_prediction":(model.decoder.last_uncorrected_trajectories_graph[:,0]+center[:,0,None]).cpu().numpy(),
       "uncorrected_anchor":model.decoder.last_uncorrected_anchors_graph[:,0].cpu().numpy()}
   for k,v in mm.items():mech[k].append(v)
 result={k:np.concatenate(v) for k,v in out.items()}
 if is_fcra:result.update({k:np.concatenate(v) for k,v in mech.items()})
 return result

def summary(v,mask):return {"samples":int(mask.sum()),**{m:float(v[m][mask].mean()) for m in METRICS}}
def changes(b,f):return {m:100*(f[m]/b[m]-1) for m in METRICS}
def rho(a,b):
 mask=np.isfinite(a)&np.isfinite(b);v=spearmanr(a[mask],b[mask]).statistic if mask.sum()>2 else np.nan;return float(v) if np.isfinite(v) else None
def describe(x):
 x=np.asarray(x);x=x[np.isfinite(x)];return {"mean":float(x.mean()),"median":float(np.median(x)),"p10":float(np.quantile(x,.1)),"p90":float(np.quantile(x,.9)),"samples":len(x)}
def diversity(p,a):
 i,j=np.triu_indices(p.shape[1],1);return {"endpoint":np.linalg.norm(p[:,i,-1]-p[:,j,-1],axis=-1).mean(-1),"trajectory":np.linalg.norm(p[:,i]-p[:,j],axis=-1).mean((-1,-2)),"anchor":np.linalg.norm(a[:,i]-a[:,j],axis=-1).mean((-1,-2))}
def cos(a,b):
 # Flatten the anchor/time and coordinate axes so this returns one cosine
 # value per candidate rather than one value per anchor.
 a=a.reshape(*a.shape[:-2],-1);b=b.reshape(*b.shape[:-2],-1)
 d=(a*b).sum(-1);n=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1);o=np.full_like(d,np.nan,dtype=np.float32);v=n>1e-8;o[v]=d[v]/n[v];return o
def write(path,rows):
 with path.open("w",newline="",encoding="utf-8-sig") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyse(split,b0,f,cache,indices,out):
 label=np.asarray(cache["need_label"])[indices].astype(bool);need=f["need"];unc=f["uncorrected_prediction"];target=f["target"]
 err=np.linalg.norm(unc-target[:,None],axis=-1);cost=err.mean(-1)+.5*err[...,-1];oracle=cost.argmin(-1)
 target_logits=-cost/TEMP;top=np.argpartition(target_logits,-TOP_M,axis=1)[:,-TOP_M:];mask=np.zeros_like(target_logits,bool);mask[np.arange(len(mask))[:,None],top]=True
 target_logits=np.where(mask,target_logits,-1e4);mx=target_logits.max(1,keepdims=True);tp=np.exp(target_logits-mx);tp/=tp.sum(1,keepdims=True)
 pred_logits=f["assign_logits"];pred=np.exp(pred_logits-pred_logits.max(1,keepdims=True));pred/=pred.sum(1,keepdims=True);pred_top=np.argpartition(pred,-TOP_M,axis=1)[:,-TOP_M:]
 assigned_top1=pred.argmax(-1);top3_hit=(pred_top==oracle[:,None]).any(1);ce=-(tp*np.log(np.maximum(pred,1e-12))).sum(1);entropy=-(f["responsibility"]*np.log(np.maximum(f["responsibility"],1e-12))).sum(1)
 applied_norm=np.linalg.norm(f["applied"],axis=-1).mean((-1,-2));target_anchor=target[:,[3,7,11]];res_target=target_anchor[:,None]-f["uncorrected_anchor"]
 rc=cos(f["residual"],res_target);weighted_cos=(tp*np.nan_to_num(rc,nan=0.)).sum(1);delta_ade=f["min_ade_20"]-b0["min_ade_20"];delta_fde=f["min_fde_20"]-b0["min_fde_20"]
 assignment={"top1_accuracy":float((assigned_top1==oracle).mean()),"top3_recall":float(top3_hit.mean()),"random_top1":.05,"random_top3":.15,"soft_cross_entropy":float(ce.mean()),"sparse_gate_entropy":describe(entropy)}
 mechanism={"need_auroc":float(roc_auc_score(label,need)),"need_auprc":float(average_precision_score(label,need)),"need_probability_vs_label_gain":rho(need,np.asarray(cache["social_gain"])[indices]),
            "correction_positive":describe(applied_norm[label]),"correction_negative":describe(applied_norm[~label]),"positive_over_negative_ratio":float(applied_norm[label].mean()/max(applied_norm[~label].mean(),1e-9)),
            "assigned_top3_residual_cosine_positive":describe(rc[label&np.ones(len(label),bool)][mask[label]].reshape(-1)) if label.any() else None,
            "weighted_alignment_vs_delta_minade_positive":rho(weighted_cos[label],delta_ade[label]),"weighted_alignment_vs_delta_minfde_positive":rho(weighted_cos[label],delta_fde[label])}
 allmask=np.ones(len(label),bool);overall_b,overall_f=summary(b0,allmask),summary(f,allmask);pos_b,pos_f=summary(b0,label),summary(f,label)
 cov={}
 for name,v in (("baseline",b0),("fcra",f)):
  ae=np.linalg.norm(v["anchor"]-target_anchor[:,None],axis=-1);cov[name]={f"step_{s}":float(ae[:,:,i].min(1).mean()) for i,s in enumerate(STEPS)};cov[name]["endpoint"]=float(np.linalg.norm(v["prediction"][:,:,-1]-target[:,None,-1],axis=-1).min(1).mean())
 cov["change_pct"]={k:100*(cov["fcra"][k]/cov["baseline"][k]-1) for k in cov["baseline"]}
 db,df=diversity(b0["prediction"],b0["anchor"]),diversity(f["prediction"],f["anchor"]);div={k:{"baseline":float(db[k].mean()),"fcra":float(df[k].mean()),"change_pct":100*(df[k].mean()/db[k].mean()-1)} for k in db}
 rows=[]
 for i,index in enumerate(indices):rows.append({"index":int(index),"scene":int(b0["scene"][i]),"need":int(label[i]),"need_probability":float(need[i]),"oracle_mode":int(oracle[i]),"predicted_mode":int(assigned_top1[i]),"top3_hit":bool(top3_hit[i]),"assignment_ce":float(ce[i]),"correction_norm":float(applied_norm[i]),"weighted_residual_cosine":float(weighted_cos[i]),"delta_minade":float(delta_ade[i]),"delta_minfde":float(delta_fde[i])})
 write(out/"paired_samples.csv",rows)
 result={"split":split,"samples":len(indices),"assignment":assignment,"mechanism":mechanism,"overall":{"baseline":overall_b,"fcra":overall_f,"change_pct":changes(overall_b,overall_f)},
         "actual_social_needed":{"baseline":pos_b,"fcra":pos_f,"change_pct":changes(pos_b,pos_f)},"anchor_coverage":cov,"candidate_diversity":div}
 (out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8");return result

def verify(path,v):
 raw=json.loads(path.read_text(encoding="utf-8"))["evaluation_best_selected"];e={"min_ade_20":raw["oracle"]["min_ade_20"],"min_fde_20":raw["oracle"]["min_fde_20"],"same_path_fde":raw["oracle"]["same_path_fde"],"top1_ade":raw["original"]["ade"],"top1_fde":raw["original"]["fde"]}
 for m,x in e.items():
  if abs(float(v[m].mean())-float(x))>2e-5:raise RuntimeError(f"reproduction {m}")

def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=128);p.add_argument("--seed",type=int,default=1);p.add_argument("--splits",nargs="+",choices=("all",*SPLITS),default=["all"]);p.add_argument("--max-samples",type=int,default=None);a=p.parse_args()
 device=torch.device(a.device);splits=list(SPLITS) if "all" in a.splits else a.splits;OUTPUT.mkdir(parents=True,exist_ok=True);cfg,cls=prepare(device);g=load("fcra_geom",WORKSPACE/"INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES"/"run_geometry_diagnostic.py");results=[]
 for split in splits:
  run=baseline_dir(split,a.seed);sp=np.load(run/"split_indices.npz");indices=np.asarray(sp["validation"],np.int64)
  if a.max_samples is not None:indices=indices[:a.max_samples]
  info=json.loads((run/"split.json").read_text());ds=g.IndexedDataset(Path(info["data_path"]),Path(info["num_path"]),indices,cfg);loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda")
  vals={}
  for name in ("baseline","fcra"):
   print(f"[infer] {split}/{name} samples={len(ds)}",flush=True);model=build(split,name,a.seed,cfg,cls,device);vals[name]=infer(model,loader,device,name=="fcra");del model
   if device.type=="cuda":torch.cuda.empty_cache()
  if a.max_samples is None:verify(run/"metrics.json",vals["baseline"]);verify(fcra_dir(split,a.seed)/"metrics.json",vals["fcra"])
  out=OUTPUT/split;out.mkdir(parents=True,exist_ok=True);results.append(analyse(split,vals["baseline"],vals["fcra"],np.load(cache_path(split,a.seed),allow_pickle=False),indices,out))
 (OUTPUT/"combined_summary.json").write_text(json.dumps({"strict_pairing":True,"results":results},indent=2,ensure_ascii=False),encoding="utf-8");print(f"[done] {OUTPUT}")
if __name__=="__main__":main()
