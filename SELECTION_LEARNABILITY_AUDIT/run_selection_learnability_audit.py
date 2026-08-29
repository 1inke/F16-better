"""EXP-150A: diagnose whether S0's frozen K=20 winner is learnable."""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parent; WS=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"); FRONT=WS/"FRONT_ENCODER_SERIES"
DATA=WS/"SDD_STANDARD_POTENTIAL_SERIES"/"processed_data"; ASSET=WS/"SCENE_QUERY_CURRENT_DECODER_SDD_SERIES"/"processed_scene_assets"
DECODER=WS/"SCENE_QUERY_CURRENT_DECODER_SDD_SERIES"/"current_scene_query_models.py"

def mod(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def rank(x):
 o=np.argsort(x,axis=-1,kind="stable"); r=np.empty_like(o,dtype=np.float32); r[np.arange(len(x))[:,None],o]=np.arange(x.shape[1]); return r
def corr(score,quality):
 rs, rq=rank(score),rank(quality); return float(np.mean([np.corrcoef(a,b)[0,1] if np.std(a)*np.std(b)>0 else 0. for a,b in zip(rs,rq)]))
def describe(ade,fde,q,logits,mask):
 ade,fde,q,logits=ade[mask],fde[mask],q[mask],logits[mask]; n=len(ade); row=np.arange(n); order=np.argsort(-logits,axis=1,kind="stable")
 out={"samples":int(n),"score_vs_negative_combined_cost_spearman":corr(logits,-q),"best_ADE_hit":float((logits.argmax(1)==ade.argmin(1)).mean()),"best_FDE_hit":float((logits.argmax(1)==fde.argmin(1)).mean()),"combined_winner_hit":float((logits.argmax(1)==q.argmin(1)).mean()),"top_scored_mode_usage":np.bincount(logits.argmax(1),minlength=20).tolist()}
 for m in (1,2,3,5,10,20):
  ids=order[:,:m]; out[f"top{m}"]={"minADE_px":float(np.take_along_axis(ade,ids,1).min(1).mean()*20),"minFDE_px":float(np.take_along_axis(fde,ids,1).min(1).mean()*20),"combined_winner_recall":float(np.any(ids==q.argmin(1)[:,None],1).mean())}
 regret_ade=ade[row,logits.argmax(1)]-ade.min(1); regret_fde=fde[row,logits.argmax(1)]-fde.min(1)
 out["regret_px"]={k:float(np.quantile(regret_ade*20,v)) for k,v in {"mean":.5,"p75":.75,"p90":.9,"p95":.95}.items()}; out["regret_px"]["mean"]=float(regret_ade.mean()*20)
 out["fde_regret_px"]={"mean":float(regret_fde.mean()*20),"p95":float(np.quantile(regret_fde*20,.95))}; return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--seeds",default="1,2,3");p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=128);p.add_argument("--force",action="store_true");a=p.parse_args()
 outdir=ROOT/"EXP_150A_S0_SELECTION_LEARNABILITY"; output=outdir/"audit_summary.json"
 if output.is_file() and not a.force: print(f"[skip] {output}");return
 os.chdir(PROJECT);sys.path[:0]=[str(FRONT),str(PROJECT)]
 from experiment_config import load_locked_q20
 from front_encoder_models import FrontEncoderASTRA
 helper=mod("exp150_helper",FRONT/"train_front_variant.py"); dec=mod("exp150_dec",DECODER)
 cfg=load_locked_q20(PROJECT/"configs"/"motif_rbf_univ.yaml");cfg.MODEL.USE_PRETRAINED_UNET=False;cfg.MODEL.USE_CACHED_UNET_FEATURES=False;cfg.DATA.MAX_AGENTS=16;cfg.MODEL.GRAPH.TOP_K=4;cfg.MODEL.GRAPH.RADIUS=6.0
 device=torch.device(a.device if torch.cuda.is_available() else "cpu");cfg.device=str(device);cfg.device_list=[device.index or 0] if device.type=="cuda" else ["cpu"]
 ds=helper.TrajectoryDataset(DATA/"sdd_standard_test_data.npy",DATA/"sdd_standard_test_num.npy",cfg,ASSET/"sdd_test_scene_ids.npy"); loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=0); scenes=np.load(ASSET/"sdd_test_scene_ids.npy")
 meta=json.loads((ASSET/"scene_asset_manifest.json").read_text()); names={int(v):k for k,v in meta["video_to_scene_id"].items()}; little=np.asarray([names[int(x)].startswith("little/") for x in scenes])
 report={"experiment":"EXP-150A frozen S0 selection learnability audit","combined_cost":"ADE + 0.25*FDE (same current coverage protocol)","seeds":{}}
 for seed in [int(x) for x in a.seeds.split(",")]:
  ckpt=WS/"SCENE_QUERY_CURRENT_DECODER_SDD_SERIES"/"EXP_147B_CURRENT_DECODER_SDD_SCENE_QUERY"/"results"/"s0_current"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_1"/"best_selected.pth"
  if not ckpt.is_file():raise FileNotFoundError(ckpt)
  model=FrontEncoderASTRA(cfg,"f16_temporal_social_cross");dec.replace_decoder(model,cfg,"s0_current");model.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=True),strict=True);model=model.to(device).eval(); A=[];F=[];Q=[];L=[]
  with torch.inference_mode():
   for past,future,num,scene,_,_ in loader:
    past,future,num=past.to(device),future.to(device),num.to(device).int();past,future=helper.normalize_to_focal_heading(past,future);center=past.mean(-2,keepdim=True);pred=model(past-center,None,None,mode="testing",num_valid=num,graph_loc=past)[2][:,0]+center[:,0,None];err=torch.linalg.vector_norm(pred-future[:,0,None],dim=-1); ade=err.mean(-1);fde=err[...,-1];A.append(ade.cpu().numpy());F.append(fde.cpu().numpy());Q.append((ade+.25*fde).cpu().numpy());L.append(model.last_mode_logits[:,0].cpu().numpy())
  ade,fde,q,logits=map(np.concatenate,(A,F,Q,L)); report["seeds"][str(seed)]={"overall":describe(ade,fde,q,logits,np.ones(len(ade),bool)),"little":describe(ade,fde,q,logits,little),"non_little":describe(ade,fde,q,logits,~little)}
 outdir.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8");print("[done] "+str(output))
if __name__=="__main__":main()
