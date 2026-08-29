"""Build leakage-safe cross-scene future banks and run the no-training oracle audit."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np,torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parent;WORKSPACE=ROOT.parent;PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");FRONT=WORKSPACE/"FRONT_ENCODER_SERIES"
DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse";CSRS=WORKSPACE/"COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES";IR1=WORKSPACE/"INTERACTION_RESIDUAL_SUPERVISION_SERIES"
CSO=WORKSPACE/"CANDIDATE_SET_OPTIMIZATION_SERIES";CACHE=ROOT/"cache";AUDIT=ROOT/"audit";SPLITS=("general","challenge");BANK_K=20;SELECT_M=3
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path)
 if s is None or s.loader is None:raise ImportError(path)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def baseline_dir(split,seed):return CSRS/"results"/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def digest(a):return hashlib.sha256(np.asarray(a,dtype=np.int64).tobytes()).hexdigest()
def local_data(raw,num):
 past=np.asarray(raw[:,:,0:8,-2:],np.float32);future=np.asarray(raw[:,0,8:20,-2:],np.float32);focal=past[:,0];disp=focal[:,-1]-focal[:,-4];speed=np.linalg.norm(disp,axis=-1);heading=np.zeros_like(disp);heading[:,0]=1.;valid=speed>.03;heading[valid]=disp[valid]/speed[valid,None]
 c,s=heading[:,0],heading[:,1];center=focal[:,-1];delta_p=past-center[:,None,None];past_local=np.empty_like(delta_p);past_local[...,0]=delta_p[...,0]*c[:,None,None]+delta_p[...,1]*s[:,None,None];past_local[...,1]=-delta_p[...,0]*s[:,None,None]+delta_p[...,1]*c[:,None,None]
 delta_f=future-center[:,None];future_local=np.empty_like(delta_f);future_local[...,0]=delta_f[...,0]*c[:,None]+delta_f[...,1]*s[:,None];future_local[...,1]=-delta_f[...,0]*s[:,None]+delta_f[...,1]*c[:,None]
 v_last=past_local[:,0,-1]-past_local[:,0,-2];v_mean=(past_local[:,0,-1]-past_local[:,0,0])/7.;v_prev=past_local[:,0,-2]-past_local[:,0,-3];dv=v_last-v_prev
 early=past_local[:,0,-4]-past_local[:,0,-7];late=past_local[:,0,-1]-past_local[:,0,-4];cross=early[:,0]*late[:,1]-early[:,1]*late[:,0];dot=(early*late).sum(-1);turn=np.stack((cross/(np.linalg.norm(early,axis=-1)*np.linalg.norm(late,axis=-1)+1e-6),dot/(np.linalg.norm(early,axis=-1)*np.linalg.norm(late,axis=-1)+1e-6)),axis=-1)
 rel=past_local[:,1:,-1];ids=np.arange(1,past.shape[1])[None];mask=ids<np.asarray(num)[:,None];dist=np.linalg.norm(rel,axis=-1);masked=np.where(mask,dist,np.inf);nearest=masked.min(-1);nearest[~np.isfinite(nearest)]=10.
 nbr_vel=(past_local[:,1:,-1]-past_local[:,1:,-4])/3.;focal_vel=(past_local[:,0,-1]-past_local[:,0,-4])[:,None]/3.;closing=-(rel*(nbr_vel-focal_vel)).sum(-1)/(dist+1e-6);strong=np.where(mask,closing,-np.inf).max(-1);strong[~np.isfinite(strong)]=0.
 features=np.concatenate((v_last,v_mean,dv,turn,np.linalg.norm(past_local[:,0,-1]-past_local[:,0,0],axis=-1,keepdims=True),(np.asarray(num)-1)[:,None]/8.,nearest[:,None],strong[:,None]),axis=-1).astype(np.float32)
 return features,future_local.astype(np.float32)
def select_diverse(futures):
 # Keep the closest-history future, then farthest-point select two alternatives.
 chosen=[0];flat=futures.reshape(len(futures),-1)
 for _ in range(1,SELECT_M):
  distance=np.linalg.norm(flat[:,None]-flat[np.asarray(chosen)][None],axis=-1).min(-1);distance[np.asarray(chosen)]=-np.inf;chosen.append(int(distance.argmax()))
 return np.asarray(chosen,np.int64)
def metric(candidates,target):
 error=np.linalg.norm(candidates-target[:,None],axis=-1);return error.mean(-1).min(-1),error[...,-1].min(-1)
def infer_baseline(split,seed,indices,device,batch_size):
 sys.path[:0]=[str(PROJECT),str(PROJECT/"scripts"),str(FRONT)];os.chdir(PROJECT);import sitecustomize  # noqa
 diag=load(f"cfb_cso_diag_{split}",CSO/"run_cso1_diagnostic.py");cfg,cls=diag.prepare(device);geom=load(f"cfb_geom_{split}",WORKSPACE/"INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES"/"run_geometry_diagnostic.py")
 info=json.loads((baseline_dir(split,seed)/"split.json").read_text(encoding="utf-8"));ds=geom.IndexedDataset(Path(info["data_path"]),Path(info["num_path"]),indices,cfg);loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda")
 model=diag.build(split,"baseline",seed,cfg,cls,device);values=diag.infer(model,loader,device);del model
 if device.type=="cuda":torch.cuda.empty_cache()
 return values
def summarize_subset(name,mask,bade,bfde,bankade,bankfde,selade,selfde,unionade,unionfde,selunionade,selunionfde):
 def mean(v):return float(v[mask].mean())
 return {"subset":name,"samples":int(mask.sum()),"baseline":{"min_ade_20":mean(bade),"min_fde_20":mean(bfde)},"bank20":{"min_ade":mean(bankade),"min_fde":mean(bankfde)},"selected3":{"min_ade":mean(selade),"min_fde":mean(selfde)},"union":{"min_ade":mean(unionade),"min_fde":mean(unionfde)},"union_improvement_pct":{"min_ade":100*(1-mean(unionade)/mean(bade)),"min_fde":100*(1-mean(unionfde)/mean(bfde))},"selected3_union":{"min_ade":mean(selunionade),"min_fde":mean(selunionfde)},"selected3_union_improvement_pct":{"min_ade":100*(1-mean(selunionade)/mean(bade)),"min_fde":100*(1-mean(selunionfde)/mean(bfde))}}
def process(split,seed,raw,num,features,futures,device,batch_size,force):
 run=baseline_dir(split,seed);sp=np.load(run/"split_indices.npz");train=np.asarray(sp["train"],np.int64);val=np.asarray(sp["validation"],np.int64);scenes=np.asarray(raw[:,0,0,0],np.int64)
 mean=features[train].mean(0);std=features[train].std(0);std[std<1e-5]=1.;z=(features-mean)/std
 cache_path=CACHE/f"{split}_seed_{seed}.npz";selected=np.zeros((len(raw),SELECT_M,12,2),np.float32);neighbors=np.full((len(raw),SELECT_M),-1,np.int64);distances=np.full((len(raw),SELECT_M),np.inf,np.float32)
 val_pos={int(index):pos for pos,index in enumerate(val)};audit20=np.empty((len(val),BANK_K,12,2),np.float32);auditdist=np.empty((len(val),BANK_K),np.float32)
 for scene in np.unique(scenes):
  bank=train[scenes[train]!=scene];query=np.flatnonzero(scenes==scene);nn=NearestNeighbors(n_neighbors=min(BANK_K,len(bank)),algorithm="auto",metric="euclidean").fit(z[bank]);d,loc=nn.kneighbors(z[query]);idx=bank[loc]
  for row,index in enumerate(query):
   future_set=futures[idx[row]];pick=select_diverse(future_set);selected[index]=future_set[pick];neighbors[index]=idx[row,pick];distances[index]=d[row,pick]
   if int(index) in val_pos:
    pos=val_pos[int(index)];audit20[pos]=future_set;auditdist[pos]=d[row]
 metadata={"split":split,"seed":seed,"bank_policy":"train-only cross-scene; no same-scene candidates","bank_k":BANK_K,"selected_m":SELECT_M,"train_indices_sha256":digest(train),"validation_indices_sha256":digest(val),"feature_dim":int(features.shape[1])}
 CACHE.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache_path,pseudo_futures=selected,neighbor_indices=neighbors,feature_distances=distances,feature_mean=mean,feature_std=std,metadata_json=np.array(json.dumps(metadata)))
 print(f"[cache] {split} {cache_path} size_mb={cache_path.stat().st_size/1048576:.1f}",flush=True)
 values=infer_baseline(split,seed,val,device,batch_size);target=values["target"];bade=np.asarray(values["min_ade_20"]);bfde=np.asarray(values["min_fde_20"]);bankade,bankfde=metric(audit20,target);selade,selfde=metric(selected[val],target);unionade=np.minimum(bade,bankade);unionfde=np.minimum(bfde,bankfde);selunionade=np.minimum(bade,selade);selunionfde=np.minimum(bfde,selfde)
 labels=np.asarray(np.load(CSRS/"teacher_cache"/f"{split}_seed_{seed}.npz",allow_pickle=False)["need_label"])[val].astype(bool);allmask=np.ones(len(val),bool)
 result={"split":split,"strict_leakage_policy":metadata["bank_policy"],"retrieval":{"bank_k":BANK_K,"selected_m":SELECT_M,"feature_distance_mean":float(auditdist.mean()),"feature_distance_p90":float(np.quantile(auditdist,.9))},"overall":summarize_subset("overall",allmask,bade,bfde,bankade,bankfde,selade,selfde,unionade,unionfde,selunionade,selunionfde),"social_needed":summarize_subset("social_needed",labels,bade,bfde,bankade,bankfde,selade,selfde,unionade,unionfde,selunionade,selunionfde)}
 AUDIT.mkdir(parents=True,exist_ok=True);(AUDIT/f"{split}_summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8");return result
def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=1);p.add_argument("--batch-size",type=int,default=128);p.add_argument("--force",action="store_true");a=p.parse_args();device=torch.device(a.device)
 raw=np.load(DATA/"eth_data_train.npy",mmap_mode="r");num=np.load(DATA/"eth_num_train.npy",mmap_mode="r");features,futures=local_data(raw,num);results=[process(s,a.seed,raw,num,features,futures,device,a.batch_size,a.force) for s in SPLITS]
 passes={r["split"]:max(r["overall"]["union_improvement_pct"].values())>=3. and max(r["overall"]["selected3_union_improvement_pct"].values())>=1. for r in results};gate=all(passes.values());combined={"gate_passed":gate,"gate_rule":"each split: Bank20 union >=3% and the actually distilled Selected3 union >=1% on minADE or minFDE","split_pass":passes,"results":results}
 (AUDIT/"combined_summary.json").write_text(json.dumps(combined,indent=2,ensure_ascii=False),encoding="utf-8");print(f"[audit] gate_passed={gate} split_pass={passes}")
if __name__=="__main__":main()
