"""EXP-169B: official V2 Full-vs-Self sparse-social counterfactual audit."""
from __future__ import annotations
import argparse,csv,importlib.util,json,os,subprocess,sys
from contextlib import nullcontext
from pathlib import Path

ASTRA=Path(r"C:\coding_toobox\envs\astra2\python.exe")
if str(Path(sys.executable).resolve()).casefold()!=str(ASTRA.resolve()).casefold():
    print(f"[bootstrap] relaunching with {ASTRA}",flush=True);raise SystemExit(subprocess.run([str(ASTRA),*sys.argv]).returncode)
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

ROOT=Path(__file__).resolve().parent;WS=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse"
CONFIG=WS/"OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES"/"motif_rbf_official_4p8.yaml";TRAINER=WS/"FRONT_ENCODER_SERIES"/"train_front_variant.py"
V2MODULE=WS/"ANCHOR_CORRECTION_VALIDATION_SERIES"/"anchor_correction_validation_models.py";CSNA=WS/"COUNTERFACTUAL_SOCIAL_NECESSITY_AUDIT"/"csna_mask.py"
SOURCE=WS/"OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES"/"EXP_156B_FOLD_LOCAL_CURRENT_DECODER"/"results";OUT=ROOT/"EXP_169B_SPARSE_SOCIAL_COUNTERFACTUAL_AUDIT"
BASELINE=WS/"OFFICIAL_ETH_UCY_FINAL_3SEED_SERIES"/"EXP_164_FINAL_V2_OFFICIAL_3SEED"/"summary.json"
FOLDS=("eth","hotel","univ","zara01","zara02")

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    with path.open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction="ignore");w.writeheader();w.writerows(rows)
def run_dir(fold,stage,seed):return SOURCE/fold/stage/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def build(fold,seed,cfg,Model,device):
    h3=run_dir(fold,"h3",seed)/"best_selected.pth";v2=run_dir(fold,"v2",seed)/"best_selected.pth";os.environ["EXP143_SOURCE_CHECKPOINT"]=str(h3)
    model=Model(cfg,"f16_temporal_social_cross");module=load(f"exp169b_v2_{fold}",V2MODULE);module.replace_decoder(model,cfg,"v2_wide_endpoint_correction")
    model.load_state_dict(torch.load(v2,map_location="cpu",weights_only=True),strict=True);return model.to(device).eval()
def obs_features(past,num):
    focal=past[:,0];v=np.diff(focal,axis=1);speed=np.linalg.norm(v,axis=-1);acc=np.linalg.norm(np.diff(v,axis=1),axis=-1)
    start,end=focal[:,0],focal[:,-1];chord=end-start;rel=focal-start[:,None];cross=np.abs(rel[...,0]*chord[:,None,1]-rel[...,1]*chord[:,None,0]);den=np.linalg.norm(chord,axis=-1)
    last=past[:,:,-1];distance=np.linalg.norm(last-last[:,:1],axis=-1);valid=np.arange(past.shape[1])[None]<num[:,None];neighbor=valid.copy();neighbor[:,0]=False;masked=np.where(neighbor,distance,np.nan);nearest=np.full(len(past),np.nan);has=neighbor.any(axis=1);nearest[has]=np.nanmin(masked[has],axis=1)
    return {"mean_step_speed":speed.mean(axis=1),"last_step_speed":speed[:,-1],"mean_acceleration":acc.mean(axis=1),"max_acceleration":acc.max(axis=1),"observation_nonlinearity":np.mean(cross/np.maximum(den[:,None],1e-6),axis=1),"valid_agent_count":num.astype(float),"neighbor_count_r2":((distance<2)&neighbor).sum(axis=1).astype(float),"neighbor_count_r4":((distance<4)&neighbor).sum(axis=1).astype(float),"neighbor_count_r6":((distance<6)&neighbor).sum(axis=1).astype(float),"nearest_neighbor_distance":nearest}
def thresholds(fold,cfg,indices):
    raw=np.load(DATA/f"{fold}_data_train.npy",mmap_mode="r");num=np.minimum(np.asarray(np.load(DATA/f"{fold}_num_train.npy",mmap_mode="r"))[indices],int(cfg.DATA.MAX_AGENTS));past=np.asarray(raw[indices,:int(cfg.DATA.MAX_AGENTS),:8,-2:],dtype=np.float32)/float(cfg.PREDICTION.TRAJ_SCALE);f=obs_features(past,num)
    return {"neighbor_r4_q25":float(np.quantile(f["neighbor_count_r4"],.25)),"valid_agents_q25":float(np.quantile(f["valid_agent_count"],.25)),"mean_acceleration_q75":float(np.quantile(f["mean_acceleration"],.75))}
@torch.inference_mode()
def infer(model,loader,device,normalize,self_only):
    intervention=load("exp169b_csna_runtime",CSNA).focal_self_only_intervention
    parts={k:[] for k in ("past","num","minADE","minFDE","sameFDE","J")};context=intervention(model) if self_only else nullcontext(model)
    with context:
        for past,future,num,_,_,_ in loader:
            past=past.to(device);future=future.to(device);num=num.to(device).int();past,future=normalize(past,future);center=past.mean(dim=-2,keepdim=True)
            pred=model(past-center,None,None,mode="testing",num_valid=num,graph_loc=past)[2]+center.unsqueeze(-3);cand=pred[:,0];target=future[:,0];error=torch.linalg.vector_norm(cand-target[:,None],dim=-1);ade=error.mean(-1);fde=error[...,-1];aw=ade.argmin(-1);row=torch.arange(len(ade),device=device)
            values={"past":past.cpu().numpy(),"num":num.cpu().numpy(),"minADE":ade.min(-1).values.cpu().numpy(),"minFDE":fde.min(-1).values.cpu().numpy(),"sameFDE":fde[row,aw].cpu().numpy(),"J":(ade.min(-1).values+.5*fde.min(-1).values).cpu().numpy()}
            for k,v in values.items():parts[k].append(v)
    return {k:np.concatenate(v) for k,v in parts.items()}
def summarize_condition(fold,scope,condition,mask,full,selfonly):
    row={"fold":fold,"scope":scope,"condition":condition,"samples":int(mask.sum())}
    for metric in ("minADE","minFDE","sameFDE","J"):
        fv=float(full[metric][mask].mean()) if mask.any() else None;sv=float(selfonly[metric][mask].mean()) if mask.any() else None;row[f"full_{metric}"]=fv;row[f"self_{metric}"]=sv;row[f"full_advantage_{metric}_pct"]=100*(sv/fv-1) if mask.any() and fv else None
    return row
def paired(fold,scope,indices,model,cfg,Dataset,normalize,device,batch,max_samples,threshold):
    if max_samples:indices=indices[:max_samples]
    base=Dataset(DATA/f"{fold}_data_train.npy",DATA/f"{fold}_num_train.npy",cfg) if scope=="inner" else Dataset(DATA/f"{fold}_data_test.npy",DATA/f"{fold}_num_test.npy",cfg)
    dataset=Subset(base,indices.tolist()) if scope=="inner" else (Subset(base,range(min(max_samples,len(base)))) if max_samples else base);loader=DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0)
    print(f"[169B] {fold}/{scope}/full n={len(dataset)}",flush=True);full=infer(model,loader,device,normalize,False);print(f"[169B] {fold}/{scope}/self",flush=True);selfonly=infer(model,loader,device,normalize,True)
    obs=obs_features(full["past"],full["num"]);low=obs["neighbor_count_r4"]<=threshold["neighbor_r4_q25"];low_valid=obs["valid_agent_count"]<=threshold["valid_agents_q25"];high_acc=obs["mean_acceleration"]>=threshold["mean_acceleration_q75"];ldha=low&high_acc
    conditions={"ALL":np.ones(len(dataset),dtype=bool),"LDHA":ldha,"LDLA":low&~high_acc,"HDHA":~low&high_acc,"HDLA":~low&~high_acc,"non_LDHA":~ldha}
    summaries=[summarize_condition(fold,scope,name,mask,full,selfonly) for name,mask in conditions.items()]
    rows=[]
    for i in range(len(dataset)):
        row={"fold":fold,"scope":scope,"sample_id":int(indices[i]) if scope=="inner" else i,"condition":next((name for name in ("LDHA","LDLA","HDHA","HDLA") if conditions[name][i]),"unknown"),"low_density":bool(low[i]),"low_valid_agents":bool(low_valid[i]),"high_acceleration":bool(high_acc[i])}
        for k,v in obs.items():row[k]=float(v[i])
        for metric in ("minADE","minFDE","sameFDE","J"):row[f"full_{metric}"]=float(full[metric][i]);row[f"self_{metric}"]=float(selfonly[metric][i]);row[f"social_gain_{metric}"]=float(selfonly[metric][i]-full[metric][i])
        rows.append(row)
    return rows,summaries
def decide(summary):
    find=lambda fold,scope,cond:next(r for r in summary if r["fold"]==fold and r["scope"]==scope and r["condition"]==cond)
    eth=find("eth","outer","LDHA");eth_gain=eth["full_advantage_J_pct"]
    inner=[find(f,"inner","LDHA")["full_advantage_J_pct"] for f in FOLDS];non=[find(f,"outer","non_LDHA")["full_advantage_J_pct"] for f in FOLDS]
    b1=eth["samples"]>0 and eth_gain>=2 and sum(x is not None and x>0 for x in inner)>=3
    b2=eth["samples"]>0 and abs(eth_gain)<1 and sum(x is not None and abs(x)<1 for x in inner)>=3
    b3=eth["samples"]>0 and eth_gain<=-2 and sum(x is not None and x<=-1 for x in inner)>=3 and sum(x is not None and x>=-1 for x in non)>=3
    if b1:case="B1_full_social_beneficial";next_exp="individual_motion_representation_audit"
    elif b2:case="B2_full_self_equivalent";next_exp="decoder_or_self_motion_priority"
    elif b3:case="B3_full_social_harmful";next_exp="EXP-169C"
    else:case="INCONCLUSIVE";next_exp="STOP_no_training"
    return {"case":case,"next_experiment":next_exp,"eth_outer_LDHA_samples":eth["samples"],"eth_outer_LDHA_full_advantage_J_pct":eth_gain,"inner_LDHA_full_advantage_J_pct":dict(zip(FOLDS,inner)),"inner_full_better_count":sum(x is not None and x>0 for x in inner),"inner_self_better_ge_1pct_count":sum(x is not None and x<=-1 for x in inner),"inner_equivalent_count":sum(x is not None and abs(x)<1 for x in inner),"outer_non_LDHA_full_advantage_J_pct":dict(zip(FOLDS,non)),"EXP169C_ALLOWED":b3}
def verify_reproduction(fold,outer_summary,seed):
    payload=json.loads(BASELINE.read_text(encoding="utf-8"));expected=next(x for x in payload["rows"] if x["fold"]==fold and int(x["seed"])==seed);row=next(x for x in outer_summary if x["condition"]=="ALL")
    targets={"minADE":expected["minADE"],"minFDE":expected["minFDE"],"sameFDE":expected["same_path_FDE"]};checks={}
    for metric,target in targets.items():
        observed=row[f"full_{metric}"];difference=observed-float(target);checks[metric]={"expected":float(target),"observed":observed,"absolute_difference":difference}
        if abs(difference)>2e-5:raise RuntimeError(f"Full-social reproduction failed: {fold}/{metric}: {difference}")
    return checks
def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--folds",nargs="+",choices=("all",*FOLDS),default=["all"]);p.add_argument("--seed",type=int,default=1);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--max-samples",type=int,default=0);p.add_argument("--output",type=Path,default=OUT);p.add_argument("--dry-run",action="store_true");a=p.parse_args();folds=FOLDS if "all" in a.folds else tuple(a.folds)
    required=[CONFIG,TRAINER,V2MODULE,CSNA,BASELINE]
    for f in folds:required += [run_dir(f,"h3",a.seed)/"best_selected.pth",run_dir(f,"v2",a.seed)/"best_selected.pth",run_dir(f,"v2",a.seed)/"split_indices.npz",DATA/f"{f}_data_train.npy",DATA/f"{f}_num_train.npy",DATA/f"{f}_data_test.npy",DATA/f"{f}_num_test.npy"]
    missing=[str(x) for x in required if not x.is_file()]
    if missing:raise FileNotFoundError("Missing:\n"+"\n".join(missing))
    if a.dry_run:print(json.dumps({"folds":folds,"seed":a.seed,"output":str(a.output),"max_samples":a.max_samples,"intervention":str(CSNA)},indent=2));return
    sys.path[:0]=[str(PROJECT),str(PROJECT/"scripts"),str(WS/"FRONT_ENCODER_SERIES")];os.chdir(PROJECT);import sitecustomize
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from train_front_variant import TrajectoryDataset,normalize_to_focal_heading
    device=torch.device(a.device);cfg=load_locked_q20(CONFIG);cfg.MODEL.USE_PRETRAINED_UNET=False;cfg.MODEL.USE_CACHED_UNET_FEATURES=False;cfg.DATA.MAX_AGENTS=9;cfg.MODEL.GRAPH.TOP_K=4;cfg.MODEL.GRAPH.RADIUS=6.;cfg.device=str(device);cfg.device_list=[device.index or 0]
    per=[];summaries=[];threshold_rows=[];reproduction={}
    for fold in folds:
        split=np.load(run_dir(fold,"v2",a.seed)/"split_indices.npz");train=np.asarray(split["train"],dtype=np.int64);val=np.asarray(split["validation"],dtype=np.int64);threshold=thresholds(fold,cfg,train);threshold_rows.append({"fold":fold,**threshold});model=build(fold,a.seed,cfg,FrontEncoderASTRA,device)
        x,s=paired(fold,"outer",np.arange(len(np.load(DATA/f"{fold}_num_test.npy",mmap_mode="r"))),model,cfg,TrajectoryDataset,normalize_to_focal_heading,device,a.batch_size,a.max_samples,threshold);per+=x;summaries+=s
        if not a.max_samples:reproduction[fold]=verify_reproduction(fold,s,a.seed)
        x,s=paired(fold,"inner",val,model,cfg,TrajectoryDataset,normalize_to_focal_heading,device,a.batch_size,a.max_samples,threshold);per+=x;summaries+=s;del model
        if device.type=="cuda":torch.cuda.empty_cache()
    complete=set(folds)==set(FOLDS) and a.max_samples==0;verdict=decide(summaries) if complete else None;a.output.mkdir(parents=True,exist_ok=True);suffix="" if complete else "_partial"
    write_csv(a.output/f"per_sample_counterfactual{suffix}.csv",per);write_csv(a.output/f"condition_bins{suffix}.csv",summaries);write_csv(a.output/f"per_fold_summary{suffix}.csv",[r for r in summaries if r["scope"]=="outer" and r["condition"] in ("ALL","LDHA","non_LDHA")]);write_csv(a.output/f"inner_proxy_summary{suffix}.csv",[r for r in summaries if r["scope"]=="inner" and r["condition"]=="LDHA"]);write_csv(a.output/f"source_thresholds{suffix}.csv",threshold_rows)
    (a.output/f"summary{suffix}.json").write_text(json.dumps({"experiment":"EXP-169B","complete":complete,"intervention":"exact CSNA focal receive-row self-loop","full_social_reproduction":reproduction,"decision":verdict,"condition_summary":summaries},indent=2),encoding="utf-8");print(json.dumps(verdict,indent=2),flush=True);print(f"[done] {a.output}")
if __name__=="__main__":main()
