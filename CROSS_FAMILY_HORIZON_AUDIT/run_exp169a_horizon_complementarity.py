"""EXP-169A: jointly-trained family horizon complementarity and reachability."""
from __future__ import annotations
import argparse,csv,importlib.util,json,math,os,subprocess,sys
from collections import Counter
from pathlib import Path

ASTRA=Path(r"C:\coding_toobox\envs\astra2\python.exe")
if str(Path(sys.executable).resolve()).casefold()!=str(ASTRA.resolve()).casefold():
    print(f"[bootstrap] relaunching with {ASTRA}",flush=True);raise SystemExit(subprocess.run([str(ASTRA),*sys.argv]).returncode)
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

ROOT=Path(__file__).resolve().parent;WS=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse"
CONFIG=WS/"OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES"/"motif_rbf_official_4p8.yaml"
TRAINER=WS/"FRONT_ENCODER_SERIES"/"train_front_variant.py"
DECODER=WS/"JOINT_HYBRID_GENERATOR_SERIES"/"hybrid_generator_models.py"
SOURCE=WS/"JOINT_HYBRID_GENERATOR_SERIES"/"EXP_168A_HYBRID_K20_10_10"/"results"
OUT=ROOT/"EXP_169A_CROSS_FAMILY_HORIZON_AUDIT"
FOLDS=("eth","hotel","univ","zara01","zara02");STEPS=(4,8,12)
HALF={4:.10,8:.125,12:.25};FULL={4:.20,8:.25,12:.50}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:raise ImportError(path)
    spec.loader.exec_module(module);return module
def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    with path.open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction="ignore");w.writeheader();w.writerows(rows)
def checkpoint(fold,seed):return SOURCE/fold/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"
def features(pos,num):
    focal=pos[:,0];velocity=np.diff(focal,axis=1);speed=np.linalg.norm(velocity,axis=-1);acc=np.diff(velocity,axis=1)
    last=pos[:,:,-1];distance=np.linalg.norm(last-last[:,:1],axis=-1);valid=np.arange(pos.shape[1])[None]<num[:,None];neighbor=valid.copy();neighbor[:,0]=False
    return {"median_speed":np.median(speed,axis=1),"mean_acceleration":np.linalg.norm(acc,axis=-1).mean(axis=1),"neighbor_count_r4":((distance<4)&neighbor).sum(axis=1).astype(float)}
def train_edges(fold,scale,max_agents):
    raw=np.load(DATA/f"{fold}_data_train.npy",mmap_mode="r");num=np.minimum(np.asarray(np.load(DATA/f"{fold}_num_train.npy",mmap_mode="r")),max_agents)
    pos=np.asarray(raw[:,:max_agents,:8,-2:],dtype=np.float32)/scale;f=features(pos,num)
    return np.quantile(f["median_speed"],[.25,.5,.75]),np.quantile(f["mean_acceleration"],[.25,.5,.75]),np.quantile(f["neighbor_count_r4"],[1/3,2/3])
def build(fold,seed,cfg,Model,device):
    model=Model(cfg,"f16_temporal_social_cross");module=load(f"exp169a_{fold}",DECODER);module.replace_decoder(model,cfg,"hybrid_k20_10_10")
    model.load_state_dict(torch.load(checkpoint(fold,seed),map_location="cpu",weights_only=True),strict=True);return model.to(device).eval()
def family_metrics(cand,target):
    error=np.linalg.norm(cand-target[:,None],axis=-1);result={}
    for step in STEPS:
        result[f"err_t{step}"]=error[:,:,step-1].min(axis=1);result[f"winner_t{step}"]=error[:,:,step-1].argmin(axis=1)
        result[f"prefix{step}"]=error[:,:,:step].mean(axis=-1).min(axis=1);result[f"prefix_winner{step}"]=error[:,:,:step].mean(axis=-1).argmin(axis=1)
    result["full_ADE"]=error.mean(axis=-1).min(axis=1);result["full_winner"]=error.mean(axis=-1).argmin(axis=1)
    result["minFDE"]=error[:,:,-1].min(axis=1);result["fde_winner"]=error[:,:,-1].argmin(axis=1)
    return result
def fam(a,b):return np.where(a<=b,"structured","free")
def mean(x):return float(np.mean(x))
def audit(fold,seed,cfg,Model,Dataset,normalize,device,batch,max_samples):
    dataset=Dataset(DATA/f"{fold}_data_test.npy",DATA/f"{fold}_num_test.npy",cfg)
    if max_samples:dataset=Subset(dataset,range(min(max_samples,len(dataset))))
    loader=DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=0);model=build(fold,seed,cfg,Model,device)
    pp=[];tt=[];nn=[]
    with torch.inference_mode():
        for past,future,num,_,_,_ in loader:
            past=past.to(device);future=future.to(device);num=num.to(device).int();past,future=normalize(past,future);center=past.mean(dim=-2,keepdim=True)
            pred=model(past-center,None,None,mode="testing",num_valid=num,graph_loc=past)[2]+center.unsqueeze(-3)
            pp.append(pred[:,0].cpu().numpy());tt.append(future[:,0].cpu().numpy());nn.append(num.cpu().numpy())
    candidates=np.concatenate(pp);target=np.concatenate(tt);num=np.concatenate(nn);structured=candidates[:,:10];free=candidates[:,10:]
    sm=family_metrics(structured,target);fm=family_metrics(free,target);um={k:np.minimum(sm[k],fm[k]) for k in sm if "winner" not in k}
    raw=np.load(DATA/f"{fold}_data_test.npy",mmap_mode="r");n=len(target);pos=np.asarray(raw[:n,:int(cfg.DATA.MAX_AGENTS),:8,-2:],dtype=np.float32)/float(cfg.PREDICTION.TRAJ_SCALE)
    obs=features(pos,num);se,ae,de=train_edges(fold,float(cfg.PREDICTION.TRAJ_SCALE),int(cfg.DATA.MAX_AGENTS));sb=np.digitize(obs["median_speed"],se,right=True)+1;ab=np.digitize(obs["mean_acceleration"],ae,right=True)+1;db=np.digitize(obs["neighbor_count_r4"],de,right=True)+1
    winner={f"t{s}":fam(sm[f"err_t{s}"],fm[f"err_t{s}"]) for s in STEPS};winner["full_ADE"]=fam(sm["full_ADE"],fm["full_ADE"]);winner["FDE"]=fam(sm["minFDE"],fm["minFDE"])
    nearest={};reachable={}
    for step in STEPS:
        free_idx=fm[f"winner_t{step}"];free_point=free[np.arange(n),free_idx,step-1];dist=np.linalg.norm(structured[:,:,step-1]-free_point[:,None],axis=-1).min(axis=1);nearest[step]=dist
        fw=winner[f"t{step}"]=="free";reachable[(step,"half")]=fw&(dist<=HALF[step]);reachable[(step,"full")]=fw&(dist<=FULL[step])
    free_full_idx=fm["full_winner"];free_path=free[np.arange(n),free_full_idx];path_distance=np.linalg.norm(structured-free_path[:,None],axis=-1).mean(axis=-1).min(axis=1)
    per=[]
    for i in range(n):
        row={"fold":fold,"seed":seed,"sample_id":i,"speed_q":int(sb[i]),"acceleration_q":int(ab[i]),"density_bin":int(db[i]),"median_speed":float(obs["median_speed"][i]),"mean_acceleration":float(obs["mean_acceleration"][i]),"neighbor_count_r4":float(obs["neighbor_count_r4"][i]),"nearest_path_distance":float(path_distance[i])}
        for step in STEPS:
            row.update({f"structured_err_t{step}":float(sm[f"err_t{step}"][i]),f"free_err_t{step}":float(fm[f"err_t{step}"][i]),f"union_err_t{step}":float(um[f"err_t{step}"][i]),f"structured_prefix_ade{step}":float(sm[f"prefix{step}"][i]),f"free_prefix_ade{step}":float(fm[f"prefix{step}"][i]),f"union_prefix_ade{step}":float(um[f"prefix{step}"][i]),f"winner_family_t{step}":winner[f"t{step}"][i],f"nearest_anchor_distance_t{step}":float(nearest[step][i]),f"reach_t{step}_half_v2":bool(reachable[(step,"half")][i]),f"reach_t{step}_full_v2":bool(reachable[(step,"full")][i])})
        row.update({"winner_family_full_ADE":winner["full_ADE"][i],"winner_family_FDE":winner["FDE"][i],"structured_full_ADE":float(sm["full_ADE"][i]),"free_full_ADE":float(fm["full_ADE"][i]),"union_full_ADE":float(um["full_ADE"][i]),"structured_minFDE":float(sm["minFDE"][i]),"free_minFDE":float(fm["minFDE"][i]),"union_minFDE":float(um["minFDE"][i])});per.append(row)
    summary={"fold":fold,"seed":seed,"samples":n}
    reachrows=[]
    for step in STEPS:
        for metric,key in (("point",f"err_t{step}"),("prefix",f"prefix{step}")):
            sv,fv,uv=mean(sm[key]),mean(fm[key]),mean(um[key]);summary[f"structured_{metric}_t{step}"]=sv;summary[f"free_{metric}_t{step}"]=fv;summary[f"union_{metric}_t{step}"]=uv;summary[f"free_gain_{metric}_t{step}_pct"]=100*(sv-uv)/sv
        fw=winner[f"t{step}"]=="free";den=max(int(fw.sum()),1)
        summary[f"free_winner_fraction_t{step}"]=float(fw.mean());summary[f"reach_t{step}_half_v2"]=float(reachable[(step,"half")].sum()/den);summary[f"reach_t{step}_full_v2"]=float(reachable[(step,"full")].sum()/den)
        reachrows.append({"fold":fold,"horizon":step,"free_winners":int(fw.sum()),"mean_nearest_distance":mean(nearest[step][fw]) if fw.any() else None,"reach_half_v2":summary[f"reach_t{step}_half_v2"],"reach_full_v2":summary[f"reach_t{step}_full_v2"]})
    for metric,key in (("full_ADE","full_ADE"),("minFDE","minFDE")):
        sv,fv,uv=mean(sm[key]),mean(fm[key]),mean(um[key]);summary[f"structured_{metric}"]=sv;summary[f"free_{metric}"]=fv;summary[f"union_{metric}"]=uv;summary[f"free_gain_{metric}_pct"]=100*(sv-uv)/sv
    transitions=Counter(f"{winner['t4'][i]}->{winner['t8'][i]}->{winner['t12'][i]}" for i in range(n));transitionrows=[{"fold":fold,"transition":k,"samples":v,"fraction":v/n} for k,v in sorted(transitions.items())]
    del model
    if device.type=="cuda":torch.cuda.empty_cache()
    return per,summary,reachrows,transitionrows
def decision(rows):
    macro=lambda key:mean([r[key] for r in rows]);positive=lambda key:sum(r[key]>0 for r in rows)
    candidates=[]
    for step in (4,8):
        candidates.append((macro(f"free_gain_prefix_t{step}_pct"),step,positive(f"free_gain_prefix_t{step}_pct"),macro(f"reach_t{step}_half_v2")))
    early=max(candidates);t12=macro("free_gain_point_t12_pct");structured_tail=macro("free_winner_fraction_t12")<.5
    early_pass=early[0]>=2 and early[2]>=4 and (t12<1 or structured_tail) and early[3]>=.60
    ep_gain=max(macro("free_gain_point_t12_pct"),macro("free_gain_minFDE_pct"));ep_positive=max(positive("free_gain_point_t12_pct"),positive("free_gain_minFDE_pct"));ep_reach=macro("reach_t12_half_v2");endpoint_pass=ep_gain>=2 and ep_positive>=4 and ep_reach>=.50
    if early_pass:case="A_early_mid_complementarity";next_exp="EXP-170A"
    elif endpoint_pass:case="B_endpoint_complementarity";next_exp="EXP-170B"
    elif max(early[0],ep_gain)>=2 and max(early[3],ep_reach)<.40:case="C_complementary_but_unreachable";next_exp="STOP"
    else:case="D_no_stable_horizon_complementarity";next_exp="STOP"
    return {"case":case,"next_experiment":next_exp,"early_best_horizon":early[1],"early_macro_gain_pct":early[0],"early_positive_folds":early[2],"early_half_budget_reach":early[3],"t12_point_gain_pct":t12,"structured_tail_more_important":structured_tail,"endpoint_gain_pct":ep_gain,"endpoint_positive_folds":ep_positive,"endpoint_half_budget_reach":ep_reach,"EXP170A_ALLOWED":early_pass,"EXP170B_ALLOWED":endpoint_pass}
def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--folds",nargs="+",choices=("all",*FOLDS),default=["all"]);p.add_argument("--seed",type=int,default=1);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--max-samples",type=int,default=0);p.add_argument("--output",type=Path,default=OUT);p.add_argument("--dry-run",action="store_true");a=p.parse_args();folds=FOLDS if "all" in a.folds else tuple(a.folds)
    required=[CONFIG,TRAINER,DECODER,*[checkpoint(f,a.seed) for f in folds],*[DATA/f"{f}_data_test.npy" for f in folds],*[DATA/f"{f}_num_test.npy" for f in folds]];missing=[str(x) for x in required if not x.is_file()]
    if missing:raise FileNotFoundError("Missing:\n"+"\n".join(missing))
    if a.dry_run:print(json.dumps({"folds":folds,"seed":a.seed,"output":str(a.output),"max_samples":a.max_samples},indent=2));return
    os.environ.pop("EXP168_DIAGNOSTICS_PATH",None);sys.path[:0]=[str(PROJECT),str(PROJECT/"scripts"),str(WS/"FRONT_ENCODER_SERIES")];os.chdir(PROJECT);import sitecustomize
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from train_front_variant import TrajectoryDataset,normalize_to_focal_heading
    device=torch.device(a.device);cfg=load_locked_q20(CONFIG);cfg.MODEL.USE_PRETRAINED_UNET=False;cfg.MODEL.USE_CACHED_UNET_FEATURES=False;cfg.DATA.MAX_AGENTS=9;cfg.MODEL.GRAPH.TOP_K=4;cfg.MODEL.GRAPH.RADIUS=6.;cfg.device=str(device);cfg.device_list=[device.index or 0]
    samples=[];summaries=[];reach=[];trans=[]
    for fold in folds:
        print(f"[169A] {fold}",flush=True);x,s,r,t=audit(fold,a.seed,cfg,FrontEncoderASTRA,TrajectoryDataset,normalize_to_focal_heading,device,a.batch_size,a.max_samples);samples+=x;summaries.append(s);reach+=r;trans+=t
    complete=set(folds)==set(FOLDS) and a.max_samples==0;verdict=decision(summaries) if complete else None;a.output.mkdir(parents=True,exist_ok=True);suffix="" if complete else "_partial"
    write_csv(a.output/f"per_sample_horizon{suffix}.csv",samples);write_csv(a.output/f"per_fold_summary{suffix}.csv",summaries);write_csv(a.output/f"reachability_summary{suffix}.csv",reach);write_csv(a.output/f"winner_transition{suffix}.csv",trans)
    (a.output/f"summary{suffix}.json").write_text(json.dumps({"experiment":"EXP-169A","complete":complete,"per_fold":summaries,"decision":verdict},indent=2),encoding="utf-8");print(json.dumps(verdict,indent=2),flush=True);print(f"[done] {a.output}")
if __name__=="__main__":main()
