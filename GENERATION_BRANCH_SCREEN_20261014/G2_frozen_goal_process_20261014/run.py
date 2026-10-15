import common as c,model as m,metrics as v
import numpy as np,torch,time,gc
SEEDS=(42,43,44)
def optimizer(net):return torch.optim.AdamW(net.process.parameters(),lr=.001,weight_decay=.0001,foreach=False)
def bootstrap(data,independent=False):
 rng=np.random.default_rng(20261014);records=[]
 for split,ix in data["split"].items():
  learned=[c.npz(c.H/f"runs/seed_{s}/{split}_metrics.npz")["vectors"] for s in SEEDS];A=[c.npz(c.OLD/f"runs/seed_{s}/{split}_metrics.npz")["vectors"] for s in SEEDS]
  for group,mask in v.masks(data["groups"][ix]):
   tracks=data["track"][ix][mask];units,inv=np.unique(tracks,return_inverse=True);counts=np.bincount(inv);T=len(units)
   for reference in ("G1","native20","prototype20","retrieval20"):
    refs=A if reference=="G1" else [c.npz(c.OLD/f"baseline/{split}_{reference}_metrics.npz")["vectors"]]*3
    diff=np.stack([learned[s][mask]-refs[s][mask] for s in range(3)])
    if independent:sums=np.stack([[diff[s,tracks==unit].sum(0) for unit in units] for s in range(3)])
    else:
     sums=np.zeros((3,T,4))
     for s in range(3):np.add.at(sums[s],inv,diff[s])
    draws=[]
    for _ in range(2000):
     c.guard();ss=rng.integers(0,3,3);tt=rng.integers(0,T,T);draws.append(sums[ss][:,tt].sum((0,1))/(3*counts[tt].sum()))
    records.append(dict(split=split,group=group,reference=reference,n=int(mask.sum()),tracks=T,metrics=["coverage","ADE","FDE","J"],estimate=diff.mean((0,1)),CI_family64=np.quantile(draws,[.05/128,1-.05/128],axis=0).T,perseed_estimate=diff.mean(1),exploratory_not_confirmatory=True))
 return records
def geometry(paths,goals,ab,base,expected):
 assert np.array_equal(base,expected["paths"]) and np.array_equal(goals,expected["goals"]) and np.array_equal(paths[:,:,-1],goals)
 s=np.arange(1,13)/12.;rebuilt=base+(s*(1-s))[None,None,:,None]*ab[:,:,:2][:,:,None]+(s*(1-s)*(2*s-1))[None,None,:,None]*ab[:,:,2:][:,:,None]
 assert np.max(abs(rebuilt-paths))<=2e-11;assert np.array_equal(v.errors(paths,expected["gt"])[1],v.errors(base,expected["gt"])[1]),"FDE_CHANGED"
 return float(np.max(abs(rebuilt-paths)))
def jump_report(paths,base,velocity,groups):
 pb=np.linalg.norm(paths[:,:,0]/.4-velocity[:,None],axis=-1);pa=np.linalg.norm(base[:,:,0]/.4-velocity[:,None],axis=-1)
 return {name:dict(B_mean=float(pb[mask].mean()),B_p95=float(np.quantile(pb[mask],.95)),A_mean=float(pa[mask].mean()),A_p95=float(np.quantile(pa[mask],.95))) for name,mask in v.masks(groups) if mask.any()}
def save_job(net,data,ix,seed,split):
 paths,goals,ab,base=m.generated(net,data,ix);a=c.npz(c.OLD/f"runs/seed_{seed}/{split}.npz");assert np.array_equal(a["sample_id"],data["ids"][ix]);err=geometry(paths,goals,ab,base,a)
 z=v.sample(data,ix,paths,goals=goals,controls=ab,base_paths=base);name=f"runs/seed_{seed}/{split}.npz";c.save(name,**z);stored=c.npz(c.H/name);met,aux=v.evaluate(stored["paths"],stored["gt"],stored["groups"],stored["native20"]);other,oa=v.evaluate(stored["paths"],stored["gt"],stored["groups"],stored["native20"],True);c.eq(met,other,2e-11)
 for key in aux:
  if aux[key].dtype.kind in "biu":assert np.array_equal(aux[key],oa[key])
  else:assert np.max(abs(aux[key]-oa[key]))<=2e-11
 c.save(f"runs/seed_{seed}/{split}_metrics.npz",sample_id=z["sample_id"],track=z["track"],groups=z["groups"],**aux)
 return met,jump_report(paths,base,data["vel"][ix],data["groups"][ix]),err
def continuation(records,ci,A,native):
 change=[r["splits"]["outer"]["change"] for r in records];diff=[r["coverage"]-A[i]["splits"]["outer"]["change"]["coverage"] for i,r in enumerate(change)];ade=[1-r["minADE"]/A[i]["splits"]["outer"]["change"]["minADE"] for i,r in enumerate(change)];row=next(x for x in ci if x["split"]=="outer" and x["group"]=="change" and x["reference"]=="G1")
 protected=all(r["splits"]["outer"][g][k]<=1.01*ref[g][k] for i,r in enumerate(records) for g in ("overall","stable","transition") for k in ("minADE","minFDE") for ref in (A[i]["splits"]["outer"],native["outer"]["native20"]))
 motion=all(not x["motion_failed"] for r in records for split in r["splits"].values() for x in split.values())
 gates=dict(coverage_mean5pp=float(np.mean(diff))>=.05,all3_coverage_positive=all(x>0 for x in diff),coverage_track_CI_lower_positive=float(row["CI_family64"][0][0])>0,change_ADE_mean5percent=float(np.mean(ade))>=.05,protected_A_and_native_1percent=protected,all_seed_split_group_five_motion_native1percent=motion)
 return dict(pass_all=all(gates.values()),gates=gates,coverage_gain_pp=np.array(diff)*100,ADE_gain_percent=np.array(ade)*100,not_final_major=True)
def main():
 started=time.perf_counter();lock=c.read(c.H/"source_manifest.json");c.stage("INITIAL_SOURCE_PARENT_COMPLETED_SHA",runs_done=0);c.verify(lock["files"]);data=c.load();records=[];outputs={};parent_result=c.read(c.OLD/"RESULT.json")
 for seed in SEEDS:
  d=c.H/f"runs/seed_{seed}";d.mkdir(parents=True,exist_ok=True);net=m.fresh(seed);frozen=m.tensor_hash(net.goal.state_dict());man=dict(c.read(c.H/"protocol.json"),seed=seed,source_sha=lock["source_sha"],source_manifest_sha=c.sha(c.H/"source_manifest.json"),G1_checkpoint_sha=c.sha(c.OLD/f"runs/seed_{seed}/checkpoint.pt"),G1_run_manifest_sha=c.sha(c.OLD/f"runs/seed_{seed}/run_manifest.json"),goal_statistics_sha=c.sha(c.H/f"goal_stats_{seed}.npz"),original_statistics_sha=c.sha(c.OLD/"statistics.npz"),split_ids_sha=c.sha(c.OLD/"split_ids.npz"),training_ids=data["ids"][data["split"]["FIT"]].tolist(),initial_all_parameters_buffers_sha=m.tensor_hash(net.state_dict()),frozen_G1_sha=frozen,trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad))
  assert man["trainable_parameters"]==5572
  if (d/"done.json").exists():
   c.done_valid(c.read(d/"done.json"),lock["source_sha"]);records.append(c.read(d/"metrics.json"));outputs.update(c.read(d/"done.json")["files"]);outputs[str(d/"done.json")]=c.sha(d/"done.json");continue
  opt=optimizer(net);g=torch.Generator().manual_seed(seed);c.write(d/"run_manifest.json",man);ck=d/"checkpoint.pt";history=m.restore(ck,net,opt,g,man) if ck.exists() else []
  for ep in range(len(history)+1,26):
   c.stage("TRAIN_G2_PROCESS_ONLY",seed=seed,epoch=ep,runs_done=len(records));history.append(m.epoch(net,opt,data,data["split"]["FIT"],g,ep));assert m.tensor_hash(net.goal.state_dict())==frozen and all(p.grad is None for p in net.goal.parameters());m.checkpoint(ck,net,opt,g,history,man);c.write(d/"history.json",history)
  assert len(history)==25 and sum(x["optimization_steps"] for x in history)==1850;state=m.tensor_hash(net.state_dict());splits={};jumps={};errs={}
  for split,ix in data["split"].items():
   c.stage("FULL_SAVED_QUALITY_CLUSTERS_MOTION",seed=seed,split=split,runs_done=len(records));splits[split],jumps[split],errs[split]=save_job(net,data,ix,seed,split);assert m.tensor_hash(net.state_dict())==state
  record=dict(seed=seed,epochs=25,splits=splits,jump=jumps,bridge_reconstruct_errors=errs,optimization_steps=1850,G1_all_parameters_buffers_EXACT=True,endpoints_FDE_EXACT=True,model_eval_state_EXACT=True);c.write(d/"metrics.json",record);ff={str(p):c.sha(p) for p in d.iterdir() if p.is_file()};c.write(d/"done.json",dict(status="COMPLETE",source_sha=lock["source_sha"],files=ff));outputs.update(ff);outputs[str(d/"done.json")]=c.sha(d/"done.json");records.append(record);del net,opt;gc.collect();torch.cuda.empty_cache()
 c.stage("BOOTSTRAP_SAVED_ARITHMETIC",runs_done=3);ci=bootstrap(data);c.write("BOOTSTRAP.json",dict(status="PASS",family_size_outer=64,records=ci));outputs[str(c.H/"BOOTSTRAP.json")]=c.sha(c.H/"BOOTSTRAP.json");gate=continuation(records,ci,parent_result["records"],parent_result["baseline_metrics"])
 c.stage("FINAL_SOURCE_PARENT_COMPLETED_SHA",runs_done=3);c.verify(lock["files"]);c.verify(outputs);c.write("ARITHMETIC_AUDIT.json",dict(status="PASS",all12_saved_quality_clusters_motion_bridge_endpoint_FDE_frozen_G1=True));c.write("PRESERVATION.json",dict(status="PASS",source_count=len(lock["files"]),old_sources_models_results_unchanged=True))
 c.write("RESULT.json",dict(status="PASS_G2_FROZEN_GOAL_PROCESS_RUN",records=records,source_sha=lock["source_sha"],output_sha=outputs,source_count=len(lock["files"]),continuation=gate,production_trainings=3,epochs=75,saved_jobs=12,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-started,final_major_not_inferred=True));c.stage("SUCCESS",runs_done=3,continuation=gate["pass_all"])
if __name__=="__main__":main()

