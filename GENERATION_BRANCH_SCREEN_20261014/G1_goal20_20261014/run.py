import common as c,model as m,metrics as v,baselines as b
import numpy as np,torch,time,gc
from pathlib import Path
SEEDS=(42,43,44)
def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,foreach=False)
def binding(data,st,seed,lock):
 d=c.H/"runs"/("seed_"+str(seed));d.mkdir(parents=True,exist_ok=True);m.seed(seed);goal=m.Goal20(st)
 return d,dict(c.read(c.H/"protocol.json"),seed=seed,source_sha=lock["source_sha"],source_manifest_sha=c.sha(c.H/"source_manifest.json"),statistics_sha=c.sha(c.H/"statistics.npz"),split_sha=c.sha(c.INPUTS[-1]),split_ids_sha=c.sha(c.H/"split_ids.npz"),initial_parameters_sha=m.tensor_hash(goal.state_dict()),train_ids=data["ids"][data["split"]["FIT"]].tolist(),parameter_count=sum(x.numel() for x in goal.parameters()),fixed_initial_goal_is_prototype20=True)
def saved_job(data,ix,p,path,**extras):
 z=v.sample(data,ix,p,**extras);c.save(path,**z);back=c.npz(c.H/path);met,aux=v.evaluate(back["paths"],back["gt"],back["groups"],back["native20"]);ind,i_aux=v.evaluate(back["paths"],back["gt"],back["groups"],back["native20"],True);c.eq(met,ind,2e-11)
 for key in aux:
  if aux[key].dtype.kind in "bui":assert np.array_equal(aux[key],i_aux[key])
  else:assert np.max(abs(aux[key]-i_aux[key]))<=2e-11
 c.save(str(Path(path).with_name(Path(path).stem+"_metrics.npz")),sample_id=z["sample_id"],groups=z["groups"],track=z["track"],**aux)
 return met
def bootstrap(data):
 sets={seed:{split:c.npz(c.H/f"runs/seed_{seed}/{split}_metrics.npz") for split in data["split"]} for seed in SEEDS}
 records=[];rng=np.random.default_rng(20261014);comparisons=("native20","prototype20","retrieval20");metric_names=("coverage","ADE","FDE","J")
 for split,ix in data["split"].items():
  for group,mask in v.masks(data["groups"][ix]):
   units,inv=np.unique(data["track"][ix][mask],return_inverse=True);counts=np.bincount(inv);T=len(units)
   for ref in comparisons:
    rz=c.npz(c.H/f"baseline/{split}_{ref}_metrics.npz");diff=np.stack([sets[s][split]["vectors"][mask]-rz["vectors"][mask] for s in SEEDS]);sums=np.zeros((3,T,4))
    for s in range(3):np.add.at(sums[s],inv,diff[s])
    draws=[]
    for _ in range(2000):
     c.guard();ss=rng.integers(0,3,3);tt=rng.integers(0,T,T);draws.append(sums[ss][:,tt].sum((0,1))/(3*counts[tt].sum()))
    records.append(dict(split=split,group=group,reference=ref,n=int(mask.sum()),tracks=T,metrics=metric_names,estimate=diff.mean((0,1)),CI_family48=np.quantile(draws,[.05/96,1-.05/96],axis=0).T,perseed_estimate=diff.mean(1),exploratory_fixeddata_seedtrack=True))
 return records
def main():
 started=time.perf_counter();lock=c.read(c.H/"source_manifest.json");c.stage("INITIAL_SOURCE_OLD_SHA",runs_done=0,runs_total=3);c.verify(lock["files"]);data=c.load();st=c.npz(c.H/"statistics.npz");assert np.array_equal(st["training_ids"],data["ids"][data["split"]["FIT"]]);outputs={};baseline_metrics={}
 for split,ix in data["split"].items():
  baseline_metrics[split]={}
  for kind in ("native20","CV1","prototype20","retrieval20"):
   c.stage("FROZEN_BASELINE_SAVE_QUALITY",split=split,port=kind,runs_done=0)
   if kind=="native20":paths=data["native"][ix].astype(float);donors=None
   else:paths,donors=b.build(data,st,ix,kind)
   name=f"baseline/{split}_{kind}.npz";extras={} if donors is None else {"donor_sample_id":donors};met=saved_job(data,ix,paths,name,**extras);baseline_metrics[split][kind]=met
   for path in (c.H/name,(c.H/name).with_name(Path(name).stem+"_metrics.npz")):outputs[str(path)]=c.sha(path)
   del paths;gc.collect()
 records=[]
 for seed in SEEDS:
  d,man=binding(data,st,seed,lock)
  if (d/"done.json").exists():
   c.done_valid(c.read(d/"done.json"),lock["source_sha"]);records.append(c.read(d/"metrics.json"));outputs.update(c.read(d/"done.json")["files"]);outputs[str(d/"done.json")]=c.sha(d/"done.json");continue
  m.seed(seed);goal=m.Goal20(st).to(m.DEVICE);assert m.tensor_hash(goal.state_dict())==man["initial_parameters_sha"];opt=optimizer(goal);g=torch.Generator().manual_seed(seed);c.write(d/"run_manifest.json",man);checkpoint=d/"checkpoint.pt";history=m.restore(checkpoint,goal,opt,g,man) if checkpoint.exists() else []
  for ep in range(len(history)+1,26):
   c.stage("TRAIN_G1",seed=seed,epoch=ep,runs_done=len(records),runs_total=3);row=m.epoch(goal,opt,data,data["split"]["FIT"],g,ep);history.append(row);m.checkpoint(checkpoint,goal,opt,g,history,man);c.write(d/"history.json",history)
  assert len(history)==25 and all(row["n"]==18909 for row in history);frozen={k:x.detach().cpu().clone() for k,x in goal.state_dict().items()};metrics={}
  for split,ix in data["split"].items():
   c.stage("FULL_SAVED_QUALITY_CLUSTERS_MOTION",seed=seed,split=split,runs_done=len(records));paths,goals,delta=m.generated(goal,data,ix);name=f"runs/seed_{seed}/{split}.npz";met=saved_job(data,ix,paths,name,goals=goals,goal_delta=delta);metrics[split]=met
   assert np.array_equal(goals,4.8*data["vel"][ix,None]+st["centers"][None]+delta.astype(float))
   del paths,goals,delta;gc.collect()
  assert all(torch.equal(x.detach().cpu(),frozen[k]) for k,x in goal.state_dict().items());result=dict(seed=seed,epoch=25,splits=metrics,parameter_count=man["parameter_count"],optimization_steps=sum(row["optimization_steps"] for row in history),all_weights_buffers_eval_EXACT=True);c.write(d/"metrics.json",result)
  ff={str(path):c.sha(path) for path in d.iterdir() if path.is_file()};c.write(d/"done.json",dict(status="COMPLETE",source_sha=lock["source_sha"],files=ff));outputs.update(ff);outputs[str(d/"done.json")]=c.sha(d/"done.json");records.append(result);del goal,opt,frozen;gc.collect();torch.cuda.empty_cache()
 c.stage("BOOTSTRAP_SAVED_ARITHMETIC",runs_done=3);ci=bootstrap(data);c.write("BOOTSTRAP.json",dict(status="PASS",family_size_outer=48,records=ci,default_rng=20261014));outputs[str(c.H/"BOOTSTRAP.json")]=c.sha(c.H/"BOOTSTRAP.json")
 n=int((data["groups"][data["split"]["outer"]]==2).sum());native=baseline_metrics["outer"]["native20"]["change"]["good"];counts=[r["splits"]["outer"]["change"]["endpoint_good"] for r in records];gate=all((count-native)/n>=.05 for count in counts)
 c.stage("FINAL_SOURCE_OLD_SHA",runs_done=3);c.verify(lock["files"]);c.verify(outputs);c.guard();c.write("ARITHMETIC_AUDIT.json",dict(status="PASS",all_saved_quality_clusters_motion_independent=True,all_trained_model_eval_states_EXACT=True,saved_baseline_jobs=16,saved_model_jobs=12,training_epochs=75,not_all_model_gradient_replay=True));c.write("PRESERVATION.json",dict(status="PASS",source_count=len(lock["files"]),old_models_not_instantiated_or_trained=True))
 c.write("RESULT.json",dict(status="PASS_G1_GOAL20_RUN",records=records,baseline_metrics=baseline_metrics,source_sha=lock["source_sha"],source_count=len(lock["files"]),output_sha=outputs,G2_upper_screen=dict(native_complete_good=native,change_windows=n,learned_endpoint_good=counts,perseed_gain_pp=[(count-native)/n*100 for count in counts],pass_all3=gate,not_major_success=True),saved_jobs=28,production_trainings=3,epochs=75,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-started,final_major_not_inferred=True));c.stage("SUCCESS",runs_done=3,G2_upper_screen=gate)
if __name__=="__main__":main()
