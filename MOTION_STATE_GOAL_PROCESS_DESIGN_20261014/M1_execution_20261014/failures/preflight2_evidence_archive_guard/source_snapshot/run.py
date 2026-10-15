import common as c,model as m,metrics as v,statistics_ci as ci
import numpy as np,torch,time,gc
SEEDS=(42,43,44)
def job(data,ix,paths,name,**extras):
 z=v.sample(data,ix,paths,**extras);c.save(name,**z);back=c.npz(c.H/name)
 met,aux=v.evaluate(back["paths"],back["gt"],back["groups"],back["native20"])
 independent,ia=v.evaluate(back["paths"],back["gt"],back["groups"],back["native20"],True);c.eq(met,independent,2e-11)
 for key,value in aux.items():
  if value.dtype.kind in "bui":assert np.array_equal(value,ia[key])
  else:assert np.max(abs(value-ia[key]))<=2e-11
 aux["vectors5"]=np.column_stack((aux["vectors"],(aux["FDE"]<=.4).any(1)))
 na,nf,nj,ng,_,_=v.errors(back["native20"],back["gt"])
 aux["native_vectors5"]=np.stack((ng.any(1),na.min(1),nf.min(1),nj.min(1),(nf<=.4).any(1)),1)
 c.save(name.replace(".npz","_metrics.npz"),sample_id=back["sample_id"],groups=back["groups"],track=back["track"],**aux)
 jump=np.sqrt(((back["paths"][:,:,0]/.4-back["velocity"][:,None])**2).sum(-1))
 for group,mask in v.masks(back["groups"]):
  met[group]["initial_velocity_jump_diagnostic"]={"mean":float(jump[mask].mean()),"p95":float(np.quantile(jump[mask],.95)),"not_part_of_11_acceleration_gate":True}
 return met
def gates(records,bootstrap):
 A=[next(r for r in records if r["seed"]==s and r["arm"]==m.ARMS[0]) for s in SEEDS]
 B=[next(r for r in records if r["seed"]==s and r["arm"]==m.ARMS[1]) for s in SEEDS]
 cr=next(r for r in bootstrap["records"] if r["split"]=="outer" and r["group"]=="change" and r["reference"]=="A_last_velocity")
 gains=[b["splits"]["outer"]["change"]["endpoint_coverage"]-a["splits"]["outer"]["change"]["endpoint_coverage"] for a,b in zip(A,B)]
 parts=dict(endpoint_all3_positive=all(x>0 for x in gains),endpoint_mean_atleast5pp=np.mean(gains)>=.05,endpoint_family26_CI_lower_positive=cr["CI_family26"][4][0]>0,
  change_FDE_each_not_worse=all(b["splits"]["outer"]["change"]["minFDE"]<=a["splits"]["outer"]["change"]["minFDE"] for a,b in zip(A,B)),
  protected_each_ADEFDE_within1percent=all(b["splits"]["outer"][g][k]<=1.01*a["splits"]["outer"][g][k] for a,b in zip(A,B) for g in ("overall","stable","transition") for k in ("minADE","minFDE")))
 ok=all(parts.values())
 return dict(S1=bool(ok),components=parts,endpoint_gain_pp=[100*x for x in gains],
  M2_parent="M1_dynamic_history_all3" if ok else "existing_G1_all3",M2_independent_not_cancelled_if_M1_negative=True,
  not_major_or_physical_safety=True,original_major_not_relaxed=True)
def binding(data,st,arm,seed,lock):
 m.seed(seed);net=m.MotionGoal(st,arm)
 return dict(c.read(c.H/"protocol.json"),arm=arm,seed=seed,source_sha=lock["source_sha"],source_manifest_sha=c.sha(c.H/"source_manifest.json"),
  statistics_sha=c.sha(c.H/"statistics.npz"),split_ids_sha=c.sha(c.H/"split_ids.npz"),
  input_sha={str(p):c.sha(p) for p in c.INPUTS},G1_centers_statistics_sha=c.sha(c.G1/"statistics.npz"),
  initial_parameters_sha=m.tensor_hash(net.state_dict()),parameter_count=sum(p.numel() for p in net.parameters()),
  train_ids=data["ids"][data["split"]["FIT"]].tolist(),old_F16_not_instantiated=True,NN_forward_inputs=["P2","raw_past","last_observed_velocity"],stage_fresh_not_warmstart=True)
def main():
 started=time.perf_counter();lock=c.read(c.H/"source_manifest.json");c.stage("INITIAL_SOURCE_INPUT_OLD_SHA",runs_done=0);c.verify(lock["files"])
 data=c.load();st=c.npz(c.H/"statistics.npz");assert np.array_equal(st["training_ids"],data["ids"][data["split"]["FIT"]])
 outputs={};records=[];init_hashes={}
 for seed in SEEDS:
  for arm in m.ARMS:
   d=c.H/f"runs/{arm}/seed_{seed}";d.mkdir(parents=True,exist_ok=True);man=binding(data,st,arm,seed,lock)
   if seed in init_hashes:assert init_hashes[seed]==man["initial_parameters_sha"]
   else:init_hashes[seed]=man["initial_parameters_sha"]
   if (d/"done.json").exists():
    done=c.read(d/"done.json");c.done_valid(done,lock["source_sha"]);assert c.read(d/"run_manifest.json")==man
    records.append(c.read(d/"metrics.json"));outputs.update(done["files"]);outputs[str(d/"done.json")]=c.sha(d/"done.json");continue
   m.seed(seed);net=m.MotionGoal(st,arm).to(m.DEVICE);assert m.tensor_hash(net.state_dict())==man["initial_parameters_sha"]
   opt=m.optimizer(net);loader=torch.Generator().manual_seed(seed);c.write(d/"run_manifest.json",man);checkpoint=d/"checkpoint.pt"
   history=m.restore(checkpoint,net,opt,loader,man) if checkpoint.exists() else []
   # Initial fixed-hash128 fixture is not one of the 24 final scientific jobs.
   fixed=c.npz(c.H/"fixed_feature_fixture.npz")["indices"]
   if not (d/"initial128.npz").exists():
    p,g,delta=m.generated(net,data,fixed);assert np.count_nonzero(delta)==0
    c.save(f"runs/{arm}/seed_{seed}/initial128.npz",sample_id=data["ids"][fixed],paths=p,goals=g)
   for ep in range(len(history)+1,26):
    c.stage("TRAIN_M1",arm=arm,seed=seed,epoch=ep,runs_done=len(records),runs_total=6)
    history.append(m.epoch(net,opt,data,data["split"]["FIT"],loader,ep));m.checkpoint(checkpoint,net,opt,loader,history,man);c.write(d/"history.json",history)
   assert len(history)==25 and sum(r["optimization_steps"] for r in history)==1850
   frozen={k:x.detach().cpu().clone() for k,x in net.state_dict().items()};mets={};assignments={}
   for split,ix in data["split"].items():
    c.stage("FULL_SAVED_QUALITY_CLUSTERS_MOTION",arm=arm,seed=seed,split=split,runs_done=len(records))
    p,g,delta=m.generated(net,data,ix);assert np.array_equal(g,4.8*data["vel"][ix,None]+st["centers"][None]+delta.astype(float))
    name=f"runs/{arm}/seed_{seed}/{split}.npz"
    mets[split]=job(data,ix,p,name,goals=g,goal_delta=delta,velocity=data["vel"][ix])
    assignments[split]={grp:{"n":z["n"],"Q_winner_counts":z["winner_counts"]} for grp,z in mets[split].items()}
    del p,g,delta;gc.collect()
   assert all(torch.equal(x.detach().cpu(),frozen[k]) for k,x in net.state_dict().items())
   result=dict(arm=arm,seed=seed,epoch=25,parameter_count=man["parameter_count"],optimization_steps=1850,splits=mets,weights_buffers_eval_EXACT=True)
   c.write(d/"metrics.json",result);c.write(d/"assignment_stats.json",assignments)
   ff={str(p):c.sha(p) for p in d.iterdir() if p.is_file()};c.write(d/"done.json",dict(status="COMPLETE",source_sha=lock["source_sha"],files=ff))
   outputs.update(ff);outputs[str(d/"done.json")]=c.sha(d/"done.json");records.append(result)
   del net,opt,frozen;gc.collect();torch.cuda.empty_cache()
 c.stage("CROSSED_TRACK_SEED_BOOTSTRAP_20000",runs_done=6);boot=ci.compute();c.write("BOOTSTRAP.json",boot);outputs.update(boot["draw_files_sha"]);outputs[str(c.H/"BOOTSTRAP.json")]=c.sha(c.H/"BOOTSTRAP.json")
 decision=gates(records,boot);c.stage("FINAL_SOURCE_OLD_OUTPUT_SHA",runs_done=6);c.verify(lock["files"]);c.verify(outputs)
 c.write("ARITHMETIC_AUDIT.json",dict(status="PASS",scientific_saved_jobs=24,initial128_jobs=6,all_quality_SciPy_clusters_fivemotion_independent=True,same_initial_state_perseed=True,not_all_model_gradient_replay=True))
 c.write("PRESERVATION.json",dict(status="PASS",source_count=len(lock["files"]),old_sources_models_results_unchanged=True,not_all_historical34k=True,old_F16_not_instantiated=True))
 c.write("RESULT.json",dict(status="PASS_M1_DYNAMIC_HISTORY_PAIRED_RUN",records=records,decision=decision,
  production_trainings=6,epochs=150,optimization_steps=11100,saved_jobs=24,source_sha=lock["source_sha"],source_count=len(lock["files"]),output_sha=outputs,
  resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-started,final_major_not_inferred_from_execution=True))
 c.stage("SUCCESS",runs_done=6,S1=decision["S1"],M2_parent=decision["M2_parent"])
if __name__=="__main__":main()
