import common as c,model as m,metrics as v,statistics_ci as ci
import numpy as np,torch,gc,time
SEEDS=(42,43,44)
def geometry(z):
 maximum=0.
 for first in range(0,len(z["paths"]),128):
  c.guard();sl=slice(first,first+128);ex=m.project_numpy(z["raw_b"][sl],z["goals"][sl],z["v0"][sl])
  for k in m.FIELDS:
   error=float(np.max(abs(z[k][sl]-ex[k])));maximum=max(maximum,error)
   assert error<=2e-11,("GEOMETRY_RECONSTRUCT",k,error)
  assert np.max(abs(z["paths"][sl,:,-1]-z["goals"][sl]))<=2e-5
 return maximum
def diagnostics(met,z):
 jump=np.linalg.norm(z["paths"][:,:,0]/.4-z["v0"][:,None],axis=-1)
 displacement=np.linalg.norm(z["goals"]-z["initial_goals"],axis=-1)
 cvgap=np.linalg.norm(z["goals"]-4.8*z["v0"][:,None],axis=-1)
 for g,mask in v.masks(z["groups"]):
  met[g]["initial_velocity_jump_diagnostic"]=dict(mean=float(jump[mask].mean()),p95=float(np.quantile(jump[mask],.95)),not_part_of_11_acceleration_gate=True)
  met[g]["goal_change_from_initial_mean"]=float(displacement[mask].mean())
  met[g]["goal_CV_gap_mean"]=float(cvgap[mask].mean())
 return met
def job(data,ix,z,name,initial_goals):
 assert not ("gt" in z)
 sample=v.sample(data,ix,z["paths"],**{k:a for k,a in z.items() if k!="paths"},initial_goals=initial_goals)
 c.save(name,**sample);back=c.npz(c.H/name);geo=geometry(back)
 met,aux=v.evaluate(back["paths"],back["gt"],back["groups"],back["native20"])
 imet,ia=v.evaluate(back["paths"],back["gt"],back["groups"],back["native20"],True);c.eq(met,imet,2e-11)
 for k,a in aux.items():
  if a.dtype.kind in "biu":assert np.array_equal(a,ia[k]),k
  else:assert np.max(abs(a-ia[k]))<=2e-11,k
 aux["vectors5"]=np.column_stack((aux["vectors"],(aux["FDE"]<=.4).any(1)))
 na,nf,nj,ng,_,_=v.errors(back["native20"],back["gt"])
 aux["native_vectors5"]=np.stack((ng.any(1),na.min(1),nf.min(1),nj.min(1),(nf<=.4).any(1)),1)
 c.save(name.replace(".npz","_metrics.npz"),sample_id=back["sample_id"],groups=back["groups"],track=back["track"],**aux)
 return diagnostics(met,back),geo
def gates(records,boot):
 A=[next(r for r in records if r["seed"]==s and r["arm"]==m.ARMS[0]) for s in SEEDS]
 B=[next(r for r in records if r["seed"]==s and r["arm"]==m.ARMS[1]) for s in SEEDS]
 cr=next(r for r in boot["records"] if r["split"]=="outer" and r["group"]=="change" and r["reference"]=="A_frozen_goal")
 gains=[b["splits"]["outer"]["change"]["coverage"]-a["splits"]["outer"]["change"]["coverage"] for a,b in zip(A,B)]
 parts=dict(coverage_all3_positive=all(x>0 for x in gains),coverage_mean_atleast5pp=np.mean(gains)>=.05,coverage_family26_CI_lower_positive=cr["CI_family26"][0][0]>0,
  change_ADE_FDE_mean_not_worse=all(np.mean([b["splits"]["outer"]["change"][k] for b in B])<=np.mean([a["splits"]["outer"]["change"][k] for a in A]) for k in ("minADE","minFDE")),
  protected_each_ADEFDE_within_A1percent=all(b["splits"]["outer"][g][k]<=1.01*a["splits"]["outer"][g][k] for a,b in zip(A,B) for g in ("overall","stable","transition") for k in ("minADE","minFDE")))
 qparts=dict(S2=all(parts.values()),change_coverage_each_atleast_native=all(b["splits"]["outer"]["change"]["good"]>=110 for b in B),
  protected_each_ADEFDE_within_native1percent=True,all48_fiveMotion=all(not z["motion_failed"] for b in B for spl in b["splits"].values() for z in spl.values()))
 data=c.load();ix=data["split"]["outer"];na,nf,_,_,_,_=v.errors(data["native"][ix].astype(float),data["gt"][ix])
 for g,mask in v.masks(data["groups"][ix]):
  if g=="change":continue
  for b in B:
   for k,ref in (("minADE",na[mask].min(1).mean()),("minFDE",nf[mask].min(1).mean())):
    if b["splits"]["outer"][g][k]>1.01*ref:qparts["protected_each_ADEFDE_within_native1percent"]=False
 return dict(S2=bool(all(parts.values())),S2_components=parts,coverage_gain_pp=[100*x for x in gains],Q2=bool(all(qparts.values())),Q2_components=qparts,M3_allowed=bool(all(qparts.values())),not_major=True,original_major_not_relaxed=True)
def binding(data,arm,s,lock,net):
 return dict(c.read(c.H/"protocol.json"),arm=arm,seed=s,source_sha=lock["source_sha"],source_manifest_sha=c.sha(c.H/"source_manifest.json"),
  statistics_sha=c.sha(c.H/f"goal_stats_{s}.npz"),split_ids_sha=c.sha(c.H/"split_ids.npz"),G1_checkpoint_sha=c.sha(c.G1/f"runs/seed_{s}/checkpoint.pt"),
  input_sha={str(p):c.sha(p) for p in c.INPUTS},initial_parameters_sha=m.tensor_hash(net.state_dict()),
  parameter_count=sum(p.numel() for p in net.parameters()),trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
  train_ids=data["ids"][data["split"]["FIT"]].tolist(),old_F16_not_instantiated=True)
def main():
 started=time.perf_counter();lock=c.read(c.H/"source_manifest.json");c.stage("INITIAL_SOURCE_INPUT_OLD_SHA",runs_done=0);c.verify(lock["files"]);data=c.load()
 outputs={};initial_records=[];records=[]
 for s in SEEDS:
  net=m.fresh(s,m.ARMS[0]);original={k:x.detach().cpu().clone() for k,x in net.state_dict().items()}
  idir=c.H/f"initial/seed_{s}";idir.mkdir(parents=True,exist_ok=True)
  if (idir/"done.json").exists():
   done=c.read(idir/"done.json");c.done_valid(done,lock["source_sha"]);outputs.update(done["files"]);initial_records.append(c.read(idir/"metrics.json"))
  else:
   mets={}
   for spl,ix in data["split"].items():
    c.stage("FULL_INITIAL_P0_SAVE_QUALITY",seed=s,split=spl,runs_done=len(records))
    z=m.generated(net,data,ix);assert not np.count_nonzero(z["raw_b"]);mets[spl],_=job(data,ix,z,f"initial/seed_{s}/{spl}.npz",z["goals"])
    del z;gc.collect()
   m.frozen_check(net,original);rec=dict(seed=s,splits=mets,initial_P0_not_native_or_G1_identity=True)
   c.write(idir/"metrics.json",rec);ff={str(p):c.sha(p) for p in idir.iterdir() if p.is_file()}
   c.write(idir/"done.json",dict(status="COMPLETE",source_sha=lock["source_sha"],files=ff));outputs.update(ff);initial_records.append(rec)
  outputs[str(idir/"done.json")]=c.sha(idir/"done.json");del net;gc.collect();torch.cuda.empty_cache()
  for arm in m.ARMS:
   d=c.H/f"runs/{arm}/seed_{s}";d.mkdir(parents=True,exist_ok=True);net=m.fresh(s,arm);man=binding(data,arm,s,lock,net)
   assert man["parameter_count"]==33306 and man["trainable_parameters"]==(7000 if arm==m.ARMS[0] else 12314)
   if (d/"done.json").exists():
    done=c.read(d/"done.json");c.done_valid(done,lock["source_sha"]);assert c.read(d/"run_manifest.json")==man
    outputs.update(done["files"]);outputs[str(d/"done.json")]=c.sha(d/"done.json");records.append(c.read(d/"metrics.json"));del net;continue
   opt=m.optimizer(net);loader=torch.Generator().manual_seed(s);c.write(d/"run_manifest.json",man);ck=d/"checkpoint.pt"
   history=m.restore(ck,net,opt,loader,man) if ck.exists() else []
   for ep in range(len(history)+1,26):
    c.stage("TRAIN_M2",arm=arm,seed=s,epoch=ep,runs_done=len(records),runs_total=6)
    history.append(m.epoch(net,opt,data,data["split"]["FIT"],loader,ep));m.frozen_check(net,original)
    m.checkpoint(ck,net,opt,loader,history,man);c.write(d/"history.json",history)
   assert len(history)==25 and sum(r["optimization_steps"] for r in history)==1850
   evalstate={k:a.detach().cpu().clone() for k,a in net.state_dict().items()};mets={};assign={}
   for spl,ix in data["split"].items():
    c.stage("FULL_FINAL_SAVE_QUALITY_CLUSTERS_MOTION",arm=arm,seed=s,split=spl,runs_done=len(records))
    z=m.generated(net,data,ix);p0=c.npz(idir/f"{spl}.npz")["goals"]
    if arm==m.ARMS[0]:assert np.array_equal(z["goals"],p0),"FROZEN_GOAL_CHANGED"
    mets[spl],_=job(data,ix,z,f"runs/{arm}/seed_{s}/{spl}.npz",p0)
    assign[spl]={g:{"n":x["n"],"Q_winner_counts":x["winner_counts"]} for g,x in mets[spl].items()}
    del z,p0;gc.collect()
   assert all(torch.equal(a.detach().cpu(),evalstate[k]) for k,a in net.state_dict().items())
   rec=dict(arm=arm,seed=s,epoch=25,parameter_count=man["parameter_count"],trainable_parameters=man["trainable_parameters"],optimization_steps=1850,splits=mets,weights_buffers_eval_EXACT=True,frozen_parent_contract_EXACT=True)
   c.write(d/"metrics.json",rec);c.write(d/"assignment_stats.json",assign);ff={str(p):c.sha(p) for p in d.iterdir() if p.is_file()}
   c.write(d/"done.json",dict(status="COMPLETE",source_sha=lock["source_sha"],files=ff));outputs.update(ff);outputs[str(d/"done.json")]=c.sha(d/"done.json");records.append(rec)
   del net,opt,evalstate;gc.collect();torch.cuda.empty_cache()
 c.stage("CROSSED_TRACK_SEED_BOOTSTRAP_20000",runs_done=6);boot=ci.compute();c.write("BOOTSTRAP.json",boot);outputs.update(boot["draw_files_sha"]);outputs[str(c.H/"BOOTSTRAP.json")]=c.sha(c.H/"BOOTSTRAP.json")
 decision=gates(records,boot);c.stage("FINAL_SOURCE_OLD_OUTPUT_SHA",runs_done=6);c.verify(lock["files"]);c.verify(outputs)
 c.write("ARITHMETIC_AUDIT.json",dict(status="PASS",scientific_saved_jobs=24,initial_P0_jobs=12,all_saved_quality_clusters_fiveMotion_geometry_independent=True))
 c.write("PRESERVATION.json",dict(status="PASS",source_count=len(lock["files"]),old_sources_models_results_unchanged=True,not_all_historical34k=True))
 c.write("RESULT.json",dict(status="PASS_M2_JOINT_GOAL_PROCESS_PAIRED_RUN",records=records,initial_records=initial_records,decision=decision,production_trainings=6,epochs=150,optimization_steps=11100,saved_jobs=24,initial_P0_jobs=12,source_sha=lock["source_sha"],source_count=len(lock["files"]),output_sha=outputs,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-started))
 c.stage("SUCCESS",runs_done=6,S2=decision["S2"],Q2=decision["Q2"])
if __name__=="__main__":main()
