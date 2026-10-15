"""Independent CPU saved-data audit. CPU checkpoint load only; no NN forward/optimizer."""
import common as c,metrics as v
import numpy as np,time,gc,traceback
def main():
 import torch,model as m,run,statistics_ci as ci
 start=time.perf_counter();lock=c.read(c.H/"source_manifest.json");res=c.read(c.H/"RESULT.json")
 assert c.read(c.H/"worker_state.json")["status"]==c.read(c.H/"exit_status.json")["status"]=="SUCCESS"
 assert res["status"]=="PASS_M2_JOINT_GOAL_PROCESS_PAIRED_RUN"
 c.verify(lock["files"]);c.verify(res["output_sha"]);data=c.load();cert=[];events=[];initials={}
 def check_saved(path,ix,ref):
  z=c.npz(path);stored=c.npz(path.with_name(path.stem+"_metrics.npz"));assert z["paths"].shape==(len(ix),20,12,2)
  for k,expected in (("sample_id",data["ids"][ix]),("groups",data["groups"][ix]),("scene",data["scene"][ix]),("track",data["track"][ix]),("gt",data["gt"][ix]),("native20",data["native"][ix].astype(float)),("v0",data["vel"][ix])):
   assert np.array_equal(z[k],expected),(str(path),k)
  assert len(np.unique(z["sample_id"]))==len(ix) and all(np.isfinite(z[k]).all() for k in m.FIELDS)
  geo=run.geometry(z);met,aux=v.evaluate(z["paths"],z["gt"],z["groups"],z["native20"],True);met=run.diagnostics(met,z);c.eq(c.safe(met),ref,2e-11)
  for k,a in aux.items():
   if a.dtype.kind in "biu":assert np.array_equal(a,stored[k]),k
   else:assert np.max(abs(a-stored[k]))<=2e-11,k
  assert np.array_equal(stored["sample_id"],z["sample_id"])
  vectors5=np.column_stack((aux["vectors"],(aux["FDE"]<=.4).any(1)));assert np.max(abs(vectors5-stored["vectors5"]))<=2e-11
  na,nf,nj,ng,_,_=v.errors(z["native20"],z["gt"],True)
  nv=np.stack((ng.any(1),na.min(1),nf.min(1),nj.min(1),(nf<=.4).any(1)),1);assert np.max(abs(nv-stored["native_vectors5"]))<=2e-11
  cert.append(dict(path=str(path),n=len(ix),geometry_error=geo,all_quality_full_SciPy_clusters_fiveMotion_PASS=True))
  return z
 for rec in res["initial_records"]:
  s=rec["seed"];d=c.H/f"initial/seed_{s}";c.done_valid(c.read(d/"done.json"),lock["source_sha"])
  for spl,ix in data["split"].items():
   z=check_saved(d/f"{spl}.npz",ix,rec["splits"][spl]);assert not np.count_nonzero(z["raw_b"]) and np.array_equal(z["goals"],z["initial_goals"])
   del z;gc.collect()
 for rec in res["records"]:
  arm=rec["arm"];s=rec["seed"];d=c.H/f"runs/{arm}/seed_{s}";man=c.read(d/"run_manifest.json");c.done_valid(c.read(d/"done.json"),lock["source_sha"])
  ck=torch.load(d/"checkpoint.pt",map_location="cpu",weights_only=False);parent=torch.load(c.G1/f"runs/seed_{s}/checkpoint.pt",map_location="cpu",weights_only=False)
  assert ck["manifest"]==man and man["source_sha"]==lock["source_sha"] and man["source_manifest_sha"]==c.sha(c.H/"source_manifest.json")
  assert man["G1_checkpoint_sha"]==c.sha(c.G1/f"runs/seed_{s}/checkpoint.pt")
  assert man["statistics_sha"]==c.sha(c.H/f"goal_stats_{s}.npz") and man["split_ids_sha"]==c.sha(c.H/"split_ids.npz")
  assert man["parameter_count"]==33306 and man["trainable_parameters"]==(7000 if arm=="frozen_goal" else 12314)
  assert np.array_equal(man["train_ids"],data["ids"][data["split"]["FIT"]])
  assert ck["epoch"]==25 and len(ck["history"])==25 and c.safe(ck["history"])==c.read(d/"history.json")
  assert all(x["n"]==18909 and x["optimization_steps"]==74 for x in ck["history"]) and sum(x["optimization_steps"] for x in ck["history"])==1850
  assert set(ck["rng"])=={"torch","cuda","loader","numpy","python"}
  op=ck["optimizer"]["param_groups"][0];assert op["lr"]==.001 and op["weight_decay"]==.0001 and op["eps"]==1e-8 and op["betas"]==(.9,.999) and not op["foreach"]
  assert all(int(x["step"].item())==1850 for x in ck["optimizer"]["state"].values())
  if s in initials:assert initials[s]==man["initial_parameters_sha"]
  else:initials[s]=man["initial_parameters_sha"]
  for k,a in parent["model"].items():
   if arm=="joint_goal" and k.startswith("head."):continue
   assert torch.equal(ck["model"]["goal."+k],a),("FROZEN_PARENT",arm,k)
  gs=c.npz(c.H/f"goal_stats_{s}.npz")
  for k in ("goal_mean","goal_std","v0_mean","v0_std"):assert np.array_equal(ck["model"][k].numpy(),gs[k])
  for spl,ix in data["split"].items():
   z=check_saved(d/f"{spl}.npz",ix,rec["splits"][spl]);p0=c.npz(c.H/f"initial/seed_{s}/{spl}.npz")
   assert np.array_equal(z["initial_goals"],p0["goals"])
   if arm=="frozen_goal":assert np.array_equal(z["goals"],p0["goals"])
   _,pf,_,pg,_,_=v.errors(p0["paths"],p0["gt"],True);_,zf,_,zg,_,_=v.errors(z["paths"],z["gt"],True)
   for group,mask in v.masks(z["groups"]):
    base=pg.any(1);good=zg.any(1)
    events.append(dict(kind="P0_to_final",arm=arm,seed=s,split=spl,group=group,n=int(mask.sum()),rescued=int((good&~base&mask).sum()),lost=int((~good&base&mask).sum()),endpoint_rescued=int(((zf<=.4).any(1)&~(pf<=.4).any(1)&mask).sum()),endpoint_lost=int((~(zf<=.4).any(1)&(pf<=.4).any(1)&mask).sum())))
   del z,p0;gc.collect()
  del ck,parent;gc.collect()
 for s in (42,43,44):
  for spl,ix in data["split"].items():
   a=c.npz(c.H/f"runs/frozen_goal/seed_{s}/{spl}_metrics.npz");b=c.npz(c.H/f"runs/joint_goal/seed_{s}/{spl}_metrics.npz")
   for group,mask in v.masks(a["groups"]):
    ga=a["vectors5"][:,0].astype(bool);gb=b["vectors5"][:,0].astype(bool);ep=a["vectors5"][:,4].astype(bool)&b["vectors5"][:,4].astype(bool)&mask
    events.append(dict(kind="A_to_B",seed=s,split=spl,group=group,n=int(mask.sum()),rescued=int((gb&~ga&mask).sum()),lost=int((~gb&ga&mask).sum()),retained=int((gb&ga&mask).sum()),common_endpointgood_n=int(ep.sum()),common_endpointgood_A=int((ga&ep).sum()),common_endpointgood_B=int((gb&ep).sum()),intersection_posttreatment_descriptive_only=True))
 boot=ci.compute(independent=True);c.eq(c.safe(boot),c.read(c.H/"BOOTSTRAP.json"),2e-11);c.eq(c.safe(run.gates(res["records"],boot)),res["decision"],2e-11)
 c.verify(lock["files"]);c.verify(res["output_sha"]);c.guard()
 c.write("completion_review/ATTRIBUTION.json",dict(status="PASS",events=events,main_denominators_unchanged=True))
 c.write("completion_review/REVIEW.json",dict(status="PASS",scientific_saved_jobs=24,initial_P0_jobs=12,checkpoint_contracts=6,history_rows=150,source_count=len(lock["files"]),output_count=len(res["output_sha"]),certificates=cert,bootstrap_draws=20000,family_size_reserved=26,per_draw_SHA_reconstructed=True,independent_track_aggregation_same_registered_draws=True,CPU_checkpoint_load_only_no_NN_forward_optimizer_or_gradient_replay=True,not_external_review=True,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-start))
 print("AUDIT_PASS",flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:traceback.print_exc();c.write("completion_review/FAILURE.json",dict(status="STOP",error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
