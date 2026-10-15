"""CPU saved-data/checkpoint audit; no NN forward, optimizer step, or external review."""
import common as c,metrics as v
import numpy as np,time,traceback,gc
def main():
 import torch,statistics_ci as ci,run
 started=time.perf_counter();lock=c.read(c.H/"source_manifest.json");result=c.read(c.H/"RESULT.json")
 assert c.read(c.H/"worker_state.json")["status"]==c.read(c.H/"exit_status.json")["status"]=="SUCCESS"
 assert result["status"]=="PASS_M1_DYNAMIC_HISTORY_PAIRED_RUN"
 c.verify(lock["files"]);c.verify(result["output_sha"]);data=c.load();st=c.npz(c.H/"statistics.npz");c.eq(c.safe(st),c.safe(c.stats(data)),1e-12)
 certificates=[];initials={}
 for record in result["records"]:
  arm=record["arm"];seed=record["seed"];d=c.H/f"runs/{arm}/seed_{seed}";man=c.read(d/"run_manifest.json")
  c.done_valid(c.read(d/"done.json"),lock["source_sha"]);assert man["source_sha"]==lock["source_sha"] and man["statistics_sha"]==c.sha(c.H/"statistics.npz")
  ck=torch.load(d/"checkpoint.pt",map_location="cpu",weights_only=False)
  assert ck["manifest"]==man and ck["epoch"]==25 and c.safe(ck["history"])==c.read(d/"history.json") and len(ck["history"])==25
  assert all(r["n"]==18909 and r["optimization_steps"]==74 for r in ck["history"]) and sum(r["optimization_steps"] for r in ck["history"])==1850
  assert set(ck["rng"])=={"torch","cuda","loader","numpy","python"} and man["parameter_count"]==34146
  for key in ("mean","std","centers","velocity_mean","velocity_std","acceleration_mean","acceleration_std"):
   assert np.array_equal(ck["model"][key].numpy(),st[key])
  op=ck["optimizer"]["param_groups"][0]
  assert op["lr"]==.001 and op["weight_decay"]==.0001 and op["betas"]==(.9,.999) and op["eps"]==1e-8 and op["foreach"]==False
  assert all(int(x["step"].item())==1850 for x in ck["optimizer"]["state"].values())
  if seed in initials:assert initials[seed]==man["initial_parameters_sha"]
  else:initials[seed]=man["initial_parameters_sha"]
  fixed=c.npz(d/"initial128.npz");fixedref=c.npz(c.H/"fixed_feature_fixture.npz")
  proto=4.8*fixedref["v0"][:,None]+st["centers"][None]
  assert np.array_equal(fixed["sample_id"],fixedref["sample_id"]) and np.array_equal(fixed["goals"],proto)
  for split,ix in data["split"].items():
   z=c.npz(d/f"{split}.npz");stored=c.npz(d/f"{split}_metrics.npz")
   assert z["paths"].shape==(len(ix),20,12,2) and np.isfinite(z["paths"]).all() and len(np.unique(z["sample_id"]))==len(ix)
   for k,ref in (("sample_id",data["ids"][ix]),("groups",data["groups"][ix]),("track",data["track"][ix]),("scene",data["scene"][ix]),("gt",data["gt"][ix]),("native20",data["native"][ix].astype(float)),("velocity",data["vel"][ix])):
    assert np.array_equal(z[k],ref),(arm,seed,split,k)
   expected=(np.arange(1,13)/12.)[None,None,:,None]*z["goals"][:,:,None]
   assert np.array_equal(z["paths"],expected) and np.array_equal(z["goals"],4.8*z["velocity"][:,None]+st["centers"][None]+z["goal_delta"].astype(float))
   met,aux=v.evaluate(z["paths"],z["gt"],z["groups"],z["native20"],True);c.eq(c.safe(met),record["splits"][split],2e-11)
   for k,a in aux.items():
    if a.dtype.kind in "bui":assert np.array_equal(a,stored[k])
    else:assert np.max(abs(a-stored[k]))<=2e-11
   assert np.array_equal(stored["sample_id"],z["sample_id"])
   vectors5=np.column_stack((aux["vectors"],(aux["FDE"]<=.4).any(1)));assert np.array_equal(vectors5,stored["vectors5"])
   a,f,j,g,_,_=v.errors(z["native20"],z["gt"],True)
   native5=np.stack((g.any(1),a.min(1),f.min(1),j.min(1),(f<=.4).any(1)),1);assert np.max(abs(native5-stored["native_vectors5"]))<=2e-11
   certificates.append(dict(arm=arm,seed=seed,split=split,n=len(ix),full_samepath_quality_SciPy_clusters_motion_goals=True))
   del z,stored;gc.collect()
  del ck;gc.collect()
 boot=ci.compute(independent=True);c.eq(c.safe(boot),c.read(c.H/"BOOTSTRAP.json"),2e-11)
 c.eq(c.safe(run.gates(result["records"],boot)),result["decision"],2e-11);c.verify(lock["files"]);c.verify(result["output_sha"]);c.guard()
 c.write("completion_review/REVIEW.json",dict(status="PASS",scientific_saved_jobs=24,checkpoint_contracts=6,history_rows=150,source_count=len(lock["files"]),output_count=len(result["output_sha"]),
  certificates=certificates,bootstrap_draws=20000,family_size_reserved=26,per_draw_SHA_reconstructed=True,independent_track_aggregation_same_registered_draws=True,
  CPU_checkpoint_load_only_no_NN_forward_optimizer_or_gradient_replay=True,not_external_review=True,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-started))
 print("AUDIT_PASS",flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:traceback.print_exc();c.write("completion_review/FAILURE.json",dict(status="STOP",error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
