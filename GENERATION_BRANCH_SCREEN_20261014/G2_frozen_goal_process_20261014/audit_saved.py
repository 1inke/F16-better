"""Full saved CPU arithmetic and checkpoint audit; no neural forward/optimizer step."""
import common as c,metrics as v
import numpy as np,time,traceback
def main():
 import torch
 import run
 started=time.perf_counter();lock=c.read(c.H/"source_manifest.json");result=c.read(c.H/"RESULT.json");assert c.read(c.H/"worker_state.json")["status"]==c.read(c.H/"exit_status.json")["status"]=="SUCCESS" and result["status"]=="PASS_G2_FROZEN_GOAL_PROCESS_RUN";c.verify(lock["files"]);c.verify(result["output_sha"]);data=c.load();parent_result=c.read(c.OLD/"RESULT.json");cert=[]
 for record in result["records"]:
  seed=record["seed"];base=c.H/f"runs/seed_{seed}";man=c.read(base/"run_manifest.json");assert man["source_sha"]==lock["source_sha"] and man["G1_checkpoint_sha"]==c.sha(c.OLD/f"runs/seed_{seed}/checkpoint.pt") and man["goal_statistics_sha"]==c.sha(c.H/f"goal_stats_{seed}.npz");c.done_valid(c.read(base/"done.json"),lock["source_sha"]);ck=torch.load(base/"checkpoint.pt",map_location="cpu",weights_only=False);old=torch.load(c.OLD/f"runs/seed_{seed}/checkpoint.pt",map_location="cpu",weights_only=False)
  assert ck["manifest"]==man and ck["epoch"]==25 and c.safe(ck["history"])==c.read(base/"history.json") and sum(x["optimization_steps"] for x in ck["history"])==1850 and all(x["n"]==18909 for x in ck["history"])
  for key,tensor in old["model"].items():assert torch.equal(tensor,ck["model"]["goal."+key]),("FROZEN_G1_CHANGED",seed,key)
  st=c.npz(c.H/f"goal_stats_{seed}.npz");source=c.npz(c.OLD/f"runs/seed_{seed}/FIT.npz");assert np.array_equal(st["training_ids"],source["sample_id"]);g=source["goals"];assert np.array_equal(st["mean"],g.mean((0,1))) and np.array_equal(st["std"],np.where(g.std((0,1))<1e-6,1.,g.std((0,1))));assert np.array_equal(ck["model"]["goal_mean"].numpy(),st["mean"]) and np.array_equal(ck["model"]["goal_std"].numpy(),st["std"]);del source,g
  assert len(ck["optimizer"]["param_groups"][0]["params"])==4 and ck["optimizer"]["param_groups"][0]["lr"]==.001 and ck["optimizer"]["param_groups"][0]["weight_decay"]==.0001;assert set(ck["rng"])=={"torch","cuda","loader","numpy","python"}
  for split,ix in data["split"].items():
   saved=c.npz(base/f"{split}.npz");original=c.npz(c.OLD/f"runs/seed_{seed}/{split}.npz");assert np.array_equal(saved["sample_id"],data["ids"][ix]) and len(np.unique(saved["sample_id"]))==len(ix)
   for key in ("sample_id","groups","track","scene","gt","native20"):assert np.array_equal(saved[key],original[key])
   assert saved["paths"].shape==(len(ix),20,12,2) and np.isfinite(saved["paths"]).all() and np.isfinite(saved["controls"]).all();error=run.geometry(saved["paths"],saved["goals"],saved["controls"],saved["base_paths"],original);met,aux=v.evaluate(saved["paths"],saved["gt"],saved["groups"],saved["native20"],True);c.eq(c.safe(met),record["splits"][split],2e-11);c.eq(run.jump_report(saved["paths"],saved["base_paths"],data["vel"][ix],saved["groups"]),record["jump"][split],2e-11);stored=c.npz(base/f"{split}_metrics.npz")
   for k in ("sample_id","track","groups"):assert np.array_equal(stored[k],saved[k])
   for k,a in aux.items():
    if a.dtype.kind in "biu":assert np.array_equal(a,stored[k])
    else:assert np.max(abs(a-stored[k]))<=2e-11
   assert np.array_equal(aux["FDE"],c.npz(c.OLD/f"runs/seed_{seed}/{split}_metrics.npz")["FDE"]);cert.append(dict(seed=seed,split=split,n=len(ix),bridge_error=error,same_goals_FDE_EXACT=True,full_quality_SciPy_clusters_motion=True))
 ci=run.bootstrap(data,True);c.eq(c.read(c.H/"BOOTSTRAP.json")["records"],c.safe(ci),2e-11);gate=run.continuation(result["records"],ci,parent_result["records"],parent_result["baseline_metrics"]);c.eq(c.safe(gate),result["continuation"],2e-11);c.verify(lock["files"]);c.verify(result["output_sha"])
 c.write("completion_review/REVIEW.json",dict(status="PASS",full12_saved_quality_clusters_motion_controls_goals_endpointFDE_fixedG1_checkpoints_75history_64CI256scalar=True,source_count=len(lock["files"]),output_count=len(result["output_sha"]),certificates=cert,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-started,CPUcheckpoint_only_not_neuralforward_optimizer_or_allgradient_replay=True,CI_independent_track_aggregation_same_registered_draws=True,not_external_review=True));print("AUDIT_PASS",flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:traceback.print_exc();c.write("completion_review/FAILURE.json",dict(status="STOP",error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise

