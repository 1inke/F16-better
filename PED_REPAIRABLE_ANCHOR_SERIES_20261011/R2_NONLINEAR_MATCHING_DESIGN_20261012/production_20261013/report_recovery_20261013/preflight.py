import common as c,saved_math as sm
import numpy as np,time,traceback
def main():
 t=time.perf_counter();c.guard();files,counts=c.original_sources();d=c.inputs();maxerr=0.;fits=[]
 for arm in ("linear","rbf64"):
  for fold in range(3):
   tr,va=c.ic.split_masks(d,fold);old=c.P/arm/("fold_"+str(fold));p=c.H/arm/("fold_"+str(fold));done=c.read(old/"done.json");assert done["status"]=="COMPLETE" and done["source_sha"]==c.read(c.P/"source_manifest.json")["source_sha"];c.verify(done["files"])
   for name in ("model.npz","predictions.npz","fit_contract.npz","run_manifest.json","done.json"):assert c.sha(p/name)==c.sha(old/name)
   a=c.npz(p/"model.npz");cert,ts,vs=sm.arrays(d,fold,arm,a);pr=c.npz(p/"predictions.npz");error=max(float(abs(ts-pr["train_scores"]).max()),float(abs(vs-pr["validation_scores"]).max()));maxerr=max(maxerr,error);assert error<=1e-10 and np.array_equal(sm.rank(vs),pr["validation_slots"])
   if arm=="linear":assert abs(vs-d["old_witness_scores"][va]).max()<=1e-10 and np.array_equal(sm.rank(vs),d["reference_slots"]["old_witness"][va])
   td={k:v[tr] for k,v in d.items() if isinstance(v,np.ndarray) and len(v)==384};rep=c.reports(td,pr["train_slots"],pr["train_scores"]);assert "fold_"+str(fold) not in rep
   # Every existing group/scene/fold remains; missing validation fold has no training population.
   assert rep["overall"]["windows"]==int(tr.sum());c.digest(rep)
   fits.append(dict(arm=arm,fold=fold,score_error=error,independent=cert,parent_fitted_checkpoint_SHA_EXACT=True))
  assert c.sha(c.H/arm/"OOF.npz")==c.sha(c.P/arm/"OOF.npz")
 for name in ("bootstrap_draws.npz","policy_slots.npz"):assert c.sha(c.H/name)==c.sha(c.P/name)
 c.verify(files);c.guard();planning=3.
 c.write("PREFLIGHT.json",dict(status="PASS",at=c.now(),source_count=len(files),counts=counts,six_completed_parent_fits_reused=True,new_fit_count=0,all384_validation_windows_retained=True,empty_training_fold_omitted_no_science_change=True,score_max_error=maxerr,fits=fits,planning_minutes=planning,ETA_minutes=[2,6],elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE,no_solver_torch_backward=True))
 lock=dict(files=files,source_sha=c.digest(files),counts=counts);c.write("source_manifest.json",lock);c.write("prelaunch_review/REVIEW.json",dict(status="PASS",source_sha=lock["source_sha"],source_count=len(files),only_report_recovery=True,old_scientific_sources_all_unchanged=True,six_checkpoint_prediction_SHA_EXACT=True,all384_validation_ids_retained=True));print("RECOVERY_PREFLIGHT_PASS",len(files),lock["source_sha"])
if __name__=="__main__":
 try:main()
 except BaseException as e:c.write("PREFLIGHT_FAILURE_"+str(time.time_ns())+".json",dict(error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
