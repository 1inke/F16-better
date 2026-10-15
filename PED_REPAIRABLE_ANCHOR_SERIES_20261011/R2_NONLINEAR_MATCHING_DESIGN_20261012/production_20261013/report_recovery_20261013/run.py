import common as c,saved_math as sm
import numpy as np,time
def main():
 t=time.perf_counter();lock=c.load_locked();d=c.inputs();c.stage("SAVED_SELECTOR_REPORT_RECOVERY",done=0,total=6);outputs={};fits=[];train_reports=[]
 for arm in ("linear","rbf64"):
  for fold in range(3):
   tr,va=c.ic.split_masks(d,fold);p=c.H/arm/("fold_"+str(fold));a=c.npz(p/"model.npz");man=c.read(p/"run_manifest.json");done=c.read(p/"done.json");c.verify(done["files"])
   cert,ts,vs=sm.arrays(d,fold,arm,a);pred=c.npz(p/"predictions.npz");assert abs(ts-pred["train_scores"]).max()<=1e-10 and abs(vs-pred["validation_scores"]).max()<=1e-10
   assert np.array_equal(sm.rank(vs),pred["validation_slots"]);fits.append(dict(arm=arm,fold=fold,fit=man["fit"],reused_parent_fit=True))
   td={k:v[tr] for k,v in d.items() if isinstance(v,np.ndarray) and len(v)==384};train_reports.append(dict(arm=arm,fold=fold,metrics=c.reports(td,pred["train_slots"],pred["train_scores"]),not_validation_or_tuning=True))
   for z in p.iterdir():
    if z.is_file():outputs[str(z)]=c.sha(z)
 for arm in ("linear","rbf64"):outputs[str(c.H/arm/"OOF.npz")]=c.sha(c.H/arm/"OOF.npz")
 pol=c.npz(c.H/"policy_slots.npz");assert np.array_equal(pol.pop("sample_id"),d["ids"]);oof={arm:c.npz(c.H/arm/"OOF.npz")["scores"] for arm in ("linear","rbf64")}
 assert abs(oof["linear"]-d["old_witness_scores"]).max()<=1e-10 and np.array_equal(pol["linear"],d["reference_slots"]["old_witness"])
 c.write("train_diagnostics.json",train_reports);summary={k:c.reports(d,v,oof.get(k)) for k,v in pol.items()};screen=c.ev.evaluate_screen(d,pol);saved=c.npz(c.H/"bootstrap_draws.npz")
 for key,value in screen["comparisons"].items():assert np.array_equal(value.pop("draws"),saved[key])
 for file in ("policy_slots.npz","bootstrap_draws.npz","train_diagnostics.json"):outputs[str(c.H/file)]=c.sha(c.H/file)
 c.stage("FINAL_SOURCE_OUTPUT_SHA",done=6,total=6);c.verify(lock["files"]);c.verify(outputs);c.guard()
 c.write("RESULT.json",dict(status="PASS_R2_REPORT_RECOVERY_EXECUTION",source_sha=lock["source_sha"],source_count=len(lock["files"]),input_windows=384,knownP_windows=263,label_counts={"P":329,"N":7027,"U":324},production_fits=6,new_recovery_fits=0,fits=fits,summary=summary,screen=screen,linear_R1_reproduction=True,output_sha=outputs,elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE,formal_increment=0,B_started=False,S2_started=False,no_new_trajectories=True,not_deployed_coverage=True,report_only_empty_training_fold_omitted=True))
 c.stage("SUCCESS",done=6,total=6,statistical_screen=screen["statistical_screen_only"])
if __name__=="__main__":main()
