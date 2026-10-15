import common as c,saved_math as sm
import numpy as np,time
def main():
 t=time.perf_counter();lock=c.load_locked();d=c.inputs();assert c.read(c.H/"PREFLIGHT.json")["status"]=="PASS"
 oof={a:np.full((384,20),np.nan) for a in ("linear","rbf64")};fits=[];outputs={};train_reports=[]
 # Complete all linear controls and validate reproduction BEFORE any treatment fit.
 for arm in ("linear","rbf64"):
  for fold in range(3):
   c.stage("FIT_SELECTOR",arm=arm,fold=fold,done=len(fits),total=6);tr,va=c.ic.split_masks(d,fold);p=c.H/arm/("fold_"+str(fold));p.mkdir(parents=True,exist_ok=True)
   assert not (p/"done.json").exists(),"NO_DUPLICATE_FIT"
   model,diag=c.m.fit_selector(d["X"][tr],d["labels"][tr],d["track"][tr],d["ids"][tr],arm,guard=c.guard)
   if arm=="linear":
    from preflight import check_linear
    reproduction=check_linear(model,d,fold)
   else:reproduction=None
   train=c.m.predict(d["X"][tr],model,c.guard);valid=c.m.predict(d["X"][va],model,c.guard);oof[arm][va]=valid
   a=c.m.model_arrays(model);c.save(str(p.relative_to(c.H)/"model.npz"),**a);c.save(str(p.relative_to(c.H)/"predictions.npz"),train_ids=d["ids"][tr],validation_ids=d["ids"][va],train_scores=train,validation_scores=valid,train_slots=c.m.rank4(train),validation_slots=c.m.rank4(valid))
   restored=c.m.restore_model(c.npz(p/"model.npz"));assert np.array_equal(valid,c.m.predict(d["X"][va],restored,c.guard))
   cert,indtrain,indvalid=sm.arrays(d,fold,arm,a);assert abs(indtrain-train).max()<=1e-10 and abs(indvalid-valid).max()<=1e-10;assert np.array_equal(sm.rank(indvalid),c.m.rank4(valid))
   target,known,w=c.m.witness_supervision(d["labels"][tr],d["track"][tr]);c.save(str(p.relative_to(c.H)/"fit_contract.npz"),train_ids=d["ids"][tr],validation_ids=d["ids"][va],target=target,known=known,weight=w)
   manifest=dict(arm=arm,fold=fold,source_sha=lock["source_sha"],source_count=len(lock["files"]),source_manifest_sha=c.sha(c.H/"source_manifest.json"),split_sha=c.digest(dict(ids=d["ids"],fold=d["fold"],tracks=d["track"])),labels_sha=c.sha(c.OLD/"labels.npz"),input_feature_order="P2_96/native_path24/vlast_dv4/bilinear12",landmark_tag="r2-fixed-rbf64-v1",bandwidth=1,landmarks=64 if arm=="rbf64" else 0,train_ids=d["ids"][tr],validation_ids=d["ids"][va],statistics_sha=c.digest({k:a[k] for k in ("mean","std","centers","center_keys","rbf_mean","rbf_std")}),weight_sha=c.digest(w),hyperparameters=c.m.PARAMS,versions=c.version(),model_sha=c.sha(p/"model.npz"),fit=diag,independent_stationarity=cert,linear_reproduction=reproduction,GT_prediction_input=False,no_optimizer_RNG_resume_claim=True)
   c.write(p/"run_manifest.json",manifest)
   current={str(z):c.sha(z) for z in p.iterdir() if z.is_file()};c.write(p/"done.json",dict(status="COMPLETE",source_sha=lock["source_sha"],files=current));assert c.done_valid(c.read(p/"done.json"),lock["source_sha"],current)
   outputs.update(current);outputs[str(p/"done.json")]=c.sha(p/"done.json");fits.append(dict(arm=arm,fold=fold,fit=diag))
   td={k:v[tr] for k,v in d.items() if isinstance(v,np.ndarray) and len(v)==384}
   train_reports.append(dict(arm=arm,fold=fold,metrics=c.reports(td,c.m.rank4(train),train),not_validation_or_tuning=True))
 policies={k:v.copy() for k,v in d["reference_slots"].items() if k!="old_witness"}
 for arm in oof:
  assert np.isfinite(oof[arm]).all();policies[arm]=c.m.rank4(oof[arm]);c.save(arm+"/OOF.npz",sample_id=d["ids"],scores=oof[arm],slots=policies[arm])
 for arm in ("linear","rbf64"):outputs[str(c.H/arm/"OOF.npz")]=c.sha(c.H/arm/"OOF.npz")
 assert np.array_equal(policies["linear"],d["reference_slots"]["old_witness"])
 c.stage("OOF_METRICS_BOOTSTRAP",done=6,total=6)
 summary={k:c.reports(d,v,oof.get(k)) for k,v in policies.items()};screen=c.ev.evaluate_screen(d,policies);draws={}
 for k,v in screen["comparisons"].items():draws[k]=v.pop("draws")
 c.save("bootstrap_draws.npz",**draws);c.save("policy_slots.npz",sample_id=d["ids"],**policies)
 c.write("train_diagnostics.json",train_reports)
 for file in ("bootstrap_draws.npz","policy_slots.npz","train_diagnostics.json"):outputs[str(c.H/file)]=c.sha(c.H/file)
 c.stage("FINAL_SOURCE_OUTPUT_SHA",done=6,total=6);c.verify(lock["files"]);c.verify(outputs)
 result=dict(status="PASS_R2_EXECUTION",source_sha=lock["source_sha"],source_count=len(lock["files"]),input_windows=384,knownP_windows=263,label_counts={"P":329,"N":7027,"U":324},production_fits=6,fits=fits,summary=summary,screen=screen,linear_R1_reproduction=True,output_sha=outputs,elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE,formal_increment=0,B_started=False,S2_started=False,no_new_trajectories=True,not_deployed_coverage=True)
 c.write("RESULT.json",result);c.stage("SUCCESS",done=6,total=6,statistical_screen=screen["statistical_screen_only"])
if __name__=="__main__":main()
