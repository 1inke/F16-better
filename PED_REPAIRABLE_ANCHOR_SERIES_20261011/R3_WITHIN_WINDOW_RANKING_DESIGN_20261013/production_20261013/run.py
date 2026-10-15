import common as c,saved_math as sm
import numpy as np,time
def main():
 t=time.perf_counter();lock=c.read(c.H/"source_manifest.json");c.verify(lock["files"]);assert c.read(c.H/"PREFLIGHT.json")["status"]=="PASS";d,maps,core,contract=c.inputs();oof={a:np.full((384,20),np.nan) for a in ("pointwise","pairwise")};fits=[];outputs={};training=[]
 for arm in ("pointwise","pairwise"):
  for fold in range(3):
   c.stage("FIT_OBJECTIVE",arm=arm,fold=fold,done=len(fits),total=6);ztr,zva,tr,va=c.fi.fold_features(d,maps,core,contract,fold,guard=c.guard);base=c.H/arm/("fold_"+str(fold));base.mkdir(parents=True,exist_ok=True);assert not (base/"done.json").exists(),"NO_DUPLICATE_FIT"
   model,diag=c.k.fit_objective(ztr,d["labels"][tr],d["track"][tr],arm,guard=c.guard)
   if arm=="pointwise":
    from preflight import reproduce
    reproduction=reproduce(model,ztr,zva,d,tr,va,fold)
   else:reproduction=None
   ts=c.k.score(ztr,model);vs=c.k.score(zva,model);oof[arm][va]=vs;ma=c.maps_arrays(maps[fold])
   value,gradient=sm.objective(ztr,d["labels"][tr],d["track"][tr],arm,model.coef,model.intercept);assert abs(value-diag["loss"])<=1e-12 and abs(gradient).max()<=1e-6
   c.save(str(base.relative_to(c.H)/"model.npz"),arm=np.array(arm),coef=model.coef,intercept=np.array([model.intercept]),**ma)
   c.save(str(base.relative_to(c.H)/"predictions.npz"),train_ids=d["ids"][tr],validation_ids=d["ids"][va],train_scores=ts,validation_scores=vs,train_slots=c.k.rank4(ts),validation_slots=c.k.rank4(vs))
   target,known,weight=c.k.pointwise_weights(d["labels"][tr],d["track"][tr]);fit_arrays=dict(train_ids=d["ids"][tr],validation_ids=d["ids"][va],labels=d["labels"][tr],tracks=d["track"][tr],target=target,known=known,pointwise_weight=weight)
   if arm=="pairwise":
    pairs=c.k.build_pairs(ztr,d["labels"][tr],d["track"][tr],guard=c.guard);fit_arrays.update(pair_window=pairs.window,pair_positive=pairs.positive,pair_negative=pairs.negative,pair_weight=pairs.weight)
   c.save(str(base.relative_to(c.H)/"fit_contract.npz"),**fit_arrays)
   td={k:v[tr] for k,v in d.items() if isinstance(v,np.ndarray) and len(v)==384};vd={k:v[va] for k,v in d.items() if isinstance(v,np.ndarray) and len(v)==384}
   tm=c.reports(td,c.k.rank4(ts),ts);vm=c.reports(vd,c.k.rank4(vs),vs);c.write(base/"metrics.json",dict(training=tm,validation=vm));training.append(dict(arm=arm,fold=fold,metrics=tm))
   norms=dict(shared_history_norm=float(np.linalg.norm(model.coef[sm.SHARED])),original_candidate_norm=float(np.linalg.norm(model.coef[np.r_[96:120,124:136]])),RBF_norm=float(np.linalg.norm(model.coef[136:])),intercept=model.intercept,zero_shared_history_difference=True)
   c.write(base/"assignment_stats.json",dict(coef_norms=norms,full20_prediction_no_label_filter=True,allPN_fit_U_excluded=True,knownP_validation=int((d["labels"][va]==1).any(1).sum()),validation=vm))
   man=dict(arm=arm,fold=fold,source_sha=lock["source_sha"],source_manifest_sha=c.sha(c.H/"source_manifest.json"),split_sha=c.digest(dict(ids=d["ids"],fold=d["fold"],tracks=d["track"])),labels_sha=c.sha(c.R/"PED_REPAIRABLE_ANCHOR_SERIES_20261011/R1_safe_repairability_supervision_20261012/record_recovery_20261012/labels.npz"),R2_model_sha=c.sha(c.OLD/("rbf64/fold_"+str(fold)+"/model.npz")),stats_sha=c.digest(ma),model_sha=c.sha(base/"model.npz"),contract_sha=c.sha(base/"fit_contract.npz"),input_order="136 original plus64 fixed R2 RBF",objective="PN_half_BCE" if arm=="pointwise" else "all_same_window_PN_softplus_negative_margin",pair_encoding="plus delta y1/minus delta y0 halfweight",C=1,S=20*int(tr.sum()),track_weight="equal tracks/equal windows inside tracks",intercept_gauge="learned_unpenalized" if arm=="pointwise" else "fixed_zero",optimizer=c.k.PARAMS,versions=c.versions(),threads=1,train_ids=d["ids"][tr],validation_ids=d["ids"][va],fit=diag,independent_stationarity=float(abs(gradient).max()),reproduction=reproduction,no_GT_prediction_input=True,no_optimizer_RNG_resume_claim=True)
   c.write(base/"run_manifest.json",man);ff={str(p):c.sha(p) for p in base.iterdir() if p.is_file()};c.write(base/"done.json",dict(status="COMPLETE",source_sha=lock["source_sha"],files=ff));assert c.done_valid(c.read(base/"done.json"),lock["source_sha"]);outputs.update(ff);outputs[str(base/"done.json")]=c.sha(base/"done.json");fits.append(dict(arm=arm,fold=fold,fit=diag,norms=norms))
 policies={k:v for k,v in d["reference_slots"].items() if k in ("linear","envelope","fixed_random","bottom","old_witness")}
 for arm in oof:
  assert np.isfinite(oof[arm]).all();policies[arm]=c.k.rank4(oof[arm]);c.save(arm+"/OOF.npz",sample_id=d["ids"],scores=oof[arm],slots=policies[arm]);outputs[str(c.H/arm/"OOF.npz")]=c.sha(c.H/arm/"OOF.npz")
 assert np.array_equal(oof["pointwise"],d["old_rbf_scores"]) and np.array_equal(policies["pointwise"],d["reference_slots"]["old_rbf"])
 c.stage("OOF_METRICS_BOOTSTRAP",done=6,total=6);summary={k:c.reports(d,v,oof.get(k)) for k,v in policies.items()};screen,draws=c.ev.evaluate_screen(d,policies);c.save("bootstrap_draws.npz",draws=draws);c.save("policy_slots.npz",sample_id=d["ids"],**policies);c.write("train_diagnostics.json",training)
 for name in ("bootstrap_draws.npz","policy_slots.npz","train_diagnostics.json"):outputs[str(c.H/name)]=c.sha(c.H/name)
 c.stage("FINAL_SOURCE_OUTPUT_SHA",done=6,total=6);c.verify(lock["files"]);c.verify(outputs);c.guard()
 c.write("RESULT.json",dict(status="PASS_R3_EXECUTION",source_sha=lock["source_sha"],source_count=len(lock["files"]),output_sha=outputs,production_fits=6,fits=fits,summary=summary,screen=screen,pointwise_R2_EXACT=True,windows=384,knownP_windows=263,label_counts={"P":329,"N":7027,"U":324},B_started=False,S2_started=False,formal_increment=0,no_new_trajectories=True,elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE));c.stage("SUCCESS",done=6,total=6,statistical_screen=screen["statistical_screen_only"])
if __name__=="__main__":main()
