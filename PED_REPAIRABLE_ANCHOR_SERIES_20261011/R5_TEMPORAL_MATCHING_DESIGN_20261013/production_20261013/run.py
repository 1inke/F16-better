import common as c,pipeline as p,saved_math as sm,raw_inputs as ri
import numpy as np,time
def main():
 t=time.perf_counter();lock=c.read(c.H/"source_manifest.json");c.stage("INITIAL_SOURCE_OLD_SHA",done=0,total=6);c.verify(lock["files"]);d,maps,core,con=c.inputs();raw=ri.raw();T=p.temporal();c.save("raw_temporal_inputs.npz",**raw,T168=T)
 oof={a:np.full((384,20),np.nan) for a in ("control","temporal")};fits=[];outputs={};training=[]
 for arm in oof:
  for fold in range(3):
   c.stage("FIT_PN_OBJECTIVE",arm=arm,fold=fold,done=len(fits),total=6);z,tr,va,stats=p.fold(fold,arm);base=c.H/arm/("fold_"+str(fold));base.mkdir(parents=True,exist_ok=True);assert not (base/"done.json").exists(),"NO_DUPLICATE_FIT"
   model,diag=c.k.fit_objective(z[tr],d["labels"][tr],d["track"][tr],"pairwise",guard=c.guard);rep=p.reproduce(model,z,d,tr,va,fold) if arm=="control" else None
   ts=c.k.score(z[tr],model);vs=c.k.score(z[va],model);oof[arm][va]=vs;ma=c.maps_arrays(maps[fold]);value,g=sm.objective(z[tr],d["labels"][tr],d["track"][tr],"pairwise",model.coef,0.);assert abs(value-diag["loss"])<=1e-12 and abs(g).max()<=1e-6
   c.save(str(base.relative_to(c.H)/"model.npz"),arm=np.array(arm),coef=model.coef,intercept=np.array([0.]),**ma,**stats)
   c.save(str(base.relative_to(c.H)/"predictions.npz"),train_ids=d["ids"][tr],validation_ids=d["ids"][va],train_scores=ts,validation_scores=vs,train_slots=c.k.rank4(ts),validation_slots=c.k.rank4(vs))
   pairs=c.k.build_pairs(z[tr],d["labels"][tr],d["track"][tr],guard=c.guard);target,known,w=c.k.pointwise_weights(d["labels"][tr],d["track"][tr])
   c.save(str(base.relative_to(c.H)/"fit_contract.npz"),train_ids=d["ids"][tr],validation_ids=d["ids"][va],labels=d["labels"][tr],tracks=d["track"][tr],target=target,known=known,pointwise_weight=w,pair_window=pairs.window,pair_positive=pairs.positive,pair_negative=pairs.negative,pair_weight=pairs.weight)
   tm=c.reports(c.subset(d,tr),c.k.rank4(ts),ts);vm=c.reports(c.subset(d,va),c.k.rank4(vs),vs);c.write(base/"metrics.json",dict(training=tm,validation=vm));training.append(dict(arm=arm,fold=fold,metrics=tm))
   norms=dict(shared_history_norm=float(np.linalg.norm(model.coef[sm.SHARED])),original_candidate_norm=float(np.linalg.norm(model.coef[np.r_[96:120,124:136]])),RBF_norm=float(np.linalg.norm(model.coef[136:200])),temporal_norm=float(np.linalg.norm(model.coef[200:])),full_norm=float(np.linalg.norm(model.coef)),intercept=0.)
   c.write(base/"assignment_stats.json",dict(coef_norms=norms,full20_prediction_no_label_filter=True,allPN_fit_U_excluded=True,validation=vm))
   man=dict(arm=arm,fold=fold,source_sha=lock["source_sha"],source_manifest_sha=c.sha(c.H/"source_manifest.json"),raw_input_SHAs={str(path):c.sha(path) for path in c.RAW_PATHS},raw_selected_T_SHA=c.sha(c.H/"raw_temporal_inputs.npz"),split_sha=c.digest(dict(ids=d["ids"],fold=d["fold"],tracks=d["track"])),labels_sha=c.sha(c.D.parent/"R1_safe_repairability_supervision_20261012/record_recovery_20261012/labels.npz"),R2_model_sha=c.sha(c.R2/("rbf64/fold_"+str(fold)+"/model.npz")),stats_sha=c.digest({**ma,**stats}),model_sha=c.sha(base/"model.npz"),contract_sha=c.sha(base/"fit_contract.npz"),dimension=z.shape[-1],newblock="168_t1..7_k1..12_DthenC",float_order="f32source_f64arithmetic_f32primitive_f64stats",objective="all_same_window_PN_softplus_negative_margin",S=20*int(tr.sum()),C=1,intercept=0,optimizer=c.k.PARAMS,versions=c.versions(),threads=1,train_ids=d["ids"][tr],validation_ids=d["ids"][va],fit=diag,independent_stationarity=float(abs(g).max()),reproduction=rep,no_GT_prediction_input=True,no_optimizer_RNG_resume_claim=True)
   c.write(base/"run_manifest.json",man);ff={str(q):c.sha(q) for q in base.iterdir() if q.is_file()};c.write(base/"done.json",dict(status="COMPLETE",source_sha=lock["source_sha"],files=ff));assert c.done_valid(c.read(base/"done.json"),lock["source_sha"]);outputs.update(ff);outputs[str(base/"done.json")]=c.sha(base/"done.json");fits.append(dict(arm=arm,fold=fold,fit=diag,norms=norms))
 pol=p.reference_slots(d)
 for arm in oof:
  assert np.isfinite(oof[arm]).all();pol[arm]=c.k.rank4(oof[arm]);c.save(arm+"/OOF.npz",sample_id=d["ids"],scores=oof[arm],slots=pol[arm]);outputs[str(c.H/arm/"OOF.npz")]=c.sha(c.H/arm/"OOF.npz")
 assert np.array_equal(oof["control"],c.npz(c.R3/"production_20261013/pairwise/OOF.npz")["scores"])
 c.stage("OOF_METRICS_SIX_BOOTSTRAP",done=6,total=6);summary={name:c.reports(d,slots,oof.get(name)) for name,slots in pol.items()};screen,draws=c.ev.evaluate_screen(d,pol);c.save("bootstrap_draws.npz",draws=draws);c.save("policy_slots.npz",sample_id=d["ids"],**pol);c.write("train_diagnostics.json",training);c.write("RANK_TRANSITIONS.json",sm.rank_transitions(d,oof["control"],oof["temporal"]))
 for name in ("raw_temporal_inputs.npz","bootstrap_draws.npz","policy_slots.npz","train_diagnostics.json","RANK_TRANSITIONS.json"):outputs[str(c.H/name)]=c.sha(c.H/name)
 c.stage("FINAL_SOURCE_OUTPUT_SHA",done=6,total=6);c.verify(lock["files"]);c.verify(outputs);c.guard()
 c.write("RESULT.json",dict(status="PASS_R5_TEMPORAL_MATCHING_EXECUTION",source_sha=lock["source_sha"],source_count=len(lock["files"]),output_sha=outputs,production_fits=6,fits=fits,summary=summary,screen=screen,control_R3_EXACT=True,windows=384,knownP_windows=263,label_counts={"P":329,"N":7027,"U":324},B_started=False,S2_started=False,formal_increment=0,no_new_trajectories=True,elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE));c.stage("SUCCESS",done=6,total=6,statistical_screen=screen["statistical_screen_only"])
if __name__=="__main__":main()
