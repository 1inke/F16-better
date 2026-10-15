import common as c,saved_math as sm,features_r5 as ft,raw_inputs as ri,pipeline as p
import numpy as np,time
def main():
 t=time.perf_counter();guard=c.guard;guard();lock=c.read(c.H/"source_manifest.json");c.verify(lock["files"]);r=c.read(c.H/"RESULT.json");assert r["status"]=="PASS_R5_TEMPORAL_MATCHING_EXECUTION";assert c.read(c.H/"worker_state.json")["status"]==c.read(c.H/"exit_status.json")["status"]=="SUCCESS";c.verify(r["output_sha"]);d,maps,core,contract=c.inputs();raw=ri.raw();T=ft.independent(raw["past"],raw["native"]);saved_raw=c.npz(c.H/"raw_temporal_inputs.npz");assert np.array_equal(T,saved_raw["T168"]);assert all(np.array_equal(raw[k],saved_raw[k]) for k in raw);pol=c.npz(c.H/"policy_slots.npz");assert np.array_equal(pol.pop("sample_id"),d["ids"]);certificates=[];maxerr=0.
 for arm in ("control","temporal"):
  oof=c.npz(c.H/arm/"OOF.npz");assert np.array_equal(oof["sample_id"],d["ids"]);restored=np.full((384,20),np.nan)
  for fold in range(3):
   guard();base=c.H/arm/("fold_"+str(fold));a=c.npz(base/"model.npz");basez,tr,va=sm.rebuilt_features(d,fold,a);mean,sd,scale=sm.temporal_stats(T,tr);assert max(abs(mean-a["temporal_mean"]).max(),abs(sd-a["temporal_raw_std"]).max(),abs(scale-a["temporal_scale"]).max())<=1e-11;assert np.array_equal(a["temporal_floor"],sd<1e-6);assert np.array_equal(a["column_order"],np.array([[t,k,v] for t in range(1,8) for k in range(1,13) for v in (0,1)]));z=basez if arm=="control" else np.concatenate((basez,(T.astype(float)-mean)/scale),axis=2);production,ptr,pva,pstats=p.fold(fold,arm);assert abs(z-production).max()<=1e-12;man=c.read(base/"run_manifest.json");assert man["source_sha"]==lock["source_sha"] and man["model_sha"]==c.sha(base/"model.npz") and man["contract_sha"]==c.sha(base/"fit_contract.npz");c.done_valid(c.read(base/"done.json"),lock["source_sha"])
   assert np.array_equal(man["train_ids"],d["ids"][tr]) and np.array_equal(man["validation_ids"],d["ids"][va]);assert man["stats_sha"]==c.digest({k:a[k] for k in list(c.maps_arrays(maps[fold]))+list(ft.statistics(T,tr))})
   fc=c.npz(base/"fit_contract.npz");assert np.array_equal(fc["labels"],d["labels"][tr]) and np.array_equal(fc["tracks"],d["track"][tr]);assert np.array_equal(fc["train_ids"],d["ids"][tr]) and np.array_equal(fc["validation_ids"],d["ids"][va]);w=sm.weights(fc["labels"],fc["tracks"]);assert np.array_equal(fc["pointwise_weight"],w) and np.array_equal(fc["known"],fc["labels"]!=-1) and np.array_equal(fc["target"],fc["labels"]==1)
   if arm in ("control","temporal"):
    delta,pw,wi,pp,nn=sm.pair_arrays(z[tr],fc["labels"],fc["tracks"]);assert all(np.array_equal(a,b) for a,b in zip((pw,wi,pp,nn),(fc["pair_weight"],fc["pair_window"],fc["pair_positive"],fc["pair_negative"])));assert np.count_nonzero(delta[:,sm.SHARED])==0 and a["intercept"][0]==0
   loss,gradient=sm.objective(z[tr],fc["labels"],fc["tracks"],"pairwise",a["coef"],float(a["intercept"][0]));assert abs(gradient).max()<=1e-6 and abs(loss-man["fit"]["loss"])<=1e-12
   pred=c.npz(base/"predictions.npz");ts=z[tr]@a["coef"]+a["intercept"][0];vs=z[va]@a["coef"]+a["intercept"][0];err=max(float(abs(ts-pred["train_scores"]).max()),float(abs(vs-pred["validation_scores"]).max()));maxerr=max(maxerr,err);assert err<=1e-10 and np.array_equal(sm.rank(ts),pred["train_slots"]) and np.array_equal(sm.rank(vs),pred["validation_slots"]);restored[va]=vs
   td={k:v[tr] for k,v in d.items() if isinstance(v,np.ndarray) and len(v)==384};vd={k:v[va] for k,v in d.items() if isinstance(v,np.ndarray) and len(v)==384};mm=c.read(base/"metrics.json");sm.equal(sm.reports(td,pred["train_slots"],pred["train_scores"]),mm["training"]);sm.equal(sm.reports(vd,pred["validation_slots"],pred["validation_scores"]),mm["validation"])
   assignment=c.read(base/"assignment_stats.json");assert assignment["full20_prediction_no_label_filter"] and assignment["allPN_fit_U_excluded"];assert assignment["coef_norms"]["shared_history_norm"]==np.linalg.norm(a["coef"][sm.SHARED]);assert assignment["coef_norms"]["RBF_norm"]==np.linalg.norm(a["coef"][136:200])
   if arm=="control":
    old=c.npz(c.R3/("production_20261013/pairwise/fold_"+str(fold)+"/model.npz"));assert np.array_equal(a["coef"],old["coef"]) and np.array_equal(a["intercept"],old["intercept"])
   certificates.append(dict(arm=arm,fold=fold,gradient_inf=float(abs(gradient).max()),score_max_error=err))
  assert abs(restored-oof["scores"]).max()<=1e-10 and np.array_equal(sm.rank(restored),pol[arm]);assert np.array_equal(oof["slots"],pol[arm])
  if arm=="control":assert np.array_equal(oof["scores"],c.npz(c.R3/"production_20261013/pairwise/OOF.npz")["scores"])
 for arm,slots in pol.items():
  scores=c.npz(c.H/arm/"OOF.npz")["scores"] if arm in ("control","temporal") else None
  if arm not in ("control","temporal"):assert np.array_equal(slots,p.reference_slots(d)[arm])
  sm.equal(sm.reports(d,slots,scores),r["summary"][arm])
 draw,hit=sm.bootstrap(d,pol);assert np.array_equal(draw,c.npz(c.H/"bootstrap_draws.npz")["draws"]);op=(d["labels"]==1).any(1);ci=np.quantile(draw,[.05/12,1-.05/12],axis=0).T;gain=True
 for j,ref in enumerate(c.ev.REFERENCES):
  delta=hit["temporal"].astype(float)-hit[ref];entry=r["screen"]["comparisons"][ref];assert entry["estimate"]==delta[op].mean() and np.array_equal(entry["CI991667"],ci[j]);assert entry["gained"]==int((op&hit["temporal"]&~hit[ref]).sum()) and entry["lost"]==int((op&~hit["temporal"]&hit[ref]).sum());gain &= entry["estimate"]>=.05 and ci[j,0]>0
 gate=bool(gain)
 for fold in range(3):
  mask=(d["fold"]==fold)&op;change=mask&(d["groups"]==2);counts={name:int(hit[name][change].sum()) for name in ("temporal",)+c.ev.REFERENCES};entry=r["screen"]["folds"][fold];assert counts==entry["change_hits"];support=int(mask.sum())>=20 and len(np.unique(d["track"][mask]))>=10;ok=bool(change.any() and all(counts["temporal"]>=counts[ref] for ref in c.ev.REFERENCES));assert support==entry["support"] and ok==entry["change_gate"];gate &= support and ok
 assert bool(gain)==r["screen"]["gain_gate"] and bool(gate)==r["screen"]["statistical_screen_only"]
 sm.equal(c.read(c.H/"RANK_TRANSITIONS.json"),sm.rank_transitions(d,c.npz(c.H/"control/OOF.npz")["scores"],c.npz(c.H/"temporal/OOF.npz")["scores"]))
 c.verify(lock["files"]);c.verify(r["output_sha"]);guard()
 c.write("completion_review/REVIEW.json",dict(status="PASS",source_sha=lock["source_sha"],source_count=len(lock["files"]),output_count=len(r["output_sha"]),all_six_maps_temporal168_contracts_manifests_pairs_train_OOF_metrics_6CIs_gates_PASS=True,score_max_error=maxerr,certificates=certificates,control_R3_coefs_intercepts_scores_slots_EXACT=True,no_fit_solver_Torch_neural_forward=True,not_external_review=True,elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE));print("AUDIT_PASS",flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:
  import traceback
  traceback.print_exc();c.write("completion_review/FAILURE.json",dict(error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
