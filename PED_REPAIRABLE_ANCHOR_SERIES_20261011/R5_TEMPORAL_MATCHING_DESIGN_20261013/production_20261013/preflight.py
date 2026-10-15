import common as c,pipeline as p,features_r5 as ft,saved_math as sm,raw_inputs as ri
import numpy as np,time,traceback
def fixture(z,y,tracks,arm):
 model,diag=c.k.fit_objective(z,y,tracks,"pairwise",guard=c.guard);again,_=c.k.fit_objective(z,y,tracks,"pairwise",guard=c.guard);assert np.array_equal(model.coef,again.coef) and model.intercept==0
 pairs=c.k.build_pairs(z,y,tracks,guard=c.guard);direct=sm.pair_arrays(z,y,tracks)
 for a,b in zip((pairs.delta,pairs.weight,pairs.window,pairs.positive,pairs.negative),direct):assert np.array_equal(a,b)
 loss,g=sm.objective(z,y,tracks,"pairwise",model.coef,0.);assert abs(loss-diag["loss"])<=1e-12 and np.max(abs(g))<=1e-6
 c.save("preflight_smoke/"+arm+".npz",coef=model.coef,intercept=np.array([0.]),features=z,labels=y,tracks=tracks)
 a=c.npz(c.H/"preflight_smoke"/(arm+".npz"));rest=c.k.RankModel("pairwise",a["coef"],0.);assert np.array_equal(c.k.score(z,model),c.k.score(z,rest))
 path=str(c.H/"preflight_smoke"/(arm+".npz"));done=dict(status="COMPLETE",source_sha="fixture",files={path:c.sha(path)});assert c.done_valid(done,"fixture")
 for bad in (dict(done,source_sha="bad"),dict(done,files={path:"bad"})):
  try:c.done_valid(bad,"fixture")
  except AssertionError:pass
  else:raise AssertionError("BAD_SOURCE_OUTPUT_ACCEPTED")
 return dict(arm=arm,actual21subset_repeat_params_scores_EXACT=True,independent_objective_stationarity_PASS=True,save_reject_source_output_PASS=True,fit=diag)
def main():
 t=time.perf_counter();c.guard();ts=time.perf_counter();files,inherit=c.sources();sha_seconds=time.perf_counter()-ts;d,m,core,con=c.inputs();raw=ri.raw();T=p.temporal()
 ts=time.perf_counter();ind=ft.independent(raw["past"],raw["native"]);scalar_seconds=time.perf_counter()-ts;assert np.array_equal(ind,T),"TEMPORAL168_SCALAR_MISMATCH"
 for chunk in (31,256):
  rebuilt=np.concatenate([ft.build(raw["past"][i:i+chunk],raw["native"][i:i+chunk]) for i in range(0,384,chunk)]);assert np.array_equal(T,rebuilt),"CHUNK_FEATURE_MISMATCH"
 controls=[];seconds=[];stat_checks=[]
 for fold in range(3):
  z,tr,va,stats=p.fold(fold,"control");base,tr2,va2=sm.rebuilt_features(d,fold,c.maps_arrays(m[fold]));assert np.array_equal(tr,tr2) and np.array_equal(va,va2);assert np.max(abs(base-z))<=1e-12
  mm,ss,scale=sm.temporal_stats(ind,tr);err=max(float(abs(mm-stats["temporal_mean"]).max()),float(abs(ss-stats["temporal_raw_std"]).max()),float(abs(scale-stats["temporal_scale"]).max()));assert err<=1e-11,"TRAIN168_STAT_MISMATCH"
  zv=ft.transform(base,ind,dict(temporal_mean=mm,temporal_scale=scale),"temporal");orig=ft.transform(z,T,stats,"temporal");normerr=float(abs(zv-orig).max());assert normerr<=1e-12,"STANDARDIZED168_MISMATCH"
  start=time.perf_counter();model,diag=c.k.fit_objective(z[tr],d["labels"][tr],d["track"][tr],"pairwise",guard=c.guard);seconds.append(time.perf_counter()-start);controls.append(dict(**p.reproduce(model,z,d,tr,va,fold),fit=diag));stat_checks.append(dict(fold=fold,stat_max_error=err,standardized_max_error=normerr,floored_columns=int(stats["temporal_floor"].sum())))
 z,tr,va,stats=p.fold(0,"control");ix=np.flatnonzero(tr);ix=ix[np.argsort(d["ids"][ix],kind="stable")][:21]
 fixtures=[]
 for arm in ("control","temporal"):
  z,tr,va,stats=p.fold(0,arm);fixtures.append(fixture(z[ix],d["labels"][ix],d["track"][ix],arm))
 z,tr,va,stats=p.fold(0,"temporal");ts=time.perf_counter();bench,diag=c.k.fit_objective(z[tr],d["labels"][tr],d["track"][tr],"pairwise",guard=c.guard);throwaway_seconds=time.perf_counter()-ts
 # No validation scientific outcome or benchmark checkpoint used/saved.
 c.verify(files);after,_=c.sources();assert files==after
 planning=(6*max([throwaway_seconds]+seconds)+scalar_seconds*2+sha_seconds*4+90)*1.5/60;assert planning<=30,("STOP_BUDGET",planning)
 c.write("PREFLIGHT.json",dict(status="PASS",source_sha=c.digest(files),source_count=len(files),inheritance=inherit,identity_raw_past_local_origin_dt_native_and_R1_136_EXACT=True,temporal168_scalar_chunk31_256_EXACT=True,independent_stats=stat_checks,control_R3_exact=controls,actual21_fixtures=fixtures,control_fit_seconds=seconds,temporal_full_fold0_throwaway_seconds=throwaway_seconds,throwaway_fit_diagnostic=diag,no_benchmark_outcome_selection_or_checkpoint=True,scalar_seconds=scalar_seconds,sha_seconds=sha_seconds,planning_minutes=planning,ETA_minutes=[max(2,planning*.6),planning*1.8],elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE,versions=c.versions(),no_neural_epoch_optimizerRNG_claim=True));print("PREFLIGHT_PASS",planning,flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:traceback.print_exc();c.write("failures/preflight1/FAILURE.json",dict(status="STOP",error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
