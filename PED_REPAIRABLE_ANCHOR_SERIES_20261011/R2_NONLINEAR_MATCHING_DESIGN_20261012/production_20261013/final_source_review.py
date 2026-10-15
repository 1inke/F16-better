import common as c,saved_math as sm
import numpy as np,time,importlib.util
def main():
 t=time.perf_counter();c.guard();assert c.read(c.H/"PREFLIGHT.json")["status"]=="PASS";d=c.inputs();old=c.npz(c.OLD/"sample_design.npz")
 fp=c.R/"PED_REPAIRABLE_ANCHOR_SERIES_20261011/stage1b_history_interaction_20261012/features.py"
 sp=importlib.util.spec_from_file_location("old_pure_features",fp);f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f)
 assert np.array_equal(d["X"],f.independent(old["p2"],old["past"],old["native"],"interaction"))
 tr,_=c.ic.split_masks(d,0);ix=np.flatnonzero(tr)[:21];errors=[]
 for arm in ("linear","rbf64"):
  a=c.npz(c.H/("preflight_smoke/"+arm+".npz"));before=c.digest(a);model=c.m.restore_model(a);score=c.m.predict(d["X"][ix],model,c.guard)
  independent=sm.features(d["X"][ix],a)@a["coef"]+a["intercept"][0];error=float(abs(independent-score).max());assert error<=1e-10 and np.array_equal(sm.rank(independent),c.m.rank4(score));assert before==c.digest(a);errors.append(dict(arm=arm,saved_score_error=error,slots_EXACT=True))
 refused=[]
 for field in ("ids","fold"):
  bad={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in d.items()}
  if field=="ids":bad["ids"][1]=bad["ids"][0]
  else:
   i=next(i for i in range(384) if np.count_nonzero(d["track"]==d["track"][i])>1);bad["fold"][i]=(bad["fold"][i]+1)%3
  try:c.ic.validate_data(bad)
  except ValueError:refused.append(field)
  else:raise AssertionError("bad identity accepted")
 files,counts=c.original_sources();source_sha=c.digest(files);c.write("source_manifest.json",dict(files=files,source_sha=source_sha,counts=counts,not_all_historical_files=True))
 c.verify(files);c.guard();c.write("prelaunch_review/REVIEW.json",dict(status="PASS",at=c.now(),source_sha=source_sha,source_count=len(files),counts=counts,actual384_feature_order_reconstruction_EXACT=True,saved_preflight_model_audit=errors,bad_ID_fold_rejected=refused,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-t,no_fit_solver_torch=True,not_external_review=True));print("SOURCE_LOCK_PRELAUNCH_PASS",len(files),source_sha)
if __name__=="__main__":main()
