import common as c,saved_math as sm
import numpy as np,time
def main():
 t=time.perf_counter();c.guard();assert c.read(c.H/"PREFLIGHT.json")["status"]=="PASS";d,maps,core,contract=c.inputs();ztr,zva,tr,va=c.fi.fold_features(d,maps,core,contract,0,guard=c.guard);errors=[]
 for arm in ("pointwise","pairwise"):
  a=c.npz(c.H/("preflight_smoke/"+arm+".npz"));model=c.k.RankModel(arm,a["coef"],float(a["intercept"][0]));z=a["features"];s=c.k.score(z,model);ind=z@a["coef"]+a["intercept"][0];assert np.array_equal(s,ind)
  value,g=sm.objective(z,a["labels"],a["tracks"],arm,a["coef"],float(a["intercept"][0]));assert abs(g).max()<=1e-6;sm.equal(sm.independent_metrics(sm.rank(s),a["labels"],s),c.ev.describe(sm.rank(s),a["labels"],s));errors.append(dict(arm=arm,score_error=0.,gradient_inf=float(abs(g).max())))
 files,counts=c.sources();source_sha=c.digest(files);c.verify(files);c.write("source_manifest.json",dict(files=files,source_sha=source_sha,counts=counts))
 c.write("prelaunch_review/REVIEW.json",dict(status="PASS",source_sha=source_sha,source_count=len(files),counts=counts,actual_saved_subset_metrics_AND_PNU_rank_recompute=True,saved_fixtures=errors,empty_groups_explicit_null=True,no_new_fit_solver_Torch=True,not_external_review=True,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-t));print("SOURCE_LOCK_PRELAUNCH_PASS",len(files),source_sha)
if __name__=="__main__":main()
