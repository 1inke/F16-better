"""CPU saved-only completion supplement. No torch/model/optimizer import or fitting."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as c,metrics as v
import numpy as np,time,gc,traceback
from scipy.cluster.hierarchy import linkage,fcluster
def norm(x):return np.sqrt((x*x).sum(-1))
def main():
 start=time.perf_counter();res=c.read(c.H/"RESULT.json");lock=c.read(c.H/"source_manifest.json");c.verify(lock["files"]);c.verify(res["output_sha"]);rows=[];cert=[]
 records=[("P0",x["seed"],x) for x in res["initial_records"]]+[(x["arm"],x["seed"],x) for x in res["records"]]
 pairs=np.triu_indices(20,1);max_endpoint=0.
 for arm,s,rec in records:
  for split in rec["splits"]:
   c.guard();d=(c.H/f"initial/seed_{s}") if arm=="P0" else (c.H/f"runs/{arm}/seed_{s}")
   path=d/f"{split}.npz";expected=res["output_sha"][str(path)];assert c.sha(path)==expected;z=c.npz(path)
   A,F,J,good,seg,q=v.errors(z["paths"],z["gt"],True);j=J.argmin(1);er=norm(z["paths"]-z["gt"][:,None])
   qall=er[np.arange(len(q)),q].reshape(-1,3,4).mean(-1);qAF=np.stack((A[np.arange(len(q)),q],F[np.arange(len(q)),q]),1)
   jAF=np.stack((A[np.arange(len(j)),j],F[np.arange(len(j)),j]),1)
   endpoints=z["paths"][:,:,-1];ep_error=float(np.max(abs(endpoints-z["goals"])));assert ep_error<=2e-5;max_endpoint=max(max_endpoint,ep_error)
   pairstats=[];epcl=[]
   for first in range(0,len(A),128):
    c.guard();p=endpoints[first:first+128];ds=norm(p[:,pairs[0]]-p[:,pairs[1]])
    pairstats.append(np.stack((ds.min(1),np.median(ds,axis=1),np.quantile(ds,.95,axis=1)),1))
    for dd in ds:epcl.append(len(np.unique(fcluster(linkage(dd/.2,method="complete"),1,criterion="distance"))))
   pairstats=np.concatenate(pairstats);epcl=np.array(epcl)
   raw=norm(z["raw_b"]);proj=norm(z["projected_a"]);minimum=norm(z["minimum_norm"]);null=norm(z["null_term"])
   rawjerk=norm(np.diff(z["raw_b"],axis=2)/.4);projjerk=norm(np.diff(z["projected_a"],axis=2)/.4)
   moment=norm(z["d"]);change=norm(z["goals"]-z["initial_goals"]);cvgap=norm(z["goals"]-4.8*z["v0"][:,None])
   for group,mask in v.masks(z["groups"]):
    expectedrec=rec["splits"][split][group];assert int(good.any(1)[mask].sum())==expectedrec["good"]
    rawpart=raw[mask];projpart=proj[mask]
    rows.append(dict(arm=arm,seed=s,split=split,group=group,n=int(mask.sum()),Q_slot_counts=np.bincount(q[mask],minlength=20),J_slot_counts=np.bincount(j[mask],minlength=20),
     Q_samepath_earlymiddlelate=qall[mask].mean(0),J_samepath_earlymiddlelate=seg[mask].mean(0),Q_samepath_ADE_FDE=qAF[mask].mean(0),J_samepath_ADE_FDE=jAF[mask].mean(0),
     false_good_if_independent_minima=int((((A.min(1)<=.2)&(F.min(1)<=.4))&~good.any(1)&mask).sum()),
     endpoint_only_clusters=float(epcl[mask].mean()),endpoint_pair_min_median_p95_windowaverages=pairstats[mask].mean(0),endpoint_maxerror=ep_error,
     raw12_accel_mean=float(rawpart.mean()),raw12_accel_p95=float(np.quantile(rawpart,.95)),projected12_accel_mean=float(projpart.mean()),projected12_accel_p95=float(np.quantile(projpart,.95)),
     raw11_jerk_mean=float(rawjerk[mask].mean()),projected11_jerk_mean=float(projjerk[mask].mean()),these_12acc_11jerk_include_boundary_NOT_old_scientific_11acc_10jerk=True,
     minimum_norm_component_L2_energy=float((minimum[mask]**2).mean()),null_component_L2_energy=float((null[mask]**2).mean()),raw_L2_energy=float((rawpart**2).mean()),
     required_moment_norm_mean=float(moment[mask].mean()),allcandidate_goalchange_mean=float(change[mask].mean()),allcandidate_goal_CV_gap_mean=float(cvgap[mask].mean()),
     first_acceleration_boundary_mean=float(projpart[:,:,0].mean())))
   assert c.sha(path)==expected;cert.append(dict(path=str(path),SHA_unchanged=True,endpoint_maxerror=ep_error))
   del z,A,F,J,good,er,raw,proj,minimum,null,rawjerk,projjerk;gc.collect()
 c.verify(lock["files"]);c.verify(res["output_sha"]);c.guard()
 c.write("completion_review/SUPPLEMENT.json",dict(status="PASS",rows=rows,certificates=cert,total_jobs=36,total_group_rows=144,endpoint_maxerror=max_endpoint,source_count=len(lock["files"]),all_source_output_SHA_before_after_PASS=True,no_torch_forward_optimizer_newtraining=True,endpoint_clustering_descriptive_not_new_deployment_rule=True,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-start))
 print("SUPPLEMENT_PASS",flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:traceback.print_exc();c.write("completion_review/SUPPLEMENT_FAILURE.json",dict(status="STOP",error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
