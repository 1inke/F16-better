"""Additional saved-only diagnostic fields required by M1; no neural forward/optimizer."""
import sys,json,hashlib,time
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parents[1];sys.path.insert(0,str(H))
import common as c,metrics as v,numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
def main():
 started=time.perf_counter();result=c.read(H/"RESULT.json");lock=c.read(H/"source_manifest.json")
 assert result["status"]=="PASS_M1_DYNAMIC_HISTORY_PAIRED_RUN";c.verify(result["output_sha"]);c.verify(lock["files"])
 rows=[];events=[];native={}
 for split in ("FIT","outer","inner","original_CAL"):
  for seed in (42,43,44):
   pair={}
   for arm in ("last_velocity","dynamic_history"):
    d=H/f"runs/{arm}/seed_{seed}";z=c.npz(d/f"{split}.npz");a=c.npz(d/f"{split}_metrics.npz");p=z["paths"];g=z["goals"];A,F,J,good,segments,q=v.errors(p,z["gt"],True);j=J.argmin(1)
    assert np.array_equal(a["good_candidate"],good)
    qA=A[np.arange(len(p)),q];qF=F[np.arange(len(p)),q];jA=A[np.arange(len(p)),j];jF=F[np.arange(len(p)),j]
    # Endpoint-only descriptive complete-link, same registered endpoint distance .20m.
    first,second=np.triu_indices(20,1);dist=np.sqrt(((g[:,first]-g[:,second])**2).sum(-1));counts=[]
    for ii,dd in enumerate(dist):
     if ii%128==0:c.guard()
     lab=fcluster(linkage(dd,method="complete"),.20,criterion="distance");counts.append(len(np.unique(lab)))
    counts=np.array(counts);pair[arm]=dict(good=good.any(1),endpoint=(F<=.4).any(1))
    ng=(v.errors(z["native20"],z["gt"],True)[3]).any(1)
    for group,mask in v.masks(z["groups"]):
     rows.append(dict(arm=arm,seed=seed,split=split,group=group,n=int(mask.sum()),complete_good=int(good.any(1)[mask].sum()),
      endpoint_good=int((F<=.4).any(1)[mask].sum()),Qwinner_ADE=float(qA[mask].mean()),Qwinner_FDE=float(qF[mask].mean()),Jclosest_ADE=float(jA[mask].mean()),Jclosest_FDE=float(jF[mask].mean()),
      endpoint_pairdistance_mean=float(dist[mask].mean()),endpoint_pairdistance_p05=float(np.quantile(dist[mask],.05)),
      endpoint_clusters_mean=float(counts[mask].mean()),Qwinner_counts=np.bincount(q[mask],minlength=20),Jclosest_counts=np.bincount(j[mask],minlength=20),
      falsegood_if_separate_minima=int(((A.min(1)<=.2)&(F.min(1)<=.4)&~good.any(1)&mask).sum()),
      native_good=int(ng[mask].sum()),all_K20_reported_not_GT_mode_semantics=True))
    for group,mask in v.masks(z["groups"]):
     if split+group not in native:
      na,nf,nj,ngc,_,_=v.errors(z["native20"],z["gt"],True)
      native[split+group]=dict(split=split,group=group,n=int(mask.sum()),good=int(ngc.any(1)[mask].sum()),endpoint_good=int((nf<=.4).any(1)[mask].sum()),ADE=float(na.min(1)[mask].mean()),FDE=float(nf.min(1)[mask].mean()),J=float(nj.min(1)[mask].mean()))
    del z,a,p,g,dist
   for group,mask in v.masks(result_groups(split,seed)):
    aa=pair["last_velocity"];bb=pair["dynamic_history"]
    events.append(dict(seed=seed,split=split,group=group,n=int(mask.sum()),
     A_complete=int(aa["good"][mask].sum()),B_complete=int(bb["good"][mask].sum()),rescued=int((~aa["good"]&bb["good"]&mask).sum()),lost=int((aa["good"]&~bb["good"]&mask).sum()),retained=int((aa["good"]&bb["good"]&mask).sum()),
     endpoint_rescued=int((~aa["endpoint"]&bb["endpoint"]&mask).sum()),endpoint_lost=int((aa["endpoint"]&~bb["endpoint"]&mask).sum())))
 c.verify(result["output_sha"]);c.verify(lock["files"]);c.write(H/"completion_review/SUPPLEMENT.json",dict(status="PASS",rows=rows,events=events,native=list(native.values()),
  input_source_sha=lock["source_sha"],scientific_output_count=len(result["output_sha"]),no_torch_model_forward_optimizer_or_retrain=True,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-started))
 print("SUPPLEMENT_PASS",len(rows),len(events),flush=True)
def result_groups(split,seed):return c.npz(H/f"runs/last_velocity/seed_{seed}/{split}_metrics.npz")["groups"]
if __name__=="__main__":main()
