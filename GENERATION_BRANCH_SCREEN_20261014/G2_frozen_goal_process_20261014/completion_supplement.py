"""CPU saved-vector supplement only; no model or bootstrap refit."""
import common as c,numpy as np
def main():
 result=c.read(c.H/"RESULT.json");parent=c.read(c.OLD/"RESULT.json");rows=[];files={}
 for seed in (42,43,44):
  for split in ("FIT","outer","inner","original_CAL"):
   a=c.OLD/f"runs/seed_{seed}/{split}_metrics.npz";b=c.H/f"runs/seed_{seed}/{split}_metrics.npz";files[str(a)]=c.sha(a);files[str(b)]=c.sha(b)
   with np.load(a,allow_pickle=False) as z:av=z["vectors"].copy();ids=z["sample_id"].copy();groups=z["groups"].copy()
   with np.load(b,allow_pickle=False) as z:bv=z["vectors"].copy();assert np.array_equal(ids,z["sample_id"]) and np.array_equal(groups,z["groups"])
   for name,mask in [("overall",np.ones(len(ids),bool))]+[(n,groups==i) for i,n in enumerate(("stable","transition","change"))]:
    ag=av[:,0].astype(bool);bg=bv[:,0].astype(bool);rows.append(dict(seed=seed,split=split,group=name,n=int(mask.sum()),A=int(ag[mask].sum()),B=int(bg[mask].sum()),rescued=int((~ag&bg&mask).sum()),lost=int((ag&~bg&mask).sum()),retained=int((ag&bg&mask).sum()),B_falsegood_if_separate_minima=int(((bv[:,1]<=.2)&(bv[:,2]<=.4)&~bg&mask).sum())))
 c.verify(files);c.write("completion_review/A_TO_B_ATTRIBUTION.json",dict(status="PASS",rows=rows,files=files,CPU_saved_vectors_only_no_forward_optimizer_newCI=True));print("ATTRIBUTION_PASS",flush=True)
if __name__=="__main__":main()

