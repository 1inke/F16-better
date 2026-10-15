"""Paired crossed whole-track/model-seed bootstrap; family26 never reduced."""
import common as c,metrics as v
import numpy as np,hashlib
N=20000;FAMILY=26
def vector(z):
 A,F,J,g,_,_=v.errors(z["paths"],z["gt"])
 return np.stack((g.any(1),A.min(1),F.min(1),J.min(1),(F<=.4).any(1)),1)
def draw_hash(ss,tt):
 return hashlib.sha256(np.asarray(ss,dtype="<i8").tobytes()+np.asarray(tt,dtype="<i8").tobytes()).hexdigest()
def aggregate(diff,inv,T,ind):
 S=np.zeros((3,T,diff.shape[-1]))
 if not ind:
  for seed in range(3):np.add.at(S[seed],inv,diff[seed])
 else:
  for t in range(T):S[:,t]=diff[:,inv==t].sum(1)
 return S
def compute(independent=False):
 rng=np.random.default_rng(20261014);initial=rng.bit_generator.state;records=[];output={}
 for split in ("FIT","outer","inner","original_CAL"):
  az=[c.npz(c.H/f"runs/frozen_goal/seed_{s}/{split}_metrics.npz") for s in (42,43,44)]
  bz=[c.npz(c.H/f"runs/joint_goal/seed_{s}/{split}_metrics.npz") for s in (42,43,44)]
  # Native full-window vectors explicitly persisted with every arm job.
  for group,mask in v.masks(az[0]["groups"]):
   c.guard();tracks,inv=np.unique(az[0]["track"][mask],return_inverse=True);T=len(tracks);counts=np.bincount(inv)
   diffs=[np.stack([b["vectors5"][mask]-a["vectors5"][mask] for a,b in zip(az,bz)]),
          np.stack([b["vectors5"][mask]-b["native_vectors5"][mask] for b in bz])]
   sums=[aggregate(d,inv,T,independent) for d in diffs];draws=[[],[]];hashes=[];seed_save=[];track_save=[]
   drawpath=c.H/f"bootstrap_draws/{split}_{group}.npz"
   saved=c.npz(drawpath) if independent else None
   for first in range(0,N,256):
    c.guard();B=min(256,N-first)
    # Interleaved per-draw seed then track draws, fixed across comparisons/metrics.
    ss=np.empty((B,3),np.int64);tt=np.empty((B,T),np.int64)
    for j in range(B):ss[j]=rng.integers(0,3,3);tt[j]=rng.integers(0,T,T);hashes.append(draw_hash(ss[j],tt[j]))
    if independent:
     assert np.array_equal(saved["seed_indices"][first:first+B],ss) and np.array_equal(saved["track_indices"][first:first+B],tt)
    else:seed_save.append(ss.astype(np.int8));track_save.append(tt.astype(np.int32))
    w=np.zeros((B,T),np.int64);np.add.at(w,(np.repeat(np.arange(B),T),tt.ravel()),1);den=3*(w@counts)
    for k,S in enumerate(sums):
     ag=np.einsum("bt,stm->bsm",w,S,optimize=False)
     selected=ag[np.arange(B)[:,None],ss].sum(1)/den[:,None];draws[k].append(selected)
   if independent:assert np.array_equal(saved["draw_sha"],np.array(hashes))
   else:
    c.save(f"bootstrap_draws/{split}_{group}.npz",seed_indices=np.concatenate(seed_save),track_indices=np.concatenate(track_save),draw_sha=np.array(hashes),tracks=tracks)
   output[str(drawpath)]=c.sha(drawpath)
   for ref,d,parts in zip(("A_frozen_goal","native20"),diffs,draws):
    dd=np.concatenate(parts);rec=dict(split=split,group=group,reference=ref,n=int(mask.sum()),tracks=T,
     metrics=["coverage","ADE","FDE","J","endpoint_coverage"],estimate=d.mean((0,1)),perseed_estimate=d.mean(1),
     CI95=np.quantile(dd,[.025,.975],axis=0).T,
     CI_family26=np.quantile(dd,[.05/(2*FAMILY),1-.05/(2*FAMILY)],axis=0).T,
     primary=(split=="outer" and group=="change"),primary_metrics=["coverage","ADE","FDE","J"],endpoint_secondary_M2=True,draw_indices_sha=c.sha(drawpath),adaptive_exploratory=True)
    records.append(rec)
 return dict(status="PASS",draws=N,family_size=FAMILY,rng="numpy_default_rng_PCG64",numpy_version=np.__version__,initial_rng_state=initial,records=records,draw_files_sha=output,secondary_CI95_not_extra_primary=True)
