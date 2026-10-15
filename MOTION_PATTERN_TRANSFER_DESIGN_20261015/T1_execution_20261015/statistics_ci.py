"""Reserved48 family; deterministic whole-track x model-seed crossed draws."""
import common as c,metrics as v
import numpy as np,hashlib
N=20000;FAMILY=48;SEEDS=(42,43,44)
def draw_hash(ss,tt):return hashlib.sha256(np.asarray(ss,dtype='<i8').tobytes()+np.asarray(tt,dtype='<i8').tobytes()).hexdigest()
def aggregate(diff,inv,T,independent):
 out=np.zeros((3,T,diff.shape[-1]))
 if independent:
  for t in range(T):out[:,t]=diff[:,inv==t].sum(1)
 else:
  for s in range(3):np.add.at(out[s],inv,diff[s])
 return out
def compute(independent=False):
 rng=np.random.Generator(np.random.PCG64(20261015));initial=rng.bit_generator.state;records=[];output={}
 for split in ('FIT','outer','inner','original_CAL'):
  az=[c.npz(c.H/f'runs/dynamic_winner/seed_{s}/{split}_metrics.npz') for s in SEEDS]
  bz=[c.npz(c.H/f'runs/fixed_motion_responsibility/seed_{s}/{split}_metrics.npz') for s in SEEDS]
  p0=c.npz(c.H/f'P0/{split}_metrics.npz')
  for group,mask in v.masks(az[0]['groups']):
   c.guard();tracks,inv=np.unique(az[0]['track'][mask],return_inverse=True);T=len(tracks);counts=np.bincount(inv)
   diffs=[np.stack([b['vectors5'][mask]-a['vectors5'][mask] for a,b in zip(az,bz)]),np.stack([b['vectors5'][mask]-p0['vectors5'][mask] for b in bz]),np.stack([b['vectors5'][mask]-b['native_vectors5'][mask] for b in bz])]
   sums=[aggregate(d,inv,T,independent) for d in diffs];draws=[[],[],[]];hashes=[];seed_save=[];track_save=[]
   path=c.H/f'bootstrap_draws/{split}_{group}.npz';saved=c.npz(path) if independent else None
   for first in range(0,N,256):
    c.guard();B=min(256,N-first);ss=np.empty((B,3),np.int64);tt=np.empty((B,T),np.int64)
    for j in range(B):ss[j]=rng.integers(0,3,3);tt[j]=rng.integers(0,T,T);hashes.append(draw_hash(ss[j],tt[j]))
    if independent:assert np.array_equal(saved['seed_indices'][first:first+B],ss) and np.array_equal(saved['track_indices'][first:first+B],tt)
    else:seed_save.append(ss.astype(np.int8));track_save.append(tt.astype(np.int32))
    weights=np.zeros((B,T),np.int64);np.add.at(weights,(np.repeat(np.arange(B),T),tt.ravel()),1);den=3*(weights@counts)
    for k,sm in enumerate(sums):
     ag=np.einsum('bt,stm->bsm',weights,sm,optimize=False);draws[k].append(ag[np.arange(B)[:,None],ss].sum(1)/den[:,None])
   if independent:assert np.array_equal(saved['draw_sha'],np.array(hashes))
   else:c.save(f'bootstrap_draws/{split}_{group}.npz',seed_indices=np.concatenate(seed_save),track_indices=np.concatenate(track_save),draw_sha=np.array(hashes),tracks=tracks)
   output[str(path)]=c.sha(path)
   for ref,d,parts in zip(('A_dynamic_winner','own_P0','native20'),diffs,draws):
    values=np.concatenate(parts)
    records.append(dict(split=split,group=group,reference=ref,n=int(mask.sum()),tracks=T,metrics=['coverage','ADE','FDE','J','endpoint_coverage'],estimate=d.mean((0,1)),perseed_estimate=d.mean(1),CI95=np.quantile(values,[.025,.975],axis=0).T,CI_family48=np.quantile(values,[.05/(2*FAMILY),1-.05/(2*FAMILY)],axis=0).T,primary_first4_only=(split=='outer' and group=='change'),endpoint_secondary=True,draw_indices_sha=c.sha(path),adaptive_exploratory=True))
 return dict(status='PASS',draws=N,family_size=48,do_not_shrink_unrun_stages=True,rng='numpy.PCG64',seed=20261015,numpy_version=np.__version__,initial_rng_state=initial,records=records,draw_files_sha=output,secondary_CI95_descriptive_only=True)
