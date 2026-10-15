"""Saved-only independent arithmetic review; no torch or model execution."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent;P=H.parent/'F16_PORTFOLIO_CALIBRATION_REPLAY_AUDIT_20261007/calibration_contract_recovery/saved'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def clust(p,b):
 out=[]
 for x,g in zip(p,b):
  q=np.linalg.norm(x[None]-x[:,None],axis=-1);q=np.maximum(q.mean(-1)/.1,q[:,:,-1]/.2);l=fcluster(linkage(q[np.triu_indices(3,1)],method='complete'),1,criterion='distance');out.append([len(np.unique(l[g])),len(np.unique(l))])
 return np.array(out)
def phys(x):
 vel=np.concatenate([x[:,:,:1],np.diff(x,axis=2)],axis=2)/.4;ac=np.diff(vel,axis=2)/.4;je=np.diff(ac,axis=2)/.4
 v=np.linalg.norm(vel,axis=-1);a=np.linalg.norm(ac,axis=-1);j=np.linalg.norm(je,axis=-1)
 return dict(speed_p95=float(np.quantile(v,.95)),accel_mean=float(a.mean()),accel_p95=float(np.quantile(a,.95)),jerk_mean=float(j.mean()),jerk_p95=float(np.quantile(j,.95)))
def ci(d,keys):
 rng=np.random.default_rng(20261007);unique,inv=np.unique(keys,return_inverse=True);draw=[]
 for _ in range(2000):
  mult=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[inv];s=rng.integers(3,size=3);draw.append(np.einsum('snm,n->m',d[s],mult)/(3*mult.sum()))
 return np.quantile(draw,[.025,.975],axis=0).T
def main():
 r=json.loads((H/'RESULT.json').read_text());locks=json.loads((H/'source_manifest.json').read_text())['files'];assert all(sha(p)==h for p,h in locks.items());mx=0.;pop={};allrecords=r['records'];assert len(r['jobs'])==9
 for st in r['jobs']:
  assert sha(st['output'])==st['output_SHA'];z=load(st['output']);tag=st['protocol']+'_'+str(st['seed']);src=load(P/f'{tag}.npz');assert np.array_equal(src['sample_id'],z['sample_id'])
  n=len(z['sample_id']);row=np.arange(n);x=src['paths'].astype(float);dist=np.sqrt(((x-src['gt'][:,None].astype(float))**2).sum(-1));a=dist.mean(-1);f=dist[:,:,-1];j=np.maximum(a/.2,f/.4);good=(a<=.2)&(f<=.4);rank=src['native_rank']
  base=good[row[:,None],rank[:,:3]].any(1);ix={'native3':rank[:,:3],'N2_L3':np.c_[rank[:,:2],np.full(n,21)],'N2_static':np.c_[rank[:,:2],np.full(n,20)],'N2_L0':np.c_[rank[:,:2],np.full(n,22)],'N2_L1':np.c_[rank[:,:2],np.full(n,23)]}
  for name,slots in ix.items():
   gg=good[row[:,None],slots];cl=clust(x[row[:,None],slots],gg);calc=np.c_[gg.any(1),a[row[:,None],slots].min(1),f[row[:,None],slots].min(1),j[row[:,None],slots].min(1),cl];mx=max(mx,float(abs(calc-z[name]).max()))
   for group,mask in [('overall',np.ones(n,bool)),('stable',z['groups']==0),('transition',z['groups']==1),('change',z['groups']==2)]:
    rr=next(v for v in allrecords if v['job']==tag and v['portfolio']==name and v['group']==group);means=calc[mask].mean(0)
    mx=max(mx,max(abs(means[k]-rr[t]) for k,t in enumerate(('coverage','minADE','minFDE','J','effective_good','distinct'))))
    assert rr['rescue']==int((gg.any(1)&~base&mask).sum()) and rr['lost']==int((~gg.any(1)&base&mask).sum())
    motion=phys(x[row[:,None],slots][mask]);ref=phys(x[mask,:20]);mx=max(mx,max(abs(motion[k]-rr['motion'][k]) for k in motion));assert rr['motion_all5_PASS']==all(motion[k]<=ref[k]*1.01 for k in motion)
  pop.setdefault(st['protocol'],[]).append(z)
 for protocol,zz in pop.items():
  for group,v in r['primary_CI'][protocol].items():
   m=np.ones(len(zz[0]['sample_id']),bool) if group=='overall' else zz[0]['groups']=={'stable':0,'transition':1,'change':2}[group];d=np.stack([z['N2_L3'][m]-z['native3'][m] for z in zz]);c=ci(d,zz[0]['track'][m]);mx=max(mx,float(abs(c-np.asarray(v['ci95'])).max()))
 assert mx<1e-10,mx;assert all(sha(p)==h for p,h in locks.items());out=H/'completion_review';out.mkdir(exist_ok=True)
 (out/'REVIEW.json').write_text(json.dumps(dict(status='PASS_SAVED_ONLY_INDEPENDENT',jobs=9,groups=180,primary_cluster_CIs=72,maximum_error=mx,all_quality_cluster_motion_rescue_lost_recomputed=True,relevant_source_input_SHA_before_after=True,no_model_forward=True,no_optimizer=True),indent=2),encoding='utf-8');print(json.dumps(dict(status='PASS',maximum_error=mx)))
if __name__=='__main__':main()
