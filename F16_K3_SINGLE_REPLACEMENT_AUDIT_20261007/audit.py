"""Saved-only paired K3 replacement. No torch/model/optimizer."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from datetime import datetime
import json,time,hashlib,ctypes,gc,sys,traceback
import numpy as np,psutil
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent;ROOT=H.parent
PREVIOUS=ROOT/'F16_PORTFOLIO_CALIBRATION_REPLAY_AUDIT_20261007/calibration_contract_recovery'
META=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'
RESOURCE=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
class Mem(ctypes.Structure):
 _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
def write(n,v):
 p=H/n;p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');os.replace(q,p)
def stage(n,**v):write('progress.json',dict(stage=n,at=datetime.now().astimezone().isoformat(),**v));print(n,v,flush=True)
def guard():
 m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));p=psutil.Process().memory_info()
 RESOURCE['peak_private_GiB']=max(RESOURCE['peak_private_GiB'],p.peak_pagefile/2**30);RESOURCE['peak_RSS_GiB']=max(RESOURCE['peak_RSS_GiB'],p.peak_wset/2**30)
 RESOURCE['min_physical_GiB']=min(RESOURCE['min_physical_GiB'],m.avail_phys/2**30);RESOURCE['min_commit_GiB']=min(RESOURCE['min_commit_GiB'],m.avail_page/2**30);RESOURCE['checks']+=1
 assert max(p.rss,p.private)<6*2**30 and min(m.avail_phys,m.avail_page)>=6*2**30,'STOP_RESOURCE'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):guard();h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify(d):
 for p,h in d.items():assert sha(p)==h,('STOP_SOURCE_OUTPUT',p)
def physics(p):
 v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4;a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4
 return {k:float(f(x)) for k,x,f in [('speed_p95',np.linalg.norm(v,axis=-1),lambda z:np.quantile(z,.95)),('accel_mean',np.linalg.norm(a,axis=-1),np.mean),('accel_p95',np.linalg.norm(a,axis=-1),lambda z:np.quantile(z,.95)),('jerk_mean',np.linalg.norm(j,axis=-1),np.mean),('jerk_p95',np.linalg.norm(j,axis=-1),lambda z:np.quantile(z,.95))]}
def cluster(p,g):
 out=[]
 for i,(x,b) in enumerate(zip(p,g)):
  if i%256==0:guard()
  d=np.linalg.norm(x[:,None]-x[None],axis=-1);D=np.maximum(d.mean(-1)/.1,d[:,:,-1]/.2)
  labels=fcluster(linkage(D[np.triu_indices(len(x),1)],method='complete'),1,criterion='distance')
  out.append([len(np.unique(labels)),len(np.unique(labels[b]))])
 return np.asarray(out,np.int32)
def bootstrap(d,tracks,reps=2000):
 # d: seed x identical window x metrics. Resample tracks and seeds.
 rng=np.random.default_rng(20261007);names,inverse=np.unique(tracks,return_inverse=True);T=len(names);N=len(tracks);S=len(d);out=[]
 for b in range(reps):
  if b%128==0:guard()
  multiplicity=np.bincount(rng.integers(T,size=T),minlength=T)[inverse];seeds=rng.integers(S,size=S)
  out.append((d[seeds]*multiplicity[None,:,None]).sum((0,1))/(S*multiplicity.sum()))
 draws=np.asarray(out);return dict(mean=d.mean((0,1)).tolist(),ci95=np.quantile(draws,[.025,.975],axis=0).T.tolist(),clusters=T,seeds=S,reps=reps)
def job(stamp,meta,smoke=False):
 tag=stamp['protocol']+'_'+str(stamp['seed']);stage('SAVED_K3_METRICS',job=tag);p=PREVIOUS/'saved'/f'{tag}.npz';assert sha(p)==stamp['output_SHA'];z=load(p)
 if smoke:z={k:(v[:128] if k in ('sample_id','paths','gt','groups','native_rank') else v) for k,v in z.items()}
 ids=z['sample_id'];N=len(ids);row=np.arange(N);assert len(np.unique(ids))==N and not np.intersect1d(ids,z['train_ids']).size
 order=np.argsort(meta['sample_id']);pos=order[np.searchsorted(meta['sample_id'][order],ids)];assert np.array_equal(meta['sample_id'][pos],ids)
 tracks=np.array([f'{s}|{int(ped)}' for s,ped in zip(meta['sequence'][pos],meta['pedestrian_id'][pos])]);assert (meta['pedestrian_id'][pos]>=0).all()
 x=z['paths'].astype(np.float64);gt=z['gt'].astype(np.float64);e=np.linalg.norm(x-gt[:,None],axis=-1);a=e.mean(-1);f=e[:,:,-1];j=np.maximum(a/.2,f/.4);g=(a<=.2)&(f<=.4);rank=z['native_rank'];assert rank.shape==(N,20)
 ix={'native3':rank[:,:3],'N2_L3':np.c_[rank[:,:2],np.full(N,21)],'N2_static':np.c_[rank[:,:2],np.full(N,20)],'N2_L0':np.c_[rank[:,:2],np.full(N,22)],'N2_L1':np.c_[rank[:,:2],np.full(N,23)]}
 nativegood=g[row[:,None],rank[:,:3]].any(1);native20good=g[:,:20].any(1);groups=z['groups'];out=dict(sample_id=ids,groups=groups,track=tracks);records=[]
 for name,slots in ix.items():
  paths=x[row[:,None],slots];aa=a[row[:,None],slots];ff=f[row[:,None],slots];jj=j[row[:,None],slots];gg=g[row[:,None],slots];cl=cluster(paths,gg);good=gg.any(1)
  vals=np.stack([good.astype(float),aa.min(1),ff.min(1),jj.min(1),cl[:,1],cl[:,0]],1);out[name]=vals
  # Exact rescue-lost identity is a core consistency requirement.
  assert int(good.sum()-nativegood.sum())==int((good&~nativegood).sum()-(nativegood&~good).sum())
  for group,m in [('overall',np.ones(N,bool)),('stable',groups==0),('transition',groups==1),('change',groups==2)]:
   if not m.any():continue
   mot=physics(paths[m]);base=physics(x[m,:20]);harm={k:(mot[k]/base[k]-1)*100 if base[k] else None for k in mot};passes=all(mot[k]<=1.01*base[k] for k in mot)
   records.append(dict(job=tag,portfolio=name,group=group,n=int(m.sum()),good=int(good[m].sum()),coverage=float(good[m].mean()),minADE=float(aa[m].min(1).mean()),minFDE=float(ff[m].min(1).mean()),J=float(jj[m].min(1).mean()),effective_good=float(cl[m,1].mean()),distinct=float(cl[m,0].mean()),rescue=int((good&~nativegood&m).sum()),lost=int((~good&nativegood&m).sum()),rescue_vs_native20=int((good&~native20good&m).sum()),motion=mot,native20_motion=base,motion_harm_percent=harm,motion_all5_PASS=passes))
 directory=H/('smoke' if smoke else 'saved');directory.mkdir(exist_ok=True);file=directory/f'{tag}.npz';assert not file.exists();np.savez_compressed(file,**out);read=load(file);assert all(np.array_equal(v,read[k]) for k,v in out.items())
 # Re-read primary arithmetic, independent of records assembly.
 for rr in records:
  m=np.ones(N,bool) if rr['group']=='overall' else groups=={'stable':0,'transition':1,'change':2}[rr['group']]
  assert abs(read[rr['portfolio']][m,0].mean()-rr['coverage'])<1e-12
 write(str(directory.relative_to(H)/f'{tag}_done.json'),dict(status='PASS_SAVED_ONLY',output_SHA=sha(file),input_SHA=stamp['output_SHA'],records=records));return out,records,file
def main(smoke=False):
 guard();stage('INITIAL_RELEVANT_SHA');result=json.loads((PREVIOUS/'RESULT.json').read_text());assert len(result['jobs'])==9 and result['no_training']
 paths=[Path(__file__),H/'DESIGN.md',META,PREVIOUS/'RESULT.json',PREVIOUS/'completion_review/REVIEW.json',PREVIOUS/'completion_review/AGGREGATE.json',*list(H.glob('*.ps1')),*list(H.glob('*.py'))]
 paths+=list((PREVIOUS/'saved').glob('*.npz'));locks={str(p):sha(p) for p in sorted(set(paths))};write('source_manifest.json',dict(files=locks,scope='relevant saved inputs/metadata/code, not all historical files'))
 if not smoke:
  pre=json.loads((H/'PREFLIGHT.json').read_text());assert pre['status']=='PASS';verify(pre['files'])
 # Independent simple numeric fixtures.
 assert cluster(np.zeros((1,20,12,2)),np.ones((1,20),bool)).tolist()==[[1,1]]
 d=np.zeros((3,12,2));d[:,:,0]=1.;b=bootstrap(d,np.repeat(['A','B','C'],4),32);assert b['ci95']==[[1.,1.],[0.,0.]]
 meta=load(META);t0=time.perf_counter();records=[];bundles={};stamps=[]
 for s in result['jobs'][:1] if smoke else result['jobs']:
  z,r,p=job(s,meta,smoke);records+=r;bundles.setdefault(s['protocol'],[]).append(z);stamps.append(dict(protocol=s['protocol'],seed=s['seed'],output=str(p),output_SHA=sha(p)));stage('JOB_DONE',done=len(stamps),total=1 if smoke else 9)
 ci={}
 if not smoke:
  for protocol,zz in bundles.items():
   assert len(zz)==3 and all(np.array_equal(zz[0]['sample_id'],v['sample_id']) for v in zz);ci[protocol]={}
   for group in ('overall','stable','transition','change'):
    mask=np.ones(len(zz[0]['sample_id']),bool) if group=='overall' else zz[0]['groups']=={'stable':0,'transition':1,'change':2}[group]
    diff=np.stack([v['N2_L3'][mask]-v['native3'][mask] for v in zz]);ci[protocol][group]=bootstrap(diff,zz[0]['track'][mask]);stage('PRIMARY_CLUSTER_BOOTSTRAP',protocol=protocol,group=group)
 verify(locks);guard();elapsed=time.perf_counter()-t0
 if smoke:write('PREFLIGHT.json',dict(status='PASS',files=locks,elapsed_seconds=elapsed,actual_rows=128,resource=RESOURCE,numeric_and_ID_and_saved_math_PASS=True,no_model_forward=True))
 else:
  write('RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_K3_SINGLE_REPLACEMENT',jobs=stamps,records=records,primary_CI=ci,metrics=['coverage','minADE','minFDE','J','effective_good','distinct'],primary='N2_L3 vs native3',no_model_forward=True,no_optimizer=True,formal_count_unchanged=True,sources_unchanged=True,resource=RESOURCE));write('ARITHMETIC_AUDIT.json',dict(status='PASS',all_saved_arrays_exact=True,rescue_lost_identity=True,all_quality_clusters_motion_recomputed=True,jobs=9))
 write('RESOURCE_USAGE.json',dict(status='PASS',**RESOURCE));stage('SUCCESS',jobs=len(stamps),elapsed_seconds=elapsed)
if __name__=='__main__':
 try:main('--smoke' in sys.argv)
 except Exception:write('FAILURE_'+str(time.time_ns())+'.json',dict(traceback=traceback.format_exc(),resource=RESOURCE));raise
