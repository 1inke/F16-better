"""Independent CPU saved-output stage2 audit; no torch or old writes."""
import os,sys,time,json,hashlib,ctypes,traceback
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import numpy as np
import psutil
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.stats import spearmanr
C=Path(__file__).resolve().parent;H=C.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):
 q=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):q.update(b)
 return q.hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding='utf-8')
class Mem(ctypes.Structure):
 _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
RES=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
def guard():
 m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
 p=psutil.Process().memory_info();RES['checks']+=1
 RES['peak_private_GiB']=max(RES['peak_private_GiB'],p.peak_pagefile/2**30);RES['peak_RSS_GiB']=max(RES['peak_RSS_GiB'],p.peak_wset/2**30)
 RES['min_physical_GiB']=min(RES['min_physical_GiB'],m.avail_phys/2**30);RES['min_commit_GiB']=min(RES['min_commit_GiB'],m.avail_page/2**30)
 if max(p.private,p.rss)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:raise MemoryError('RESOURCE6')
MAX=0.;CHECKS=0;CIS=0
def eq(x,y,tol=1e-10):
 global MAX,CHECKS
 CHECKS+=1
 if x is None or y is None:assert x is None and y is None;return
 x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape
 e=float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.size else 0.;MAX=max(MAX,e)
 assert e<=tol,(e,tol)
def errors(p,g):
 d=np.linalg.norm(p-(g[:,None] if p.ndim==4 else g),axis=-1);return d.mean(-1),d[...,-1]
def ci(x,track):
 global CIS
 CIS+=1;u,inv=np.unique(track,return_inverse=True);s=np.bincount(inv,weights=x);n=np.bincount(inv)
 rng=np.random.default_rng(20261009);vals=[]
 for start in range(0,2000,100):
  ix=rng.integers(0,len(u),(min(100,2000-start),len(u)));vals.extend((s[ix].sum(1)/n[ix].sum(1)).tolist())
 return np.quantile(vals,[.025,.5,.975]).tolist()
def motion(p,m):
 x=p[m];v=np.diff(np.concatenate((np.zeros((len(x),20,1,2)),x),axis=2),axis=2)/.4;acc=np.diff(v,axis=2)/.4;j=np.diff(acc,axis=2)/.4
 v=np.linalg.norm(v,axis=-1).ravel();acc=np.linalg.norm(acc,axis=-1).ravel();j=np.linalg.norm(j,axis=-1).ravel()
 return dict(speedp95=float(np.quantile(v,.95)),accelmean=float(acc.mean()),accelp95=float(np.quantile(acc,.95)),jerkmean=float(j.mean()),jerkp95=float(np.quantile(j,.95)))
def clusters(p,aa,ff):
 I,J=np.triu_indices(20,1);dc=np.empty(len(p),int);ec=np.empty(len(p),int);good=(aa<=.2)&(ff<=.4)
 for i in range(len(p)):
  if i%256==0:guard()
  d=np.linalg.norm(p[i,I]-p[i,J],axis=-1);d=np.maximum(d.mean(1)/.10,d[:,-1]/.20)
  lab=fcluster(linkage(d,method='complete'),t=1.,criterion='distance');dc[i]=len(np.unique(lab));ec[i]=len(np.unique(lab[good[i]]))
 return dc,ec
def reconstruct(data,name,syn):
 o=data['obs'].copy();valid=data['valid'].copy();ix=np.arange(len(o));meta={}
 if name.startswith('syn_'):
  ix=syn['index'];o=o[ix];valid=np.full(len(ix),2,dtype=int)
  side,d=name.split('_')[1:];d={'03':.3,'06':.6,'10':1.,'15':1.5,'25':2.5}[d]
  norm=syn['leftnormal']*(1 if side=='left' else -1);vn=syn['velocity'];vt=(o[:,0,-1]-o[:,0,-2]).astype(float);w=vn-vt
  pos=o[:,0,-1].astype(float)-5*w+d*norm
  o[:,1]=(pos[:,None]+vn[:,None]*np.arange(-7,1)[None,:,None]).astype(np.float32);o[:,2:]=o[:,0,None]
  wa=(o[:,1,-1]-o[:,1,-2]).astype(float)-vt;rel=(o[:,1,-1]-o[:,0,-1]).astype(float)
  tc=-(rel*wa).sum(1)/(wa*wa).sum(1);dist=np.linalg.norm(rel+tc[:,None]*wa,axis=1)
  assert abs(tc-5).max()<=.002 and abs(dist-d).max()<=2e-5
  meta=dict(away=-norm,tca=tc,closest=dist,distance=np.full(len(ix),d),side=np.full(len(ix),1 if side=='left' else -1))
 elif name[0]=='L':
  keep=int(name[1:]);o[:,:,:8-keep]=o[:,:,8-keep:9-keep]
 elif name=='reverse_prefix':o[:,:,:6]=o[:,:,:6][:,:,::-1].copy()
 elif name=='shuffle_prefix':
  for i,ID in enumerate(data['ids']):
   rng=np.random.default_rng(int(hashlib.sha256(f'20261009-history:{int(ID)}'.encode()).hexdigest()[:16],16));o[i,:,:6]=o[i,:,rng.permutation(6)].transpose(1,0,2)
 elif name=='noNbr':valid[:]=1;o[:,1:]=o[:,0,None]
 elif name=='farNbr':
  v=o[:,0,-1]-o[:,0,-2];speed=np.linalg.norm(v,axis=1);u=np.tile([1.,0.],(len(o),1));m=speed>=.02;u[m]=v[m]/speed[m,None];o[:,1:]+=10*np.stack((-u[:,1],u[:,0]),1)[:,None,None].astype(np.float32)
 elif name=='replaceNbr':
  for i in range(len(o)):
   j=(i+1)%len(o);o[i,1:]=data['obs'][j,1:]-data['obs'][j,0,-1]+data['obs'][i,0,-1];valid[i]=data['valid'][j]
 elif name.startswith('leave'):
  slot=int(name[5:])
  for i in range(len(o)):
   if slot<valid[i]:o[i,slot:8]=o[i,slot+1:9];valid[i]-=1;o[i,valid[i]:]=o[i,0]
 elif name=='rotate':
  angle=np.array([2*np.pi*int(hashlib.sha256(f'20261009-rotation:{int(ID)}'.encode()).hexdigest()[:12],16)/16**12 for ID in data['ids']])
  Q=np.stack((np.cos(angle),-np.sin(angle),np.sin(angle),np.cos(angle)),1).reshape(-1,2,2)
  o=np.einsum('bij,bntj->bnti',Q,o).astype(np.float32);meta['Q']=Q
 elif name=='mirror':o[...,0]*=-1
 elif name=='translate':o+=np.array([10.,-7.],np.float32)
 elif name=='scale08':o*=.8
 elif name=='scale12':o*=1.2
 else:raise ValueError(name)
 return o,valid,ix,meta
def main():
 t=time.perf_counter();guard();r=read(H/'RESULT.json');assert r['status']=='PASS_STAGE2_RAW_HISTORY_NEIGHBOR_GEOMETRY' and len(r['jobs'])==480
 assert read(H/'worker_state.json')['status']=='SUCCESS';assert read(H/'exit_status.json')['status']=='SUCCESS'
 for f in ('ARITHMETIC_AUDIT.json','FROZEN_AUDIT.json','RESOURCE_USAGE.json'):assert read(H/f)['status']=='PASS'
 frozen=read(H/'FROZEN_AUDIT.json')['models'];assert len(frozen)==15 and all(x['weights_bool_buffers_EXACT'] for x in frozen)
 locks=read(H/'source_manifest.json')['files'].copy();assert len(locks)==363
 for job in r['jobs']:
  locks.update(job['outputs']);dp=Path(next(k for k in job['outputs'] if k.endswith('outputs.npz'))).parent/'done.json';d=read(dp)
  assert d==job and d['source_sha']==sha(H/'source_manifest.json');locks[str(dp)]=sha(dp)
 for p in H.glob('data/*'):locks[str(p)]=sha(p)
 for p in H.glob('jobs/*/baseline.npz'):locks[str(p)]=sha(p)
 for f in ('RESULT.json','worker_state.json','exit_status.json','ARITHMETIC_AUDIT.json','FROZEN_AUDIT.json','RESOURCE_USAGE.json'):locks[str(H/f)]=sha(H/f)
 locks[str(Path(__file__))]=sha(__file__)
 def verify():
  for i,(p,d) in enumerate(locks.items()):
   if i%32==0:guard()
   assert sha(p)==d,('SHA',p)
 verify();write(C/'input_manifest.json',dict(files=locks))
 points=[];quality=[];temporal=[];geometry=[];attention=[];dose=[];rows=0;seen=set()
 for job in r['jobs']:
  guard();fold=job['fold'];seed=job['seed'];name=job['intervention'];key=(fold,seed,name);assert key not in seen;seen.add(key)
  pth=Path(next(k for k in job['outputs'] if k.endswith('outputs.npz')));z=load(pth);m=read(pth.parent/'metrics.json');cls=load(pth.parent/'clusters.npz')
  data=load(H/f'data/{fold}_selected.npz');syn=load(H/f'data/{fold}_synthetic.npz');o,n,ix,meta=reconstruct(data,name,syn)
  eq(o,z['obs'],0);eq(n,z['valid'],0);assert len(np.unique(z['sample_id']))==len(ix)==job['n']
  assert np.array_equal(z['sample_id'],data['ids'][ix]) and np.array_equal(z['track'],data['track'][ix]);eq(z['GT'],data['gt'][ix],0)
  for k,v in meta.items():eq(v,z[k],0)
  for k in (k for k in z if k.startswith('mask_')):eq(z[k],data[k][ix],0)
  base=load(H/f'jobs/{fold}_seed{seed}/baseline.npz');eq(z['base_paths'],base['paths'][ix],0);eq(z['base_logits'],base['logits'][ix],0)
  pred=z['joint_first2'][:,0]
  if name=='rotate':pred=np.einsum('bji,bktj->bkti',z['Q'],pred)
  if name=='mirror':pred=pred.copy();pred[...,0]*=-1
  if name=='scale08':pred=pred/.8
  if name=='scale12':pred=pred/1.2
  eq(pred,z['paths'],0);eq(z['logits'],z['joint_logits_first2'][:,0],0)
  p=z['paths'];g=z['GT'];b=z['base_paths'];aa,ff=errors(p,g);ba,bf=errors(b,g)
  dc,ec=clusters(p,aa,ff);eq(dc,cls['distinct'],0);eq(ec,cls['effectivegood'],0);eq(z['sample_id'],cls['sample_id'],0)
  good=((aa<=.2)&(ff<=.4)).any(1);bg=((ba<=.2)&(bf<=.4)).any(1);J=np.maximum(aa/.2,ff/.4);slot=J.argmin(1)
  for rec in m['quality']:
   mask=z['mask_'+rec['subset']];N=int(mask.sum());eq(N,rec['n'],0)
   vals=dict(minADE=aa[mask].min(1).mean(),minFDE=ff[mask].min(1).mean(),J=J[mask].min(1).mean(),coverage=int(good[mask].sum()),base_coverage=int(bg[mask].sum()),rescue=int((mask&good&~bg).sum()),lost=int((mask&~good&bg).sum()),distinct=dc[mask].mean(),effectivegood=ec[mask].mean())
   for k,v in vals.items():eq(v,rec[k])
   mot=motion(p,mask);bm=motion(b,mask)
   for k,v in mot.items():eq(v,rec['motion'][k])
   harms={k:float(mot[k]/bm[k]-1) if bm[k] else None for k in mot}
   quality.append(dict(fold=fold,seed=seed,intervention=name,**rec,motion_harm_vs_baseline=harms))
   err=np.linalg.norm(p[np.arange(len(p)),slot]-g,axis=-1)
   temporal.append(dict(fold=fold,seed=seed,intervention=name,subset=rec['subset'],n=N,Jclosest_single_slot_early=float(err[mask,:4].mean()),middle=float(err[mask,4:8].mean()),late=float(err[mask,8:].mean())))
  signedreads={}
  for method in ('top1','meanK'):
   top=z['logits'].argmax(1);bt=z['base_logits'].argmax(1)
   point=p[np.arange(len(p)),top] if method=='top1' else p.mean(1);bp=b[np.arange(len(p)),bt] if method=='top1' else b.mean(1)
   ad,fd=errors(point,g);bad,bfd=errors(bp,g);resp=np.linalg.norm(point-bp,axis=-1).mean(1);end=np.linalg.norm(point[:,-1]-bp[:,-1],axis=-1)
   if 'away' in z:
    nb=load(H/f'jobs/{fold}_seed{seed}/noNbr/outputs.npz');eq(z['noNbr_paths'],nb['paths'][ix],0);eq(z['noNbr_logits'],nb['logits'][ix],0)
    ref=nb['paths'][ix,np.argmax(nb['logits'][ix],1)] if method=='top1' else nb['paths'][ix].mean(1)
    signed=((point[:,-1]-ref[:,-1])*z['away']).sum(1);signedreads[method]=signed
   for rec in (v for v in m['point_records'] if v['method']==method):
    mask=z['mask_'+rec['subset']];eq(int(mask.sum()),rec['n'],0);eq(len(np.unique(z['track'][mask])),rec['tracks'],0)
    for k,v in dict(ADE=ad[mask].mean(),FDE=fd[mask].mean(),delta_ADE=(ad-bad)[mask].mean(),delta_FDE=(fd-bfd)[mask].mean(),response_meanpath=resp[mask].mean(),response_endpoint=end[mask].mean(),delta_ADE_track_CI=ci((ad-bad)[mask],z['track'][mask]),delta_FDE_track_CI=ci((fd-bfd)[mask],z['track'][mask])).items():eq(v,rec[k])
    if 'away' in z:
     for k,v in dict(away_endpoint=signed[mask].mean(),away_fraction=(signed[mask]>0).mean(),away_track_CI=ci(signed[mask],z['track'][mask]),away_fraction_track_CI=ci((signed[mask]>0).astype(float),z['track'][mask])).items():eq(v,rec[k])
    points.append(dict(fold=fold,seed=seed,intervention=name,**rec))
  if name in ('rotate','mirror','translate','scale08','scale12'):
   fixed=p[np.arange(len(p)),z['base_logits'].argmax(1)];orig=b[np.arange(len(b)),z['base_logits'].argmax(1)]
   geometry.append(dict(fold=fold,seed=seed,intervention=name,n=len(p),same_slot_meanpath=float(np.linalg.norm(p-b,axis=-1).mean()),same_slot_max=float(abs(p-b).max()),fixed_baseline_top1_meanpath=float(np.linalg.norm(fixed-orig,axis=-1).mean()),argmax_switch_fraction=float((z['logits'].argmax(1)!=z['base_logits'].argmax(1)).mean())))
  if name.startswith('leave'):
   slotnum=int(name[5:]);active=data['valid']>slotnum;att=z['baseline_attention'];eq(att,base['attention'][:,slotnum],0)
   effects=np.linalg.norm(p-b,axis=-1).mean((1,2));valid=active&np.isfinite(att);rho,pv=spearmanr(att[valid],effects[valid]) if valid.sum()>2 and np.ptp(att[valid])>0 and np.ptp(effects[valid])>0 else (np.nan,np.nan)
   attention.append(dict(fold=fold,seed=seed,slot=slotnum,n=int(valid.sum()),rho=float(rho) if np.isfinite(rho) else None,p_unadjusted=float(pv) if np.isfinite(pv) else None,attention_effects=att[valid].tolist(),same_slot_effects=effects[valid].tolist()))
  if 'away' in z:
   jp=z['joint_first2'];lg=z['joint_logits_first2'];target=jp[np.arange(len(jp)),0,lg[:,0].argmax(1)];nei=jp[np.arange(len(jp)),1,lg[:,1].argmax(1)];near=np.linalg.norm(target-nei,axis=-1).min(1)
   expected=dict(n=len(jp),min_distance_mean=float(near.mean()),collision_fraction_below_02=float((near<.2).mean()),tca_max_error=float(abs(z['tca']-5).max()),distance_max_error=float(abs(z['closest']-z['distance']).max()))
   for k,v in expected.items():eq(v,m['synthetic_joint_top1'][k])
   dose.append(dict(fold=fold,seed=seed,intervention=name,ids=z['sample_id'].tolist(),track=z['track'].tolist(),distance=float(z['distance'][0]),side=int(z['side'][0]),signed={k:v.tolist() for k,v in signedreads.items()},joint=m['synthetic_joint_top1']))
  rows+=len(p)
  if len(seen)%32==0:print('REVIEW',len(seen),480,flush=True)
 assert len(seen)==480
 attpool=[]
 for fold in ('eth','hotel','univ','zara01','zara02'):
  for seed in (1,2,3):
   group=[v for v in attention if v['fold']==fold and v['seed']==seed];x=np.concatenate([v['attention_effects'] for v in group]);y=np.concatenate([v['same_slot_effects'] for v in group])
   rho,pv=spearmanr(x,y) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else (np.nan,np.nan)
   attpool.append(dict(fold=fold,seed=seed,n=len(x),rho=float(rho) if np.isfinite(rho) else None,p_unadjusted=float(pv) if np.isfinite(pv) else None,not_independent_slots=True,not_causal=True))
 slopes=[]
 for fold in ('eth','hotel','univ','zara01','zara02'):
  for seed in (1,2,3):
   for side in (1,-1):
    arr=sorted([v for v in dose if v['fold']==fold and v['seed']==seed and v['side']==side],key=lambda v:v['distance']);dist=np.array([v['distance'] for v in arr]);assert len(arr)==5
    for method in ('top1','meanK'):
     y=np.array([v['signed'][method] for v in arr]);assert all(v['ids']==arr[0]['ids'] for v in arr);center=dist-dist.mean();slope=(center[:,None]*y).sum(0)/(center@center)
     slopes.append(dict(fold=fold,seed=seed,side=side,method=method,n=y.shape[1],tracks=len(set(arr[0]['track'])),distance_means=y.mean(1).tolist(),slope_mean=float(slope.mean()),slope_track_CI=ci(slope,np.array(arr[0]['track'])),fraction_window_monotone_decreasing=float((np.diff(y,axis=0)<=0).all(0).mean()),mean_monotone_decreasing=bool((np.diff(y.mean(1))<=0).all()),not_corrected_significance=True))
 verify()
 write(C/'SUMMARY.json',dict(point_records=points,quality=quality,temporal=temporal,geometry=geometry,attention_per_slot=attention,attention_pooled=attpool,dose_response=slopes,synthetic=dose,primary_P2_family_300_no_valid_corrected_p_test_implemented=True))
 write(C/'RESOURCE_USAGE.json',dict(status='PASS',**RES))
 write(C/'REVIEW.json',dict(status='PASS',science_jobs=480,full_cluster_windows=rows,metric_checks=CHECKS,max_error=MAX,bootstrap_intervals_recomputed=CIS,source_locked_files=363,input_output_files=len(locks),initial_final_SHA_unchanged=True,rawobs_reconstruction_all480=True,synthetic_closest_inverse_geometry_attention_verified=True,no_torch=True,no_model_forward=True,no_optimizer=True,not_external_peer_review=True,elapsed_seconds=time.perf_counter()-t,SUMMARY_SHA=sha(C/'SUMMARY.json')))
 print(json.dumps(read(C/'REVIEW.json')),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  write(C/'FAILURE_v2.json',dict(status='FAILURE',traceback=traceback.format_exc(),resource=RES));raise

