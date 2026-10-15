"""Independent saved-only review. No torch, model, optimizer or old writes."""
import os,sys,time,json,hashlib,ctypes,traceback
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np
import psutil
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.stats import spearmanr
C=Path(__file__).resolve().parent;H=C.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding='utf-8')
class Mem(ctypes.Structure):
 _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
RES=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
def guard():
 m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
 p=psutil.Process().memory_info()
 RES['peak_private_GiB']=max(RES['peak_private_GiB'],p.peak_pagefile/2**30);RES['peak_RSS_GiB']=max(RES['peak_RSS_GiB'],p.peak_wset/2**30)
 RES['min_physical_GiB']=min(RES['min_physical_GiB'],m.avail_phys/2**30);RES['min_commit_GiB']=min(RES['min_commit_GiB'],m.avail_page/2**30);RES['checks']+=1
 if max(p.private,p.rss)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:raise MemoryError('RESOURCE6')
MAX=0.;CHECKS=0
def eq(a,b,tol=1e-10):
 global MAX,CHECKS
 CHECKS+=1
 if a is None or b is None:assert a is None and b is None;return
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape
 err=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
 MAX=max(MAX,err);assert err<=tol,(err,tol)
def errors(p,g):
 e=np.sqrt(np.sum((p-g[:,None] if p.ndim==4 else p-g)**2,axis=-1));return e.mean(-1),e[...,-1]
def cv(o,k=1):return ((o[:,0,-1]-o[:,0,-1-k])/k).astype(float)[:,None]*np.arange(1,13)[None,:,None]
def ca(o,cap):
 v=np.diff(o[:,0].astype(float),axis=1);a=v[:,-1]-v[:,-2];a*=np.minimum(1.,cap/np.maximum(np.linalg.norm(a,axis=1),1e-300))[:,None]
 return np.cumsum(v[:,-1,None]+np.cumsum(.8**np.arange(1,13)[None,:,None]*a[:,None],axis=1),axis=1)
def kalman(o,q,r):
 y=o[:,0].astype(float);T=np.array([[1.,1.],[0.,1.]]);Q=q*np.array([[.25,.5],[.5,1.]])
 x=np.stack((y[:,0],y[:,1]-y[:,0]),-1);P=np.tile(np.eye(2)[None],(len(y),1,1))
 for t in range(1,8):
  x=x@T.T;P=T@P@T.T+Q;K=P[:,:,0]/(P[:,0,0]+r)[:,None];x+=(y[:,t]-x[:,:,0])[:,:,None]*K[:,None,:];P-=K[:,:,None]*P[:,None,0,:]
 return x[:,:,0][:,None]+x[:,:,1][:,None]*np.arange(1,13)[None,:,None]-y[:,-1,None]
def ci(vals,tracks,den=None,ratio=False):
 uniq,inv=np.unique(tracks,return_inverse=True);s=np.bincount(inv,weights=vals);n=np.bincount(inv,weights=den) if den is not None else np.bincount(inv)
 rng=np.random.default_rng(20261009);out=[]
 for start in range(0,2000,100):
  ix=rng.integers(0,len(uniq),(min(100,2000-start),len(uniq)));a=s[ix].sum(1);b=n[ix].sum(1);ok=b>0
  out.extend((1-a[ok]/b[ok] if ratio else a[ok]/b[ok]).tolist())
 return np.quantile(out,[.025,.5,.975]).tolist() if out else None
def perm(rm,rg,b):
 den=(rg*rg).sum()
 if den==0:return None
 rng=np.random.default_rng(20261009);groups=[np.flatnonzero(b==k) for k in np.unique(b)];out=[]
 for _ in range(1000):
  ix=np.arange(len(rm))
  for g in groups:ix[g]=rng.permutation(g)
  out.append(float(1-((rg-rm[ix])**2).sum()/den))
 return np.array(out)
def summary(p,g,last,delta):
 aa,ff=errors(p,g);J=np.maximum(aa/.2,ff/.4).min(1);dev=np.linalg.norm(p-last[:,None],axis=-1).mean(-1)>delta
 return dict(minADE=aa.min(1),minFDE=ff.min(1),J=J,coverage=(J<=1).astype(float),nonCV_hit=((aa<=.2)&(ff<=.4)&dev).any(1).astype(float),nonCV_candidate_count=dev.sum(1),spread=np.linalg.norm(p[:,:,-1]-p[:,:,-1].mean(1,keepdims=True),axis=-1).mean(1),**{f'coverage_{a:.2f}':((aa<=a)&(ff<=2*a)).any(1).astype(float) for a in (.1,.15,.2,.3,.4)})
def cluster(p,g):
 aa,ff=errors(p,g);good=(aa<=.2)&(ff<=.4);n=len(p);distinct=np.empty(n,int);effective=np.empty(n,int);I,J=np.triu_indices(20,1)
 for start in range(0,n,256):
  guard()
  for i in range(start,min(start+256,n)):
   dist=np.linalg.norm(p[i,I]-p[i,J],axis=-1);d=np.maximum(dist.mean(1)/.10,dist[:,-1]/.20)
   labs=fcluster(linkage(d,method='complete'),t=1.,criterion='distance')
   distinct[i]=len(np.unique(labs));effective[i]=len(np.unique(labs[good[i]]))
 return distinct,effective
def motion(p,m):
 if not m.any():return None
 x=p[m];v=np.diff(np.concatenate((np.zeros((len(x),20,1,2)),x),2),axis=2)/.4;a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4
 vn=np.linalg.norm(v,axis=-1).ravel();an=np.linalg.norm(a,axis=-1).ravel();jn=np.linalg.norm(j,axis=-1).ravel()
 return dict(speedp95=float(np.quantile(vn,.95)),accelmean=float(an.mean()),accelp95=float(np.quantile(an,.95)),jerkmean=float(jn.mean()),jerkp95=float(np.quantile(jn,.95)))
def main():
 t=time.perf_counter();guard();assert read(H/'RESULT.json')['status']=='PASS_STAGE1_E0_E1_E2_E7'
 assert read(H/'worker_state.json')['status']=='SUCCESS' if (H/'worker_state.json').exists() else True
 P=read(H/'protocol.json');locks=read(H/'source_manifest.json')['files'].copy();locks.update(read(H/'PREFLIGHT.json')['calibration_sha'])
 for p in H.glob('data/*'):locks[str(p)]=digest(p)
 result=read(H/'RESULT.json')
 for dp in result['jobs']:
  d=read(dp);assert d['status']=='SUCCESS' and d['source_manifest_sha']==digest(H/'source_manifest.json')
  locks[dp]=digest(dp);locks.update(d['output_sha'])
  if 'checkpoint' in d:assert digest(d['checkpoint'])==d['checkpoint_sha'] and d['frozen_weights_buffers_EXACT']
 for f in ('RESULT.json','ARITHMETIC_AUDIT.json','FROZEN_AUDIT.json','RESOURCE_USAGE.json'):locks[str(H/f)]=digest(H/f)
 locks[str(Path(__file__))]=digest(__file__)
 def verify():
  for i,(p,s) in enumerate(locks.items()):
   if i%32==0:guard()
   assert digest(p)==s,('SHA',p)
 verify();write(C/'input_manifest.json',dict(files=locks))
 points=[];candidates=[];paired=[];holm=[];motionrows=[];nullcount=0;bootstrapcount=0;rows=0
 for fold in P['folds']:
  data=load(H/f'data/{fold}_evaluation.npz');o=data['obs'];g=data['gt'];tracks=data['track'];last=cv(o);cal=read(H/f'calibration/{fold}.json')
  baselines=dict(CV_last=last,CV_avg3=cv(o,3),CA=ca(o,cal['ca_cap']),Kalman=kalman(o,cal['kalman']['q'],cal['kalman']['r']))
  strong=baselines[cal['strongest_simple']];ba,bf=errors(strong,g);rg=g-last;den=(rg*rg).sum((1,2));cache=[]
  for seed in P['model_seeds']:
   print('REVIEW',fold,seed,flush=True);guard();path=H/'jobs'/f'{fold}_seed{seed}';z=load(path/'outputs.npz');s=read(path/'summary.json');ns=load(path/'nulls.npz');p=z['model_paths']
   assert np.array_equal(z['sample_id'],data['ids']) and np.array_equal(z['track'],tracks);eq(z['GT'],g,0);eq(z['CV'],last,0)
   masks={k[5:]:v for k,v in z.items() if k.startswith('mask_')};cs=summary(p,g,last,cal['delta_nonCV_candidate'])
   for k,v in cs.items():eq(v,z['candidate_'+k])
   dc,ec=cluster(p,g);np.savez_compressed(C/f'{fold}_seed{seed}_clusters.npz',sample_id=data['ids'],distinct=dc,effectivegood=ec)
   u=o[:,0,-1]-o[:,0,-2];ss=np.linalg.norm(u,axis=1);defined=ss>=.02;direction=np.zeros_like(u,dtype=float);direction[:,0]=1.;direction[defined]=u[defined]/ss[defined,None];normal=np.stack((-direction[:,1],direction[:,0]),1)
   reads=dict(top1=p[np.arange(len(p)),z['model_logits'].argmax(1)],meanK=p.mean(1))
   for name,pnt in {**baselines,**reads}.items():
    aa,ff=errors(pnt,g);eq(aa,z[name+'_ADE']);eq(ff,z[name+'_FDE']);rm=pnt-last;num=((g-pnt)**2).sum((1,2));norm=np.sqrt((rm*rm).sum((1,2))*den);cos=np.divide((rm*rg).sum((1,2)),norm,out=np.full(len(p),np.nan),where=norm>0)
    for rec in (r for r in s['point_records'] if r['method']==name):
     m=masks[rec['subset']];n=int(m.sum());assert n==rec['n'] and len(np.unique(tracks[m]))==rec['tracks']
     for k,val in dict(ADE=float(aa[m].mean()) if n else None,FDE=float(ff[m].mean()) if n else None,skill=float(1-num[m].sum()/den[m].sum()) if den[m].sum()>0 else None,cos_mean=float(np.nanmean(cos[m])) if np.isfinite(cos[m]).any() else None).items():eq(val,rec[k])
     eq(int(np.isfinite(cos[m]).sum()),rec['cos_defined_n'],0)
     if n:eq(float((aa-ba)[m].mean()),rec['ADE_delta_strongest_simple']);eq(float((ff-bf)[m].mean()),rec['FDE_delta_strongest_simple'])
     if name in reads and n:
      eq(ci(num[m],tracks[m],den[m],True),rec['skill_track_CI']);eq(ci((aa-ba)[m],tracks[m]),rec['ADE_delta_track_CI']);eq(ci((ff-bf)[m],tracks[m]),rec['FDE_delta_track_CI']);bootstrapcount+=3
      nul=perm(rm[m],rg[m],z['bin'][m])
      if nul is not None:
       eq(nul,ns[name+'__'+rec['subset']]);eq(float(np.quantile(nul,.95)),rec['null95'])
       pv=float((1+(nul>=rec['skill']).sum())/1001);eq(pv,rec['p_permutation_plus1'],0);nullcount+=1000
       if rec['subset']=='nonCV_cue_positive':holm.append(dict(fold=fold,seed=seed,method=name,n=n,tracks=rec['tracks'],skill=rec['skill'],skill_CI=rec['skill_track_CI'],raw_p=pv,ADE_delta=rec['ADE_delta_strongest_simple'],ADE_CI=rec['ADE_delta_track_CI'],FDE_delta=rec['FDE_delta_strongest_simple'],FDE_CI=rec['FDE_delta_track_CI'],strongest_simple=cal['strongest_simple']))
      for axis,vec in [('longitudinal',direction),('lateral',normal)]:
       gs=(rg[:,-1]*vec).sum(1);ps=(rm[:,-1]*vec).sum(1);ok=m&defined&(abs(gs)>1e-12);count=int(ok.sum());eq(count,rec[axis+'_sign_n'],0);eq(float((np.sign(gs[ok])==np.sign(ps[ok])).mean()) if count else None,rec[axis+'_sign_accuracy']);eq(float(max((gs[ok]>0).mean(),(gs[ok]<0).mean())) if count else None,rec[axis+'_GT_majority_baseline'])
     points.append(dict(fold=fold,seed=seed,**rec))
   for rec in s['candidate_records']:
    m=masks[rec['subset']]
    for k,v in cs.items():eq(float(v[m].mean()),rec[k])
    for key,y in [('spread_vs_CV_FDE',z['CV_last_FDE']),('spread_vs_top1_FDE',z['top1_FDE'])]:
     r=rec[key]
     if r is not None:rho,pv=spearmanr(cs['spread'][m],y[m]);eq(rho,r['rho']);eq(pv,r['p_unadjusted_exploratory'])
    candidates.append(dict(kind='model',fold=fold,seed=seed,**rec,distinct=float(dc[m].mean()),effectivegood=float(ec[m].mean())))
    motionrows.append(dict(kind='model',fold=fold,seed=seed,subset=rec['subset'],n=int(m.sum()),**motion(p,m)))
   cache.append(dict(cs=cs,masks=masks,dc=dc,ec=ec));rows+=len(p);del ns,z
  for i,rngseed in enumerate(P['random_control_seeds']):
   guard();path=H/'jobs'/f'{fold}_random{rngseed}';z=load(path/'outputs.npz');s=read(path/'summary.json');p=z['paths'];assert np.array_equal(z['sample_id'],data['ids']);cs=summary(p,g,last,cal['delta_nonCV_candidate'])
   for k,v in cs.items():eq(v,z[k])
   dc,ec=cluster(p,g);np.savez_compressed(C/f'{fold}_random{rngseed}_clusters.npz',sample_id=data['ids'],distinct=dc,effectivegood=ec)
   model=cache[i];masks=model['masks']
   for rec in s['candidate_records']:
    m=masks[rec['subset']]
    for k,v in cs.items():eq(float(v[m].mean()),rec[k])
    candidates.append(dict(kind='random',fold=fold,seed=rngseed,**rec,distinct=float(dc[m].mean()),effectivegood=float(ec[m].mean())))
    motionrows.append(dict(kind='random',fold=fold,seed=rngseed,subset=rec['subset'],n=int(m.sum()),**motion(p,m)))
    pair=dict(fold=fold,model_seed=i+1,control_rng=rngseed,subset=rec['subset'],n=int(m.sum()),tracks=len(np.unique(tracks[m])))
    for k in ('minADE','minFDE','J','coverage','nonCV_hit'):
     d=model['cs'][k]-cs[k];pair[k+'_delta']=float(d[m].mean());pair[k+'_track_CI']=ci(d[m],tracks[m]);bootstrapcount+=1
    d=model['ec']-ec;pair['effectivegood_delta']=float(d[m].mean());pair['effectivegood_track_CI']=ci(d[m],tracks[m]);bootstrapcount+=1
    mg=model['cs']['coverage'].astype(bool);cg=cs['coverage'].astype(bool)
    pair['model_rescue_vs_control']=int((m&mg&~cg).sum());pair['model_lost_vs_control']=int((m&~mg&cg).sum())
    paired.append(pair)
   rows+=len(p);del z,p
  del cache
 assert len(holm)==30
 order=np.argsort([r['raw_p'] for r in holm],kind='stable');running=0.
 for rank,i in enumerate(order):
  running=max(running,(len(holm)-rank)*holm[i]['raw_p']);holm[i]['holm_p']=min(1.,running);holm[i]['conditional_null_rejected_0.05']=holm[i]['holm_p']<.05
 verify();write(C/'SUMMARY.json',dict(P1=holm,point_records=points,candidate_records=candidates,paired_E7=paired,motion=motionrows,denominators=[dict(fold=f,**read(H/f'data/{f}_identity.json')) for f in P['folds']],Holm_family=30,random_seeds_are_not_model_replicates=True))
 write(C/'RESOURCE_USAGE.json',dict(status='PASS',**RES))
 outs={str(p):digest(p) for p in C.glob('*') if p.is_file() and p.name not in ('REVIEW.json','input_manifest.json','audit_saved.py')}
 write(C/'REVIEW.json',dict(status='PASS',science_jobs=30,full_cluster_windows=rows,metric_checks=CHECKS,max_error=MAX,permutation_values_recomputed=nullcount,bootstrap_intervals_recomputed=bootstrapcount,source_calibration_done_output_initial_final_SHA_unchanged=True,input_files=len(locks),saved_output_sha=outs,elapsed_seconds=time.perf_counter()-t,no_model_forward=True,no_torch=True,no_optimizer=True,no_gradient_replay=True,not_external_peer_review=True))
 print(json.dumps(read(C/'REVIEW.json')),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  write(C/'FAILURE.json',dict(status='FAILURE',traceback=traceback.format_exc(),resource=RES));raise

