"""Frozen native-support audit. Only NumPy/SciPy saved data; no model imports."""
import os,sys,json,time,hashlib,importlib.util,traceback,inspect
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent;P=H.parent/'experiment1_motion_process'
sp=importlib.util.spec_from_file_location('saved_cpu',P/'completion_review/audit_saved.py');A=importlib.util.module_from_spec(sp);sp.loader.exec_module(A)
ARMS=('position3','acceleration3');SEEDS=(42,43,44);SPLITS=('outer','inner','original_CAL');PORTS=('native20','bottom20','policy20','bank24')
def write(name,x):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist()),encoding='utf-8');os.replace(t,p)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
 with t.open('wb') as f:np.savez_compressed(f,**x)
 os.replace(t,p)
def dist(p,q):
 e=np.sqrt(np.sum((p-q)**2,-1));return np.maximum(e.mean(-1)/.1,e[...,-1]/.2)
def labels(paths):
 d=np.asarray([dist(paths[i],paths[j]) for i in range(len(paths)-1) for j in range(i+1,len(paths))])
 return fcluster(linkage(d,method='complete'),1.,criterion='distance')
def allowed(m,base):return all(m[k]<=base[k]*1.01 for k in base)
def policy(native,learned,independent=False):
 assert native.shape==(20,12,2) and learned.shape==(4,12,2)
 assert np.isfinite(native).all() and np.isfinite(learned).all()
 lab=labels(native);rep=np.asarray([np.flatnonzero(lab==lab[i])[0] for i in range(20)])
 cur=native.copy();origin=np.arange(20);base=A.motion(native[None]);events=[]
 for j in range(4):
  eligible=[i for i in range(20) if origin[i]==i and rep[i]!=i]
  if not eligible:events.append(dict(learned=j,reason='no_removable',removed=-1));continue
  if independent:
   redundancy=[float(dist(native[i],native[rep[i]])) for i in eligible]
   index=sorted(zip(redundancy,eligible))[0][1];old=float(dist(native[index],native[rep[index]]))
   novel=min(float(dist(learned[j],cur[i])) for i in range(20) if i!=index)
  else:
   dd=dist(native[eligible],native[rep[eligible]]);index=eligible[int(np.argmin(dd))];old=float(dd.min())
   novel=float(dist(learned[j],np.delete(cur,index,axis=0)).min())
  event=dict(learned=j,removed=index,redundancy=old,novelty=novel)
  if not novel>old:event['reason']='novelty';events.append(event);continue
  temp=cur.copy();temp[index]=learned[j];mot=A.motion(temp[None]);bad=[k for k in base if mot[k]>1.01*base[k]]
  event['failed_motion']=bad
  if bad:event['reason']='physical';events.append(event);continue
  cur=temp;origin[index]=20+j;event['reason']='accepted';events.append(event)
 cert=dist(native,native[rep])
 assert np.max(cert)<=1. and all(np.array_equal(cur[r],native[r]) and origin[r]==r for r in np.unique(rep))
 assert all(np.array_equal(cur[i],native[origin[i]]) if origin[i]<20 else np.array_equal(cur[i],learned[origin[i]-20]) for i in range(20))
 return cur,origin,rep,cert,events
def integrate(native,learned,independent=False):
 out=[];orig=[];reps=[];cert=[];events=[]
 for i in range(len(native)):
  if i%32==0:A.guard()
  p,o,r,c,e=policy(native[i],learned[i],independent);out.append(p);orig.append(o);reps.append(r);cert.append(c);events.append(e)
 return np.asarray(out),np.asarray(orig),np.asarray(reps),np.asarray(cert),events
def evaluate(z):
 metrics={};vectors=[];cluster=[]
 nativegood=A.error(z['native20'],z['gt'])[3].any(1);bottomgood=A.error(z['bottom20'],z['gt'])[3].any(1)
 for port in PORTS:
  p=z[port];a,f,j,g,one=A.error(p,z['gt']);good=g.any(1)
  cl=[]
  for i in range(len(p)):
   if i%32==0:A.guard()
   lab=labels(p[i]);cl.append([len(np.unique(lab)),len(np.unique(lab[g[i]]))])
  cl=np.asarray(cl);cluster.append(cl);vectors.append(np.stack([good,a.min(1),f.min(1),j.min(1)],1));metrics[port]={}
  for name,mask in A.masks(z['groups']):
   n=int(mask.sum())
   if not n:metrics[port][name]=dict(n=0);continue
   mot=A.motion(p[mask]);base=A.motion(z['native20'][mask])
   metrics[port][name]=dict(n=n,good=int(good[mask].sum()),coverage=float(good[mask].mean()),minADE=float(a[mask].min(1).mean()),minFDE=float(f[mask].min(1).mean()),J=float(j[mask].min(1).mean()),native_rescue=int((good&~nativegood&mask).sum()),native_lost=int((~good&nativegood&mask).sum()),native_retained=int((good&nativegood&mask).sum()),bottom_rescue=int((good&~bottomgood&mask).sum()),bottom_lost=int((~good&bottomgood&mask).sum()),effective_distinct=float(cl[mask,0].mean()),effective_good=float(cl[mask,1].mean()),sensitivity_03_05_good=int(((a<=.3)&(f<=.5)).any(1)[mask].sum()),single_Jclosest_early_middle_late=one[mask].mean(0),motion=mot,native_motion=base,motion_all5_1pct=allowed(mot,base))
 return metrics,np.stack(vectors,1),np.stack(cluster,1)
def sourcejob(arm,seed,split):
 z=A.npz(P/'runs'/arm/f'seed_{seed}'/f'{split}.npz')
 assert len(np.unique(z['sample_id']))==len(z['sample_id'])
 assert z['native20'].shape==(len(z['sample_id']),20,12,2) and z['refined4'].shape==(len(z['sample_id']),4,12,2)
 keep=np.ones(z['native20'].shape[:2],bool);keep[np.arange(len(keep))[:,None],z['slots']]=False
 assert np.array_equal(z['refined20'][keep],z['native20'][keep])
 assert np.array_equal(z['refined20'][np.arange(len(keep))[:,None],z['slots']],z['refined4'])
 return z
def build(z):
 policy_,origin,rep,cert,events=integrate(z['native20'],z['refined4'])
 out={k:z[k] for k in ('sample_id','groups','track','gt','slots','refined4')}
 out.update(native20=z['native20'],bottom20=z['refined20'],policy20=policy_,bank24=np.concatenate([z['native20'],z['refined4']],1),origin=origin,representative=rep,certificate=cert)
 return out,events
def audit(z,events):
 p,o,r,c,e=integrate(z['native20'],z['refined4'],True)
 assert np.array_equal(p,z['policy20']) and np.array_equal(o,z['origin']) and np.array_equal(r,z['representative'])
 assert A.compare(c,z['certificate'])<=1e-12 and A.digest(e)==A.digest(events)
 return evaluate(z)
def boot(saved):
 rng=np.random.default_rng(20261011);out=[]
 for arm in ARMS:
  for split in SPLITS:
   zz=[saved[(arm,s,split)] for s in SEEDS];z=zz[0]
   for x in zz:assert all(np.array_equal(z[k],x[k]) for k in ('sample_id','groups','track'))
   _,inv=np.unique(z['track'],return_inverse=True);T=inv.max()+1
   for group,mask in A.masks(z['groups']):
    counts=np.bincount(inv[mask],minlength=T);active=np.flatnonzero(counts)
    for ri,ref in enumerate(('native20','bottom20')):
     diff=np.stack([x['vectors'][:,2]-x['vectors'][:,ri] for x in zz]);sums=np.zeros((3,T,4))
     for i in range(3):np.add.at(sums[i],inv[mask],diff[i,mask])
     draws=[]
     for rep in range(2000):
      if rep%64==0:A.guard()
      ss=rng.integers(0,3,3);tt=rng.choice(active,len(active),replace=True);draws.append(sums[ss][:,tt].sum((0,1))/(3*counts[tt].sum()))
     out.append(dict(arm=arm,split=split,group=group,reference=ref,n=int(mask.sum()),tracks=len(active),metrics=['coverage','ADE','FDE','J'],estimate=diff[:,mask].mean((0,1)),CI95=np.quantile(draws,[.025,.975],axis=0).T,exploratory=True))
 assert len(out)==48
 return out
def summary(events,z):
 reasons={k:sum(e['reason']==k for row in events for e in row) for k in ('accepted','novelty','physical','no_removable')}
 n=(z['origin']>=20).sum(1)
 return dict(reasons=reasons,insertions_histogram=np.bincount(n,minlength=5),removed_histogram=np.bincount([e['removed'] for row in events for e in row if e['reason']=='accepted'],minlength=20),native_cluster_representatives_mean=float(np.mean([len(np.unique(r)) for r in z['representative']])),max_certificate=float(z['certificate'].max()))
def skip(path,manifest):
 d=A.read(path);assert d['source_sha']==A.digest(manifest),'SOURCE_INCOMPATIBLE';A.verify(d['files']);return d
def main():
 start=time.perf_counter();A.guard();source=A.read(H/'source_manifest.json')['files'];A.verify(source)
 saved={};records=[];outputs={}
 for arm in ARMS:
  for seed in SEEDS:
   for split in SPLITS:
    folder=H/'jobs'/arm/f'seed_{seed}'/split;done=folder/'done.json';write('progress.json',dict(stage='FROZEN_POLICY_SAVED_ARITHMETIC',arm=arm,seed=seed,split=split))
    if done.exists():skip(done,source)
    else:
     old=sourcejob(arm,seed,split);z,events=build(old);met,v,cl=evaluate(z);folder.mkdir(parents=True,exist_ok=True)
     save(folder/'paths.npz',z);save(folder/'metrics.npz',dict(vectors=v,clusters=cl));write(str((folder/'events.json').relative_to(H)),events);write(str((folder/'metrics.json').relative_to(H)),dict(metrics=met,integration=summary(events,z)))
     reread=A.npz(folder/'paths.npz');mm,vv,cc=audit(reread,A.read(folder/'events.json'))
     assert A.compare(mm,met)<=1e-12 and np.array_equal(vv,v) and np.array_equal(cc,cl)
     files={str(p):A.sha(p) for p in folder.iterdir() if p.is_file()};write(str(done.relative_to(H)),dict(status='PASS',source_sha=A.digest(source),files=files))
    d=skip(done,source);outputs.update(d['files']);outputs[str(done)]=A.sha(done)
    z=A.npz(folder/'paths.npz');m=A.read(folder/'metrics.json');v=A.npz(folder/'metrics.npz')['vectors']
    saved[(arm,seed,split)]=dict(vectors=v,**{k:z[k] for k in ('sample_id','groups','track')});records.append(dict(arm=arm,seed=seed,split=split,**m))
    print('JOB_PASS',arm,seed,split,flush=True)
 write('progress.json',dict(stage='PAIRED_TRACK_BOOTSTRAP'));ci=boot(saved);write('BOOTSTRAP.json',dict(records=ci,scalar_CIs=192))
 outputs[str(H/'BOOTSTRAP.json')]=A.sha(H/'BOOTSTRAP.json')
 write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=18,independent_policy_loops=True,all_saved_quality_complete_link_motion_recomputed=True,bootstrap_records=48,scalar_CIs=192,no_model_replay=True))
 write('progress.json',dict(stage='FINAL_SOURCE_SHA'));A.verify(source);A.verify(outputs);A.guard()
 write('PRESERVATION.json',dict(status='PASS',source_count=len(source),source_initial_final=True,old_sources_outputs_models_immutable=True))
 write('RESOURCE_USAGE.json',A.resource)
 write('RESULT.json',dict(status='PASS_NATIVE_SUPPORT_INTEGRATION_DIAGNOSTIC',records=records,output_sha=outputs,source_count=len(source),elapsed_seconds=time.perf_counter()-start))
if __name__=='__main__':
 try:main()
 except BaseException:
  write('FAILURE_'+str(time.time_ns())+'.json',dict(status='STOP',traceback=traceback.format_exc(),resource=A.resource));raise
