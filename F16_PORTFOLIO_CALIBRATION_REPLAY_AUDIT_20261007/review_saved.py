"""Independent saved-only NumPy/SciPy review; no model import/forward."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(n,v):(H/n).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def physics(p):
 v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4
 s=np.linalg.norm(v,axis=-1);a=np.linalg.norm(np.diff(v,axis=2)/.4,axis=-1)
 j=np.linalg.norm(np.diff(v,n=2,axis=2)/(.4**2),axis=-1)
 return [float(np.quantile(s,.95)),float(a.mean()),float(np.quantile(a,.95)),float(j.mean()),float(np.quantile(j,.95))]
def cluster(p,g):
 out=[]
 for x,b in zip(p,g):
  if len(x)==1:out.append([1,int(b[0])]);continue
  d=np.linalg.norm(x[:,None]-x[None],axis=-1);D=np.maximum(d.mean(-1)/.1,d[:,:,-1]/.2)
  lab=fcluster(linkage(D[np.triu_indices(len(x),1)],method='complete'),1,criterion='distance')
  out.append([len(np.unique(lab)),len(np.unique(lab[b]))])
 return np.asarray(out)
def main():
 started=time.time();r=json.loads((H/'RESULT.json').read_text());locked=json.loads((H/'source_manifest.json').read_text())['files']
 assert r['status']=='PASS_DIAGNOSTIC_ONLY_PORTFOLIO_CALIBRATION_REPLAY' and len(r['jobs'])==9
 before={p:sha(p) for p in locked};assert before==locked
 rec=[];maxerr=0.;motionfails=[];cal=[]
 for stamp in r['jobs']:
  tag=stamp['protocol']+'_'+str(stamp['seed']);p=H/'saved'/f'{tag}.npz';assert sha(p)==stamp['output_SHA']
  with np.load(p,allow_pickle=False) as z:z={k:z[k] for k in z.files}
  x=z['paths'].astype(np.float64);gt=z['gt'].astype(np.float64);e=np.linalg.norm(x-gt[:,None],axis=-1);a=e.mean(-1);f=e[:,:,-1];j=np.maximum(a/.2,f/.4);g=(a<=.2)&(f<=.4);N=len(x);row=np.arange(N);rank=z['native_rank'];gr=z['groups']
  assert len(np.unique(z['sample_id']))==N and not np.intersect1d(z['sample_id'],z['train_ids']).size
  ports={'native1':rank[:,:1],'native3':rank[:,:3],'native20':np.tile(np.arange(20),(N,1)),'C1':np.c_[rank[:,0],np.full(N,20),np.full(N,21)],'C2':np.c_[rank[:,0],np.full(N,21),np.full(N,22)],'C3':np.c_[rank[:,0],np.full(N,21),np.full(N,23)],'full24':np.tile(np.arange(24),(N,1))}
  masks={'overall':np.ones(N,bool),'stable':gr==0,'transition':gr==1,'change':gr==2};pop={k:g[row[:,None],ix].any(1) for k,ix in ports.items()}
  for name,ix in ports.items():
   paths=x[row[:,None],ix];cl=cluster(paths,g[row[:,None],ix]);published=r['summaries'][tag]['portfolios'][name]
   maxerr=max(maxerr,abs(cl[:,0].mean()-published['effective_distinct_mean']),abs(cl[:,1].mean()-published['effective_good_mean']))
   for group,m in masks.items():
    if not m.any():continue
    q=g[row[:,None],ix];vals=[float(q[m].any(1).mean()),float(a[row[:,None],ix][m].min(1).mean()),float(f[row[:,None],ix][m].min(1).mean()),float(j[row[:,None],ix][m].min(1).mean())]
    saved=next(v for v in published['groups'] if v['group']==group)
    for key,val in zip(('coverage','minADE','minFDE','J'),vals):maxerr=max(maxerr,abs(val-saved[key]))
    mot=physics(paths[m]);ref=physics(x[m,:20]);old=[saved['motion'][k] for k in ('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95')];maxerr=max(maxerr,max(abs(np.array(mot)-old)))
    passes=all(v<=b*1.01 for v,b in zip(mot,ref))
    if not passes:motionfails.append(dict(job=tag,portfolio=name,group=group,harm_percent=[(v/b-1)*100 if b else None for v,b in zip(mot,ref)]))
    base=pop['native3'];fullbase=pop['native20'];pg=pop[name]
    rec.append(dict(job=tag,protocol=stamp['protocol'],seed=stamp['seed'],portfolio=name,group=group,n=int(m.sum()),coverage=vals[0],minADE=vals[1],minFDE=vals[2],J=vals[3],rescue_vs_native3=int((pg&~base&m).sum()),lost_vs_native3=int((~pg&base&m).sum()),rescue_vs_native20=int((pg&~fullbase&m).sum()),motion_pass=passes))
  ix=ports['C1'];gg=g[row[:,None],ix];aa=a[row[:,None],ix];ff=f[row[:,None],ix];jj=j[row[:,None],ix];u=-2*jj+3*gg;t=np.exp(u-u.max(1,keepdims=True));t/=t.sum(1,keepdims=True)
  for mode in ('A','B','C'):
   prob=z['prob_'+mode].astype(float);pick=prob.argmax(1);conf=prob.max(1);truth=gg[row,pick];ece=sum(m.mean()*abs(conf[m].mean()-truth[m].mean()) for lo in np.arange(10)/10 if (m:=(conf>lo)&(conf<=lo+.1)).any())
   tier=z['tier_'+mode];ret=tier!=3;success=np.where(tier==1,pop['native1'],np.where(tier==2,pop['C1'],False));brier=((conf-truth)**2).mean();ce=-(t*np.log(np.clip(prob,1e-12,1))).sum(1).mean()
   prior=r['summaries'][tag]['preprocessing'][mode]
   for key,val in [('selected_good_ECE',ece),('selected_good_Brier',brier),('utility_crossentropy',ce),('retained_fraction',ret.mean()),('all_window_success',success.mean())]:maxerr=max(maxerr,abs(val-prior[key]))
   for group,m in masks.items():
    rr=m&ret
    if not m.any():continue
    ec=sum(b.mean()*abs(conf[m][b].mean()-truth[m][b].mean()) for lo in np.arange(10)/10 if (b:=(conf[m]>lo)&(conf[m]<=lo+.1)).any())
    cal.append(dict(job=tag,group=group,mode=mode,ECE=float(ec),utility_CE=float(-(t[m]*np.log(np.clip(prob[m],1e-12,1))).sum(1).mean()),uniform_CE=float(np.log(3)),utility_Brier=float(((prob[m]-t[m])**2).sum(1).mean()),uniform_utility_Brier=float(((1/3-t[m])**2).sum(1).mean()),weighted_ADE=float((prob[m]*aa[m]).sum(1).mean()),native1_ADE=float(aa[m,0].mean()),all_success=float(success[m].mean()),retained_fraction=float(rr.sum()/m.sum()),retained_success=float(success[rr].mean()) if rr.any() else None,native1_retained=float(pop['native1'][rr].mean()) if rr.any() else None,native3_retained=float(pop['native3'][rr].mean()) if rr.any() else None,expected_outputs=float(np.where(tier[m]==1,1,np.where(tier[m]==2,3,0)).mean())))
 assert maxerr<1e-10, maxerr
 aggregate={}
 for group in masks:
  aggregate[group]={}
  for name in ports:
   rows=[v for v in rec if v['group']==group and v['portfolio']==name]
   aggregate[group][name]={k:float(np.mean([v[k] for v in rows])) for k in ('coverage','minADE','minFDE','J','rescue_vs_native3','lost_vs_native3','rescue_vs_native20')}
  aggregate[group]['calibration']={mode:{k:float(np.mean([v[k] for v in cal if v['mode']==mode and v['group']==group and v[k] is not None])) for k in ('ECE','utility_CE','uniform_CE','utility_Brier','uniform_utility_Brier','weighted_ADE','native1_ADE','all_success','retained_fraction','retained_success','native1_retained','native3_retained','expected_outputs')} for mode in ('A','B','C')}
 assert all(sha(p)==h for p,h in before.items())
 write('completion_review/REVIEW.json',dict(status='PASS_SAVED_ONLY_NUMPY_SCIPY',jobs=9,source_input_checkpoint_count=len(locked),saved_summary_max_error=maxerr,all_full_clusters_motion_recomputed=True,no_torch_model_forward=True,no_optimizer=True,all_relevant_SHA_before_after=True,elapsed_seconds=time.time()-started))
 write('completion_review/AGGREGATE.json',dict(aggregation='unweighted protocol-by-seed macro, not 9 independent datasets',groups=aggregate,records=rec,calibration=cal,motion_failures=motionfails))
 print(json.dumps(dict(review='PASS',maxerror=maxerr,groups=aggregate,motion_failure_count=len(motionfails)),indent=2))
if __name__=='__main__':
 (H/'completion_review').mkdir(exist_ok=True);main()
