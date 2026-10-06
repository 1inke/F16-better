"""CPU saved-output review; no torch/model/optimizer. Does not repeat complete-link clustering."""
from pathlib import Path
import json,hashlib,time,gc,sys
import numpy as np
HERE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(HERE))
from resource_guard import guard
ROOT=HERE.parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def verify(files):
 for i,(p,h) in enumerate(files.items()):
  if i%1000==0:guard.check()
  assert sha(p)==h,p
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main():
 start=time.perf_counter();review=read(HERE/'prelaunch_review/REVIEW.json');source=review['source_hashes'];old=read(HERE/'old_results_manifest.json')['files'];locks=read(HERE/'ARITHMETIC_AUDIT.json')['outputs']
 verify(source);verify(old);verify(locks)
 result=read(HERE/'RESULT.json');assert result['status']=='PASS_DIAGNOSTIC_ONLY_ENDPOINT_ACCEL_PROJECTION'
 futurepath=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'
 with np.load(futurepath,allow_pickle=False) as z:ids=z['sample_id'];future=z['future']
 with np.load(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz',allow_pickle=False) as z:groups=z['change_group'];logits=z['logits']
 maxerror=0.;endpointmax=0.;referencemax=0.;alphaerr=0.;rows=[]
 for record in result['records']:
  split=record['split'];a,b=(0,25912) if split=='FIT' else (25912,28791)
  dst=next(iter(record['outputs']));z=load(dst);assert np.array_equal(z['sample_id'],ids[a:b])
  native=load(HERE/('jobs/native_'+split+'.npz')) if record['model'] else z
  base=native['native_trajectory'].astype(np.float64)
  prefixes=('original_','projected_') if record['model'] else ('native_',)
  for prefix in prefixes:
   path=z[prefix+'trajectory'].astype(np.float64);err=np.linalg.norm(path-future[a:b,None].astype(np.float64),axis=-1)
   aa=err.mean(-1);ff=err[:,:,-1];good=(aa<=.2)&(ff<=.4)
   selected=logits[a:b].argmax(-1);row=np.arange(b-a);oracle=np.maximum(aa/.2,ff/.4).argmin(-1)
   exact={'coverage':good.any(-1),'raw_good':good.sum(-1),'loose_coverage':((aa<=.3)&(ff<=.5)).any(-1),'selected_good':good[row,selected],'oracle_slot':oracle}
   for k,v in exact.items():assert np.array_equal(v,z[prefix+k]),(dst,prefix,k)
   assert np.array_equal(z[prefix+'effective_good']==0,~good.any(-1))
   scalar={'minADE':aa.min(-1),'minFDE':ff.min(-1),'joint_J':np.maximum(aa/.2,ff/.4).min(-1)}
   for k,v in scalar.items():
    delta=float(np.abs(v-z[prefix+k]).max());maxerror=max(maxerror,delta);assert delta<=1e-12
   v=np.diff(path,axis=2,prepend=np.zeros_like(path[:,:,:1]))/.4
   ac=np.diff(v,axis=2)/.4;je=np.diff(ac,axis=2)/.4
   motion={k:np.linalg.norm(x,axis=-1) for k,x in [('speed',v),('accel',ac),('jerk',je)]}
   for k,x in motion.items():assert np.array_equal(x,z[prefix+'motion_'+k]),(dst,k)
   for group,g in zip(('overall','stable','transition','change'),record['groups'][prefix[:-1]]):
    mask=np.ones(b-a,bool) if group=='overall' else groups[a:b]==('stable','transition','change').index(group)
    assert int(good[mask].any(-1).sum())==g['coverage_total']
    for k,x in motion.items():
     arr=x[mask].reshape(-1)
     for stat,value in [('mean',float(arr.mean())),('p95',float(np.quantile(arr,.95)))]:
      delta=abs(value-g[k+'_'+stat]);maxerror=max(maxerror,delta);assert delta<=1e-12
   rows.append(dict(split=split,model=record['model'],prefix=prefix,rows=b-a,all_same_candidate_metrics_and_pooled_motion_PASS=True))
   if prefix=='projected_':
    mb=native['native_motion_accel'].mean((1,2));mp=motion['accel'].mean((1,2));assert np.all(mp<=1.01*mb)
    u=z['increments'].astype(np.float64);up=z['projected_increments'];D=np.linalg.norm(np.diff(u,axis=2)/.4**2,axis=-1).mean((1,2))
    nominal=np.ones(b-a);nz=D!=0;nominal[nz]=np.clip(.01*mb[nz]/D[nz],0,1)
    reserve=8*((512*np.finfo(np.float64).eps/2)/(1-512*np.finfo(np.float64).eps/2))*(np.max(np.abs(base),axis=(1,2,3))+12*np.max(np.abs(u),axis=(1,2,3)))/.4**2
    safe=np.ones(b-a);safe[nz]=np.clip(np.maximum(0,.01*mb[nz]-reserve[nz])/D[nz],0,1)
    ae=float(np.abs(safe-z['projection_alpha']).max());alphaerr=max(alphaerr,ae);assert ae<=1e-12
    ep=float(np.abs((path-base)[:,:,-1]-u.sum(2)).max());endpointmax=max(endpointmax,ep);assert ep<=2e-5
    mu=u.mean(2,keepdims=True);ref=base+np.cumsum(mu+nominal[:,None,None,None]*(u-mu),2)
    re=float(np.abs(path-ref).max());referencemax=max(referencemax,re);assert re<=2e-5
    assert np.all(np.abs(up)<=z['limit'].astype(np.float64))
  del z,native,base,path,err,v,ac,je,motion;gc.collect();guard.check()
 verify(locks);verify(source);verify(old)
 out=dict(status='PASS_CPU_SAVED_OUTPUT_REVIEW',records=rows,snapshots=len(rows),row_instances=sum(x['rows'] for x in rows),max_saved_metric_group_summary_error=maxerror,max_endpoint_error=endpointmax,max_nominal_reference_error=referencemax,max_alpha_recompute_error=alphaerr,outputs_SHA=len(locks),source_SHA=len(source),old_SHA=len(old),all_before_after_SHA_unchanged=True,no_torch=True,no_model_forward=True,no_optimizer=True,no_complete_link_second_reconstruction=True,production_full_cluster_reconstruction_previously_PASS=True,memory=guard.check(),seconds=time.perf_counter()-start)
 (HERE/'completion_review/REVIEW.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 print(json.dumps({k:out[k] for k in ('status','snapshots','row_instances','max_saved_metric_group_summary_error','max_endpoint_error','max_nominal_reference_error','max_alpha_recompute_error','outputs_SHA','source_SHA','old_SHA','memory','seconds')}))
if __name__=='__main__':main()

