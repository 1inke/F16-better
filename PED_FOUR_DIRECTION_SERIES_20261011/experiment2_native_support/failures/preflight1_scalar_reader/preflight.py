"""Actual CPU preflight and resource benchmark; no torch."""
import sys,time,json,traceback,inspect
sys.dont_write_bytecode=True
import run as R
import numpy as np
A=R.A;H=R.H;P=R.P
def locks():
 files=dict(A.read(P/'source_manifest.json')['files']);initial=len(files)
 def add(p,value=None):
  p=str(p);value=value or A.sha(p)
  if p in files:assert files[p]==value,('OLD_DIGEST_CONFLICT',p)
  files[p]=value
 for p,v in A.read(P/'RESULT.json')['output_sha'].items():add(p,v)
 for name in ('RESULT.json','RESULT_ANALYSIS.md','RESULT_ANALYSIS.json','ARITHMETIC_AUDIT.json','PRESERVATION.json','RESOURCE_USAGE.json','source_manifest.json','PREFLIGHT.json'):
  add(P/name)
 for p in (P/'completion_review').rglob('*'):
  if p.is_file() and p.suffix in ('.py','.json'):add(p)
 for p in H.iterdir():
  if p.is_file() and (p.suffix in ('.py','.ps1') or p.name in ('README.md','DESIGN.md','protocol.json')):add(p)
 add(H/'completion_review/audit_saved.py')
 return files,initial
def fixtures():
 assert tuple(inspect.signature(R.policy).parameters)==('native','learned','independent')
 p=np.zeros((20,12,2));p[:,:,0]=np.arange(20)[:,None]
 l=np.zeros((4,12,2));l[:,:,0]=100.
 out,o,r,c,e=R.policy(p,l);assert np.array_equal(out,p) and all(x['reason']=='no_removable' for x in e)
 p=np.zeros((20,12,2));out,o,r,c,e=R.policy(p,np.zeros((4,12,2)))
 assert len(np.unique(R.labels(p)))==1 and np.array_equal(out,p) and all(x['reason']=='novelty' for x in e)
 l=np.zeros((4,12,2));l[:,:,0]=np.arange(1,13)[None]*.1
 out,o,r,c,e=R.policy(p,l);assert np.array_equal(out,p) and all(x['reason']=='physical' for x in e) and np.max(c)==0.
 base={k:1. for k in A.motion(p[None])}
 for key in base:
  m=dict(base);m[key]=1.0100001;assert not R.allowed(m,base)
 assert R.allowed(dict(base),base)
 p[:,:,0]=np.arange(1,13)[None]*.4;p[1,:,0]+=.001
 l=np.repeat(p[:1],4,0);l[:,:,0]+=.01
 out,o,r,c,e=R.policy(p,l);out2,o2,r2,c2,e2=R.policy(p,l,True)
 assert np.array_equal(out,out2) and np.array_equal(o,o2) and A.digest(e)==A.digest(e2)
 assert all(o[i]==i for i in np.unique(r)) and c.max()<=1
 assert all(x['removed']>=1 for x in e if x['removed']!=-1)
 # Same-candidate good cannot combine separate ADE and FDE minima.
 path=np.zeros((1,2,12,2));path[0,0,-1,0]=.5;path[0,1,:-1,0]=.3
 a,f,j,g,_=A.error(path,np.zeros((1,12,2)));assert a.min()<.2 and f.min()<.4 and not g.any()
 return dict(unique_singletons=True,copies_one_cluster=True,strict_novelty_tie=True,zero_native_physical=True,each_five_literal_reject=True,joint_representatives_preserved=True,physical_tie_order=True,two_policy_loops=True,signature_no_GT_group_logit_ID=True,same_candidate_disjoint_minima=True)
def main():
 start=time.perf_counter();A.guard();files,inherited=locks();A.verify(files);tests=fixtures()
 source=A.read(P/'RESULT.json');assert source['status']=='PASS_MOTION_PROCESS_PAIRED_RUN'
 cache=next(p for p in files if p.endswith('eth_train.npz') and 'train_dev' in p)
 with np.load(cache,allow_pickle=False) as cc:
  ids=cc['sample_id'].copy();logs=cc['logits'].copy();group=cc['change_group'].copy();native=cc['candidate'][...,:24].reshape(-1,20,12,2).astype(float)
 lookup={int(x):i for i,x in enumerate(ids)}
 allrows=0
 for arm in R.ARMS:
  for seed in R.SEEDS:
   manifest=A.read(P/'runs'/arm/f'seed_{seed}'/'run_manifest.json')
   assert manifest['source_files']==A.read(P/'source_manifest.json')['files']
   for split,N in zip(R.SPLITS,(1035,1856,2879)):
    z=R.sourcejob(arm,seed,split);assert len(z['sample_id'])==N
    ix=np.asarray([lookup[int(x)] for x in z['sample_id']]);assert np.array_equal(z['native20'],native[ix]) and np.array_equal(z['groups'],group[ix])
    slots=np.asarray([sorted(sorted(range(20),key=lambda k:(-float(logs[i,k]),k))[16:]) for i in ix]);assert np.array_equal(slots,z['slots']);allrows+=N
 timings=[];errors=[]
 for arm in R.ARMS:
  z=R.sourcejob(arm,42,'outer');z={k:v[:128] for k,v in z.items()}
  out,ev=R.build(z);met,v,cl=R.evaluate(out);folder=H/'preflight_smoke'/arm
  R.save(folder/'paths.npz',out);R.write(str((folder/'events.json').relative_to(H)),ev)
  mm,vv,cc=R.audit(A.npz(folder/'paths.npz'),A.read(folder/'events.json'))
  err=A.compare(met,mm);assert err<=1e-12 and np.array_equal(v,vv) and np.array_equal(cl,cc);errors.append(err)
  # Policy does not inspect posthoc GT; recompute with identical paths and changed GT.
  zz=dict(z);zz['gt']=zz['gt']+100.;other,_=R.build(zz);assert np.array_equal(other['policy20'],out['policy20'])
 for i in range(8):
  arm=R.ARMS[i%2];z=R.sourcejob(arm,42,'outer');z={k:v[i*128:(i+1)*128] for k,v in z.items()};t=time.perf_counter()
  out,ev=R.build(z);met,v,cl=R.evaluate(out);mm,vv,cc=R.audit(out,ev);assert A.compare(met,mm)<=1e-12
  timings.append(dict(n=len(z['sample_id']),seconds=time.perf_counter()-t))
 # Real completed-job skip and intentional corruption rejected, isolated fixture.
 folder=H/'preflight_smoke';R.write('preflight_smoke/witness.json',dict(value=1))
 ff={str(folder/'witness.json'):A.sha(folder/'witness.json')};R.write('preflight_smoke/done.json',dict(source_sha=A.digest(files),files=ff));R.skip(folder/'done.json',files)
 for case in ('source','output'):
  bad=dict(files)
  if case=='source':bad['intentional_bad_source']='0'
  else:R.write('preflight_smoke/witness.json',dict(value=2))
  try:R.skip(folder/'done.json',bad)
  except AssertionError:pass
  else:raise AssertionError('BAD_'+case+'_NOT_REJECTED')
 R.write('preflight_smoke/witness.json',dict(value=1))
 A.verify(files);A.guard()
 seconds_per_row=sum(x['seconds'] for x in timings)/sum(x['n'] for x in timings)
 nominal=seconds_per_row*allrows
 assert nominal/60<30,'STOP_LONG_TASK_BUDGET'
 R.write('source_manifest.json',dict(files=files,inherited_count=inherited,old_digests_extended_not_overwritten=True))
 R.write('PREFLIGHT.json',dict(status='PASS',fixtures=tests,actual_all18_ID_native_group_slot_manifest_SHA_PASS=True,actual_two128_saved_full_quality_complete_link_motion_policy_max_error=max(errors),GT_independence_actual=True,completed_analysis_SHA_skip_bad_source_output_reject=True,no_torch_forward_optimizer=True,source_count=len(files),inherited_count=inherited,rows_across_source_jobs=allrows,benchmark=timings,nominal_double_pipeline_seconds=nominal,benchmark_not_production_guarantee=True,elapsed_seconds=time.perf_counter()-start,resource=A.resource))
 R.write('prelaunch_review/REVIEW.json',dict(status='PASS',source_count=len(files),all_current_scientific_sources_locked=True,preflight_actual_CPU_not_training=True,DA1='Bounded integration question; geometric certificate is not strict GT protection; perwindow quantile is not pooled guarantee; adaptive exploration; all old major gates retained.',not_external_review=True))
 print('PREFLIGHT_PASS',len(files),nominal,A.resource,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  R.write('failures/preflight_'+str(time.time_ns())+'.json',dict(status='STOP',traceback=traceback.format_exc(),resource=A.resource));raise
