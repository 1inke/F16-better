"""Actual saved128 runner/reader review and final immutable source binding."""
import common as c,prepare as prep,run,audit_saved as audit,decision,statistics_ci as ci
import numpy as np,ast,copy,time,traceback

def decision_fixtures():
 def metric(coverage):return dict(coverage=coverage,minADE=.1,minFDE=.2,good=int(coverage*100),rescued=10,lost=0,motion_failed=[])
 records=[]
 for arm,coverage in (('dynamic_winner',.4),('fixed_motion_responsibility',.5)):
  for seed in (42,43,44):records.append(dict(arm=arm,seed=seed,splits={s:{g:metric(coverage) for g in ('overall','stable','transition','change')} for s in ('FIT','outer','inner','original_CAL')}))
 P0={'outer':{'change':metric(.3)}};boot={'records':[]}
 for group in ('overall','stable','transition','change'):
  for ref in ('A_dynamic_winner','own_P0','native20'):
   boot['records'].append(dict(split='outer',group=group,reference=ref,CI_family48=[[.01,.2],[-.1,.1],[-.1,.1],[-.1,.1]],perseed_estimate=[[.1,-.1,-.1,0]]*3))
 positive=decision.gates(records,P0,boot);assert positive['S_T1'] and positive['Q'] and positive['T2_permitted']
 negative=copy.deepcopy(records);negative[3]['splits']['inner']['change']['motion_failed']=['accel_mean']
 gate=decision.gates(negative,P0,boot);assert gate['S_T1'] and not gate['Q'] and not gate['T2_permitted']
 late=copy.deepcopy(records);late[3]['splits']['outer']['change']['coverage']=.29
 assert not decision.gates(late,P0,boot)['S_T1']
 return dict(positive_allgates=True,one_literal_motion_failure_blocks=True,no_ownP0_gain_blocks=True,not_scientific_results=True)

def bootstrap_fixture():
 rng=np.random.Generator(np.random.PCG64(20261015));inv=np.array([0,0,1,2,2,2]);T=3;counts=np.bincount(inv)
 d=rng.normal(size=(3,6,5));a=ci.aggregate(d,inv,T,False);b=ci.aggregate(d,inv,T,True);assert np.array_equal(a,b)
 error=0.
 for _ in range(32):
  ss=rng.integers(0,3,3);tt=rng.integers(0,T,T);weight=np.bincount(tt,minlength=T)
  aggregate=np.einsum('t,stm->sm',weight,a,optimize=False)[ss].sum(0)/(3*np.sum(weight*counts))
  sample=np.concatenate([np.flatnonzero(inv==t) for t in tt]);direct=d[ss][:,sample].mean((0,1));error=max(error,float(np.max(abs(aggregate-direct))))
 assert error<1e-12
 return dict(crossed_wholetrack_seed_weighting_max_error=error,draws=32,not_scientific_bootstrap=True)

def main():
 start=time.perf_counter();c.guard();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS_NUMERIC_AND_BUDGET' and pre['planning_minutes']<=30
 snap=c.read(c.H/'preflight_source_snapshot.json');c.verify(snap['files'])
 for p in c.H.glob('*.py'):ast.parse(p.read_text(encoding='utf8'),filename=str(p))
 data=c.load();dictionary=c.npz(c.H/'dictionary.npz');prep.attach(data,dictionary);ix=data['split']['outer'][:128]
 certificates=[]
 for arm in ('dynamic_winner','fixed_motion_responsibility'):
  z=c.npz(c.H/f'preflight_smoke/{arm}.npz')
  met=run.job(data,ix,z['paths'],z['correction'],f'prelaunch_review/saved_smoke/{arm}.npz','prelaunch_fixture')
  path=c.H/f'prelaunch_review/saved_smoke/{arm}.npz';saved=c.npz(path);ind,aux=audit.final_metrics(saved);stored=c.npz(path.with_name(path.stem+'_metrics.npz'))
  c.eq(c.safe(ind),met,2e-11)
  for key,a in aux.items():
   if a.dtype.kind in 'bui':assert np.array_equal(a,stored[key])
   else:assert np.max(abs(a-stored[key]))<=2e-11
  # Actual completed-analysis reuse is allowed only when its input/output
  # source contract is compatible; no second calculation, no optimizer resume.
  again=run.job(data,ix,z['paths'],z['correction'],f'prelaunch_review/saved_smoke/{arm}.npz','prelaunch_fixture');assert again==met
  certificates.append(dict(arm=arm,actual128_sameID=True,full_runner_saved_math_independent_quality_clusters_motion_assignment_EXACT=True,compatible_done_SHA_skip=True))
 gates=decision_fixtures();stats=bootstrap_fixture();files,inherit=c.locks()
 # All provenance and immutable dictionary/preflight artifacts inherit their
 # actual digests. Never replace an inherited digest with the current value.
 names=['dictionary.npz','dictionary_trace.json','dictionary_source_manifest.json','dictionary_review/REVIEW.json','DICTIONARY_PREFLIGHT.json','statistics.npz','split_ids.npz','fixed_feature_fixture.npz','PREFLIGHT.json','BENCHMARK.json','preflight_source_snapshot.json']
 paths=[c.H/n for n in names]+[p for folder in ('failures','preflight_smoke','prelaunch_review/saved_smoke') for p in (c.H/folder).rglob('*') if p.is_file()]
 for p in paths:
  key=str(p);value=c.sha(p)
  if key in files:assert files[key]==value,('IMMUTABLE_DIGEST_CHANGED',key)
  else:files[key]=value
 c.verify(snap['files']);c.verify(files);c.guard();source=c.digest(files)
 c.write('source_manifest.json',dict(source_sha=source,files=files,inheritance=inherit))
 c.write('prelaunch_review/REVIEW.json',dict(status='PASS',at=c.now(),source_sha=source,source_count=len(files),actual128_saved_full_pipeline=certificates,decision_fixtures=gates,bootstrap_fixture=stats,full_numeric_GPU_resume_evidence='PREFLIGHT.json',planning_minutes=pre['planning_minutes'],old_sources_models_results_initial_final='PASS',same_assistant_not_external_review=True,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-start))
 c.write('LAUNCH_GATE.json',dict(status='PASS',at=c.now(),source_sha=source,planning_minutes=pre['planning_minutes'],registered_budget_minutes=30,scientific_sources_locked=True))
 c.write('runtime_state.json',dict(status='PRELAUNCH_ALL_GATES_PASS',at=c.now(),production_started=False,active_worker=False,source_sha=source,source_count=len(files),planning_minutes=pre['planning_minutes'],ETA_minutes=pre['ETA_minutes'],planned_runs=6,completed_runs=0,planned_epochs=150,planned_final_jobs=24,planned_shared_P0_jobs=4,first_progress_query_done=False,progress_query_count=0,startup_check_count=0,T2_started=False,T3_started=False,whole_series_complete=False))
 print(dict(status='FINAL_PRELAUNCH_PASS',source_count=len(files),source_sha=source,planning_minutes=pre['planning_minutes']),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  c.write('prelaunch_review/FAILURE.json',dict(status='STOP',at=c.now(),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
