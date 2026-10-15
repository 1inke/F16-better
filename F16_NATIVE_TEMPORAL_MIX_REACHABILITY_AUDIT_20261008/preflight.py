"""Resource-first CPU fixture, actual saved-data pipeline, SHA and throughput audit."""
import sys,time,traceback,json,inspect
from pathlib import Path
import run as q
import numpy as np
def main():
    started=time.perf_counter();q.guard();initial,count=q.locks();q.verify(initial);identities=q.identity_check();fixture={}
    t=np.arange(1,13)/12.;p=np.stack((t+.03*np.sin(3*t),.12*t**3),-1)[None,None];r=np.stack((.02*t*t,-.03*t**3),-1)[None,None]
    out,alpha=q.RAY['project_numpy'](p,r);ind=q.roots_independent(p,r);assert np.max(abs(alpha-ind))<=2e-12;lo=np.zeros_like(alpha);hi=np.ones_like(alpha)
    for _ in range(70):
        mid=(lo+hi)/2;ok=np.logical_and.reduce([np.all(np.sum((b+mid[...,None,None]*d)**2,-1)<=1.01**2*np.sum(b*b,-1),-1) for b,d in zip(q.derivatives(p),q.derivatives(r))]);lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid)
    assert abs(alpha.item()-lo.item()*(1-2.**-20))<1e-12
    z=np.zeros_like(p);assert np.array_equal(q.RAY['project_numpy'](z,r)[0],z);assert np.array_equal(q.RAY['project_numpy'](p,z)[0],z)
    try:q.RAY['audit_numpy'](z,z+np.broadcast_to(np.array([1e-7,0.]),z.shape));raise RuntimeError('Substantive defect accepted')
    except AssertionError:pass
    constant=np.stack((t,t*.3),-1)[None,None];assert q.shape_diagnostic(constant,.05*constant).max()<1e-12
    assert list(inspect.signature(q.generation).parameters)==['native','slots','partner']
    fixture=dict(identity=True,zero_budget=True,curved_active=True,independent_bisection_error=abs(alpha.item()-lo.item()*(1-2.**-20)),constant_shape_fit=True,substantive_defect_reject=True,GT_free_generation_contract=True)
    full=q.inputs('outer');z={k:x[:128] for k,x in full.items()};g=q.generation(z['native20'],z['slots'],7);p=q.whole(z['native20'],z['slots'],g['refined4']);q.physical_review(z,7,g);met,extra=q.metric(z,p);im,ix=q.metric(z,p,True);err=q.compare(met,im);assert err<=1e-12 and all(np.array_equal(extra[k],ix[k]) for k in extra)
    d=q.H/'preflight_smoke';d.mkdir(exist_ok=True);path=d/'paths.npz';assert not path.exists();np.savez_compressed(path,**g,**extra,sample_id=z['sample_id']);reload=q.npz(path);assert all(np.array_equal(reload[k],v) for k,v in g.items());pp=q.whole(z['native20'],z['slots'],reload['refined4']);qm,_=q.metric(z,pp,True);assert q.compare(met,qm)<=1e-12
    smoke=dict(full_original_SciPy_K20_complete_link=True,full_saved_samepath_quality_clusters_motion_max_error=err,allgroups_motion=[(k,v['all5_literal_PASS']) for k,v in met.items()],actual_rows=128,no_model_forward_or_gradient=True)
    # Exact completed-SHA skip is analysis recovery only. Deliberate source/output errors.
    q.write('preflight_smoke/done.json',dict(status='PASS',source_sha='fixture',files={str(path):q.sha(path)}));q.done_valid(d,'fixture')
    try:q.done_valid(d,'wrong-source');raise RuntimeError('Bad source accepted')
    except AssertionError:pass
    q.write('preflight_bad_output/done.json',dict(status='PASS',source_sha='fixture',files={str(path):'0'*64}))
    try:q.done_valid(q.H/'preflight_bad_output','fixture');raise RuntimeError('Bad output accepted')
    except AssertionError:pass
    # Eight fixed128 complete pipelines; result numbers are discarded, not used for policy choice.
    samples=[]
    for i in range(8):
        q.guard();start=time.perf_counter();x=q.generation(z['native20'],z['slots'],i);p=q.whole(z['native20'],z['slots'],x['refined4']);q.metric(z,p);q.metric(z,p,True);q.physical_review(z,i,x);samples.append(time.perf_counter()-start)
    q.verify(initial);files,n=q.locks();assert files==initial;assert n==count;q.write('source_manifest.json',dict(files=files,inherited_related_count=count,not_full_historical_digest=True));elapsed=time.perf_counter()-started
    N=1035+1856+2879;compute=float(np.mean(samples))*20*N/128;planning=max(3.,(compute+elapsed+60)/60);q.write('PREFLIGHT.json',dict(status='PASS',source_sha=q.digest(files),source_count=len(files),inherited_count=count,identity_audit=identities,fixture=fixture,actual_saved_pipeline=smoke,analysis_completedSHA_skip_badsource_badoutput_rejected=True,not_optimizer_RNG_resume=True,benchmark_8_saved128_seconds=samples,benchmark_mean=float(np.mean(samples)),nominal_compute_seconds=compute,initial_finalSHA_fixture_IO_elapsed_seconds=elapsed,planning_minutes=planning,estimated_elapsed_minutes=[max(2.,planning*.7),planning*1.7],not_production_guarantee=True,no_benchmark_model_or_outcome_selection=True,uncompressed_new_paths_alpha_metrics_estimate_GiB=20*N*(4*12*2*8+4*2*8+20+4*8)/2**30,resource=q.resource))
    q.write('prelaunch_review/REVIEW.json',dict(status='PASS',source_SHA=q.digest(files),all_current_scientific_code_locked=True,DA1='PASS_WITH_LIMITATIONS_GTfree_fixedbank_not_GToracle_equalK20_gain',no_training_model_or_optimizer=True,not_external_review=True));print('PREFLIGHT_PASS',len(files),compute,planning,q.resource)
if __name__=='__main__':
    try:main()
    except BaseException:
        q.write('failures/preflight1/FAILURE.json',dict(status='FAILURE',traceback=traceback.format_exc(),resource=q.resource,not_production_failure=True));q.write('runtime_state.json',dict(status='STOP_PREFLIGHT_FAILURE_WAIT_REVIEW',new_worker_started=False,formal_count_increment=0));raise
