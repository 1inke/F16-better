"""CPU saved-only completion check; no model/torch/optimizer."""
import sys,os
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util,time,gc
import numpy as np
BASE=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('eqk20_saved_review',BASE/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);r.H=OUT
start=time.perf_counter();r.guard();result=r.read(BASE/'RESULT.json');ci=r.read(BASE/'BOOTSTRAP.json');audit=r.read(BASE/'ARITHMETIC_AUDIT.json');preserve=r.read(BASE/'PRESERVATION.json')
assert result['status']=='PASS_DIAGNOSTIC_ONLY_CANONICAL_EQUAL_K20' and result['jobs']==27
assert audit['status']==preserve['status']=='PASS' and audit['independent_max_error']==0 and audit['scalar_CIs']==288
source=r.read(BASE/'source_manifest.json')['files'];assert len(source)==470 and r.digest(source)==result['source_sha'];r.verify(source);r.verify(result['output_sha'])
locked={str(BASE/n):r.sha(BASE/n) for n in ('RESULT.json','BOOTSTRAP.json','ARITHMETIC_AUDIT.json','PRESERVATION.json','RESOURCE_USAGE.json')}
data=r.cache();rows={};errors=[];models={};checked=[]
for rec in result['records']:
    tag,seed,name=rec['grouping'],rec['seed'],rec['split'];d=r.P/'runs'/tag/f'seed_{seed}';path=d/f'{name}.npz';job=BASE/'jobs'/f'{tag}_{seed}_{name}.json';saved=r.npz(job.with_suffix('.npz'));z=r.npz(path)
    scores=r.identity(z,data,d,name);met,v=r.evaluation(z,scores,True);err=r.compare({'metrics':rec['metrics'],'decomposition':rec['decomposition']},met);assert err<=1e-12
    assert np.array_equal(saved['sample_id'],z['sample_id']) and all(np.array_equal(v[k],saved[k]) for k in v)
    assert r.sha(path)==rec['input_sha'] and r.sha(d/'checkpoint.pt')==rec['checkpoint_sha']
    man=r.read(d/'run_manifest.json');done=r.read(d/'done.json');assert done['status']=='PASS' and done['run_manifest_sha']==r.digest(man) and man['seed']==seed and man['grouping']==tag and man['epochs']==25;r.verify(done['files'])
    models[str(d/'checkpoint.pt')]=rec['checkpoint_sha'];rows.setdefault((tag,name),[]).append((seed,{k:saved[k] for k in ('sample_id','groups','track','vectors')}));errors.append(err);checked.append(dict(grouping=tag,seed=seed,split=name,rows=len(z['sample_id']),full_independent_quality_clusters_motion_decomposition_max_error=err))
    print('SAVED_REVIEW',len(checked),tag,seed,name,flush=True);del z,saved,v;gc.collect()
assert len(models)==9
newci=r.bootstrap(rows);cierr=r.compare(newci,ci['records']);assert cierr==0 and len(newci)==72
fail=[];passes=0
for rec in result['records']:
    for group,m in rec['metrics']['replacement20'].items():
        if not m['n']:continue
        base=m['native20_motion'];bad={k:100*(value/base[k]-1) for k,value in m['motion'].items() if value>base[k]*1.01}
        assert m['motion_all5_1pct']==(not bad)
        if bad:fail.append(dict(grouping=rec['grouping'],seed=rec['seed'],split=rec['split'],group=group,harm_percent=bad))
        else:passes+=1
aggregate=[]
for tag in ('scene_family','target_track','recording_time'):
    for split in ('outer','inner','original_CAL'):
        rr=[x for x in result['records'] if x['grouping']==tag and x['split']==split];assert len(rr)==3
        for group in r.GROUPS:
            native=rr[0]['metrics']['native20'][group];n=native['n']
            if not n:continue
            treatment=[x['metrics']['replacement20'][group] for x in rr];c=next(x for x in ci['records'] if x['grouping']==tag and x['split']==split and x['group']==group and x['port']=='replacement20')
            aggregate.append(dict(grouping=tag,split=split,group=group,n=n,native_good=native['good'],replacement_good=[x['good'] for x in treatment],rescued=[x['rescued'] for x in treatment],lost=[x['lost'] for x in treatment],native_coverage=100*native['coverage'],replacement_mean_coverage=100*np.mean([x['coverage'] for x in treatment]),coverage_pp=100*c['estimate'][0],coverage_CI95_pp=(100*np.asarray(c['CI95'][0])).tolist(),ADE_gain_percent=100*(1-np.mean([x['minADE'] for x in treatment])/native['minADE']),FDE_gain_percent=100*(1-np.mean([x['minFDE'] for x in treatment])/native['minFDE']),continuous_CI95=c['CI95'][1:],native_effective_good=native['effective_good'],replacement_effective_good=[x['effective_good'] for x in treatment],replacement_distinct=[x['effective_distinct'] for x in treatment],motion_all_seeds_PASS=all(x['motion_all5_1pct'] for x in treatment),single_slot_early_middle_late=[x['single_Jclosest_early_middle_late'] for x in treatment],decomposition=[x['decomposition'][group] for x in rr]))
r.verify(source);r.verify(result['output_sha']);r.verify(locked);r.guard()
r.write('REVIEW.json',dict(status='PASS_CPU_SAVED_ONLY',jobs=27,models_SHA=9,complete_quality_clusters_motion_decomposition_recompute=True,independent_max_error=max(errors),scalar_CIs=288,deterministic_CI_recompute_max_error=cierr,source470_before_after_PASS=True,all_inherited385_digests_unchanged=True,output_SHA_PASS=True,all_checkpoints_unchanged=True,not_all_historical_SHA=True,no_model_forward_backward_optimizer=True,not_external_review=True,resource=r.RESOURCE,elapsed_seconds=time.perf_counter()-start,records=checked))
r.write('SUMMARY.json',dict(status='PASS',aggregate=aggregate,motion=dict(total=passes+len(fail),PASS=passes,FAIL=len(fail),failures=fail),major_success=False))
print('COMPLETE',max(errors),cierr,passes,len(fail),r.RESOURCE,flush=True)
