"""Read-only saved-output validation and descriptive summaries; no torch/model/optimizer."""
from pathlib import Path
import json, hashlib, os, time
from datetime import datetime
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import psutil
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;ROOT=PARENT.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def verify(files):
    for p,h in files.items():assert sha(p)==h,('SHA mismatch',p)
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (keys or z.files)}
def dump(p,v):
    assert HERE in p.parents
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def masks(g,reg):
    clear=np.logical_and(g['GT_event'].astype(bool),~g['GT_uncertain'].astype(bool))
    m={'all':np.ones(len(reg),bool),'clear':clear,
       'specific_first_compatible':clear&g['cue_first_compatible'].astype(bool),
       'specific_coarse_path_failure':clear&g['cue_first_compatible'].astype(bool)&(g['class_0']==7)}
    m.update({s:reg==i for i,s in enumerate(('stable','transition','change'))})
    m.update({'scene_'+str(s):g['scene_family']==s for s in np.unique(g['scene_family'])})
    return m
def main():
    start=time.perf_counter();pre=read(PARENT/'preflight.json');result=read(PARENT/'RESULT.json')
    assert result['status']=='PASS_DIAGNOSTIC_ONLY_ORIGINAL_OBJECTIVE'
    assert read(PARENT/'exit_status.json')['status']==read(PARENT/'independent_worker_state.json')['status']=='SUCCESS'
    assert read(PARENT/'ARITHMETIC_AUDIT.json')['status']=='PASS'
    assert read(PARENT/'RESOURCE_USAGE.json')['status']=='PASS'
    old=read(PARENT/'old_results_manifest.json')['files'];locks={**old,**pre['source_hashes']}
    direct=[PARENT/p for p in ('RESULT.json','ARITHMETIC_AUDIT.json','RESOURCE_USAGE.json','preflight.json','FIT_objective.npz','CAL_objective.npz','independent_worker_state.json','exit_status.json')]
    direct.append(Path(__file__))
    nativepath=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
    groupdir=ROOT/'F16_CUE_RESPONSE_CONCORDANCE_20261004/recovery_memory_20261004/jobs'
    direct.extend([nativepath,groupdir/'FIT_native.npz',groupdir/'CAL_native.npz'])
    extra={str(p):sha(p) for p in direct};verify(locks);verify(extra)
    native=load(nativepath,['sample_id','change_group']);assert len(np.unique(native['sample_id']))==28791
    methods=('native','learned1','learned2','learned3','pulse_full','single_witness')
    summary={};maximum=0.;outputs_n=0;blocks_n=0
    for pop,a,c in [('FIT',0,25912),('CAL',25912,28791)]:
        values=load(PARENT/(pop+'_objective.npz'));assert np.array_equal(values['sample_id'],native['sample_id'][a:c])
        assert len(values['sample_id'])==c-a
        parts=[]
        for b in range(0,c-a,64):
            stop=min(b+64,c-a);stem=PARENT/'blocks'/f'{pop}_{b:05d}_{stop:05d}'
            done=read(stem.with_suffix('.done.json'))
            assert done['status']=='SUCCESS' and done['source_hashes']==pre['source_hashes']
            assert len(done['outputs'])==2;verify(done['outputs']);outputs_n+=2;blocks_n+=1
            audit=read(stem.with_suffix('.audit.json'));assert audit['status']=='PASS' and audit['max_error']<1e-9
            z=load(stem.with_suffix('.npz'));assert np.array_equal(z['sample_id'],native['sample_id'][a+b:a+stop]);parts.append(z)
        for k,v in values.items():
            joined=np.concatenate([z[k] for z in parts]);assert np.array_equal(joined,v),(pop,k,'concatenation')
            assert np.isfinite(v).all()
        del parts
        groups=load(groupdir/(pop+'_native.npz'),['sample_id','scene_family','GT_event','GT_uncertain','cue_first_compatible','class_0'])
        assert np.array_equal(groups['sample_id'],values['sample_id'])
        summary[pop]={}
        for group,mask in masks(groups,native['change_group'][a:c]).items():
            prior=result['all_summaries'][pop][group];n=int(mask.sum());assert n==prior['n']
            viable=mask&(values['witness_slot']>=0);assert int(viable.sum())==prior['witness_available']
            block={'n':n,'witness_available':int(viable.sum()),'methods':{},'directional_counts':{}}
            for method in methods:
                v={k.split('__',1)[1]:x for k,x in values.items() if k.startswith(method+'__')}
                stats={k:float(x[mask].mean()) for k,x in v.items() if k!='winner_slot'} if n else {}
                if n:
                    for k,x in stats.items():
                        err=abs(x-prior['methods'][method][k]);maximum=max(maximum,err);assert err<1e-12,(pop,group,method,k,err)
                    base_total=values['native__total'];base_q=values['native__quality']
                    expected=dict(coverage_count=int(v['coverage'][mask].sum()),
                        objective_lower_than_native=int((v['total'][mask]<base_total[mask]-1e-12).sum()),
                        quality_lower_total_not_lower=int(((v['quality'][mask]<base_q[mask]-1e-12)&(v['total'][mask]>=base_total[mask]-1e-12)).sum()))
                    for k,x in expected.items():assert x==prior['methods'][method][k]
                    cov=expected['coverage_count'];bad=int(v['good_present_winner_bad'][mask].sum())
                    stats.update(expected);stats.update(winner_bad_count=bad,winner_bad_percent_of_covered=100*bad/cov if cov else None,
                        quality_lower_count=int((v['quality'][mask]<base_q[mask]-1e-12).sum()),
                        single_witness_total_lower_on_available=int((viable&(v['total']<base_total-1e-12)).sum()) if method=='single_witness' else None,
                        coverage_without_witness=int((mask&v['coverage'].astype(bool)&~viable).sum()))
                block['methods'][method]=stats
            for label,subset in [('all_available',viable),('native_no_good_available',viable&~values['native__coverage'].astype(bool))]:
                grad=values['native__derivative_toward_single_quality'][subset]
                mismatch=(values['witness_slot'][subset]!=values['native__winner_slot'][subset])
                assert np.all(grad[mismatch]==0),'nonwinner coordinate receives no hard-quality gradient'
                single_total=values['single_witness__total'][subset];base_total=values['native__total'][subset]
                block['directional_counts'][label]=dict(n=int(subset.sum()),
                    replacement_slot_not_native_winner=int(mismatch.sum()),
                    native_quality_direction_exact_zero=int((grad==0).sum()),
                    native_quality_direction_negative=int((grad<0).sum()),
                    native_quality_direction_positive=int((grad>0).sum()),
                    single_witness_total_lower=int((single_total<base_total-1e-12).sum()))
            summary[pop][group]=block
    assert blocks_n==450 and outputs_n==900
    verify(locks);verify(extra)
    report=dict(status='PASS',no_torch=True,no_model_forward=True,no_optimizer=True,blocks=blocks_n,outputs_SHA=outputs_n,
        old_files=len(old),source_files=len(pre['source_hashes']),full_old_and_sources_before_after=True,
        direct_inputs_and_review_source_before_after_SHA=extra,all_saved_arrays_concatenation_exact=True,all_group_aggregates_independently_recomputed=True,
        saved_metrics_arithmetic_evidence='Completed worker numpy per-block checks; supplement verifies their flags, SHA, all saved-array concatenation and all group aggregates, not fresh model/path/autograd replay.',
        independent_summary_max_error=maximum,groups=summary,elapsed_seconds=time.perf_counter()-start,
        peak_private_GiB=psutil.Process().memory_info().peak_pagefile/2**30,
        at=datetime.now().astimezone().isoformat())
    dump(HERE/'REVIEW.json',report)
    lines=['# All fixed-population/group summaries','',
        'Saved coordinate-objective contributions; population denominators, not minibatch losses. Oracle rows are not deployable predictions.',
        '','| Population/group | N | Method | Strict coverage | Quality | Total | Winner bad / covered |',
        '|---|---:|---|---:|---:|---:|---:|']
    for pop,groups in summary.items():
        for group,g in groups.items():
            if not g['n']:continue
            for method,v in g['methods'].items():
                lines.append(f"| {pop}/{group} | {g['n']} | {method} | {v['coverage_count']} | {v['quality']:.9f} | {v['total']:.9f} | {v['winner_bad_count']} / {v['coverage_count']} |")
    (HERE/'GROUP_TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','direct_inputs_and_review_source_before_after_SHA')},ensure_ascii=False),flush=True)
    for pop,groups in summary.items():
        for group in ('all','specific_first_compatible','specific_coarse_path_failure'):
            g=groups[group]
            print(json.dumps(dict(population=pop,group=group,n=g['n'],directional_counts=g['directional_counts'],
                winner_bad={m:dict(count=v.get('winner_bad_count'),covered=v.get('coverage_count'),pct=v.get('winner_bad_percent_of_covered')) for m,v in g['methods'].items()})),flush=True)
if __name__=='__main__':main()
