"""Saved-output arithmetic only. No model, optimizer, new held-test arrays or worker query."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
D = ROOT / 'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003'
INPUTS = [D/'candidate_audit.npz', D/'features/dev.npz', D/'labels/dev.npz',
          D/'candidate_audit.json',
          ROOT/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/summary.json']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def load(path): return json.loads(path.read_text(encoding='utf-8-sig'))

def categories(a, f, at, ft, mask):
    n=int(mask.sum())
    labels={'both_pass':(a<=at)&(f<=ft),
            'ADE_only_fail':(a>at)&(f<=ft),
            'FDE_only_fail':(a<=at)&(f>ft),
            'both_fail':(a>at)&(f>ft)}
    failed=int((mask&~labels['both_pass']).sum())
    rows={}
    for name, flag in labels.items():
        m=mask&flag; count=int(m.sum())
        rows[name]=dict(n=count,pct_population=100*count/n,
            pct_among_failed=None if name=='both_pass' or not failed else 100*count/failed,
            mean_ADE=float(a[m].mean()) if count else None,
            mean_FDE=float(f[m].mean()) if count else None,
            contribution_ADE=float(a[m].sum()/n), contribution_FDE=float(f[m].sum()/n))
    assert sum(r['n'] for r in rows.values())==n
    assert abs(sum(r['contribution_ADE'] for r in rows.values())-a[mask].mean())<1e-6
    assert abs(sum(r['contribution_FDE'] for r in rows.values())-f[mask].mean())<1e-6
    return dict(n=n,failed_n=failed,mean_ADE=float(a[mask].mean()),mean_FDE=float(f[mask].mean()),categories=rows)

def main():
    assert not (OUT/'RESULT.json').exists(), 'Do not overwrite completed result'
    before={str(p):sha(p) for p in INPUTS}
    d=np.load(D/'candidate_audit.npz',allow_pickle=False)
    feat=np.load(D/'features/dev.npz',allow_pickle=False)
    lab=np.load(D/'labels/dev.npz',allow_pickle=False)
    ids=d['sample_id']; assert len(ids)==1516 and len(np.unique(ids))==1516
    assert np.array_equal(ids,feat['sample_id']) and np.array_equal(ids,lab['sample_id'])
    assert feat['paths'].shape==(1516,20,12,2) and np.isfinite(feat['paths']).all()
    error=np.linalg.norm(feat['paths']-lab['future'][:,None],axis=-1)
    a=error.mean(-1).astype(np.float64); f=error[:,:,-1].astype(np.float64)
    np.testing.assert_allclose(a,d['ade'],atol=2e-5,rtol=0)
    np.testing.assert_allclose(f,d['fde'],atol=2e-5,rtol=0)
    selected=feat['logits'].argmax(1); assert np.array_equal(selected,d['selected'])
    idx=np.arange(len(ids)); ia=a.argmin(1); iff=f.argmin(1)
    routes={'selected':(a[idx,selected],f[idx,selected]),
            'independent_minima':(a.min(1),f.min(1)),
            'minADE_same_trajectory':(a[idx,ia],f[idx,ia]),
            'minFDE_same_trajectory':(a[idx,iff],f[idx,iff])}
    masks={'all':np.ones(len(ids),bool),'clear_event':d['GT_event'].astype(bool),
           'no_clear_event':~d['GT_event'].astype(bool)&~d['GT_uncertain'].astype(bool),
           'boundary_uncertain':d['GT_uncertain'].astype(bool),'near_static':d['near_static'].astype(bool)}
    assert sum(int(masks[k].sum()) for k in ('clear_event','no_clear_event','boundary_uncertain'))==1516
    result={}
    for at,ft in ((.2,.4),(.3,.5),(.5,1.),(.2,.35)):
        key=f'{at:g}/{ft:g}'
        rows={name:{group:categories(aa,ff,at,ft,mask) for group,mask in masks.items()} for name,(aa,ff) in routes.items()}
        rows['samepath_coverage']={}
        good=((a<=at)&(f<=ft)).any(1)
        separate=(a.min(1)<=at)&(f.min(1)<=ft)
        for group,mask in masks.items():
            rows['samepath_coverage'][group]=dict(n=int(mask.sum()),samepath_good_n=int((mask&good).sum()),
                independent_both_pass_n=int((mask&separate).sum()),
                split_candidate_tradeoff_n=int((mask&separate&~good).sum()),
                no_samepath_good_n=int((mask&~good).sum()))
        result[key]=rows
    ref=load(D/'candidate_audit.json')['rows'][0]
    assert abs(routes['selected'][0].mean()-ref['mean_selected_ADE'])<1e-6
    assert abs(routes['independent_minima'][1].mean()-ref['mean_oracle_FDE'])<1e-6
    attribution={}
    for name,(aa,ff) in routes.items():
        attribution[name]={}
        for group in ('clear_event','no_clear_event','boundary_uncertain','near_static'):
            mask=masks[group]
            attribution[name][group]=dict(n=int(mask.sum()),weight=float(mask.mean()),
                mean_ADE=float(aa[mask].mean()),mean_FDE=float(ff[mask].mean()),
                contribution_ADE=float(aa[mask].sum()/len(ids)),contribution_FDE=float(ff[mask].sum()/len(ids)),
                share_total_ADE=float(aa[mask].sum()/aa.sum()),share_total_FDE=float(ff[mask].sum()/ff.sum()))
    saved=load(INPUTS[-1])['rows']; weights=np.array([r['test_samples'] for r in saved])
    metrics=('min_ade_20','min_fde_20','same_path_fde','top1_ade','top1_fde')
    macro={k:float(np.mean([r[k] for r in saved])) for k in metrics}
    micro={k:float(np.average([r[k] for r in saved],weights=weights)) for k in metrics}
    targets={}
    for at,ft in ((.20,.35),(.20,.33),(.18,.30),(.176,.308)):
        targets[f'{at:g}/{ft:g}']=dict(ADE_reduction_from_existing_macro_pct=100*(1-at/macro['min_ade_20']),
            FDE_reduction_from_existing_macro_pct=100*(1-ft/macro['min_fde_20']))
    assert before=={str(p):sha(p) for p in INPUTS}
    output=dict(status='PASS_SAVED_OUTPUT_ARITHMETIC_ONLY',no_training=True,no_new_heldtest_array_read=True,
        n_dev=1516,source_hashes_before_after_unchanged=before,thresholds=result,
        group_average_attribution=attribution,existing_five_fold_saved_summary=dict(rows=saved,macro=macro,micro=micro,
            parameters=saved[0]['parameters'],seed_count=1,total_sample_count=int(weights.sum()),
            caveat='Existing reported aggregates only; not a new held-test evaluation or independent confirmation.'),
        proposed_targets_not_adopted_or_preregistered=targets,
        limitations=['DEV counts are exploratory single-checkpoint descriptions, not five-fold proportions.',
            'Independent minima can be from different candidates.', 'Samepath thresholds are diagnostic, not benchmark SOTA rules.',
            'No posthoc bad-sample exclusion or formal-gate relaxation.', 'Old formal summaries are historical aggregate evidence, not new SOTA audit.'])
    (OUT/'RESULT.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for key in ('0.2/0.4','0.3/0.5'):
        print(key)
        for route in ('selected','independent_minima'):
            for group in ('all','clear_event'):
                z=result[key][route][group]
                print(route,group,z['n'],{k:(v['n'],round(v['pct_population'],2),None if v['pct_among_failed'] is None else round(v['pct_among_failed'],2)) for k,v in z['categories'].items()})
        print('samepath',result[key]['samepath_coverage']['all'])
    print('MACRO',macro,'MICRO',micro,'TARGETS',targets)
    print('ATTRIBUTION',attribution['independent_minima'])

if __name__=='__main__': main()
