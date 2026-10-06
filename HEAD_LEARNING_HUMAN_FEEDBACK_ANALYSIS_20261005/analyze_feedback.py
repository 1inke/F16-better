"""Frozen saved-path audit of real human feedback. No torch/model/optimizer."""
from pathlib import Path
import sys, os, json, hashlib, csv, time, ctypes
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import psutil
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VIEW = ROOT/'HEAD_LEARNING_VISUAL_REVIEW_20261005'
TRIAL = ROOT/'F16_HEAD_LEARNING_DIAGNOSIS_20261005'
USER = Path('C:/Users/linke/Downloads/human_head_learning_observations_20261005.json')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def serial(v): return v.tolist() if isinstance(v,np.ndarray) else v.item()
def save(name,obj):
    p = HERE/name
    assert HERE in p.parents
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=serial),encoding='utf-8')

class Mem(ctypes.Structure):
    _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]

def guard():
    m=Mem(); m.length=ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    p=psutil.Process().memory_info()
    assert max(p.rss,p.private)<6*2**30 and min(m.avail_phys,m.avail_page)>=6*2**30, 'STOP: resource reserve'
    return dict(peak_RSS_GiB=p.peak_wset/2**30,peak_private_GiB=p.peak_pagefile/2**30,physical_available_GiB=m.avail_phys/2**30,commit_available_GiB=m.avail_page/2**30)

def flag(old,new):
    return np.where(~old&new,'rescued',np.where(old&~new,'lost',np.where(old&new,'retained','missing')))

def main():
    assert not (HERE/'AUDIT.json').exists(), 'Completed reports immutable; use a new folder'
    start=time.perf_counter(); guard()
    watched=[USER, VIEW/'data/exact_arrays.npz', VIEW/'data/common.js', VIEW/'audit.json',
             VIEW/'README.md', VIEW/'build_review.py', TRIAL/'RESULT.json',
             TRIAL/'RESULT_ANALYSIS.md', TRIAL/'completion_review/REVIEW.json',
             ROOT/'GENERATION_PULSE_RESPONSE_SERIES_20261004/run_head.py', HERE/'analyze_feedback.py', HERE/'METHOD.md']
    watched += [TRIAL/f'runs/{arm}/seed{s}/{pop}_epoch120.npz' for arm in ('zero','mean_std') for s in (7,8,9) for pop in ('FIT','CAL')]
    before={str(p):sha(p) for p in watched}
    assert read(VIEW/'audit.json')['status']=='PASS_FIT_CAL_SAVED_OUTPUT_VISUAL_REVIEW'
    assert read(TRIAL/'completion_review/REVIEW.json')['status']=='PASS'
    feedback=read(USER)
    assert feedback['study']=='head_learning_20261005'
    assert feedback['source_sha']==sha(TRIAL/'RESULT.json')
    txt=(VIEW/'data/common.js').read_text(encoding='utf-8')
    meta=json.loads(txt.split('=',1)[1].rstrip(';\n'))
    assert feedback['source_sha']==meta['source_sha']
    with np.load(VIEW/'data/exact_arrays.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    ids=data['sample_id']; pop=data['population']; n=len(ids)
    assert n==len(np.unique(ids))==649
    assert np.count_nonzero(pop=='FIT')==256 and np.count_nonzero(pop=='CAL')==393
    cases={c['id']:c for c in meta['cases']}; annotations=feedback['annotations']
    assert len({a['sample_id'] for a in annotations})==len(annotations)
    index={int(v):i for i,v in enumerate(ids)}
    selected=data['logits'].argmax(-1)
    keys=['native']+[f'{arm}{s}' for s in (7,8,9) for arm in ('fixed','learned')]
    rowix=np.arange(n); M={}; rows=[]; validations=[]
    # This is descriptive, not a new pass/fail definition of conservatism.
    steps=np.arange(1,13,dtype=np.float64)
    past=data['past'].astype(np.float64)
    cv_last=past[:,-1,None]+steps[None,:,None]*(past[:,-1]-past[:,-2])[:,None]
    x=np.arange(4,dtype=np.float64); xc=x-x.mean()
    slope=(past[:,-4:]*xc[None,:,None]).sum(1)/np.square(xc).sum()
    cv_ols=past[:,-1,None]+steps[None,:,None]*slope[:,None]
    for key in keys:
        path=data[key].astype(np.float64)
        assert path.shape==(n,20,12,2) and np.isfinite(path).all()
        error=np.linalg.norm(path-data['future'][:,None].astype(np.float64),axis=-1)
        ade=error.mean(-1); fde=error[:,:,-1]
        good=(ade<=.2)&(fde<=.4)
        closest=np.maximum(ade/.2,fde/.4).argmin(-1)
        e=error[rowix,closest]; chosen=error[rowix,selected]
        values=dict(ade=ade,fde=fde,good=good,coverage=good.any(-1),closest=closest,error=error,
                    closest_error=e,selected_error=chosen,selected_good=good[rowix,selected])
        for cvname,cv in [('last',cv_last),('ols4',cv_ols)]:
            distance=np.linalg.norm(path-cv[:,None],axis=-1).mean(-1)
            pick=distance[rowix,selected]
            # Ties occupy the same minimum rank; no claim of independent calibrated probabilities.
            values['cv_'+cvname+'_rank']=1+(distance<pick[:,None]-1e-12).sum(-1)
            values['cv_'+cvname+'_distance']=pick
        M[key]=values
        for i,id_ in enumerate(ids):
            assert np.max(abs(ade[i]-np.asarray(cases[int(id_)]['ade'][key])))<1e-12
            assert np.max(abs(fde[i]-np.asarray(cases[int(id_)]['fde'][key])))<1e-12
    for a in annotations:
        i=index[a['sample_id']]; s=a['seed']; key=f'learned{s}'
        assert a['source_sha']==feedback['source_sha'] and a['reference']=='native' and a['thresholds']==[.2,.4]
        actual=str(flag(M['native']['coverage'],M[key]['coverage'])[i])
        assert a['classification']==actual
        validations.append(dict(sample_id=a['sample_id'],population=str(pop[i]),seed=s,classification=actual,
                                path_mode=a['path_mode'],highlighted_slot_1based=a['highlighted_slot_1based'],note=a['note']))
    for s in (7,8,9):
        key=f'learned{s}'; changes=flag(M['native']['coverage'],M[key]['coverage'])
        for i,id_ in enumerate(ids):
            v=M[key]; old=M['native']; e=v['closest_error'][i]; oe=old['closest_error'][i]
            slot=v['closest'][i]; oslot=old['closest'][i]; se=v['selected_error'][i]
            # Native-best slot held fixed is a separate test from best-to-best comparison.
            held=v['error'][i,oslot]
            r=dict(sample_id=int(id_),population=str(pop[i]),seed=s,classification=str(changes[i]),
                   annotated=any(a['sample_id']==id_ and a['seed']==s for a in annotations),
                   native_closest_slot_1based=int(oslot)+1,learned_closest_slot_1based=int(slot)+1,
                   inherited_selected_slot_1based=int(selected[i])+1,
                   native_good=bool(old['coverage'][i]),learned_good=bool(v['coverage'][i]),
                   native_selected_good=bool(old['selected_good'][i]),learned_selected_good=bool(v['selected_good'][i]),
                   native_ADE=float(oe.mean()),learned_ADE=float(e.mean()),native_FDE=float(oe[-1]),learned_FDE=float(e[-1]),
                   native_first_half=float(oe[:6].mean()),learned_first_half=float(e[:6].mean()),
                   native_second_half=float(oe[6:].mean()),learned_second_half=float(e[6:].mean()),
                   native_early4=float(oe[:4].mean()),learned_early4=float(e[:4].mean()),
                   native_middle4=float(oe[4:8].mean()),learned_middle4=float(e[4:8].mean()),
                   native_late4=float(oe[8:].mean()),learned_late4=float(e[8:].mean()),
                   held_slot_first_half=float(held[:6].mean()),held_slot_second_half=float(held[6:].mean()),
                   selected_ADE=float(se.mean()),selected_FDE=float(se[-1]),
                   selected_first_half=float(se[:6].mean()),selected_second_half=float(se[6:].mean()),
                   early_improves_late_worsens=bool(e[:6].mean()<oe[:6].mean() and e[6:].mean()>oe[6:].mean()),
                   held_early_improves_late_worsens=bool(held[:6].mean()<oe[:6].mean() and held[6:].mean()>oe[6:].mean()),
                   late_larger_than_early=bool(e[6:].mean()>e[:6].mean()),
                   endpoint_worse=bool(e[-1]>oe[-1]),
                   good_available_selection_miss=bool(v['coverage'][i] and not v['selected_good'][i]),
                   minADE_slot_1based=int(v['ade'][i].argmin())+1,minFDE_slot_1based=int(v['fde'][i].argmin())+1,
                   independent_mins_pass_no_samepath=bool(v['ade'][i].min()<=.2 and v['fde'][i].min()<=.4 and not v['coverage'][i]),
                   native_cv_last_rank=int(old['cv_last_rank'][i]),learned_cv_last_rank=int(v['cv_last_rank'][i]),
                   native_cv_ols4_rank=int(old['cv_ols4_rank'][i]),learned_cv_ols4_rank=int(v['cv_ols4_rank'][i]),
                   native_selected_CV_distance=float(old['cv_last_distance'][i]),learned_selected_CV_distance=float(v['cv_last_distance'][i]))
            rows.append(r)
    aggregates=[]
    for popname in ('FIT','CAL'):
        for s in (7,8,9):
            for group in ('all','rescued','lost','retained','missing','annotated'):
                rr=[r for r in rows if r['population']==popname and r['seed']==s and
                    (group=='all' or (group=='annotated' and r['annotated']) or r['classification']==group)]
                if not rr: continue
                nrow=len(rr)
                result=dict(population=popname,seed=s,group=group,n=nrow)
                for f in ('native_good','learned_good','native_selected_good','learned_selected_good','early_improves_late_worsens',
                          'held_early_improves_late_worsens','late_larger_than_early','endpoint_worse',
                          'good_available_selection_miss','independent_mins_pass_no_samepath'):
                    result[f]=sum(r[f] for r in rr)
                for f in ('native_first_half','learned_first_half','native_second_half','learned_second_half',
                          'native_FDE','learned_FDE','selected_ADE','selected_FDE'):
                    result[f+'_mean']=float(np.mean([r[f] for r in rr]))
                for arm in ('native','learned'):
                    for cv in ('last','ols4'):
                        result[f'{arm}_cv_{cv}_top5_count']=sum(r[f'{arm}_cv_{cv}_rank']<=5 for r in rr)
                aggregates.append(result)
    annotated_rows=[r for r in rows if r['annotated']]
    assert len(annotated_rows)==len(annotations)==38
    # Independent scalar arithmetic and candidate qualification for all uploaded cases.
    maxerr=0.
    for r in annotated_rows:
        i=index[r['sample_id']]; key=f"learned{r['seed']}"; k=r['learned_closest_slot_1based']-1
        errors=[float(np.hypot(float(data[key][i,k,t,0])-float(data['future'][i,t,0]),
                               float(data[key][i,k,t,1])-float(data['future'][i,t,1]))) for t in range(12)]
        maxerr=max(maxerr,abs(sum(errors)/12-r['learned_ADE']),abs(errors[-1]-r['learned_FDE']))
    assert maxerr<1e-12
    expected={7:(167,28,30),8:(161,32,40),9:(165,29,33)}
    for s,(total,rescue,lost) in expected.items():
        a=next(x for x in aggregates if x['population']=='CAL' and x['seed']==s and x['group']=='all')
        assert a['learned_good']==total
        assert len([r for r in rows if r['population']=='CAL' and r['seed']==s and r['classification']=='rescued'])==rescue
        assert len([r for r in rows if r['population']=='CAL' and r['seed']==s and r['classification']=='lost'])==lost
    after={str(p):sha(p) for p in watched}; assert before==after
    for name,rr in [('case_metrics.csv',rows),('annotation_case_metrics.csv',annotated_rows),('aggregates.csv',aggregates)]:
        with (HERE/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    save('annotations_verified.json',validations)
    save('RESULT.json',dict(status='PASS_SAVED_FEEDBACK_DESCRIPTIVE_ONLY',annotations=len(annotations),
                           populations=dict(FIT=256,CAL=393),aggregates=aggregates,annotated_cases=annotated_rows,
                           no_model_forward=True,no_optimizer=True,no_rescore=True))
    save('AUDIT.json',dict(status='PASS',direct_files_SHA_unchanged=len(before),input_hashes=before,
                          independent_scalar_annotated_metric_max_error=maxerr,resource=guard(),
                          seconds=time.perf_counter()-start,scope='Direct saved inputs only; not full historical manifest/model replay/cluster recomputation'))
    print(json.dumps(dict(annotations=38,audit='PASS',summary=[a for a in aggregates if a['population']=='CAL' and a['group'] in ('all','annotated','rescued','lost')]),ensure_ascii=False))

if __name__=='__main__':main()
