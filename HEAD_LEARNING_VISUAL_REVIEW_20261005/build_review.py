"""Read-only final diagnostic outputs to offline case viewer; no torch/inference."""
from pathlib import Path
import os, sys, json, csv, hashlib, time, ctypes
sys.dont_write_bytecode = True
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import psutil
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
TRIAL=ROOT/'F16_HEAD_LEARNING_DIAGNOSIS_20261005'
TEMPLATE=ROOT/'PULSE_RESPONSE_VISUAL_REVIEW_20261004'
CACHE=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
ENC=ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'
GROUPS=ROOT/'F16_CUE_RESPONSE_CONCORDANCE_20261004/recovery_memory_20261004/jobs'
SEEDS=(7,8,9); KEYS=['native']+[f'{arm}{s}' for s in SEEDS for arm in ('fixed','learned')]

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (keys or z.files)}
def default(v):return v.tolist() if isinstance(v,np.ndarray) else v.item()
def dump(p,v):
    assert HERE in p.parents
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=default),encoding='utf-8')
def js(p,target,v):
    assert HERE in p.parents
    p.write_text(target+'='+json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False,default=default)+';\n',encoding='utf-8')
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    v=psutil.Process().memory_info();assert max(v.rss,v.private)<6*2**30 and min(m.avail_phys,m.avail_page)>=6*2**30, 'Resource reserve; STOP, do not relax'
    return dict(peak_RSS_GiB=v.peak_wset/2**30,peak_private_GiB=v.peak_pagefile/2**30,available_physical_GiB=m.avail_phys/2**30,available_commit_GiB=m.avail_page/2**30)
def metrics(p,gt):
    e=np.sqrt(np.sum((p.astype(np.float64)-gt[:,None].astype(np.float64))**2,axis=-1))
    return e.mean(-1),e[:,:,-1]
def flags(old,new):return np.where(~old&new,'rescued',np.where(old&~new,'lost',np.where(old&new,'retained','missing')))

def main():
    assert not (HERE/'audit.json').exists(),'Completed output immutable; choose new folder'
    begin=time.perf_counter();guard()
    assert read(TRIAL/'independent_worker_state.json')['status']==read(TRIAL/'exit_status.json')['status']=='SUCCESS'
    assert read(TRIAL/'completion_review/REVIEW.json')['status']=='PASS'
    watched=[CACHE,ENC,GROUPS/'FIT_native.npz',GROUPS/'CAL_native.npz']
    for folder in (TRIAL,TEMPLATE):
        watched += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.tmp','.pending')]
    before={str(p):sha(p) for p in watched}
    print('READ_ONLY_SHA',len(before),flush=True);guard()
    ids=load(TRIAL/'split_ids.npz');indices=load(TRIAL/'split_indices.npz')
    cache=load(CACHE,['sample_id','motion','logits','candidate','change_group'])
    enc=load(ENC,['sample_id','past'])
    assert np.array_equal(cache['sample_id'],enc['sample_id'])
    ix=np.concatenate([indices[p] for p in ('FIT','CAL')]);allids=np.concatenate([ids[p] for p in ('FIT','CAL')])
    assert len(allids)==len(np.unique(allids))==649 and np.array_equal(cache['sample_id'][ix],allids)
    assert np.max(abs(enc['past'][ix]-cache['motion'][ix,:16].reshape(-1,8,2)))<=2e-5
    past=enc['past'][ix].copy();scores=cache['logits'][ix].copy();formal=cache['change_group'][ix].copy()
    base=cache['candidate'][ix,:,:24].reshape(649,20,12,2).astype(np.float32)
    paths={k:[] for k in KEYS};effective={k:[] for k in KEYS};distinct={k:[] for k in KEYS}
    future=[];allmasks={};poplabels=[];events=[];scene=[];kinds=[];onsets=[];uncertain=[]
    expected=read(TRIAL/'completion_review/GROUP_ROWS.json');maxerror=0.
    for pop in ('FIT','CAL'):
        initial=load(TRIAL/f'runs/zero/seed7/{pop}_epoch0.npz')
        assert np.array_equal(initial['sample_id'],ids[pop]) and np.array_equal(initial['native'],initial['trajectory'])
        n=len(ids[pop]);paths['native'].append(initial['native']);future.append(initial['future'])
        effective['native'].append(initial['effective_good']);distinct['native'].append(initial['all_distinct'])
        masks={k[5:]:initial[k] for k in initial if k.startswith('mask_')}
        for k,v in masks.items():allmasks.setdefault(k,[]).append(v)
        group=load(GROUPS/f'{pop}_native.npz',['sample_id','GT_event','GT_uncertain','GT_kind','GT_onset','scene_family'])
        local=indices[pop]-(25912 if pop=='CAL' else 0)
        assert np.array_equal(group['sample_id'][local],ids[pop])
        for name,bag in [('GT_event',events),('GT_uncertain',uncertain),('GT_kind',kinds),('GT_onset',onsets),('scene_family',scene)]:bag.append(group[name][local])
        poplabels += [pop]*n
        for s in SEEDS:
            for arm,label in [('zero','fixed'),('mean_std','learned')]:
                z=load(TRIAL/f'runs/{arm}/seed{s}/{pop}_epoch120.npz')
                assert np.array_equal(z['sample_id'],ids[pop]) and np.array_equal(z['native'],initial['native']) and np.array_equal(z['future'],initial['future'])
                assert all(np.array_equal(z['mask_'+k],v) for k,v in masks.items())
                assert z['trajectory'].shape==(n,20,12,2) and np.isfinite(z['trajectory']).all()
                a,f=metrics(z['trajectory'],z['future']);good=((a<=.2)&(f<=.4)).any(-1)
                for name,value in [('minade',a.min(-1)),('minfde',f.min(-1))]:
                    e=float(np.max(abs(value-z[name])));maxerror=max(maxerror,e);assert e<1e-12
                assert np.array_equal(good,z['coverage'])
                for g,mask in masks.items():
                    row=next(r for r in expected if r['arm']==arm and r['seed']==s and r['pop']==pop and r['epoch']==120 and r['group']==g)
                    assert int((good&mask).sum())==row['good_windows'] and int(mask.sum())==row['n']
                key=f'{label}{s}';paths[key].append(z['trajectory']);effective[key].append(z['effective_good']);distinct[key].append(z['all_distinct'])
        guard()
    paths={k:np.concatenate(v) for k,v in paths.items()};effective={k:np.concatenate(v) for k,v in effective.items()};distinct={k:np.concatenate(v) for k,v in distinct.items()}
    future=np.concatenate(future);masks={k:np.concatenate(v) for k,v in allmasks.items()}
    assert np.array_equal(base,paths['native']);del cache,enc,base
    kinds=np.concatenate(kinds);onsets=np.concatenate(onsets);uncertain=np.concatenate(uncertain);scene=np.concatenate(scene)
    metrics_map={k:metrics(v,future) for k,v in paths.items()};ade={k:v[0] for k,v in metrics_map.items()};fde={k:v[1] for k,v in metrics_map.items()}
    selected=scores.argmax(-1);counts=[];rows=[];reps=[];unions={};flags_main={}
    for ta,tf in ((.2,.4),(.3,.5)):
        yes={k:((ade[k]<=ta)&(fde[k]<=tf)).any(-1) for k in KEYS}
        for s in SEEDS:
            new=f'learned{s}'
            for ref in ('native',f'fixed{s}'):
                changes=flags(yes[ref],yes[new])
                if ta==.2:flags_main[(s,ref)]=changes
                for pop in ('FIT','CAL'):
                    pm=np.array(poplabels)==pop
                    for g,gm in masks.items():
                        mask=pm&gm
                        counts.append(dict(pop=pop,group=g,seed=s,reference=ref,ade_limit=ta,fde_limit=tf,n=int(mask.sum()),reference_good=int((yes[ref]&mask).sum()),learned_good=int((yes[new]&mask).sum()),**{f:int((changes[mask]==f).sum()) for f in ('rescued','lost','retained','missing')}))
                for i,id_ in enumerate(allids):
                    oldmargin=np.maximum(ade[ref][i]/ta,fde[ref][i]/tf);newmargin=np.maximum(ade[new][i]/ta,fde[new][i]/tf)
                    oldslot=int(oldmargin.argmin());newslot=int(newmargin.argmin())
                    rows.append(dict(sample_id=int(id_),population=poplabels[i],seed=s,reference=ref,transition=str(changes[i]),ade_limit=ta,fde_limit=tf,coarse_failure=bool(masks['coarse'][i]),scene=str(scene[i]),old_margin=float(oldmargin.min()),new_margin=float(newmargin.min()),old_closest_slot=oldslot+1,new_closest_slot=newslot+1,old_ADE=float(ade[ref][i,oldslot]),old_FDE=float(fde[ref][i,oldslot]),new_ADE=float(ade[new][i,newslot]),new_FDE=float(fde[new][i,newslot]),new_raw_good_count=int(((ade[new][i]<=ta)&(fde[new][i]<=tf)).sum()),effective_good_count_primary=int(effective[new][i])))
    for pop in ('FIT','CAL'):
        pm=np.array(poplabels)==pop;fs=np.stack([flags_main[(s,'native')] for s in SEEDS],-1)
        unions[pop]={f:dict(ids=allids[pm&(fs==f).any(-1)].tolist(),all_three_ids=allids[pm&(fs==f).all(-1)].tolist()) for f in ('rescued','lost')}
        for s in SEEDS:
            delta=np.maximum(ade[f'learned{s}']/.2,fde[f'learned{s}']/.4).min(-1)-np.maximum(ade['native']/.2,fde['native']/.4).min(-1)
            for flag in ('rescued','lost'):
                choice=np.flatnonzero(pm&(flags_main[(s,'native')]==flag));order=choice[np.lexsort((allids[choice],abs(delta[choice])))]
                if len(order):
                    for rule,i in [('median_change',order[(len(order)-1)//2]),('largest_change',order[-1])]:reps.append(dict(pop=pop,seed=s,transition=flag,rule=rule,id=int(allids[i]),row=int(i)))
    (HERE/'data/cases').mkdir(parents=True,exist_ok=True)
    meta=[];maxround=0.
    for i,id_ in enumerate(allids):
        record=dict(id=int(id_),row=i,pop=poplabels[i],clear=bool(masks['clear'][i]),uncertain=bool(uncertain[i]),group=int(formal[i]),event_kind=int(kinds[i]),onset=int(onsets[i]),scene=str(scene[i]),masks={k:bool(v[i]) for k,v in masks.items()},ade={k:ade[k][i].tolist() for k in KEYS},fde={k:fde[k][i].tolist() for k in KEYS},effective={k:int(effective[k][i]) for k in KEYS},distinct={k:int(distinct[k][i]) for k in KEYS},selected=int(selected[i]))
        meta.append(record)
        display=dict(id=int(id_),past=np.round(past[i].astype(float),6),gt=np.round(future[i].astype(float),6),scores=scores[i],neighbors=[],paths={k:np.round(paths[k][i].astype(float),6) for k in KEYS})
        js(HERE/f'data/cases/{id_}.js',f'window.PULSE_CASES[{int(id_)}]',display)
        if i%64==0:guard()
    for k in KEYS:
        a,f=metrics(np.round(paths[k].astype(float),6),np.round(future.astype(float),6));error=max(float(abs(a-ade[k]).max()),float(abs(f-fde[k]).max()));maxround=max(maxround,error);assert error<4e-6
    initial=next(x for x in reps if x['pop']=='CAL' and x['seed']==7 and x['transition']=='lost' and x['rule']=='median_change')
    common=dict(cases=meta,keys=KEYS,seeds=list(SEEDS),counts=counts,unions=unions,representatives=reps,event_labels=['无明确事件','转向','速度变化','启动','停止'],dt=.4,source_sha=before[str(TRIAL/'RESULT.json')],default_id=initial['id'],default_seed=7,default_filter='lost',default_pop='CAL')
    js(HERE/'data/common.js','window.PULSE_REVIEW',common)
    csvout(HERE/'case_classification.csv',rows);csvout(HERE/'counts.csv',counts)
    dump(HERE/'summary.json',dict(counts=counts,unions=unions,representatives=reps,default=initial))
    np.savez_compressed(HERE/'data/exact_arrays.npz',sample_id=allids,population=np.array(poplabels),past=past,future=future,logits=scores,**paths)
    for p,h in before.items():assert sha(Path(p))==h,p
    dump(HERE/'input_hashes.json',before)
    dump(HERE/'audit.json',dict(status='PASS_FIT_CAL_SAVED_OUTPUT_VISUAL_REVIEW',IDs=649,FIT=256,CAL=393,all_final_6_runs=True,K=20,exclusions=0,all_saved_group_counts_match=True,saved_metric_max_error=maxerror,display_rounding_metric_max_error=maxround,old_files_before_after_SHA=len(before),old_SHA_scope='Entire current diagnosis and prior pulse visualizer plus directly-read FITCAL metadata/cache; not full-history25004 rerun',resource=guard(),elapsed_seconds=time.perf_counter()-begin,training=False,model_forward=False,optimizer=False,DEV_test_read=False,classification='SAMEcandidate ADE<=.2 AND FDE<=.4; .3/.5 sensitivity only',selection='Native argmax actual; new trajectories only inherit original slot, not rescored',pulse_profile='Not available in saved diagnostic outputs; no new inference to synthesize it'))
    print(json.dumps(dict(status='PASS',default=initial,counts=[x for x in counts if x['group']=='all' and x['reference']=='native' and x['ade_limit']==.2],unions={p:{f:dict(unique=len(v['ids']),all_three=len(v['all_three_ids'])) for f,v in groups.items()} for p,groups in unions.items()},resource=guard()),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
