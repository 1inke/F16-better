"""Saved-data completion supplement; no torch, model forward or optimizer."""
from pathlib import Path
import os, sys, json, hashlib, time, ctypes
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import psutil
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent

class Mem(ctypes.Structure):
    _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
resource = dict(peak_RSS_GiB=0., peak_private_GiB=0., min_physical_GiB=1e10, min_commit_GiB=1e10)
def guard():
    v = psutil.Process().memory_info(); m=Mem(); m.length=ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    for k,x in [('peak_RSS_GiB',v.peak_wset/2**30),('peak_private_GiB',v.peak_pagefile/2**30)]: resource[k]=max(resource[k],x)
    resource['min_physical_GiB']=min(resource['min_physical_GiB'],m.avail_phys/2**30)
    resource['min_commit_GiB']=min(resource['min_commit_GiB'],m.avail_page/2**30)
    assert max(v.rss,v.private)<6*2**30 and min(m.avail_phys,m.avail_page)>=6*2**30, 'STOP_RESOURCE'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def verify(files):
    for i,(p,digest) in enumerate(files.items()):
        assert sha(p)==digest, ('SHA',p)
        if i%128==0: guard()
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def scalar(a,arm,scales):
    original=float(a['primary'].astype(float).mean()+.05*a['div_num'].astype(float).sum()/max(1,a['div_den'].sum())+.01*a['residual'].astype(float).mean()+.01*a['smooth'].astype(float).mean()+.1*a['protection'].astype(float).sum()/max(1,a['protected'].sum()))
    extra=0.
    for g in range(3):
        m=a['group']==g
        if m.any(): extra+=m.mean()*max(a['accel'][m].astype(float).mean()-1.01*a['native_accel'][m].astype(float).mean(),0)/scales[g]
    return dict(original_total=original,P=float(extra),added_total=float(original+(.1*extra if arm=='mean_std' else 0.)),coverage=int(a['good'].sum()),n=len(a['good']))
def main():
    start=time.perf_counter(); guard()
    worker=read(EXP/'independent_worker_state.json'); assert worker['status']=='SUCCESS'
    assert read(EXP/'results/exit_status.json')['status']=='SUCCESS'
    pf=read(EXP/'preflight.json'); sources=pf['source_hashes']; old=read(EXP/'old_results_manifest.json')['files']
    models=read(EXP/'results/model_lock.json')['models']; pop=read(EXP/'results/physical_population/summary.json')
    assert pop['status']=='PASS' and pop['weights_buffers_fixed']
    for filename in ('independent_audit.json','saved_metrics_recompute.json','frozen_audit.json'):
        assert read(EXP/'results'/filename)['status']=='PASS'
    assert worker['final_old_SHA_files']==len(old)==30298 and len(sources)==484 and len(models)==18
    verify(sources); verify(old); verify(models); verify(pop['output_SHA'])
    scales=np.array(read(EXP/'physical_accel_scales.json')['scales'],float)
    maximum=0.; records=[]
    for r in pop['records']:
        a=load(EXP/'results/physical_population'/f"seed{r['seed']}_{r['arm']}_{r['policy']}_{r['split']}.npz")
        assert len(a['sample_id'])==len(np.unique(a['sample_id']))==r['n']
        actual=scalar(a,r['arm'],scales)
        for k,v in actual.items():
            error=abs(v-r[k]); maximum=max(maximum,error); assert error<=1e-10,(k,error)
        for g,name in enumerate(('stable','transition','change')):
            m=a['group']==g
            records.append(dict(arm=r['arm'],seed=r['seed'],policy=r['policy'],split=r['split'],group=name,n=int(m.sum()),good=int(a['good'][m].sum()),accel_mean=float(a['accel'][m].astype(float).mean()),native_accel_mean=float(a['native_accel'][m].astype(float).mean()),accel_harm_pct=float(100*(a['accel'][m].astype(float).mean()/a['native_accel'][m].astype(float).mean()-1))))
        del a; guard()
    q=read(EXP/'results/quality/summary.json'); cache=load(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_dev.npz')
    ids=cache['sample_id']; groups=cache['change_group']; native=((cache['ade']<=.2)&(cache['fde']<=.4)).any(-1)
    assert native.sum()==q['native_good']==733
    for arm in ('zero','mean_std'):
        for seed in (1,2,3):
            parent=EXP/'results/models'/arm/f'seed_{seed}'/'balanced_protected'
            rows=read(parent/'objective_history.json')['rows']; assert len(rows)==80
            assert [r['epoch'] for r in rows]==list(range(1,81))
            for row in rows:
                assert row['n']==25912 and sum(row['groups_n'])==25912
                assert abs(row['total']-row['original']-row['added'])<2e-6
                assert abs(row['added']-(.1*row['P'] if arm=='mean_std' else 0.))<2e-6
            for policy in ('constrained','legacy','final'):
                a=load(EXP/'results/quality'/f'seed{seed}_{arm}_{policy}_quality.npz'); assert np.array_equal(a['sample_id'],ids)
                for r in q['records']:
                    if (r['arm'],r['seed'],r['policy'])!=(arm,seed,policy): continue
                    m=np.ones(len(ids),bool) if r['group']=='overall' else groups==('stable','transition','change').index(r['group'])
                    actual=dict(good=int(a['coverage'][m].sum()),rescued=int((~native[m]&a['coverage'][m]).sum()),lost=int((native[m]&~a['coverage'][m]).sum()),effective_good_total=int(a['effective_good'][m].sum()),inherited_selected_good=int(a['selected_good'][m].sum()))
                    assert all(r[k]==v for k,v in actual.items()),(r,actual)
                assert np.all(a['effective_good']<=a['raw_good']) and np.all(a['effective_good']<=a['all_distinct'])
                del a
            guard()
    # Fresh raw controls: actual full-array policy repetition, not an assumed repeat.
    repeats=[]
    for seed in (1,2,3):
        for policy in ('constrained','legacy','final'):
            new=load(EXP/'results'/f'seed{seed}_zero_{policy}.npz')
            prior=load(ROOT/'GENERATION_CANDIDATE_STATE_SERIES_20261005/results'/f'seed{seed}_mean_std_{policy}.npz')
            exact=set(new)==set(prior) and all(np.array_equal(new[k],prior[k]) for k in new)
            repeats.append(dict(seed=seed,policy=policy,all_arrays_bitwise_exact=bool(exact)))
            del new,prior; guard()
    verify(pop['output_SHA']); verify(models); verify(sources); verify(old)
    result=dict(status='PASS_SAVED_NUMPY_SUPPLEMENT',maximum_population_component_error=maximum,sources=len(sources),old_files=len(old),models=len(models),population_outputs=len(pop['output_SHA']),all_before_after_SHA=True,physical_group_records=records,raw_control_repeats=repeats,memory=resource,elapsed_seconds=time.perf_counter()-start,
        scope='36 saved population originaltotal/P independent NumPy recomputation, all18 saved quality group counts and six80epoch objective traces; full geometry clusters, all metrics/CIs and frozen replay verified by completed worker, NOT rerun here; no torch/model/optimizer or new data; same assistant not external review.')
    (HERE/'REVIEW.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','maximum_population_component_error','sources','old_files','models','population_outputs','memory','elapsed_seconds')}),flush=True)
if __name__=='__main__': main()
