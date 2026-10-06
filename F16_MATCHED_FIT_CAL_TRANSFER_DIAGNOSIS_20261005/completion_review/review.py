"""Independent CPU saved-output arithmetic review; no torch/model/optimizer."""
from pathlib import Path
import ctypes, hashlib, json, time, psutil
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
HERE=Path(__file__).resolve().parent;SERIES=HERE.parent;ROOT=SERIES.parent
INPUTS={};START=time.perf_counter();RSS=PRIVATE=0.;PHYS=COMMIT=float('inf')
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('tp','ap','tc','ac','tv','av','extra')]
def guard():
    global RSS,PRIVATE,PHYS,COMMIT
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    p=psutil.Process().memory_info();RSS=max(RSS,p.peak_wset/2**30);PRIVATE=max(PRIVATE,p.peak_pagefile/2**30)
    PHYS=min(PHYS,m.ap/2**30);COMMIT=min(COMMIT,m.ac/2**30)
    assert max(p.rss,p.private)<6*2**30 and min(m.ap,m.ac)>=6*2**30,'STOP_RESOURCE'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(path):
    path=Path(path);INPUTS[str(path)]=sha(path);return json.loads(path.read_text(encoding='utf-8-sig'))
def load(path,keys=None):
    path=Path(path);INPUTS[str(path)]=sha(path)
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in (z.files if keys is None else keys)}
def verify(files,label):
    for n,(path,digest) in enumerate(files.items()):
        if n%500==0:guard()
        assert sha(path)==digest,(label,path)
    print(label,len(files),'PASS',flush=True)
def save(name,obj):
    path=HERE/name;assert not path.exists(),('PRESERVE',path)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def masks(g):return [('overall',np.ones(len(g),bool))]+[(k,g==i) for i,k in enumerate(('stable','transition','change'))]
guard();assert not (HERE/'REVIEW.json').exists()
result=read(SERIES/'RESULT.json');assert result['status']=='PASS_DIAGNOSTIC_ONLY'
assert read(SERIES/'worker_state.json')['status']==read(SERIES/'exit_status.json')['status']=='SUCCESS'
old=read(SERIES/'old_results_manifest.json')['files'];src=read(SERIES/'preflight.json')['source_hashes']
assert len(old)==29871 and len(src)==433
verify(old,'old-before');verify(src,'source-before')
ar=read(SERIES/'ARITHMETIC_AUDIT.json');frozen=read(SERIES/'FROZEN_REPLAY_AUDIT.json')
assert ar['status']==frozen['status']=='PASS' and len(frozen['records'])==18
assert len(ar['all_saved_output_SHA'])==76;verify(ar['all_saved_output_SHA'],'outputs76')
cache=load(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz',['sample_id','logits','change_group'])
labels=load(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz',['sample_id','future'])
assert np.array_equal(cache['sample_id'],labels['sample_id']) and len(np.unique(cache['sample_id']))==28791
models={x['model']['path']:x['model']['sha256'] for x in result['records'] if x['model'] is not None};assert len(models)==18;verify(models,'models18')
native={};checked=[];maxerr=0.;pair=np.triu_indices(20,1)
for rec in result['records']:
    guard();a,b=(0,25912) if rec['split']=='FIT' else (25912,28791)
    z=load(SERIES/'jobs'/(rec['job']+'.npz'));done=read(SERIES/'jobs'/(rec['job']+'.done.json'))
    assert done['status']=='SUCCESS' and done['source_signature']==sha(SERIES/'preflight.json')
    ids=z['sample_id'];path=z['trajectory'];gt=labels['future'][a:b];scores=cache['logits'][a:b];group=cache['change_group'][a:b]
    assert np.array_equal(ids,cache['sample_id'][a:b]) and path.shape==(b-a,20,12,2) and np.isfinite(path).all()
    for start in range(0,len(path),256):
        guard();sl=slice(start,min(len(path),start+256));x=path[sl].astype(np.float64)
        e=np.sqrt(((x-gt[sl,None].astype(np.float64))**2).sum(-1));ade=e.mean(-1);fde=e[:,:,-1]
        good=(ade<=.2)&(fde<=.4);row=np.arange(len(x));slot=np.maximum(ade/.2,fde/.4).argmin(-1);selected=scores[sl].argmax(-1)
        values=dict(coverage=good.any(-1),loose_coverage=((ade<=.3)&(fde<=.5)).any(-1),raw_good=good.sum(-1),selected_good=good[row,selected],
            selected_slot=selected,oracle_slot=slot,minADE=ade.min(-1),minFDE=fde.min(-1),selected_ADE=ade[row,selected],selected_FDE=fde[row,selected],
            disjoint_minima=(ade.min(-1)<=.2)&(fde.min(-1)<=.4)&~good.any(-1),early=e[row,slot,:4].mean(-1),middle=e[row,slot,4:8].mean(-1),late=e[row,slot,8:].mean(-1))
        effective=np.empty(len(x),np.int16);distinct=effective.copy()
        for j in range(len(x)):
            d=np.sqrt(((x[j,pair[0]]-x[j,pair[1]])**2).sum(-1))
            lab=fcluster(linkage(np.maximum(d.mean(-1)/.1,d[:,-1]/.2),method='complete'),1.,criterion='distance')
            effective[j]=len(np.unique(lab[good[j]]));distinct[j]=len(np.unique(lab))
        values.update(effective_good=effective,all_distinct=distinct)
        v=np.diff(x,axis=2,prepend=np.zeros_like(x[:,:,:1]))/.4;acc=np.diff(v,axis=2)/.4;jrk=np.diff(acc,axis=2)/.4
        for k,d in [('speed',v),('accel',acc),('jerk',jrk)]:values['motion_'+k]=np.sqrt((d*d).sum(-1))
        for key,value in values.items():
            error=float(np.abs(value.astype(float)-z[key][sl].astype(float)).max());maxerr=max(maxerr,error)
            assert error<=1e-12,(rec['job'],key,error)
    if rec['model'] is None:native[rec['split']]={k:v for k,v in z.items() if k not in ('trajectory','sample_id')}
    nq=native[rec['split']];rows=[]
    for name,mask in masks(group):
        stored=next(r for r in rec['groups'] if r['group']==name);assert stored['n']==int(mask.sum())
        for key in ('coverage','loose_coverage','raw_good','effective_good','selected_good','disjoint_minima'):assert stored[key+'_total']==int(z[key][mask].sum())
        for key in ('minADE','minFDE','early','middle','late','selected_ADE','selected_FDE'):assert abs(stored[key]-float(z[key][mask].mean()))<1e-12
        for key in ('all_distinct','effective_good'):
            histkey='distinct_histogram' if key=='all_distinct' else 'effective_good_histogram'
            assert stored[histkey]==np.bincount(z[key][mask],minlength=21).tolist()
        assert stored['selection_miss']==int((z['coverage'][mask]&~z['selected_good'][mask]).sum())
        rescued=int((~nq['coverage'][mask]&z['coverage'][mask]).sum());lost=int((nq['coverage'][mask]&~z['coverage'][mask]).sum())
        assert (stored['rescued'],stored['lost'])==(rescued,lost)
        for key in ('speed','accel','jerk'):
            values=z['motion_'+key][mask].reshape(-1)
            assert abs(stored[key+'_mean']-float(values.mean()))<1e-12
            assert abs(stored[key+'_p95']-float(np.quantile(values,.95)))<1e-12
        rows.append(stored)
    checked.append(dict(job=rec['job'],split=rec['split'],model=rec['model'],metadata=rec['metadata'],groups=rows))
    del z,path;guard();print(rec['job'],'ALL-COLUMNS-GROUPS PASS',flush=True)
assert len(checked)==38
verify(models,'models-after');verify(src,'source-after');verify(old,'old-after');verify(INPUTS,'inputs-after')
save('ALL_GROUPS_VERIFIED.json',dict(status='PASS',records=checked))
save('REVIEW.json',dict(status='PASS_CPU_SAVED_FULL_ARITHMETIC_REVIEW',jobs=38,outputs_SHA=76,models=18,
    unique_IDs=28791,FIT=25912,CAL=2879,all_metric_cluster_motion_columns_recomputed=True,all_group_aggregates_recomputed=True,max_error=maxerr,
    source_count=len(src),old_file_count=len(old),before_after_SHA=True,input_count=len(INPUTS),no_torch_model_optimizer=True,
    scope='Saved coordinate arithmetic, same registered SciPy complete-link definition; not external reviewer or replay of all models',
    seconds=time.perf_counter()-START,memory=dict(peak_RSS_GiB=RSS,peak_private_GiB=PRIVATE,min_physical_GiB=PHYS,min_commit_GiB=COMMIT)))
save('INPUT_HASHES.json',INPUTS)
print('REVIEW PASS',maxerr,'seconds',time.perf_counter()-START,flush=True)
