"""Bounded independent saved-output review; no torch/model/optimizer/GT forward."""
from pathlib import Path
import json,csv,hashlib,time,ctypes,psutil
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
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
def load(path):
    path=Path(path);INPUTS[str(path)]=sha(path)
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def csvrows(path):
    INPUTS[str(path)]=sha(path)
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def verify(mapping,label):
    for n,(path,digest) in enumerate(mapping.items()):
        if n%500==0:guard()
        assert sha(path)==digest,(label,path)
    print(label,len(mapping),'PASS',flush=True)
def save(name,obj):
    p=HERE/name;assert not p.exists(),('PRESERVE',p)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
assert not (HERE/'REVIEW.json').exists();guard()
old=read(SERIES/'old_results_manifest.json')['files'];sources=read(SERIES/'preflight.json')['source_hashes']
assert len(old)==28514 and len(sources)==382
verify(old,'old-before');verify(sources,'source-before')
worker=read(SERIES/'independent_worker_state.json');ex=read(SERIES/'results/exit_status.json')
assert worker['status']==ex['status']=='SUCCESS'
for name in ('independent_audit','saved_metrics_recompute','frozen_audit','audit'):
    a=read(SERIES/f'results/{name}.json');assert a['status']=='PASS',(name,a['status'])
locks=read(SERIES/'results/model_lock.json')['models'];assert len(locks)==18;verify(locks,'model-locks')
summary=read(SERIES/'results/summary.json');assert summary['status']=='COMPLETE'
quality=read(SERIES/'results/quality/summary.json');assert quality['status']=='PASS_SAVED_QUALITY_ARITHMETIC'
arrays=load(ROOT/'PULSE_RESPONSE_VISUAL_REVIEW_20261004/data/exact_arrays.npz')
ids=arrays['sample_id'];gt=arrays['future'].astype(np.float64);scores=arrays['logits'];clear=arrays['clear_event']
assert len(ids)==len(np.unique(ids))==1516 and clear.sum()==469
i,j=np.triu_indices(20,1);ii=np.arange(len(ids))
def quantities(path):
    path=path.astype(np.float64);error=np.linalg.norm(path-gt[:,None],axis=-1);a=error.mean(-1);f=error[:,:,-1]
    good=(a<=.2)&(f<=.4);eff=np.zeros(len(path),np.int16);distinct=eff.copy()
    for n in range(len(path)):
        if n%256==0:guard()
        distance=np.linalg.norm(path[n,i]-path[n,j],axis=-1)
        labels=fcluster(linkage(np.maximum(distance.mean(-1)/.10,distance[:,-1]/.20),method='complete'),1.,criterion='distance')
        distinct[n]=len(np.unique(labels));eff[n]=len(np.unique(labels[good[n]]))
    slot=np.maximum(a/.2,f/.4).argmin(-1);chosen=error[ii,slot]
    return a,f,dict(coverage=good.any(-1),raw_good=good.sum(-1),effective_good=eff,all_distinct=distinct,
        selected_good=good[ii,scores.argmax(-1)],oracle_slot=slot,early=chosen[:,:4].mean(-1),middle=chosen[:,4:8].mean(-1),late=chosen[:,8:].mean(-1))
na,nf,nq=quantities(arrays['native']);assert nq['coverage'].sum()==733 and nq['effective_good'].sum()==929
rows=[];histories=[];maxerror=0.;allq={}
for arm in ('zero','mean_std'):
  for seed in (1,2,3):
    directory=SERIES/f'results/models/{arm}/seed_{seed}/balanced_protected'
    complete=read(directory/'complete.json');history=csvrows(directory/'history.csv')
    assert complete['status']=='COMPLETE' and complete['epochs']==80 and [int(x['epoch']) for x in history]==list(range(1,81))
    ce=complete['best_feasible_epoch'];chosen=next((r for r in history if int(r['epoch'])==ce),None)
    histories.append(dict(arm=arm,seed=seed,selected_epoch=ce,legacy_epoch=complete['legacy_epoch'],
        train_first=float(history[0]['train_loss']),train_final=float(history[-1]['train_loss']),
        train_loss_gain_pct=100*(1-float(history[-1]['train_loss'])/float(history[0]['train_loss'])),
        infeasible_epochs=sum(r['cal_feasible'].lower()=='false' for r in history),
        CAL_first={k:float(history[0][k]) for k in ('overall_minade20','overall_minfde20','stable_minade20','stable_minfde20','change_minade20','change_minfde20')},
        CAL_selected=None if chosen is None else {k:float(chosen[k]) for k in ('overall_minade20','overall_minfde20','stable_minade20','stable_minfde20','change_minade20','change_minfde20')},
        CAL_final={k:float(history[-1][k]) for k in ('overall_minade20','overall_minfde20','stable_minade20','stable_minfde20','change_minade20','change_minfde20')}))
    for policy in ('constrained','legacy','final'):
      guard();pred=load(SERIES/f'results/seed{seed}_{arm}_{policy}.npz');saved=load(SERIES/f'results/quality/seed{seed}_{arm}_{policy}_quality.npz')
      assert np.array_equal(ids,pred['sample_id']) and np.array_equal(ids,saved['sample_id'])
      path=pred['trajectory'];assert path.shape==(1516,20,12,2) and np.isfinite(path).all()
      a,f,q=quantities(path)
      for key,value in q.items():assert np.allclose(value,saved[key],atol=1e-10,rtol=0),(arm,seed,policy,key)
      for key,value in [('minade20',a.min(-1)),('minfde20',f.min(-1))]:
        diff=float(np.max(np.abs(value-pred[key])));maxerror=max(maxerror,diff);assert diff<=1e-6
      covered=q['coverage'];native=nq['coverage'];allq[arm,seed,policy]=q
      row=dict(arm=arm,seed=seed,policy=policy,good=int(covered.sum()),raw_good_total=int(q['raw_good'].sum()),effective_good_total=int(q['effective_good'].sum()),
        rescued=int((~native&covered).sum()),lost=int((native&~covered).sum()),
        clear_good=int(covered[clear].sum()),clear_rescued=int((clear&~native&covered).sum()),clear_lost=int((clear&native&~covered).sum()),
        inherited_selected_good=int(q['selected_good'].sum()),
        lost_effective_on_retained_coverage=int(np.maximum(nq['effective_good']-q['effective_good'],0)[covered&native].sum()),
        gained_effective_on_retained_coverage=int(np.maximum(q['effective_good']-nq['effective_good'],0)[covered&native].sum()),
        native_multiple_to_one=int(((nq['effective_good']>1)&(q['effective_good']==1)&covered).sum()),
        changed_distinct_count=int((q['all_distinct']!=nq['all_distinct']).sum()),
        raw_good_histogram={str(k):int((q['raw_good']==k).sum()) for k in np.unique(q['raw_good'])},
        effective_good_histogram={str(k):int((q['effective_good']==k).sum()) for k in np.unique(q['effective_good'])},
        all_distinct_histogram={str(k):int((q['all_distinct']==k).sum()) for k in np.unique(q['all_distinct'])})
      rows.append(row)
raw_repeated=[]
for seed in (1,2,3):
    for policy in ('constrained','legacy','final'):
        current=load(SERIES/f'results/seed{seed}_zero_{policy}.npz')
        prior=load(ROOT/f'GENERATION_CANDIDATE_STATE_SERIES_20261005/results/seed{seed}_mean_std_{policy}.npz')
        assert np.array_equal(current['sample_id'],prior['sample_id'])
        exact=np.array_equal(current['trajectory'],prior['trajectory'])
        raw_repeated.append(dict(seed=seed,policy=policy,trajectories_bitwise_exact=bool(exact),maximum_path_difference=float(np.max(np.abs(current['trajectory']-prior['trajectory'])))))
comparisons=[]
for seed in (1,2,3):
    new=allq['mean_std',seed,'constrained'];control=allq['zero',seed,'constrained']
    comparisons.append(dict(seed=seed,rescued=int((~control['coverage']&new['coverage']).sum()),lost=int((control['coverage']&~new['coverage']).sum()),
        clear_rescued=int((clear&~control['coverage']&new['coverage']).sum()),clear_lost=int((clear&control['coverage']&~new['coverage']).sum()),
        coverage_gain=int(new['coverage'].sum()-control['coverage'].sum()),effective_good_delta=int(new['effective_good'].sum()-control['effective_good'].sum())))
verify(INPUTS,'direct-input-after');verify(locks,'model-after');verify(old,'old-after');verify(sources,'source-after');guard()
report=dict(status='PASS_SAVED_OUTPUT_COMPLETION_REVIEW',no_model_forward=True,no_optimizer=True,no_new_data=True,
    independent_min_metric_max_error=maxerror,all18_quality_columns_recomputed=True,all18_complete_link_clusters_reconstructed=True,
    cluster_algorithm_scope='New reconstruction from saved float64 geometry, same preregistered SciPy complete linkage; not independent clustering algorithm/external review.',
    old_files_before_after=28514,source_before_after=382,model_SHA=18,saved_predictions=18,explicit_IDs=1516,K20_shape=[1516,20,12,2],
    native_good=733,native_effective_good_total=929,native_raw_good_total=int(nq['raw_good'].sum()),native_inherited_selected_good=int(nq['selected_good'].sum()),
    clear_event_n=469,clear_native_good=int(nq['coverage'][clear].sum()),rows=rows,paired_constrained_coverage=comparisons,histories=histories,raw_previous_actual_comparison=raw_repeated,
    production_actual_seconds=(__import__('datetime').datetime.fromisoformat(worker['finished_at'])-__import__('datetime').datetime.fromisoformat(worker['started_at'])).total_seconds(),
    production_resource=read(SERIES/'RESOURCE_USAGE.json'),review_resource=dict(peak_RSS_GiB=RSS,peak_private_GiB=PRIVATE,min_physical_GiB=PHYS,min_commit_GiB=COMMIT),
    review_elapsed_seconds=time.perf_counter()-START)
save('INPUT_HASHES.json',INPUTS);save('REVIEW.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('rows','histories')}),flush=True)
