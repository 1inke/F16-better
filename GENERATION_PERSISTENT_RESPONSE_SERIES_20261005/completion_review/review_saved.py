"""Bounded NumPy/JSON saved-output review. No model import, forward or optimizer."""
from pathlib import Path
import json,hashlib,csv,time,ctypes
from datetime import datetime
import numpy as np
HERE=Path(__file__).resolve().parent; SERIES=HERE.parent; ROOT=SERIES.parent
START=time.perf_counter(); INPUTS={}; CHECKS=0
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('physical','available_physical','page','available_page','virtual','available_virtual','extended')]
def guard():
    global CHECKS
    mem=MEMORYSTATUSEX();mem.length=ctypes.sizeof(mem)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    assert min(mem.available_physical,mem.available_page)>=6*1024**3,'System reserve below6GiB: stop'
    CHECKS+=1
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): value.update(block)
    return value.hexdigest()
def read(path):
    path=Path(path);INPUTS[str(path)]=sha(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))
def load(path):
    path=Path(path);INPUTS[str(path)]=sha(path)
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(name,obj):
    path=HERE/name;assert not path.exists(),f'Preserve existing {path}'
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rows(path):
    INPUTS[str(path)]=sha(path)
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def verify(mapping,label):
    for i,(path,digest) in enumerate(mapping.items()):
        if i%500==0:guard()
        assert sha(path)==digest,f'{label} SHA {path}'
    print(label,len(mapping),'PASS',flush=True)
    return len(mapping)
assert not (HERE/'REVIEW.json').exists()
guard();claim=read(HERE/'CLAIM_INTENT_MANIFEST.json')
old=read(SERIES/'old_results_manifest.json')['files']
sources=read(SERIES/'preflight.json')['source_hashes']
first_attempt=read(HERE/'failures/FORMAT_ASSERTION_FAILURE.json')
assert first_attempt['actual_passed_before_failure']['old_files']==len(old)==27164
assert first_attempt['actual_passed_before_failure']['scientific_sources']==len(sources)==347
verify(sources,'source-before-schema-repair')
state=read(SERIES/'independent_worker_state.json');exit=read(SERIES/'results/exit_status.json')
assert state['status']==exit['status']=='SUCCESS'
audit=read(SERIES/'results/independent_audit.json');recompute=read(SERIES/'results/saved_metrics_recompute.json')
assert audit['status']==recompute['status']=='PASS'
locks=read(SERIES/'results/model_lock.json')['models'];verify(locks,'18-model-locks')
arrays=load(ROOT/'PULSE_RESPONSE_VISUAL_REVIEW_20261004/data/exact_arrays.npz')
ids=arrays['sample_id'];assert len(ids)==len(np.unique(ids))==1516
gt=arrays['future'].astype(np.float64);native=arrays['native'].astype(np.float64);scores=arrays['logits'];selected=scores.argmax(-1)
classification=read(ROOT/'F16_LOW_CUE_FAILURE_STRATIFICATION_20261004/CLASSIFICATION.json')['cases']
lookup={int(c['sample_id']):c for c in classification};clear=np.array([lookup[int(i)]['GT_clear_event'] for i in ids]);assert clear.sum()==469
def metrics(path):
    error=np.linalg.norm(path.astype(np.float64)-gt[:,None],axis=-1);ade=error.mean(-1);fde=error[:,:,-1];good=(ade<=.2)&(fde<=.4)
    return ade,fde,good
na,nf,ng=metrics(native);coverage=ng.any(-1);assert coverage.sum()==733
assert all(bool(lookup[int(i)]['native_good_candidate'])==bool(g) for i,g in zip(ids,coverage))
review_rows=[];maxerror=0.;repeated=[];history=[]
quality=read(SERIES/'results/quality/summary.json')
for arm in ('zero','mean_std'):
  for seed in (1,2,3):
    folder=SERIES/f'results/models/{arm}/seed_{seed}/balanced_protected'
    complete=read(folder/'complete.json');hist=rows(folder/'history.csv')
    assert complete['status']=='COMPLETE' and complete['epochs']==80 and len(hist)==80
    assert [int(r['epoch']) for r in hist]==list(range(1,81))
    history.append(dict(arm=arm,seed=seed,complete=complete,first_loss=float(hist[0]['train_loss']),last_loss=float(hist[-1]['train_loss']),cal_infeasible_epochs=sum(r['cal_feasible'].lower()=='false' for r in hist),CAL_first={k:float(hist[0][k]) for k in ('overall_minade20','overall_minfde20','change_minade20','change_minfde20')},CAL_last={k:float(hist[-1][k]) for k in ('overall_minade20','overall_minfde20','change_minade20','change_minfde20')}))
    for policy in ('constrained','legacy','final'):
      guard();pred=load(SERIES/f'results/seed{seed}_{arm}_{policy}.npz');q=load(SERIES/f'results/quality/seed{seed}_{arm}_{policy}_quality.npz')
      assert np.array_equal(pred['sample_id'],ids) and np.array_equal(q['sample_id'],ids)
      path=pred['trajectory'];assert path.shape==(1516,20,12,2) and np.isfinite(path).all()
      a,f,g=metrics(path);covered=g.any(-1);slot=np.maximum(a/.2,f/.4).argmin(-1)
      for key,v in [('coverage',covered),('raw_good',g.sum(-1)),('selected_good',g[np.arange(1516),selected]),('oracle_slot',slot)]:assert np.array_equal(q[key],v),key
      for key,v in [('minade20',a.min(-1)),('minfde20',f.min(-1))]:
        diff=float(np.max(np.abs(pred[key]-v)));maxerror=max(maxerror,diff);assert diff<=1e-6,key
      row=dict(arm=arm,seed=seed,policy=policy,good=int(covered.sum()),native_good=733,raw_good_total=int(g.sum()),effective_good_total=int(q['effective_good'].sum()),inherited_selected_good=int(g[np.arange(1516),selected].sum()),rescued=int((~coverage&covered).sum()),lost=int((coverage&~covered).sum()),clear_native_good=int(coverage[clear].sum()),clear_good=int(covered[clear].sum()),clear_rescued=int((clear&~coverage&covered).sum()),clear_lost=int((clear&coverage&~covered).sum()),effective_good_histogram={str(k):int((q['effective_good']==k).sum()) for k in np.unique(q['effective_good'])},all_distinct_histogram={str(k):int((q['all_distinct']==k).sum()) for k in np.unique(q['all_distinct'])})
      review_rows.append(row)
      if arm=='zero':
        oldpred=load(ROOT/f'GENERATION_PULSE_RESPONSE_SERIES_20261004/results/seed{seed}_mean_std_{policy}.npz')
        assert np.array_equal(ids,oldpred['sample_id'])
        repeated.append(dict(seed=seed,policy=policy,trajectory_bitwise_equal=bool(np.array_equal(path,oldpred['trajectory'])),max_error=float(np.abs(path-oldpred['trajectory']).max())))
allstudies=[]
for name in ('GENERATION_TIME_RESPONSE_SERIES_20261004','GENERATION_PULSE_RESPONSE_SERIES_20261004',SERIES.name):
  folder=ROOT/name;s=read(folder/'results/summary.json');i=read(folder/'results/independent_audit.json');r=read(folder/'results/saved_metrics_recompute.json')
  assert i['status']==r['status']=='PASS'
  hs=[]
  for arm in ('zero','mean_std'):
    for seed in (1,2,3):
      hfolder=folder/f'results/models/{arm}/seed_{seed}/balanced_protected'
      h=rows(hfolder/'history.csv');assert len(h)==80
      comp=read(hfolder/'complete.json');assert comp['epochs']==80
      hs.append(dict(arm=arm,seed=seed,first_loss=float(h[0]['train_loss']),last_loss=float(h[-1]['train_loss']),cal_infeasible=sum(x['cal_feasible'].lower()=='false' for x in h),complete=comp,CAL_final_change_ADE=float(h[-1]['change_minade20']),CAL_final_change_FDE=float(h[-1]['change_minfde20'])))
  allstudies.append(dict(folder=name,summary=s,histories=hs,independent_audit=i,saved_metrics_recompute=r))
save('ALL_THREE_POLICY_GROUPS_AND_CIS.json',allstudies)
verify(INPUTS,'direct-inputs-after');verify(old,'all27164prior-after');verify(sources,'source-after')
elapsed=(datetime.fromisoformat(state['finished_at'])-datetime.fromisoformat(state['started_at'])).total_seconds()
report=dict(status='PASS_SAVED_OUTPUT_SUPPLEMENT',no_model_forward=True,no_optimizer=True,no_new_data=True,independent_recomputed_min_max_error=maxerror,quality_counts_and_slots_recomputed=True,complete_link_recluster_scope='Production worker independently audited; this supplement verifies saved quality counts, not fresh all-cluster reconstruction',old_files_before_after=len(old),scientific_sources_before_after=len(sources),model_locks=len(locks),saved_predictions=18,explicit_IDs=1516,K20_shape=[1516,20,12,2],native_good=733,native_effective_good_total=quality['native_effective_good_total'],native_inherited_selected_good=int(ng[np.arange(1516),selected].sum()),clear_event469_native_good=int(coverage[clear].sum()),rows=review_rows,histories=history,repeated_control=repeated,production_actual_seconds=elapsed,production_resource=read(SERIES/'RESOURCE_USAGE.json'),resource_checks=CHECKS,review_elapsed_seconds=time.perf_counter()-START,source_scope='Saved arithmetic, CSV and SHA; no all-model replay or external independent review')
save('INPUT_HASHES.json',INPUTS);save('REVIEW.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('rows','histories','repeated_control')}),flush=True)
