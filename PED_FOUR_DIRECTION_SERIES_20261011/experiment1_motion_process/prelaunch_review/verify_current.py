"""CPU current-source and actual saved-smoke review, no torch/optimizer."""
import importlib.util,json,time,hashlib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent.parent
s=importlib.util.spec_from_file_location('saved_cpu_audit',H/'completion_review/audit_saved.py')
a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
start=time.perf_counter();a.guard();pre=a.read(H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30
files=a.read(H/'source_manifest.json')['files'];a.verify(files);assert a.digest(files)==pre['source_sha']
cache=next(Path(k) for k in files if k.endswith('results\\cache\\train_dev\\eth_train.npz'))
split=next(Path(k) for k in files if k.endswith('GENERATION_GROUPED_TRANSFER_SERIES_20261006\\identity_audit\\splits.npz'))
c=a.npz(cache);sp=a.npz(split);index=sp['outer_eval_index'][:128];checks=[]
for arm in a.ARMS:
    path=H/'prelaunch_review'/f'{arm}_smoke128.npz';z=a.npz(path)
    assert np.array_equal(z['sample_id'],c['sample_id'][index])
    assert np.array_equal(z['native20'],c['candidate'][index,...,:24].reshape(-1,20,12,2).astype(float))
    met,assignment,vectors,clusters,bounds=a.audit_job(z,arm)
    assert vectors.shape==(128,3,4) and clusters.shape==(128,3,2)
    checks.append(dict(arm=arm,saved128_ID_nativecache_controls_fixed16_bound_fullquality_clusters_motion=True,smoke_sha=a.sha(path),not_scientific_result=True))
extra={str(p):a.sha(p) for p in [Path(__file__),H/'completion_review/audit_saved.py']}
a.verify(files);a.verify(extra);a.guard()
a.write('PRELAUNCH_CPU_REVIEW.json',dict(status='PASS',checks=checks,locked_scientific_files=len(files),extra_review_source_sha=extra,initial_final_SHA_PASS=True,no_torch_or_model_or_optimizer=True,resource=a.resource,seconds=time.perf_counter()-start))
print(json.dumps(dict(status='PASS',locked_files=len(files),seconds=time.perf_counter()-start,extra_sources=extra)),flush=True)
