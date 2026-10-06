"""Read-only single failed-block diagnosis and preservation SHA review.
No model creation/forward/optimizer, no tolerance change, no task launch.
"""
import os, sys
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
from datetime import datetime
import json, hashlib, importlib.util, time
import numpy as np
import torch
torch.set_num_threads(2)
ROOT = Path('C:/Users/linke/Documents/New project')
HERE = ROOT/'F16_CURVE_REACHABILITY_DIAGNOSIS_20261004'
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()
locks = read(HERE/'preflight.json')['source_hashes']
source_mismatch = [p for p,s in locks.items() if sha(p)!=s]
assert not source_mismatch, source_mismatch
spec = importlib.util.spec_from_file_location('locked_curve_failure_diagnosis', HERE/'diagnosis.py')
d = importlib.util.module_from_spec(spec);sys.modules[spec.name] = d;spec.loader.exec_module(d)
# Load only the immutable train arrays needed for the single failing block.
with np.load(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz', allow_pickle=False) as z:
    ids = z['sample_id'];motion = z['motion'];candidate = z['candidate']
with np.load(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz', allow_pickle=False) as z:
    assert np.array_equal(ids, z['sample_id'])
    future = z['future']
past = motion[:,:16].reshape(-1,8,2)
speed = np.linalg.norm(past[:,-1]-past[:,-2],axis=-1).astype(np.float32)
L = (1.5*(speed+.03)).astype(np.float32).astype(float)
# Confirm the unchanged speed formula against already saved production blocks.
limit_max = 0.0
for name in ('FIT_00000_00064', 'FIT_10880_10944'):
    with np.load(HERE/'chunks'/(name+'.npz'), allow_pickle=False) as z:
        ix = np.searchsorted(ids, z['sample_id'])
        assert np.array_equal(ids[ix],z['sample_id'])
        limit_max = max(limit_max,float(np.max(np.abs(L[ix]-z['limit']))))
assert limit_max == 0
ix = np.arange(10944,11008)
base = candidate[ix,...,:24].reshape(-1,20,12,2);gt = future[ix]
began = time.perf_counter();z = d.compute(base,gt,L[ix])
x = z['pulse_params'].reshape(-1,6);ell = np.repeat(L[ix],20)
delta,u,profile = d.production_decode(x,ell)
float_error = float(np.max(np.abs(d.decode(x,ell)-delta)))
zerosum = float(np.max(np.abs(profile.sum(-1))))
arg = int(np.argmax(np.abs(profile.sum(-1))))
audit_exception = None
try:
    d.audit(base,gt,L[ix],z)
except AssertionError:
    audit_exception = 'AssertionError'
report = dict(status='FAILURE_DIAGNOSIS_ONLY_NOT_COMPLETION',
    reviewed_at=datetime.now().astimezone().isoformat(),
    failed_block='FIT_10944_11008',rows=64,immutable_computation_seconds=time.perf_counter()-began,
    model_creation=False,model_forward=False,model_optimizer=False,DEV_access=False,
    scientific_sources_modified=False,scientific_protocol_or_tolerance_changed=False,
    limit_formula_saved_blocks_exact_max_error=limit_max,
    failed_block_reproduced_assertion=audit_exception,
    float32_vs_float64_path_max_m=float_error,float_path_threshold_m=2e-5,
    float_path_clause_pass=float_error<2e-5,
    profile_zero_sum_max=zerosum,zero_sum_threshold=6e-6,
    zero_sum_clause_pass=zerosum<6e-6,
    max_zero_sum_sample_id=int(ids[ix[arg//20]]),max_zero_sum_candidate_slot=arg%20,
    max_zero_sum_parameters=x[arg].tolist(),
    endpoint_decomposition_max_m=float(np.max(np.abs(delta[:,-1]-6*ell[:,None]*x[:,:2]))))
print('FAILED_BLOCK_MEASUREMENTS '+json.dumps(report),flush=True)
done = sorted(p for p in (HERE/'chunks').glob('*.done.json') if p.name.startswith(('FIT_','CAL_')))
mismatches=[];outputs=0;rows=0
for p in done:
    meta=read(p);assert meta['source_hashes']==locks
    assert meta['status']=='SUCCESS';rows+=meta['rows']
    for path,digest in meta['outputs'].items():
        outputs+=1
        if sha(path)!=digest:mismatches.append(path)
report.update(scientific_done_blocks=len(done),scientific_rows_saved=rows,
    done_output_files_verified=outputs,done_output_SHA_mismatch=mismatches,
    saved_source_identity_exact=True,synthetic_resume_fixture_excluded=True)
old=read(HERE/'old_results_manifest.json')['files'];old_mismatch=[]
began=time.perf_counter()
for n,(p,digest) in enumerate(old.items(),1):
    if sha(p)!=digest:old_mismatch.append(p)
    if n%4000==0:print('PRESERVATION_SHA '+str(n)+'/'+str(len(old)),flush=True)
source_mismatch=[p for p,s in locks.items() if sha(p)!=s]
report.update(old_files_verified=len(old),old_SHA_mismatch=old_mismatch,
    source_files_verified=len(locks),source_SHA_mismatch=source_mismatch,
    preservation_hash_seconds=time.perf_counter()-began,
    full_population_arithmetic_audit_completed=False,full_population_frozen_replay_audit_completed=False,
    population_classification_conclusion=None,automatic_restart=False,
    interpretation='Numeric audit failure is not evidence of model quality or RAM failure. Single-block fitting is a failure reproduction, not a completed diagnosis or deployable result.')
print('FINAL_FAILURE_REVIEW '+json.dumps(report),flush=True)
