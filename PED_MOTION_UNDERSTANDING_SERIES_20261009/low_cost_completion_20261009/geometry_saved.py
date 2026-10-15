"""Bounded, saved-only E5 set geometry supplement. No torch or model imports."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, time, json, hashlib, ctypes, traceback
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import psutil
from scipy.optimize import linear_sum_assignment

H = Path(__file__).resolve().parent
S = H.parent / 'stage2_raw_interventions'
START = time.perf_counter()
RESOURCE = dict(peak_private_GiB=0., peak_RSS_GiB=0., min_physical_GiB=100., min_commit_GiB=100., checks=0)
LOCKS = {}
RECORDS = []

class Mem(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(k, ctypes.c_ulonglong) for k in ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended')]

def guard():
    m = Mem(); m.length = ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    p = psutil.Process().memory_info()
    RESOURCE['peak_private_GiB'] = max(RESOURCE['peak_private_GiB'], p.peak_pagefile / 2**30)
    RESOURCE['peak_RSS_GiB'] = max(RESOURCE['peak_RSS_GiB'], p.peak_wset / 2**30)
    RESOURCE['min_physical_GiB'] = min(RESOURCE['min_physical_GiB'], m.avail_phys / 2**30)
    RESOURCE['min_commit_GiB'] = min(RESOURCE['min_commit_GiB'], m.avail_page / 2**30)
    RESOURCE['checks'] += 1
    if max(p.private, p.rss) > 6*2**30 or min(m.avail_phys, m.avail_page) < 6*2**30:
        raise MemoryError('STOP_RESOURCE_6GiB')
    if time.perf_counter() - START > 120:
        raise TimeoutError('STOP_SHORT_BUDGET_120_SECONDS')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda: f.read(1048576), b''): h.update(x)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(name, value):
    (H/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def lock(path, expected=None):
    path = Path(path).resolve(); got = sha(path)
    assert expected is None or expected == got, ('SHA', str(path))
    assert str(path) not in LOCKS or LOCKS[str(path)] == got
    LOCKS[str(path)] = got

def costs(base, changed):
    return np.linalg.norm(base[:, None] - changed[None], axis=-1).mean(-1)

def fixtures():
    p = np.arange(20*12*2, dtype=float).reshape(20,12,2) / 100
    for q in (p, p[::-1]):
        d = costs(p, q); r,c = linear_sum_assignment(d)
        assert d[r,c].mean() == 0
        assert .5*(d.min(0).mean()+d.min(1).mean()) == 0
    assert np.trace(costs(p,p[::-1])) > 0
    q = p + np.array([.3, .4])
    d = costs(p,q); r,c = linear_sum_assignment(d)
    assert abs(d[r,c].mean()-.5) < 1e-12

def run():
    guard(); fixtures()
    for path in (Path(__file__), H/'DESIGN.md', S/'source_manifest.json'):
        lock(path)
    source = sha(S/'source_manifest.json')
    for fold in ('eth','hotel','univ','zara01','zara02'):
        for seed in (1,2,3):
            for name in ('rotate','mirror','translate','scale08','scale12'):
                guard(); job = S/f'jobs/{fold}_seed{seed}/{name}'
                done = read(job/'done.json'); lock(job/'done.json')
                assert done['status'] == 'SUCCESS' and done['source_sha'] == source
                assert done['fold']==fold and done['seed']==seed and done['intervention']==name
                path = job/'outputs.npz'; lock(path,done['outputs'][str(path)])
                with np.load(path,allow_pickle=False) as z:
                    base = z['base_paths'].astype(float); changed = z['paths'].astype(float)
                    ids=z['sample_id'].copy(); masks={k[5:]:z[k].astype(bool) for k in z.files if k.startswith('mask_')}
                n=len(ids)
                assert base.shape==changed.shape==(n,20,12,2) and len(np.unique(ids))==n
                assert np.isfinite(base).all() and np.isfinite(changed).all()
                vals=np.empty((n,5)); assignments=np.empty((n,20),np.int16)
                for start in range(0,n,128):
                    guard()
                    end=min(n,start+128)
                    dist=np.linalg.norm(base[start:end,:,None]-changed[start:end,None,:],axis=-1).mean(-1)
                    for offset,d in enumerate(dist):
                        i=start+offset; r,c=linear_sum_assignment(d); assignments[i]=c
                        same=np.trace(d)/20; matched=d[r,c].mean()
                        assert matched<=same+1e-12
                        nearest=.5*(d.min(0).mean()+d.min(1).mean())
                        assert nearest<=matched+1e-12
                        # Rebuild each selected edge without the broadcast cost tensor.
                        direct=np.linalg.norm(base[i,r]-changed[i,c],axis=-1).mean()
                        assert abs(direct-matched)<=1e-12
                        ep=np.linalg.norm(base[i,r,-1]-changed[i,c,-1],axis=-1).mean()
                        vals[i]=(same,matched,nearest,ep,float((c!=r).mean()))
                dest=f'{fold}_seed{seed}_{name}.npz'
                np.savez_compressed(H/dest,sample_id=ids,measurements=vals,assignment=assignments)
                groups={}
                for group,mask in masks.items():
                    assert mask.shape==(n,)
                    groups[group]=dict(n=int(mask.sum()),means=vals[mask].mean(0).tolist() if mask.any() else None)
                RECORDS.append(dict(fold=fold,seed=seed,intervention=name,n=n,groups=groups,
                    measurement_columns=['same_slot_path_m','matched_path_m','symmetric_nearest_path_m','matched_endpoint_m','moved_slot_fraction'],
                    all_p95=np.quantile(vals,.95,axis=0).tolist(),output=dest,output_sha=sha(H/dest)))
                print(f'{len(RECORDS)}/75 {fold} {seed} {name}',flush=True)
    write('input_manifest.json',LOCKS)
    for path,digest in LOCKS.items():
        guard(); assert sha(path)==digest,('FINAL_SHA',path)
    for record in RECORDS: assert sha(H/record['output'])==record['output_sha']
    result=dict(status='PASS_SAVED_ONLY_GEOMETRY_SET_SUPPLEMENT',jobs=len(RECORDS),records=RECORDS,
        input_files=len(LOCKS),initial_final_SHA_PASS=True,fixtures_PASS=True,selected_edge_arithmetic_recomputed=True,
        no_torch_forward_optimizer=True,no_GT_matching=True,not_external_review=True,
        descriptive_after_outcomes=True,resource=RESOURCE,elapsed_seconds=time.perf_counter()-START)
    write('RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try: run()
    except Exception:
        write('FAILURE.json',dict(status='STOP',traceback=traceback.format_exc(),completed_records=RECORDS,
                                 resource=RESOURCE,elapsed_seconds=time.perf_counter()-START))
        raise
