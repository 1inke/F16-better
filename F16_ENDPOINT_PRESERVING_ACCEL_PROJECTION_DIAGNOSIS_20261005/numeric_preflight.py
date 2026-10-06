"""Small first gate before data/model replay. Does not train or modify old files."""
from pathlib import Path
import ctypes
import hashlib
import json
import math
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import psutil
import torch
from projection import project

HERE = Path(__file__).resolve().parent
DT = .4
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

class Mem(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(k, ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]

def resource():
    p = psutil.Process().memory_info()
    v = Mem(); v.length = ctypes.sizeof(v)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(v))
    assert max(p.rss, p.private) <= 6*2**30 and min(v.avail_phys, v.avail_page) >= 6*2**30, 'STOP_RESOURCE'
    return dict(peak_RSS_GiB=p.peak_wset/2**30, peak_private_GiB=p.peak_pagefile/2**30,
                physical_available_GiB=v.avail_phys/2**30, commit_available_GiB=v.avail_page/2**30)

def acceleration_metric(saved):
    # Same geometry definition as the saved-output physical audit, not latent u.
    p = np.asarray(saved, dtype=np.float64)
    velocity = np.diff(p, axis=2, prepend=np.zeros_like(p[:, :, :1])) / DT
    a = np.diff(velocity, axis=2) / DT
    return np.linalg.norm(a, axis=-1).mean((1, 2))

def scalar_reference(base, u):
    b = np.asarray(base, dtype=np.float64); c = np.asarray(u, dtype=np.float64)
    mb = acceleration_metric(b)
    d = np.linalg.norm(np.diff(c, axis=2) / DT**2, axis=-1).mean((1, 2))
    alpha = np.array([1. if y == 0 else min(1., max(0., .01*x/y)) for x,y in zip(mb,d)])
    mu = np.empty((len(b),20,1,2),np.float64)
    for i in range(len(b)):
        for k in range(20):
            for xy in range(2): mu[i,k,0,xy] = math.fsum(float(x) for x in c[i,k,:,xy]) / 12
    up = mu + alpha[:,None,None,None]*(c-mu)
    return b + np.cumsum(up,axis=2), up, alpha

def fixtures():
    zero = np.zeros((1,20,12,2),np.float32)
    constant = np.broadcast_to(np.array([.01,.02],np.float32),zero.shape).copy()
    wave = np.broadcast_to(np.linspace(-.02,.02,12,dtype=np.float32)[None,None,:,None],zero.shape).copy()
    steps = np.arange(1,13,dtype=np.float32)[None,None,:,None]
    moving = np.broadcast_to(steps*np.array([.0625,.125],np.float32),zero.shape).copy()
    curved = moving.copy(); curved[:,:,:,0] += steps[0,0,:,0]**2*.001
    return [('identity',zero,zero), ('static_constant_drift_D0',zero,constant),
            ('static_varying_Dpositive',zero,constant+wave),
            ('constant_velocity_D0',moving,constant), ('curved_varying',curved,constant+wave)]

def main():
    started = time.perf_counter()
    sources = [HERE/'projection.py', HERE/'numeric_preflight.py', HERE/'DESIGN.md', HERE/'README.md']
    initial = {str(p):sha(p) for p in sources}
    rows = []; arrays = {}; resources = [resource()]
    for device in ('cpu','cuda:0'):
        assert device == 'cpu' or torch.cuda.is_available()
        for name, base, u in fixtures():
            b = torch.from_numpy(base).to(device); c = torch.from_numpy(u).to(device)
            with torch.no_grad(): path, up, detail = project(b,c)
            if device.startswith('cuda'): torch.cuda.synchronize()
            p = path.cpu().numpy(); v = up.cpu().numpy()
            reference, ref_u, ref_alpha = scalar_reference(base,u)
            native_metric = acceleration_metric(base); projected_metric = acceleration_metric(p)
            endpoint_error = float(np.abs((p-base)[:,:,-1] - u.astype(np.float64).sum(2)).max())
            path_error = float(np.abs(p-reference).max())
            # Static speed limit is .045; all fixtures use increments within it.
            clip_error = float(max(0., np.abs(v).max()-.045))
            strict = bool(np.all(projected_metric <= 1.01*native_metric))
            row = dict(device=device,fixture=name,alpha=detail['alpha'].cpu().tolist(),
                       independent_alpha=ref_alpha.tolist(),native_saved_acceleration_mean=native_metric.tolist(),
                       projected_saved_acceleration_mean=projected_metric.tolist(),
                       strict_one_percent_bound_pass=strict,endpoint_error_m=endpoint_error,
                       independent_path_error_m=path_error,clip_excess_m=clip_error,
                       consistency_2e_5_pass=endpoint_error<=2e-5 and path_error<=2e-5,
                       reference_saved_acceleration_mean=acceleration_metric(reference).tolist())
            rows.append(row)
            key=device.replace(':','_')+'_'+name
            arrays[key+'_base']=base; arrays[key+'_u']=u; arrays[key+'_path']=p; arrays[key+'_projected_u']=v
            resources.append(resource())
        bad = torch.full((1,20,12,2),float('nan'),device=device)
        rejected=False
        try: project(bad,torch.zeros_like(bad))
        except AssertionError: rejected=True
        assert rejected, 'bad input not rejected'
    final = {str(p):sha(p) for p in sources}
    assert initial == final
    failures = [r for r in rows if not r['strict_one_percent_bound_pass'] or not r['consistency_2e_5_pass'] or r['clip_excess_m']>0]
    status = 'STOP_NUMERIC_STRICT_GUARD_WAIT_HUMAN' if failures else 'PASS_FIRST_NUMERIC_GATE_ONLY'
    out = HERE/'failures'/'numeric_gate_1' if failures else HERE/'numeric_preflight'
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'fixtures.npz',**arrays)
    report=dict(status=status,time=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                elapsed_seconds=time.perf_counter()-started,process_id=psutil.Process().pid,
                rows=rows,failed_count=len(failures),source_before=initial,source_after=final,
                source_unchanged=True,fixture_output_sha=sha(out/'fixtures.npz'),resources=resources,
                production_started=False,model_replay_performed=False,old_files_read_or_written=False,
                complete_old_manifest_SHA_audit_performed=False,no_optimizer=True,
                explanation='2e-5 consistency tolerance is NOT added to strict 1% physical gate. No coefficient/floor/threshold changed.')
    (out/'REPORT.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=status,failed_count=len(failures),report=str(out/'REPORT.json'),elapsed_seconds=report['elapsed_seconds'],resources=resources[-1])))
    return 2 if failures else 0

if __name__=='__main__': raise SystemExit(main())
