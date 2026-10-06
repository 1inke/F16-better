"""Fixed repair synthetic gate; no old model/data replay or optimizer."""
from pathlib import Path
import hashlib,json,math,time,ctypes
from datetime import datetime,timezone,timedelta
import numpy as np
import torch,psutil
from projection import project
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
def resource():
    p=psutil.Process().memory_info();m=Mem();m.length=ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    assert max(p.rss,p.private)<=6*2**30 and min(m.avail_phys,m.avail_page)>=6*2**30,'STOP_RESOURCE'
    return dict(peak_RSS_GiB=p.peak_wset/2**30,peak_private_GiB=p.peak_pagefile/2**30,physical_available_GiB=m.avail_phys/2**30,commit_available_GiB=m.avail_page/2**30)
def accel(p):
    p=np.asarray(p,np.float64);v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4
    return np.linalg.norm(np.diff(v,axis=2)/.4,axis=-1).mean((1,2))
def reference(b,u):
    b=b.astype(np.float64);u=u.astype(np.float64)
    mb=accel(b);d=np.linalg.norm(np.diff(u,axis=2)/.16,axis=-1).mean((1,2))
    a=np.divide(.01*mb,d,out=np.ones_like(d),where=d!=0).clip(0,1)
    mean=u.mean(2,keepdims=True);up=mean+a[:,None,None,None]*(u-mean)
    return b+np.cumsum(up,axis=2)
def fixtures():
    zero=np.zeros((1,20,12,2),np.float32)
    const=np.broadcast_to(np.array([.01,.02],np.float32),zero.shape).copy()
    wave=np.broadcast_to(np.linspace(-.02,.02,12,dtype=np.float32)[None,None,:,None],zero.shape).copy()
    step=np.arange(1,13,dtype=np.float32)[None,None,:,None]
    moving=np.broadcast_to(step*np.array([.0625,.125],np.float32),zero.shape).copy()
    curved=moving.copy();curved[:,:,:,0]+=step[0,0,:,0]**2*.001
    rows=[('identity',zero,zero),('static_constant_drift_D0',zero,const),('static_varying_Dpositive',zero,const+wave),('constant_velocity_D0',moving,const),('curved_varying',curved,const+wave)]
    aligned=np.zeros_like(zero);aligned[:,:,:,0]=step[0,0,:,0]**2*.001
    delta=np.zeros_like(zero);delta[:,:,:,0]=np.arange(12,dtype=np.float32)*.001
    rows.extend([('aligned_tight',aligned,delta),('opposite',aligned,-delta)])
    rng=np.random.default_rng(20261006);velocity=rng.uniform(-.1,.1,(64,20,12,2)).astype(np.float32)
    base=velocity.cumsum(2);u=rng.uniform(-.04,.04,base.shape).astype(np.float32)
    base[:4]=0;rows.append(('fixed_random64',base,u))
    return rows
def main():
    start=time.perf_counter();res=[resource()]
    files=[HERE/x for x in ('README.md','REPAIR_DESIGN.md','projection.py','numeric_preflight.py')]
    before={str(p):sha(p) for p in files}
    original=json.loads((PARENT/'failures/numeric_gate_1/REPORT.json').read_text(encoding='utf-8'))
    for p,h in original['source_after'].items():assert sha(p)==h
    rows=[];arrays={}
    for device in ('cpu','cuda:0'):
        assert device=='cpu' or torch.cuda.is_available()
        for name,b,u in fixtures():
            tb=torch.from_numpy(b).to(device);tu=torch.from_numpy(u).to(device)
            with torch.no_grad():path,up,part=project(tb,tu)
            p=path.cpu().numpy();uu=up.cpu().numpy()
            mb=accel(b);mp=accel(p)
            endpoint=float(np.abs((p-b.astype(np.float64))[:,:,-1]-u.astype(np.float64).sum(2)).max())
            referr=float(np.abs(p-reference(b,u)).max())
            clip=float(max(0.,np.abs(uu).max()-max(.045,float(np.abs(u).max()))))
            flags=dict(strict_guard=bool(np.all(mp<=1.01*mb)),endpoint=endpoint<=2e-5,reference=referr<=2e-5,clip=clip==0)
            adiff=float((part['alpha_nominal']-part['alpha']).max().cpu())
            rows.append(dict(device=device,fixture=name,windows=len(b),flags=flags,mb=mb.tolist(),mp=mp.tolist(),endpoint_error=endpoint,reference_error=referr,clip_excess=clip,alpha_roundoff_difference=adiff,roundoff_reserve_max=float(part['roundoff_reserve'].max().cpu())))
            key=device.replace(':','_')+'_'+name
            arrays.update({key+'_base':b,key+'_u':u,key+'_path':p,key+'_up':uu})
            res.append(resource())
        bad=torch.full((1,20,12,2),float('nan'),device=device)
        try:project(bad,torch.zeros_like(bad))
        except AssertionError:pass
        else:raise AssertionError('badinput not rejected')
    after={str(p):sha(p) for p in files};assert before==after
    for p,h in original['source_after'].items():assert sha(p)==h
    failures=[r for r in rows if not all(r['flags'].values())]
    dst=HERE/('failures/repair_numeric_gate1' if failures else 'numeric_preflight');dst.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dst/'fixtures.npz',**arrays)
    report=dict(status='STOP_NUMERIC_REPAIR_WAIT_HUMAN' if failures else 'PASS_NUMERIC_REPAIR_SYNTHETIC_ONLY',at=datetime.now(timezone(timedelta(hours=8))).isoformat(),seconds=time.perf_counter()-start,rows=rows,failed_count=len(failures),source_hashes=before,source_after=after,parent_failed4_sources_unchanged=True,fixture_sha=sha(dst/'fixtures.npz'),resource=res,model_replay=False,production_started=False,all_prior_SHA_audit=False)
    (dst/'REPORT.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],failed_count=len(failures),rows=len(rows),max_endpoint=max(r['endpoint_error'] for r in rows),max_reference=max(r['reference_error'] for r in rows),max_alpha_roundoff=max(r['alpha_roundoff_difference'] for r in rows),memory=res[-1],report=str(dst/'REPORT.json'))))
    return 2 if failures else 0
if __name__=='__main__':raise SystemExit(main())
