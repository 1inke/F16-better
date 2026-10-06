"""Registered ALLK physical acceleration, FIT-only scales, no forward GT."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch

def window_accel(path):
    origin=path.new_zeros((*path.shape[:2],1,2))
    velocity=torch.diff(torch.cat((origin,path),2),dim=2)/.4
    return torch.linalg.vector_norm(torch.diff(velocity,dim=2)/.4,dim=-1).mean((1,2))

def numpy_accel(path):
    path=np.asarray(path)
    origin=np.zeros((*path.shape[:2],1,2),dtype=path.dtype)
    velocity=np.diff(np.concatenate((origin,path),2),axis=2)/.4
    return np.linalg.norm(np.diff(velocity,axis=2)/.4,axis=-1).mean((1,2))

def penalty(path,base,groups,scales,details=False):
    current=window_accel(path);native=window_accel(base.detach())
    result=current.sum()*0.;counts=[];active=[];means=[];references=[]
    for group in range(3):
        mask=groups==group;n=int(mask.sum())
        counts.append(n)
        if n:
            mean=current[mask].mean();ref=native[mask].mean()
            gap=mean-1.01*ref
            result=result+(n/len(groups))*gap.relu()/scales[group]
            active.append(bool((gap.detach()>0).item()))
            means.append(float(mean.detach()));references.append(float(ref.detach()))
        else:active.append(False);means.append(None);references.append(None)
    if details:return result,dict(counts=counts,active=active,means=means,native_means=references)
    return result

def stats(base,groups,fit,chunk):
    sums=np.zeros(3,np.float64);counts=np.zeros(3,np.int64)
    for start in range(0,len(fit),chunk):
        idx=fit[start:start+chunk]
        vals=window_accel(torch.from_numpy(base[idx])).numpy().astype(np.float64)
        for group in range(3):
            keep=groups[idx]==group;sums[group]+=vals[keep].sum(dtype=np.float64);counts[group]+=int(keep.sum())
    assert (counts>0).all()
    return np.maximum(sums/counts,.001),counts

def prepare(h,cache,fit,base):
    path=h.HERE/'physical_accel_scales.npz';meta=h.HERE/'physical_accel_scales.json'
    ids=np.ascontiguousarray(cache['sample_id'][fit])
    provenance=dict(fit_n=len(fit),FIT_only=True,no_GT_scale=True,dt=.4,
        ids_SHA=hashlib.sha256(ids.tobytes()).hexdigest(),ids_dtype=str(ids.dtype),
        source_SHA=h.p.sha256(Path(__file__)),floor=.001,groups=['stable','transition','change'])
    if path.exists():
        saved=json.loads(meta.read_text(encoding='utf-8'));assert saved['provenance']==provenance,'scale ID/source mismatch'
        z=np.load(path,allow_pickle=False);assert np.array_equal(z['fit_ids'],ids)
        scale=z['scales'];assert scale.shape==(3,) and np.isfinite(scale).all() and (scale>=.001).all()
        assert h.p.sha256(path)==saved['npz_SHA'];return scale
    assert len(fit)==25912
    scale,counts=stats(base,cache['change_group'],fit,256)
    check=np.concatenate((fit[:517],fit[-516:]))
    a,na=stats(base,cache['change_group'],check,31);b,nb=stats(base,cache['change_group'],check,256)
    assert np.array_equal(na,nb) and np.max(abs(a-b))<1e-12
    scale=scale.astype(np.float32)
    np.savez(path,scales=scale,fit_ids=ids,counts=counts)
    h.m.write_json(meta,dict(status='PASS_FIT_ONLY',provenance=provenance,scales=scale.tolist(),counts=counts.tolist(),
        chunk1033_max=float(np.max(abs(a-b))),npz_SHA=h.p.sha256(path)))
    return scale

def load(here):
    z=np.load(here/'physical_accel_scales.npz',allow_pickle=False)
    return z['scales'].copy()

def fixtures():
    times=torch.arange(1,13,dtype=torch.float64)
    path=torch.zeros(4,20,12,2,dtype=torch.float64)
    path[...,0]=times.square()[None,None]*.01
    native=path*.5;group=torch.tensor([0,0,1,2]);scale=torch.tensor([.1,.2,.3],dtype=torch.float64)
    path.requires_grad_();value,info=penalty(path,native,group,scale,True)
    expected=sum((info['counts'][g]/4)*max(info['means'][g]-1.01*info['native_means'][g],0)/float(scale[g]) for g in range(3))
    assert abs(float(value)-expected)<1e-12
    assert np.max(abs(window_accel(path).detach().numpy()-numpy_accel(path.detach().numpy())))<1e-12
    gradient=torch.autograd.grad(value,path)[0];assert torch.isfinite(gradient).all() and gradient.norm()>0
    direction=torch.linspace(-.01,.01,path.numel(),dtype=torch.float64).reshape_as(path)
    derivative=float((gradient*direction).sum())
    errors=[]
    for eps in (1e-4,1e-5):
        finite=float((penalty(path.detach()+eps*direction,native,group,scale)-penalty(path.detach()-eps*direction,native,group,scale))/(2*eps))
        errors.append(abs(finite-derivative));assert abs(finite-derivative)<1e-7+1e-3*abs(derivative)
    assert float(penalty(native,native,group,scale))==0
    assert float(penalty(native*.9,native,group,scale))==0
    assert float(penalty(native*1.01,native,group,scale))<1e-12
    one=torch.zeros(4,dtype=torch.long);onevalue,oneinfo=penalty(path,native,one,scale,True)
    assert oneinfo['counts']==[4,0,0] and not any(oneinfo['active'][1:])
    assert abs(float(onevalue)-(info['means'][0]-1.01*info['native_means'][0])/.1)<1e-12
    return dict(status='PASS',numpy_max=0.,finite_difference_errors=errors,
        native_zero=True,below_zero=True,tolerance=True,empty_groups=True,mixed_weights=True)
