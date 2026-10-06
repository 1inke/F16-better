"""Preregistered past/native-only FIT median route; no GT/group/logits."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from resource_guard import guard
def statistic(observation,base):
    past=observation[:,:16].reshape(-1,8,2)
    observed=(past[:,-1]-past[:,-2])/.4
    velocity=torch.diff(base,dim=2,prepend=torch.zeros_like(base[:,:,:1]))/.4
    return (velocity-observed[:,None,None]).square().sum(-1).mean(-1).sqrt()
def load(root):
    with np.load(Path(root)/'route_threshold.npz',allow_pickle=False) as z:return z['median'].copy()
def prepare(h,cache,fit,values):
    path=h.HERE/'route_threshold.npz';meta=h.HERE/'route_threshold.json'
    ids=cache['sample_id'][fit]
    provenance=dict(source_SHA=h.p.sha256(Path(__file__)),fit_ids_SHA=hashlib.sha256(ids.tobytes()).hexdigest(),
        points=518240,dt=.4,q_dtype='float32',median_dtype='float64',chunk=256,ties_expert0=True,no_GT=True)
    if path.exists():
        report=json.loads(meta.read_text(encoding='utf-8'));assert report['provenance']==provenance
        assert h.p.sha256(path)==report['npz_SHA']
        with np.load(path,allow_pickle=False) as z:assert np.array_equal(z['fit_id'],ids)
        return report
    parts=[]
    for start in range(0,len(fit),256):
        guard.check();idx=fit[start:start+256]
        parts.append(statistic(torch.from_numpy(values[0][idx]),torch.from_numpy(values[1][idx])).numpy())
    q=np.concatenate(parts);assert q.shape==(25912,20) and np.isfinite(q).all()
    median=np.median(q.astype(np.float64))
    route=q.astype(np.float64)>median;counts=[int((~route).sum()),int(route.sum())]
    assert min(counts)>=.1*q.size,('STOP degenerate FIT routing',counts)
    check=np.r_[fit[:517],fit[-516:]]
    def chunks(n):
        return np.concatenate([statistic(torch.from_numpy(values[0][check[s:s+n]]),torch.from_numpy(values[1][check[s:s+n]])).numpy() for s in range(0,len(check),n)])
    a,b=chunks(31),chunks(256);assert np.array_equal(a,b)
    assert np.median(a.astype(np.float64))==np.median(b.astype(np.float64))
    scene_counts={}
    # Scene/track provenance from existing raw train dataset only; no future features.
    for key in ('scene','scene_id','scene_name','scene_family'):
        if key in cache and np.asarray(cache[key]).ndim==1:
            for label in np.unique(cache[key][fit]):
                mask=cache[key][fit]==label;scene_counts[str(label)]=[int((~route[mask]).sum()),int(route[mask].sum())]
    np.savez(path,median=np.asarray(median,np.float64),fit_id=ids)
    report=dict(status='PASS_FIT_ONLY_ROUTE',provenance=provenance,median=float(median),candidate_counts=counts,
        windows_with_branch=[int((~route).any(-1).sum()),int(route.any(-1).sum())],ties=int((q.astype(np.float64)==median).sum()),
        per_slot_expert1=route.sum(0).tolist(),scene_counts=scene_counts,chunk_exact=True,npz_SHA=h.p.sha256(path))
    h.m.write_json(meta,report);return report
def fixtures():
    obs=torch.zeros(3,940);base=torch.zeros(3,20,12,2)
    obs[1,:16]=torch.tensor([[0.,0.],[.4,0.],[.8,0.],[1.2,0.],[1.6,0.],[2.,0.],[2.4,0.],[2.8,0.]]).flatten()
    base[1,:,:,0]=torch.arange(1,13)*.4;base[2,:,:,0]=torch.arange(1,13)*.2
    actual=statistic(obs,base).numpy()
    expected=np.zeros_like(actual,dtype=np.float64)
    for i in range(3):
        past=obs[i,:16].numpy().astype(np.float64).reshape(8,2);vobs=(past[-1]-past[-2])/.4
        for k in range(20):
            velocity=np.diff(base[i,k].numpy().astype(np.float64),axis=0,prepend=np.zeros((1,2)))/.4
            expected[i,k]=np.sqrt(np.mean(np.sum((velocity-vobs)**2,axis=-1)))
    error=float(np.max(abs(actual-expected)));assert error<2e-5 and np.max(actual[1])<2e-5
    assert ((torch.tensor([1.,2.,3.],dtype=torch.float64)>2).long()==torch.tensor([0,0,1])).all()
    return dict(status='PASS',scalar_max=error,constant_motion_near_zero=True,tie_to_expert0=True)
