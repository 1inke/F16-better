"""Native predicted-state32; fixed FIT-only scalers, chunk-bounded, no GT."""
from pathlib import Path
import json,math
import numpy as np
import torch
from resource_guard import guard

def state32(base):
    assert base.ndim==4 and base.shape[1:]==(20,12,2)
    v=torch.diff(base,dim=2,prepend=torch.zeros_like(base[:,:,:1]))/.4
    a=torch.diff(v,dim=2)/.4
    first,last=v[:,:,0],v[:,:,-1]
    cross=first[...,0]*last[...,1]-first[...,1]*last[...,0]
    dot=(first*last).sum(-1)
    heading=torch.atan2(cross,dot)
    heading=torch.where((first.norm(dim=-1)<.03)|(last.norm(dim=-1)<.03),torch.zeros_like(heading),heading)
    result=torch.cat((v.flatten(2),v.mean(2),last,base[:,:,-1],a.norm(dim=-1).mean(2)[...,None],heading[...,None]),-1)
    assert result.shape==(*base.shape[:2],32) and torch.isfinite(result).all()
    return result

def load_scalers(root):
    with np.load(Path(root)/'state_scalers.npz',allow_pickle=False) as saved:
        return saved['mean'].copy(),saved['std'].copy()

def prepare_scalers(root,sample_ids,fit,base,sha):
    root=Path(root);path=root/'state_scalers.npz';metadata=root/'state_scalers.json'
    provenance=dict(features_source_SHA=sha(Path(__file__)),fit_ids_n=len(fit),candidate_points=len(fit)*20,
                    FIT25912_only=True,GT_used=False,dt=.4,std_floor=.001,chunk_rows=256)
    assert len(fit)==25912 and len(np.unique(sample_ids[fit]))==25912
    if path.exists():
        report=json.loads(metadata.read_text(encoding='utf-8'))
        assert report['provenance']==provenance and report['scalers_SHA']==sha(path)
        with np.load(path,allow_pickle=False) as saved:
            assert np.array_equal(saved['fit_id'],sample_ids[fit])
        return
    total=np.zeros(32,np.float64);squares=total.copy();count=0
    minimum=np.full(32,np.inf);maximum=np.full(32,-np.inf)
    # Features evaluated in production float32 arithmetic, statistics in float64.
    with torch.no_grad():
        for start in range(0,len(fit),256):
            guard.check();ids=fit[start:start+256]
            x=state32(torch.from_numpy(base[ids])).numpy().reshape(-1,32).astype(np.float64)
            total+=x.sum(0);squares+=(x*x).sum(0);count+=len(x)
            minimum=np.minimum(minimum,x.min(0));maximum=np.maximum(maximum,x.max(0))
    mean=total/count;std=np.sqrt(np.maximum(0.,squares/count-mean*mean))
    assert count==518240 and np.isfinite(mean).all() and np.isfinite(std).all()
    assert (std>1e-7).all(),('nonvarying registered state dimension',std.tolist())
    m32=mean.astype(np.float32);s32=np.maximum(std,.001).astype(np.float32)
    np.savez_compressed(path,mean=m32,std=s32,fit_id=sample_ids[fit])
    report=dict(status='PASS_FIT_ONLY_SCALERS',provenance=provenance,scalers_SHA=sha(path),
                raw_std=std.tolist(),minimum=minimum.tolist(),maximum=maximum.tolist(),
                deterministic_redundancies=True,full_rank_claim=False)
    metadata.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FIT_STATE_SCALERS PASS '+json.dumps(dict(points=count,raw_std_min=float(std.min()),raw_std_max=float(std.max()))),flush=True)

def validate_state_math():
    # Independent scalar numpy equations: stationary, constant motion and turn.
    base=np.zeros((3,20,12,2),np.float32)
    base[1,...,0]=np.arange(1,13,dtype=np.float32)*.4
    base[2,...,0]=np.cumsum(np.linspace(.4,.02,12,dtype=np.float32))
    base[2,...,1]=np.cumsum(np.linspace(.02,.4,12,dtype=np.float32))
    expected=np.empty((3,20,32),np.float64)
    for b in range(3):
        for k in range(20):
            positions=base[b,k].astype(np.float64)
            velocity=np.diff(positions,axis=0,prepend=np.zeros((1,2)))/.4
            accel=np.diff(velocity,axis=0)/.4
            f,l=velocity[0],velocity[-1]
            h=0. if min(np.linalg.norm(f),np.linalg.norm(l))<.03 else math.atan2(f[0]*l[1]-f[1]*l[0],np.dot(f,l))
            expected[b,k]=np.r_[velocity.ravel(),velocity.mean(0),l,positions[-1],np.linalg.norm(accel,axis=-1).mean(),h]
    actual=state32(torch.from_numpy(base)).numpy()
    error=float(np.max(np.abs(actual-expected)))
    assert error<2e-5 and np.count_nonzero(actual[0])==0
    assert np.allclose(actual[1,:,24:28],1.*np.array([1,0,1,0]),atol=2e-5)
    assert (actual[2,:,-1]>.9).all()
    return dict(status='PASS_INDEPENDENT_SCALAR_MATH',max_error=error,order_locked=True,no_GT_inputs=True)
