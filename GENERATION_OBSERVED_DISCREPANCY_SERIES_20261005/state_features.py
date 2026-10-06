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

def observed_reference(observation):
    assert observation.ndim==2 and observation.shape[1]==940
    past=observation[:,:16].reshape(-1,8,2)
    velocity=(past[:,-1]-past[:,-2])/.4
    result=torch.cat((velocity[:,None].expand(-1,12,-1).flatten(1),
                      velocity,velocity,4.8*velocity,torch.zeros_like(velocity)),-1)
    assert result.shape==(len(observation),32) and torch.isfinite(result).all()
    return result[:,None]

def validate_reference_math():
    obs=np.zeros((4,940),np.float32)
    obs[1,:16]=(np.arange(8)[:,None]*np.array([.4,-.2])).ravel()
    obs[2,:16]=(np.arange(8)[:,None]*np.array([-.08,.12])+np.array([10.,-12.])).ravel()
    obs[3,:16]=np.array([3.,-7.]*8,np.float32)
    actual=observed_reference(torch.from_numpy(obs)).numpy()[:,0]
    expected=np.zeros_like(actual,dtype=np.float64)
    for row in range(4):
        past=obs[row,:16].reshape(8,2).astype(np.float64)
        v=(past[-1]-past[-2])/.4
        expected[row]=np.r_[np.tile(v,12),v,v,4.8*v,0.,0.]
    error=float(np.max(np.abs(actual-expected)))
    assert error<2e-5 and np.count_nonzero(actual[[0,3]])==0
    assert np.count_nonzero(actual[:,-2:])==0
    # Zero future discrepancy before common rawmu/std normalization.
    constant=np.zeros((4,20,12,2),np.float32)
    for row in range(4):constant[row]=np.arange(1,13)[:,None]*.4*expected[row,24:26]
    raw=state32(torch.from_numpy(constant));delta=raw-torch.from_numpy(actual[:,None])
    e=float(delta[...,:30].abs().max());assert e<2e-5
    assert torch.equal(raw[...,-2:],delta[...,-2:])
    shifted=torch.from_numpy(obs.copy());shifted[:,:16]+=2.
    translation_error=float((observed_reference(shifted)-observed_reference(torch.from_numpy(obs))).abs().max())
    assert translation_error<2e-5
    return dict(status='PASS_INDEPENDENT_REFERENCE30',scalar_error=error,constant_motion_error=e,
                coordinate_translation_error=translation_error,centered_dimensions=30,
                final2_unchanged=True,zero_observed_velocity_zero_reference=True,
                common_raw_standardization_not_claimed_zero=True)

def fit_discrepancy_numeric(root,values,fit):
    mean,std=load_scalers(root);total=np.zeros(32);squares=total.copy();count=0
    minimum=np.full(32,np.inf);maximum=-minimum.copy();maxref=0.
    with torch.no_grad():
        for start in range(0,len(fit),256):
            guard.check();ids=fit[start:start+256]
            raw=state32(torch.from_numpy(values[1][ids]))
            reference=observed_reference(torch.from_numpy(values[0][ids])).expand_as(raw)
            delta=raw-reference
            assert torch.equal(raw[...,-2:],delta[...,-2:])
            z=((delta-torch.from_numpy(mean))/torch.from_numpy(std)).numpy().reshape(-1,32).astype(np.float64)
            assert np.isfinite(z).all()
            total+=z.sum(0);squares+=(z*z).sum(0);count+=len(z)
            minimum=np.minimum(minimum,z.min(0));maximum=np.maximum(maximum,z.max(0))
            maxref=max(maxref,float(reference.abs().max()))
    variance=np.maximum(0.,squares/count-(total/count)**2)
    assert count==518240 and (variance>1e-12).all()
    report=dict(status='PASS_FULL_FIT_NUMERIC_ONLY',points=count,GT_used=False,CAL_DEV_used=False,
                common_raw_scalers=True,centered_dimensions=30,final2_unchanged=True,
                standardized_mean=(total/count).tolist(),standardized_std=np.sqrt(variance).tolist(),
                minimum=minimum.tolist(),maximum=maximum.tolist(),reference_abs_max=maxref,
                not_effectiveness_or_full_rank_proof=True)
    (Path(root)/'fit_discrepancy_numeric.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
