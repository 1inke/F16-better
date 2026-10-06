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

def load_scalers(root,centered=False):
    with np.load(Path(root)/('centered_scalers.npz' if centered else 'state_scalers.npz'),allow_pickle=False) as saved:
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

def centered_features(observation,base):
    raw=state32(base)
    delta=raw-observed_reference(observation).expand_as(raw)
    assert torch.equal(raw[...,-2:],delta[...,-2:])
    return delta

def moments(x):
    x=np.asarray(x,np.float64).reshape(-1,32)
    mu=x.mean(0);sigma=np.sqrt(np.maximum(0.,(x*x).mean(0)-mu*mu))
    return mu,sigma

def prepare_centered_scalers(root,sample_ids,fit,values,sha):
    root=Path(root);path=root/'centered_scalers.npz';meta=root/'centered_scalers.json'
    rawmean,rawstd=load_scalers(root)
    provenance=dict(features_source_SHA=sha(Path(__file__)),rawscalers_SHA=sha(root/'state_scalers.npz'),
                    fit_ids_n=len(fit),candidate_points=len(fit)*20,FIT25912_only=True,
                    GT_used=False,CAL_DEV_used=False,dt=.4,std_floor=.001,chunk_rows=256,
                    first30_only=True,last2_raw_scalers_unchanged=True)
    assert len(fit)==25912 and len(np.unique(sample_ids[fit]))==25912
    if path.exists():
        report=json.loads(meta.read_text(encoding='utf-8'))
        assert report['provenance']==provenance and report['scalers_SHA']==sha(path)
        with np.load(path,allow_pickle=False) as saved:
            assert np.array_equal(saved['fit_id'],sample_ids[fit])
            assert np.array_equal(saved['mean'][-2:],rawmean[-2:]) and np.array_equal(saved['std'][-2:],rawstd[-2:])
        return
    total=np.zeros(32,np.float64);squares=total.copy();count=0
    lo=np.full(32,np.inf);hi=-lo.copy()
    with torch.no_grad():
        for start in range(0,len(fit),256):
            guard.check();ids=fit[start:start+256]
            x=centered_features(torch.from_numpy(values[0][ids]),torch.from_numpy(values[1][ids])).numpy().reshape(-1,32).astype(np.float64)
            total+=x.sum(0);squares+=(x*x).sum(0);count+=len(x)
            lo=np.minimum(lo,x.min(0));hi=np.maximum(hi,x.max(0))
    mu=total/count;sigma=np.sqrt(np.maximum(0.,squares/count-mu*mu))
    assert count==518240 and np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma>1e-7).all()
    mean=rawmean.copy();std=rawstd.copy()
    mean[:30]=mu[:30].astype(np.float32);std[:30]=np.maximum(sigma[:30],.001).astype(np.float32)
    assert np.array_equal(mean[-2:],rawmean[-2:]) and np.array_equal(std[-2:],rawstd[-2:])
    np.savez_compressed(path,mean=mean,std=std,fit_id=sample_ids[fit])
    meta.write_text(json.dumps(dict(status='PASS_FIT_ONLY_CENTERED_SCALERS',provenance=provenance,scalers_SHA=sha(path),
        centered_mean_float64=mu.tolist(),centered_std_float64=sigma.tolist(),minimum=lo.tolist(),maximum=hi.tolist(),
        control_scalers_mean=rawmean.tolist(),control_scalers_std=rawstd.tolist()),indent=2),encoding='utf-8')
    print('FIT_CENTERED_SCALERS PASS '+str(count),flush=True)

def validate_centered_statistics(root,values,fit):
    # Same actual float32 feature values, independent direct versus chunked float64 statistics.
    ids=fit[:1033]
    with torch.no_grad():
        x=centered_features(torch.from_numpy(values[0][ids]),torch.from_numpy(values[1][ids])).numpy()
    expected=moments(x);maximum=0.
    for chunk in (31,256):
        total=np.zeros(32);squares=total.copy();n=0
        for start in range(0,len(ids),chunk):
            z=x[start:start+chunk].reshape(-1,32).astype(np.float64)
            total+=z.sum(0);squares+=(z*z).sum(0);n+=len(z)
        mu=total/n;std=np.sqrt(np.maximum(0.,squares/n-mu*mu))
        error=max(float(np.abs(mu-expected[0]).max()),float(np.abs(std-expected[1]).max()))
        assert error<1e-10,error
        maximum=max(maximum,error)
    rawmean,rawstd=load_scalers(root);mean,std=load_scalers(root,True)
    assert np.array_equal(mean[-2:],rawmean[-2:]) and np.array_equal(std[-2:],rawstd[-2:])
    metadata=json.loads((Path(root)/'centered_scalers.json').read_text(encoding='utf-8'))
    registered_mu=np.asarray(metadata['centered_mean_float64']);registered_std=np.asarray(metadata['centered_std_float64'])
    assert np.array_equal(mean[:30],registered_mu[:30].astype(np.float32))
    assert np.array_equal(std[:30],np.maximum(registered_std[:30],.001).astype(np.float32))
    # Input affine transform is invertible; no claim final weights/optimizer are equivalent.
    a=(x-rawmean)/rawstd;b=(x-mean)/std
    inverse=b*std+mean
    error=float(np.abs(inverse-x).max())
    assert error<2e-5 and not np.array_equal(a[...,:30],b[...,:30]),error
    return dict(status='PASS_INDEPENDENT_STATISTICS_AND_AFFINE',fixture_rows=1033,chunk_sizes=[31,256],
                chunk_statistic_max_error=maximum,inverse_feature_max_error=error,
                first30_registered=True,last2_exact=True,CAL_DEV_used=False,GT_used=False)

def fit_discrepancy_numeric(root,values,fit):
    scalers={arm:load_scalers(root,arm=='mean_std') for arm in ('zero','mean_std')}
    totals={arm:np.zeros(32) for arm in scalers};squares={arm:np.zeros(32) for arm in scalers};count=0
    with torch.no_grad():
        for start in range(0,len(fit),256):
            guard.check();ids=fit[start:start+256]
            delta=centered_features(torch.from_numpy(values[0][ids]),torch.from_numpy(values[1][ids]))
            for arm,(mean,std) in scalers.items():
                z=((delta-torch.from_numpy(mean))/torch.from_numpy(std)).numpy().reshape(-1,32).astype(np.float64)
                assert np.isfinite(z).all()
                totals[arm]+=z.sum(0);squares[arm]+=(z*z).sum(0)
            count+=len(ids)*20
    assert count==518240
    records={}
    for arm in scalers:
        mu=totals[arm]/count;variance=np.maximum(0.,squares[arm]/count-mu*mu)
        assert (variance>1e-12).all()
        records[arm]=dict(standardized_mean=mu.tolist(),standardized_std=np.sqrt(variance).tolist())
    assert np.max(np.abs(np.asarray(records['mean_std']['standardized_mean'])[:30]))<1e-4
    assert np.max(np.abs(np.asarray(records['mean_std']['standardized_std'])[:30]-1))<1e-4
    assert np.array_equal(totals['zero'][-2:],totals['mean_std'][-2:])
    report=dict(status='PASS_FULL_FIT_NUMERIC_ONLY',points=count,GT_used=False,CAL_DEV_used=False,
                same_centered_features=True,centered_dimensions=30,final2_unchanged=True,records=records,
                not_effectiveness_or_full_rank_proof=True)
    (Path(root)/'fit_discrepancy_numeric.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
