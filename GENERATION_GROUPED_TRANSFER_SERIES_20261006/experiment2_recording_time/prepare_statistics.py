"""Same fixed mathematics, only common adaptation-training rows fit statistics."""
import adapter as h
import numpy as np,torch
def state_stats(base,idx,chunk):
    sums=np.zeros(32,np.float64);squares=sums.copy();count=0
    for start in range(0,len(idx),chunk):
        h.guard.check();x=h.state.state32(torch.from_numpy(base[idx[start:start+chunk]])).numpy().reshape(-1,32).astype(np.float64)
        sums+=x.sum(0);squares+=(x*x).sum(0);count+=len(x)
    mean=sums/count;std=np.sqrt(np.maximum(0,squares/count-mean*mean));return mean,std,count
def accumulate_fixed_features(features,chunk):
    sums=np.zeros(32,np.float64);squares=sums.copy();count=0
    for start in range(0,len(features),chunk):
        x=features[start:start+chunk].reshape(-1,32).astype(np.float64)
        sums+=x.sum(0);squares+=(x*x).sum(0);count+=len(x)
    mean=sums/count;return mean,np.sqrt(np.maximum(0,squares/count-mean*mean)),count
def prepare(cache,values,splits):
    idx=splits['common_statistics_index'];ids=cache['sample_id'][idx];base=values[1]
    assert len(idx)==12544 and np.array_equal(ids,splits['common_statistics_ids'])
    assert not set(idx)&set(splits['outer_eval_index']) and not set(idx)&set(splits['inner_cal_index'])
    assert all(not (h.HERE/name).exists() for name in ('state_scalers.npz','physical_accel_scales.npz','route_threshold.npz','common_statistics.npz')),'Preserve existing statistics; no silent refit'
    mean,std,count=state_stats(base,idx,256);assert count==len(idx)*20 and (std>1e-7).all()
    check=np.r_[idx[:517],idx[-516:]]
    # Production feature arithmetic is fixed256rows. Audit accumulation chunking
    # on those SAME float32 features, not an alternate atan2 CPU vector backend.
    locked_features=np.concatenate([h.state.state32(torch.from_numpy(base[check[start:start+256]])).numpy() for start in range(0,len(check),256)])
    alternate_features=np.concatenate([h.state.state32(torch.from_numpy(base[check[start:start+31]])).numpy() for start in range(0,len(check),31)])
    backend_delta=float(abs(locked_features-alternate_features).max())
    a=accumulate_fixed_features(locked_features,31);b=accumulate_fixed_features(locked_features,256)
    error=max(float(abs(a[0]-b[0]).max()),float(abs(a[1]-b[1]).max()));assert error<1e-11
    scales,counts=h.physical.stats(base,cache['change_group'],idx,256)
    pa,pn=h.physical.stats(base,cache['change_group'],check,31);pb,pm=h.physical.stats(base,cache['change_group'],check,256)
    assert np.array_equal(pn,pm) and abs(pa-pb).max()<1e-12
    targets=h.p.classify_velocity(values[2][:,1:],values[3][:,-1]/12,h.LABEL_CFG)
    classes=np.bincount(targets[idx],minlength=7);assert (classes>0).all()
    weights=np.sqrt(len(idx)/(7*classes)).clip(.5,3);weights/=np.mean(weights[targets[idx]])
    qualityscale=np.asarray([cache['ade'][idx].min(-1).mean(),cache['fde'][idx].min(-1).mean()],np.float32).clip(1e-6)
    q=[]
    for start in range(0,len(idx),256):
        take=idx[start:start+256];q.append(h.route.statistic(torch.from_numpy(values[0][take]),torch.from_numpy(base[take])).numpy())
    median=float(np.median(np.concatenate(q).astype(np.float64)))
    np.savez_compressed(h.HERE/'state_scalers.npz',mean=mean.astype(np.float32),std=np.maximum(std,.001).astype(np.float32),fit_id=ids)
    np.savez_compressed(h.HERE/'physical_accel_scales.npz',scales=scales.astype(np.float32),fit_ids=ids,counts=counts)
    np.savez_compressed(h.HERE/'route_threshold.npz',median=np.asarray(median,np.float64),fit_id=ids)
    np.savez_compressed(h.HERE/'common_statistics.npz',quality_weights=weights.astype(np.float32),quality_counts=classes,protection_scales=qualityscale,fit_ids=ids)
    result=dict(status='PASS_COMMON_TRAIN_STATISTICS',rows=len(idx),candidate_points=count,ids_SHA=h.sha(h.HERE/'identity_audit/splits.npz'),no_validation_statistics=True,state_chunk1033_error=error,feature_backend31_vs256_delta_descriptive=backend_delta,feature_production_chunk=256,statistic_threshold_unchanged=1e-11,physical_chunk_error=float(abs(pa-pb).max()),physical_scales=scales,physical_counts=counts,quality_counts=classes,quality_weights=weights,protection_scales=qualityscale,route_median_unused_uniform=median,state_math=h.state.validate_state_math(),physical_fixtures=h.physical.fixtures())
    h.write(h.HERE/'statistics_review.json',result);return result
