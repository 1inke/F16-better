"""FIT/CAL-only descriptive reconstruction of cached observed social tap; no GT/dev."""
from pathlib import Path
import sys,json,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'GENERATION_TOKEN_SEQUENCE_SERIES'))
import run_head as x
start=time.perf_counter()
raw=x.n.RAW_LOAD(x.p.TRAIN_CACHE);enc=x.n.RAW_LOAD(x.n.HERE/'train_encoder.npz')
np.testing.assert_array_equal(raw['sample_id'],enc['sample_id'])
assert raw['social'].shape==(28791,96) and enc['tokens'].shape==(28791,8,96)
assert np.isfinite(raw['social']).all() and np.isfinite(enc['tokens']).all()
fit,cal=x.p.internal_split(len(raw['sample_id']),.1)
assert len(fit)==25912 and len(cal)==2879 and not(set(raw['sample_id'][fit])&set(raw['sample_id'][cal]))
f=fit[:4096];q=cal[:512];social=raw['social'].astype(np.float64)
records=[]
for name,features in [('tokenmean',enc['tokens'].mean(1)),('past_last_mean',np.concatenate([enc['past'].reshape(-1,16),enc['tokens'][:,-1],enc['tokens'].mean(1)],1))]:
    features=features.astype(np.float64)
    fm=features[f].mean(0);fs=features[f].std(0).clip(.001)
    train=np.column_stack([(features[f]-fm)/fs,np.ones(len(f))])
    target=social[f];weights,_,rank,_=np.linalg.lstsq(train,target,rcond=None)
    residual=social[q]-np.column_stack([(features[q]-fm)/fs,np.ones(len(q))])@weights
    scale=np.sqrt(((social[q]-target.mean(0))**2).mean())
    records.append(dict(features=name,fit_n=len(f),cal_n=len(q),rank=int(rank),normalized_rmse=float(np.sqrt((residual**2).mean())/max(scale,1e-12)),max_abs=float(abs(residual).max())))
sources=[x.p.TRAIN_CACHE,x.n.HERE/'train_encoder.npz',Path(__file__),x.n.f.FRONT_MODELS,x.n.f.FORMAL_RUNNER if hasattr(x.n.f,'FORMAL_RUNNER') else Path(x.n.f.__file__)]
out=dict(status='PASS_DESCRIPTIVE_ONLY',fit_n=len(fit),cal_n=len(cal),fit_subset_ids=raw['sample_id'][f].tolist(),cal_subset_ids=raw['sample_id'][q].tolist(),social_dim=96,fit_social_std_mean=float(social[f].std(0).mean()),fit_social_norm_mean=float(np.linalg.norm(social[f],axis=1).mean()),reconstruction=records,source_hashes={str(v):x.p.sha256(Path(v)) for v in sources},no_gt_no_dev=True,seconds=time.perf_counter()-start,note='Linear proxy reconstruction only: not nonlinear information proof, useful-feature proof, intent evidence or formal experiment. Social already feeds frozen tokens. Real source replay alignment still required before formaltrain.')
x.m.write_json(Path(__file__).resolve().parent/'GENERATION_SOCIAL_TAP_DIAGNOSIS.json',out)
print(json.dumps({k:v for k,v in out.items() if not k.endswith('_ids') and k!='source_hashes'},indent=2))
