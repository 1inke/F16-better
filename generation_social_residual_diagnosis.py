"""FIT/CAL-only fixed ridge residualization feasibility, no GT/dev/test."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'GENERATION_SOCIAL_TAP_SERIES'))
import run_head as a
raw=a.n.RAW_LOAD(a.p.TRAIN_CACHE)
enc=a.n.RAW_LOAD(a.n.HERE/'train_encoder.npz')
assert np.array_equal(raw['sample_id'],enc['sample_id'])
fit,cal=a.p.internal_split(len(raw['sample_id']),.1)
assert len(fit)==25912 and len(cal)==2879
features=np.concatenate([enc['past'].reshape(-1,16),enc['tokens'][:,-1],enc['tokens'].mean(1)],1).astype(np.float64)
social=raw['social'].astype(np.float64)
xm=features[fit].mean(0);xs=features[fit].std(0).clip(.001)
ym=social[fit].mean(0);ys=social[fit].std(0).clip(.001)
X=np.column_stack([(features-xm)/xs,np.ones(len(features))]);Y=(social-ym)/ys
ridge=.001
penalty=np.eye(X.shape[1])*ridge;penalty[-1,-1]=0
weights=np.linalg.solve(X[fit].T@X[fit]/len(fit)+penalty,X[fit].T@Y[fit]/len(fit))
R=Y-X@weights
rm=R[fit].mean(0);rs=R[fit].std(0).clip(.001)
records={}
for name,ids in [('fit',fit),('cal',cal)]:
 residual=(R[ids]-rm)/rs
 records[name]=dict(n=len(ids),raw_standardized_social_rms=float(np.sqrt(np.mean(Y[ids]**2))),residual_rms=float(np.sqrt(np.mean(R[ids]**2))),residual_scaled_rms=float(np.sqrt(np.mean(residual**2))),scaled_abs_p99=float(np.quantile(abs(residual),.99)),finite=bool(np.isfinite(residual).all()))
assert all(v['finite'] for v in records.values())
destination=ROOT/'GENERATION_SOCIAL_RESIDUAL_TRANSFORM.npz'
np.savez_compressed(destination,feature_mean=xm.astype('float32'),feature_std=xs.astype('float32'),social_mean=ym.astype('float32'),social_std=ys.astype('float32'),weights=weights.astype('float32'),residual_mean=rm.astype('float32'),residual_std=rs.astype('float32'),fit_ids=raw['sample_id'][fit],cal_ids=raw['sample_id'][cal])
sources=[Path(__file__),a.p.TRAIN_CACHE,a.n.HERE/'train_encoder.npz',destination]
out=dict(status='PASS_FEASIBILITY_ONLY',ridge_fixed=ridge,feature_order='past16+lasttoken96+meantoken96',fit_n=len(fit),cal_n=len(cal),fit_only_transform=True,no_gt_dev_test=True,records=records,fit_residual_std_min=float(rs.min()),fit_residual_std_max=float(rs.max()),source_hashes={str(p):a.p.sha256(p) for p in sources},note='Residualization removes FIT-linear predictable social components, may remove useful signal/amplify noise; does not create information or prove redundancy causes prediction errors. Full and residual inputs both nonzero; no CAL selection of ridge/scalers/features.')
a.m.write_json(ROOT/'GENERATION_SOCIAL_RESIDUAL_DIAGNOSIS.json',out)
print(json.dumps(out,indent=2))
