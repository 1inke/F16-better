"""Saved-data D5c slopes and D7/probe construction verification."""
import os,sys,time,importlib.util,traceback
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
H=Path(__file__).resolve().parent;P=H.parent;T=time.perf_counter()
sp=importlib.util.spec_from_file_location('sup_saved_utils',P/'claim_evidence_followup_20261009/saved_reader_recovery/analyze_saved.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a);a.H=H;a.START=T;a.Q=a.read(H/'protocol.json')
sp=importlib.util.spec_from_file_location('verify_probe_features',H/'probes/run.py');pr=importlib.util.module_from_spec(sp);sp.loader.exec_module(pr);pr.a=a
checks=0;maxerror=0.
def close(x,y,tol=1e-10):
    global checks,maxerror
    x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape
    error=float(abs(x-y).max()) if x.size else 0.;maxerror=max(maxerror,error);assert error<=tol;checks+=x.size
def slope(x,y,t):
    if len(t)==0:return dict(value=None,CI=None,n=0,tracks=0)
    u,ix=np.unique(t,return_inverse=True);mat=np.stack([np.bincount(ix,weights=z,minlength=len(u)) for z in (x,y,x*x,x*y,np.ones(len(x)))],1)
    def calc(m):sx,sy,sx2,sxy,n=m.T;den=sx2-sx*sx/n;return np.divide(sxy-sx*sy/n,den,out=np.full_like(den,np.nan),where=den>1e-12)
    value=calc(mat.sum(0)[None])[0];rng=np.random.default_rng(a.Q['stat_seed']);vals=[]
    for b in range(0,2000,100):
        ii=rng.integers(0,len(u),(100,len(u)));vv=calc(mat[ii].sum(1));vals.extend(vv[np.isfinite(vv)].tolist())
    return dict(value=float(value) if np.isfinite(value) else None,CI=np.quantile(vals,[.025,.5,.975]).tolist() if vals else None,n=len(t),tracks=len(u))
def main():
    a.guard();a.lock(Path(__file__));assert abs(slope(np.arange(10.),2*np.arange(10.)+3,np.arange(10))['value']-2)<1e-12
    for folder in [H,H/'probes']:
        a.lock(folder/'input_manifest.json');a.EXPECTED.update(a.read(folder/'input_manifest.json')['files']);rr=a.read(folder/'RESULT.json');a.lock(folder/'RESULT.json');a.EXPECTED.update(rr['output_sha'])
    a.lock(H/'probes/run.py');dose=[];shift=a.read(H/'D7_SHIFT.json')['records'];d7=a.read(H/'D7.json')['records'];a.lock(H/'D7.json');a.lock(H/'D7_SHIFT.json')
    for fold in a.Q['folds']:
        f=a.load(H/f'FEATURES_{fold}.npz');d=a.load(P/f'data/{fold}_evaluation.npz');bank=a.load(P/f'stage3_knn_memory/data/{fold}_bank.npz')
        for r in [r for r in shift if r['fold']==fold]:
            key=r['feature'].replace('_mps','');tx=f['train_'+key];xx=f['test_'+key];close(float(ks_2samp(tx,xx).statistic),r['KS_distance'])
        pred={kind:a.old_job(P/f'stage3_knn_memory/jobs/{fold}_{kind}_all_train',3)['pred'] for kind in ('velocity_only','position_velocity')};pred['position_only']=a.load(H/f'D7_position_{fold}.npz')['pred'];cv=a.cv(d['obs']);den=((d['gt']-cv)**2).sum((1,2));t=d['track']
        for seed in a.Q['seeds']:
            z=a.old_job(P/f'jobs/{fold}_seed{seed}');tp=z['model_paths'][np.arange(len(t)),z['model_logits'].argmax(1)];tn=((tp-d['gt'])**2).sum((1,2))
            for r in [r for r in d7 if r['fold']==fold and r['seed']==seed]:
                m=np.ones(len(t),bool) if r['subset']=='all' else z['mask_nonCV_cue_positive'];num=((pred[r['method']]-d['gt'])**2).sum((1,2));value=a.ci(num[m],den[m],t[m],True);assert value==r['skill'];assert a.ci((tn-num)[m],den[m],t[m])==r['skill_gain_vs_top1']
            for method in ('top1','meanK'):
                v=a.load(H/f'D5c_{fold}_seed{seed}_{method}.npz');m=v['C3']&v['away_valid'];x=v['closest'][m];tt=v['track'][m];dose.append(dict(fold=fold,seed=seed,method=method,subset='original_real_C3_away_defined',model_slope=slope(x,v['response'][m],tt),GT_slope=slope(x,v['GTresponse'][m],tt),model_minus_GT_slope=slope(x,(v['response']-v['GTresponse'])[m],tt),post_T0_supplement_exploratory_not_primary=True))
        z=a.load(H/f'probes/PROBES_{fold}.npz');rp=a.D/f'{fold}_data_train.npy';cp=a.D/f'{fold}_num_train.npy';a.lock(rp);a.lock(cp);raw=np.load(rp,mmap_mode='r');count=np.load(cp,mmap_mode='r');obs=np.array(raw[z['train_ids'],:,:8,-2:],np.float32);valid=np.minimum(count[z['train_ids']].astype(int),9);tx=pr.features(obs,valid);xx=pr.features(d['obs'],d['valid']);res=(bank['gt']-a.cv(obs)).reshape(-1,24);close(res.mean(0),z['PCA_mean']);targets=(res-z['PCA_mean'])@z['PCA_basis'];nb=a.neighborhood(obs,valid);g=((bank['gt'][:,-1]-a.cv(obs)[:,-1])*nb['away']).sum(1);mask=nb['C3']&nb['away_valid']
        for j,name in enumerate(('self','self_social')):
            for kind,Y,mm in [('PCA',targets,np.ones(len(obs),bool)),('g',g[:,None],mask)]:
                X=tx[j][mm];mu=z[name+'_'+kind+'_mu'];sd=z[name+'_'+kind+'_std'];ym=z[name+'_'+kind+'_target_mean'];W=z[name+'_'+kind+'_weights'];close(X.mean(0),mu);close(Y[mm].mean(0),ym);sx=(X-mu)/sd;left=(sx.T@sx/len(X)+np.eye(X.shape[1])*.01)@W;right=sx.T@(Y[mm]-ym)/len(X);close(left,right,tol=1e-9)
            mu=z[name+'_PCA_mu'];sd=z[name+'_PCA_std'];ym=z[name+'_PCA_target_mean'];W=z[name+'_PCA_weights'];pp=cv+(((xx[j]-mu)/sd@W+ym)@z['PCA_basis'].T+z['PCA_mean']).reshape(-1,12,2);close(pp,z['pred_self' if j==0 else 'pred_social'])
        print('SUPPLEMENT',fold,checks,flush=True)
    a.emit('D5c_SLOPES',dict(records=dose));
    for p,s in a.LOCKS.items():a.guard();assert a.sha(p)==s
    a.write('SUPPLEMENT_REVIEW.json',dict(status='PASS_D7_ALL_METRICS_KS_AND_ALL_PROBE_NORMAL_EQUATIONS',checks=checks,max_error=maxerror,elapsed_seconds=time.perf_counter()-T,no_torch_forward_optimizer=True,not_full_second_probe_refit=True,source_SHA_preserved=True,resource=a.RES));a.write('supplement_input_manifest.json',dict(files=a.LOCKS))
if __name__=='__main__':
    try:main()
    except Exception:a.write('SUPPLEMENT_FAILURE.json',dict(status='STOP',traceback=traceback.format_exc()));raise
