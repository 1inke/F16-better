"""Saved sufficient-statistic audit, independent candidate ranking arithmetic."""
import os,sys,time,json,importlib.util,traceback
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
H=Path(__file__).resolve().parent;P=H.parent;T=time.perf_counter()
sp=importlib.util.spec_from_file_location('audit_utils',P/'claim_evidence_followup_20261009/saved_reader_recovery/analyze_saved.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a);a.H=H;a.START=T;a.Q=a.read(H/'protocol.json');a.Q['max_compute_seconds']=600
checks=0;maxerror=0.
def check(x,y):
    global checks,maxerror
    if isinstance(x,dict):
        assert x.keys()==y.keys(),(x.keys(),y.keys())
        for k in x:check(x[k],y[k])
    elif isinstance(x,(list,np.ndarray)):
        assert len(x)==len(y)
        for xx,yy in zip(x,y):check(xx,yy)
    elif isinstance(x,(float,int,np.floating,np.integer)) and not isinstance(x,bool):
        assert y is not None;err=abs(float(x)-float(y));maxerror=max(maxerror,err);assert err<=1e-9+1e-12*abs(float(y)),(x,y,err);checks+=1
    else:assert x==y;checks+=1
def verify(folder):
    result=a.read(folder/'RESULT.json');manifest=a.read(folder/'input_manifest.json')
    for p,s in manifest['files'].items():a.lock(p,s)
    for p,s in result['output_sha'].items():a.lock(p,s)
    for name,s in result['summary_sha'].items():a.lock(folder/(name+'.json'),s)
    return result
def main():
    a.guard();root=verify(H);probe=verify(H/'probes')
    for p in [Path(__file__),H/'DESIGN.md',H/'protocol.json']:a.lock(p)
    D2=a.read(H/'D2.json')['records'];D1=a.read(H/'D1a.json')['records'];D3=a.read(H/'D3.json')['records'];D6=a.read(H/'D6.json')['records'];D5c=a.read(H/'D5c.json')['records']
    for fold in a.Q['folds']:
        for seed in a.Q['seeds']:
            for method in ('top1','meanK'):
                z=a.load(H/f'D2_{fold}_seed{seed}_{method}.npz');t=z['track'];cv=z['CV_pass'];good=z['model_pass']
                for r in [r for r in D2 if r['fold']==fold and r['seed']==seed and r['method']==method]:
                    m=z['mask_'+r['subset']];check(int(m.sum()),r['n']);check(len(np.unique(t[m])),r['tracks']);check(a.ci(z['num'][m],z['den'][m],t[m],True),r['skill']);check(a.ci(z['dot'][m],z['r2'][m],t[m]),r['lambda_star']);check(a.mean_ci(z['r2'][m],t[m]),r['mean_residual_squared']);check(float((z['den']-z['num'])[m].sum()),r['delta_SSE_sum'])
                    for key,x in [('ADE_delta_CV','delta_ADE_CV'),('ADE_delta_strongest','delta_ADE_strongest'),('FDE_delta_strongest','delta_FDE_strongest')]:check(a.mean_ci(z[x][m],t[m]),r[key])
                    for name,mm in [('kept',cv&good),('lost',cv&~good),('rescue',~cv&good),('both_fail',~cv&~good)]:check(int((mm&m).sum()),r[name])
                    for c in r['curve']:check(a.ratio(float((2*c['lambda_value']*z['dot'][m]-c['lambda_value']**2*z['r2'][m]).sum()),float(z['den'][m].sum())),c['skill'])
                    if r['subset'] in ('all','P1','fallback','no_fallback','start','start_fallback','start_no_fallback'):
                        rr=next(d for d in D1 if (d['fold'],d['seed'],d['method'],d['subset'])==(fold,seed,method,r['subset']))
                        for k in rr:check(rr[k],r[k])
            z=a.load(H/f'D3D6_{fold}_seed{seed}.npz');old=a.old_job(P/f'jobs/{fold}_seed{seed}');p=old['model_paths'];cv=old['CV'];mu=p.mean(1);check(np.argmin(np.sum((p-mu[:,None])**2,axis=(2,3)),1),z['medoid']);check(np.argmin(np.sum((p-cv[:,None])**2,axis=(2,3)),1),z['nearest_CV']);check(np.sum((p-mu[:,None])**2,axis=(2,3)).mean(1),z['spread']);t=z['track']
            for r in [r for r in D6 if r['fold']==fold and r['seed']==seed]:
                m=np.ones(len(t),bool) if r['subset']=='all' else z['mask_P1'];good=(z['ADE_all']<=r['tau'])&(z['FDE_all']<=2*r['tau']);present=good.any(1);mixed=present&~good.all(1);mm=m&present;check(int(mm.sum()),r['good_present']);check(int((m&mixed).sum()),r['mixed_n'])
                hits={key:a.mean_ci(good[np.arange(len(t)),z[idx]][mm].astype(float),t[mm]) for key,idx in [('top1','top'),('nearest_meanK_squared_medoid','medoid'),('nearest_CV_squared','nearest_CV')]};hits['uniform_random_expectation']=a.mean_ci(good.mean(1)[mm],t[mm]);check(hits,r['conditional_hits']);check(a.mean_ci((good[np.arange(len(t)),z['top']].astype(float)-good[np.arange(len(t)),z['medoid']])[mm],t[mm]),r['top1_minus_medoid'])
                values=[]
                for i in np.flatnonzero(m&mixed):
                    n=int(good[i].sum());scores=z['logits'][i];value=(rankdata(scores)[good[i]].sum()-n*(n+1)/2)/(n*(20-n));values.append(value)
                check(a.mean_ci(np.array(values),t[m&mixed]),r['within_window_logit_AUROC'])
            for r in [r for r in D3 if r['fold']==fold and r['seed']==seed]:
                m=np.ones(len(t),bool) if r['subset']=='all' else z['mask_P1'];check(a.ci(z['spread'][m],z['error'][m],t[m]),r['spread_over_err']);check(a.mean_ci(z['spread'][m],t[m]),r['spread_mean'])
                y=~z['CV_pass'][m];ss=z['spread'][m];n=int(y.sum());ng=len(y)-n;point=float((rankdata(ss)[y].sum()-n*(n+1)/2)/(n*ng)) if n and ng else None;check(point,r['spread_discriminates_CV_fail']['value'])
                for b in r['bins']:
                    mm=m&(z['spread_bin']==b['bin']);check(int(mm.sum()),b['n']);check(a.ci(z['error'][mm],z['spread'][mm],t[mm]),b['err_over_spread'])
        for method in ('top1','meanK'):
            for seed in a.Q['seeds']:
                z=a.load(H/f'D5c_{fold}_seed{seed}_{method}.npz')
                for r in [r for r in D5c if r['fold']==fold and r['seed']==seed and r['method']==method]:
                    m=z['C3']&z['away_valid']&(z['bin']==r['bin']);check(int(m.sum()),r['n']);check(a.mean_ci(z['response'][m],z['track'][m]),r['model_response']);check(a.mean_ci(z['GTresponse'][m],z['track'][m]),r['GT_response'])
        bank=a.load(P/f'stage3_knn_memory/data/{fold}_bank.npz');pos=a.load(H/f'D7_position_{fold}.npz');d=a.load(P/f'data/{fold}_evaluation.npz');lookup={int(ID):i for i,ID in enumerate(bank['ids'])};ix=np.array([[lookup[int(ID)] for ID in row] for row in pos['neighbor_ids']]);base=(bank['obs'][:,-1]-bank['obs'][:,-2]).astype(float)[:,None]*np.arange(1,13)[None,:,None];check(a.cv(d['obs'])+(bank['gt']-base)[ix].mean(1),pos['pred'])
        X=(bank['obs'][:,-1].astype(float)-pos['stats_mean'])/pos['stats_std'];query=(d['position'].astype(float)-pos['stats_mean'])/pos['stats_std']
        for i in np.linspace(0,len(d['ids'])-1,8,dtype=int):
            dd=np.linalg.norm(X-query[i],axis=1);ID=bank['ids'][np.lexsort((bank['ids'],dd))[:32]];check(ID,pos['neighbor_ids'][i])
        # Pure linear probe consistency, all saved point/CI summaries, train/test disjointness.
        pr=a.load(H/f'probes/PROBES_{fold}.npz');assert not np.intersect1d(pr['train_tracks'],pr['track']).size;check(pr['PCA_basis'].T@pr['PCA_basis'],np.eye(2));den=((pr['GT']-pr['CV'])**2).sum((1,2))
        for r in [r for r in a.read(H/'probes/D4.json')['records'] if r['fold']==fold]:
            key='pred_self' if r['probe']=='self' else 'pred_social';num=((pr[key]-pr['GT'])**2).sum((1,2));m=np.ones(len(den),bool) if r['subset']=='all' else pr['mask_P1'] if r['subset']=='P1' else pr[r['subset'].replace('nonCV_','')];old=a.old_job(P/f'jobs/{fold}_seed{r["seed"]}')
            if r['subset'].startswith('nonCV_'):m=m&old['mask_nonCV']
            check(a.ci(num[m],den[m],pr['track'][m],True),r['skill']);tp=old['model_paths'][np.arange(len(den)),old['model_logits'].argmax(1)];tn=((tp-pr['GT'])**2).sum((1,2));check(a.ci((tn-num)[m],den[m],pr['track'][m]),r['skill_gain_top1'])
        for r in [r for r in a.read(H/'probes/D5a.json')['records'] if r['fold']==fold]:
            m=pr['g_mask'];t=pr['track'][m];g=pr['g'][m];dm=(g-pr['g_train_mean'])**2;g0=(pr['g_self'][m]-g)**2;g1=(pr['g_social'][m]-g)**2
            if r['probe']=='social_minus_self':check(a.mean_ci(g0-g1,t),r['paired_MSE_gain']);check(a.ci(g0-g1,dm,t),r['gain_vs_TRAIN_mean'])
            else:
                e=g0 if r['probe']=='self' else g1;check(a.ci(e,dm,t,True),r['skill_vs_TRAIN_mean']);check(a.ci(e,g*g,t,True),r['skill_vs_zero'])
        print('AUDIT',fold,checks,flush=True)
    for p,s in a.LOCKS.items():a.guard();assert a.sha(p)==s,('FINAL_SHA',p)
    a.write('REVIEW.json',dict(status='PASS_SAVED_ARITHMETIC_AND_POSITION_KNN_AUDIT',checks=checks,max_numeric_error=maxerror,input_files=len(a.LOCKS),resource=a.RES,elapsed_seconds=time.perf_counter()-T,not_external_review=True,not_all_model_or_probe_second_fit=True,not_all_nearest_queries_second_search=True,D3_AUC_CI_not_rebootstrapped=True));a.write('review_input_manifest.json',dict(files=a.LOCKS))
if __name__=='__main__':
    try:main()
    except Exception:a.write('REVIEW_FAILURE.json',dict(status='STOP_REVIEW',traceback=traceback.format_exc(),checks=checks));raise
