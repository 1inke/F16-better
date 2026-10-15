import os,sys,time,importlib.util,gc
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
sp=importlib.util.spec_from_file_location('v3_supplement',Path(__file__).parent/'saved.py');v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v);a=v.a;H=v.H;P=v.P;Q=v.Q;a.START=time.perf_counter();checks=0;maxerr=0.
def ck(x,y):
    global checks,maxerr
    x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape
    if x.dtype.kind in 'bUS' or y.dtype.kind in 'bUS':assert np.array_equal(x,y)
    else:
        ok=np.isfinite(x)&np.isfinite(y);assert np.array_equal(np.isfinite(x),np.isfinite(y));error=abs(x[ok]-y[ok]);assert np.allclose(x,y,atol=1e-10,rtol=1e-10,equal_nan=True);maxerr=max(maxerr,float(error.max()) if len(error) else 0)
    checks+=x.size
def main():
    a.guard();a.lock(Path(__file__));prior=a.read(H/'RESULT_SAVED.json');G2=a.read(H/'G2.json')['records'];G7=a.read(H/'G7.json')['records'];a.lock(H/'G2.json');a.lock(H/'G7.json');rr=[]
    for fold in Q['folds']:
        d=a.load(P/f'data/{fold}_evaluation.npz');selected=a.load(P/f'stage2_raw_interventions/data/{fold}_selected.npz');obs=selected['obs'].astype(float);valid=np.arange(1,9)[None]<selected['valid'][:,None];vel=obs[:,:,-1]-obs[:,:,-2];rel=obs[:,:1,-1]-obs[:,1:,-1];diff=rel[:,:,None]+(vel[:,:1]-vel[:,1:])[:,:,None]*np.arange(1,13)[None,None,:,None];distance=np.linalg.norm(diff,axis=-1);minimum=distance.min(2);at=distance.argmin(2);directions=diff[np.arange(len(obs))[:,None],np.arange(8)[None],at];unit=np.divide(directions,minimum[:,:,None],out=np.zeros_like(directions),where=minimum[:,:,None]>0);weights=np.exp(-minimum)/(1+(at+1)/12)*valid;vec=(weights[:,:,None]*unit).sum(1);norm=np.linalg.norm(vec,axis=1);wd=np.divide(vec,norm[:,None],out=np.zeros_like(vec),where=norm[:,None]>0);lookup={int(ID):i for i,ID in enumerate(d['ids'])};ix=np.array([lookup[int(ID)] for ID in selected['ids']]);edges=np.array(a.read(P/f'calibration/{fold}.json')['speed_edges']);a.lock(P/f'calibration/{fold}.json')
        for seed in Q['seeds']:
            old=a.old_job(P/f'jobs/{fold}_seed{seed}');point=a.load(H/f'POINTS_{fold}_seed{seed}.npz');base=dict(GT=point['GT'],CV=point['CV'],track=point['track'],P1=point['P1']);g=point['GT']-point['CV'];s=np.linalg.norm((d['obs'][:,0,-1]-d['obs'][:,0,-2]).astype(float),axis=1);mu=old['model_paths'].mean(1);spread=((old['model_paths']-mu[:,None])**2).sum((2,3)).mean(1);cp=v.groups(base)['strict_CV_pass'];masks=dict(all=np.ones(len(s),bool),P1=base['P1'],stationary=s<.02,CV_pass=cp,CV_fail=~cp,**{f'speed{b}':np.searchsorted(edges,s,side='right')==b for b in range(4)})
            for row in G7:
                if row['fold']!=fold or row['seed']!=seed:continue
                m=masks[row['subset']];stat=v.dist(spread[m]);ck(stat['q'],row['spread']['q']);av=v.auc(spread[m],~cp[m]);assert (av is None)==(row['spread_CV_failure_AUC'] is None)
                if av is not None:ck(av,row['spread_CV_failure_AUC'])
            for row in G2:
                if row['fold']!=fold or row['seed']!=seed or row['subset']=='speed_bin':continue
                m=v.groups(base)[row['subset']];r=point[row['method']]-point['CV'];dot=(r*g).sum();u=(r*r).sum();z=v.normstats(r,g,point['track'],m)
                for key in ('lambda_star_TEST_only','global_signed_cosine','oracle_skill_cosine_squared','original_skill'):ck(z[key],row[key])
                ck(z['window_cosine']['q'],row['window_cosine']['q'])
            oldno=a.old_job(P/f'stage2_raw_interventions/jobs/{fold}_seed{seed}/noNbr',2);cv=a.cv(selected['obs']);ca,cf=a.err(cv,selected['gt']);subset=dict(all=np.ones(len(obs),bool),P1=old['mask_nonCV_cue_positive'][ix],strict_CV_pass=(ca<=.2)&(cf<=.4),C3=a.neighborhood(selected['obs'],selected['valid'])['C3'])
            for method in ('top1','meanK'):
                x=a.load(H/f'RESPONSE_{fold}_seed{seed}_{method}.npz');ck(x['weighted_direction'],wd);ck(x['weight'],weights);select=lambda pp,ll:pp[np.arange(len(obs)),ll.argmax(1)] if method=='top1' else pp.mean(1);dr=select(oldno['base_paths'],oldno['base_logits'])-select(oldno['paths'],oldno['logits']);ck(np.linalg.norm(dr.reshape(len(obs),24),axis=1),x['magnitude']);ck((dr[:,-1]*wd).sum(1),x['weighted_response'])
                for name,m in subset.items():
                    near=m&x['nearest_defined'];wm=m&x['weighted_defined'];rr.append(dict(fold=fold,seed=seed,method=method,subset=name,n=int(m.sum()),tracks=len(np.unique(x['track'][m])),path_L2_m=v.dist(x['magnitude'][m]),path_RMS_m=v.dist(x['magnitude'][m]/np.sqrt(24)),endpoint_response_magnitude_m=v.dist(np.linalg.norm(dr[m,-1],axis=1)),signed_nearest=a.mean_ci(x['nearest_response'][near],x['track'][near]),signed_weighted=a.mean_ci(x['weighted_response'][wm],x['track'][wm]),positive_nearest=a.mean_ci((x['nearest_response'][near]>0).astype(float),x['track'][near]),positive_weighted=a.mean_ci((x['weighted_response'][wm]>0).astype(float),x['track'][wm]),model_slope_weighted=v.slope(x['closest'],x['weighted_response'],x['track'],wm&x['C3']),GT_slope_weighted=v.slope(x['closest'],x['GT_weighted'],x['track'],wm&x['C3'])))
            del old;gc.collect()
    a.emit('G6_POPULATIONS',dict(records=rr));
    for p,s in a.LOCKS.items():assert a.sha(p)==s
    a.write('supplement_input_manifest.json',dict(files=a.LOCKS));a.write('SUPPLEMENT_REVIEW.json',dict(status='PASS_FULL_G2_G7_POINT_ARITHMETIC_G6_WEIGHT_RESPONSE_AND_POPULATIONS',checks=checks,max_absolute_error=maxerr,input_files=len(a.LOCKS),resource=a.RES,elapsed_seconds=time.perf_counter()-a.START,no_model_replay=True,not_all_second_CI_rebuild=True,summary_sha=a.OUT))
if __name__=='__main__':main()
