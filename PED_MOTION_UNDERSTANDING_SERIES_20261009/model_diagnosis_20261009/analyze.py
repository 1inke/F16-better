"""T0 model diagnostics; only saved data, new position-only kNN; no torch."""
import os,sys,time,json,gc,traceback,importlib.util
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,ks_2samp,spearmanr
from scipy.spatial import cKDTree
H=Path(__file__).resolve().parent;P=H.parent;V=P/'claim_evidence_v2_20261009/numeric_fixture_recovery'
sp=importlib.util.spec_from_file_location('old_saved_math',P/'claim_evidence_followup_20261009/saved_reader_recovery/analyze_saved.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
Q=a.read(H/'protocol.json');a.Q=Q;a.H=H;a.START=time.perf_counter()
sp=importlib.util.spec_from_file_location('v2_math',V/'analyze_v2.py');v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
v.a=a
def rows(name,x):a.emit(name,dict(records=x,exploratory=True,track_clustered=True))
def stat(x,t):return a.mean_ci(np.asarray(x,float),t)
def total(x):return float(np.sum(x))
def auc(scores,y):
    pos=int(y.sum());neg=len(y)-pos
    return float((rankdata(scores)[y].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None
def auc_ci(scores,y,t):
    if not len(t):return dict(value=None,CI=None,n=0,tracks=0)
    u,inv=np.unique(t,return_inverse=True);order=np.argsort(scores,kind='stable');s=scores[order];yy=y[order];iv=inv[order]
    _,gi=np.unique(s,return_inverse=True);G=gi.max()+1;rng=np.random.default_rng(Q['stat_seed']);values=[]
    for j in range(Q['bootstrap_reps']):
        if j%100==0:a.guard()
        w=np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u))[iv]
        p=np.bincount(gi,weights=w*yy,minlength=G);n=np.bincount(gi,weights=w*(~yy),minlength=G);d=p.sum()*n.sum()
        if d>0:values.append(float((p*(np.cumsum(n)-.5*n)).sum()/d))
    return dict(value=auc(scores,y),CI=np.quantile(values,[.025,.5,.975]).tolist() if values else None,n=len(t),tracks=len(u))
def lambda_ci(dot,r2,t):return a.ci(dot,r2,t)
def subsets(z,d,start):
    c1,c2,c3=v.cues(d['obs'],d['valid']);fallback=np.linalg.norm(d['obs'][:,0,-1]-d['obs'][:,0,-4],axis=-1)<=.03
    goodcv=(z['CV_last_ADE']<=.2)&(z['CV_last_FDE']<=.4)
    return dict(all=np.ones(len(start),bool),P1=z['mask_nonCV_cue_positive'],strict_CV_pass=goodcv,strict_CV_fail=~goodcv,nonCV=z['mask_nonCV'],cue_nonCV=z['mask_nonCV_cue_positive'],start=start,fallback=fallback,no_fallback=~fallback,start_fallback=start&fallback,start_no_fallback=start&~fallback,C1=c1,C2=c2,C3=c3)
def position_knn(bank,d):
    assert not np.intersect1d(bank['track'],d['track']).size
    X=bank['obs'][:,-1].astype(float);mu=X.mean(0);sd=np.where(X.std(0)<1e-6,1.,X.std(0));X=(X-mu)/sd;queries=(d['position'].astype(float)-mu)/sd;tree=cKDTree(X);ixs=[];ds=[]
    for begin in range(0,len(queries),128):
        a.guard();batch=queries[begin:begin+128];dist,_=tree.query(batch,k=32,workers=1)
        for j,row in enumerate(batch):
            pool=np.array(tree.query_ball_point(row,np.nextafter(dist[j,-1]+1e-12,np.inf)),int);dd=np.linalg.norm(X[pool]-row,axis=1);order=np.lexsort((bank['ids'][pool],dd))[:32];ixs.append(pool[order]);ds.append(dd[order])
    ix=np.array(ixs);dis=np.array(ds)
    for j in np.linspace(0,len(queries)-1,min(32,len(queries)),dtype=int):
        dd=np.linalg.norm(X-queries[j],axis=1);bix=np.lexsort((bank['ids'],dd))[:32];assert np.array_equal(ix[j],bix)
    cvbank=(bank['obs'][:,-1]-bank['obs'][:,-2]).astype(float)[:,None]*np.arange(1,13)[None,:,None]
    pred=a.cv(d['obs'])+(bank['gt']-cvbank)[ix].mean(1)
    a.save(f'D7_position_{d["fold"]}.npz',sample_id=d['ids'],track=d['track'],pred=pred,GT=d['gt'],neighbor_ids=bank['ids'][ix],distances=dis,stats_mean=mu,stats_std=sd,train_ids=bank['ids'])
    return pred
def main():
    a.guard()
    for mf in [V/'input_manifest.json',P/'stage3_knn_memory/completion_review/input_manifest.json']:
        a.lock(mf);a.EXPECTED.update(a.read(mf)['files'])
    for p in [Path(__file__),H/'DESIGN.md',H/'protocol.json',V/'analyze_v2.py',P/'claim_evidence_followup_20261009/saved_reader_recovery/analyze_saved.py',Path('C:/Users/linke/Downloads/ped_model_diagnosis_and_understanding_plan.md')]:a.lock(p)
    assert auc(np.zeros(4),np.array([True,False,True,False]))==.5
    rm=np.ones((2,12,2));rg=rm*.25;assert total(rm*rg)/total(rm*rm)==.25
    a.write('PREFLIGHT.json',dict(status='PASS_NUMERIC_FIXTURES',no_torch=True,resource=a.RES))
    out={k:[] for k in ('D1a','D2','D3','D6','D7','D5c')};shift=[]
    for fold in Q['folds']:
        d=a.load(P/f'data/{fold}_evaluation.npz');d['fold']=fold;bank=a.load(P/f'stage3_knn_memory/data/{fold}_bank.npz');pos=position_knn(bank,d)
        rp=a.D/f'{fold}_data_train.npy';cp=a.D/f'{fold}_num_train.npy';a.lock(rp);a.lock(cp);raw=np.load(rp,mmap_mode='r');counts=np.load(cp,mmap_mode='r');obs=np.array(raw[bank['ids'],:,:8,-2:],np.float32);valid=np.minimum(counts[bank['ids']].astype(int),9);assert np.array_equal(obs[:,0],bank['obs'])
        cal=a.read(P/f'calibration/{fold}.json');a.lock(P/f'calibration/{fold}.json');R=a.read(V/'CALIBRATION.json')[fold]['R'];a.lock(V/'CALIBRATION.json')
        tr=a.neighborhood(obs,valid,R);te=a.neighborhood(d['obs'],d['valid'],R);cvtrain=(bank['obs'][:,-1]-bank['obs'][:,-2]).astype(float)[:,None]*np.arange(1,13)[None,:,None];aatr,fftr=a.err(cvtrain,bank['gt']);dist=a.neighborhood(obs,valid)['nearest']
        trainfeat=dict(speed_mps=tr['speed']/.4,density=tr['density'],stationary=(tr['speed']<.02).astype(float));testfeat=dict(speed_mps=te['speed']/.4,density=te['density'],stationary=(te['speed']<.02).astype(float))
        for name in trainfeat:shift.append(dict(fold=fold,feature=name,train_n=len(obs),test_n=len(te['speed']),KS_distance=float(ks_2samp(trainfeat[name],testfeat[name]).statistic),train_mean=a.mean(trainfeat[name]),test_mean=a.mean(testfeat[name]),train_p95=float(np.quantile(trainfeat[name],.95)),test_p95=float(np.quantile(testfeat[name],.95)),not_iid_significance=True))
        # D5c bin edges based solely on train future-CV closest distances, not test response.
        tv=(obs[:,:,-1]-obs[:,:,-2]).astype(float);tx=obs[:,:,-1].astype(float);pr=tx[:,:,None]+tv[:,:,None]*np.arange(1,13)[None,None,:,None];legal=np.arange(1,9)[None]<valid[:,None];trainclose=np.where(legal[:,:,None],np.linalg.norm(pr[:,1:]-pr[:,:1],axis=-1),np.inf).min((1,2));edges=np.unique(np.quantile(trainclose[trainclose<1.],[.25,.5,.75]));nb=a.neighborhood(d['obs'],d['valid']);sp=np.linalg.norm(d['obs'][:,0,-1]-d['obs'][:,0,-2],axis=1);start=(sp<.02)&(np.linalg.norm(d['gt'][:,-1],axis=1)>cal['cv_fde_quantiles'][1])
        a.save(f'FEATURES_{fold}.npz',sample_id=d['ids'],track=d['track'],train_speed=trainfeat['speed_mps'],test_speed=testfeat['speed_mps'],train_density=tr['density'],test_density=te['density'],train_stationary=trainfeat['stationary'],test_stationary=testfeat['stationary'],start=start,train_closest=trainclose,D5c_edges=edges)
        del obs,raw,counts,pr;gc.collect()
        kn={kind:a.old_job(P/f'stage3_knn_memory/jobs/{fold}_{kind}_all_train',3)['pred'] for kind in ('velocity_only','position_velocity')};kn['position_only']=pos
        for seed in Q['seeds']:
            a.guard();z,d0=v.read_model(fold,seed);assert np.array_equal(d0['ids'],d['ids']);p=z['model_paths'];GT=z['GT'];CV=z['CV'];top=z['model_logits'].argmax(1);idx=np.arange(len(p));mu=p.mean(1);tp=p[idx,top];den=((GT-CV)**2).sum((1,2));t=z['track'];groups=subsets(z,d,start);cvpass=groups['strict_CV_pass'];strong=cal['strongest_simple'];baseA=z[strong+'_ADE'];baseF=z[strong+'_FDE']
            for method,pred in [('top1',tp),('meanK',mu)]:
                rm=pred-CV;rg=GT-CV;r2=(rm*rm).sum((1,2));dot=(rm*rg).sum((1,2));num=((pred-GT)**2).sum((1,2));aa,ff=a.err(pred,GT);good=(aa<=.2)&(ff<=.4);delta=den-num
                for tag,m in groups.items():
                    n=int(m.sum());rec=dict(fold=fold,seed=seed,method=method,subset=tag,n=n,tracks=len(np.unique(t[m])),skill=a.ci(num[m],den[m],t[m],True),lambda_star=lambda_ci(dot[m],r2[m],t[m]),mean_residual_squared=stat(r2[m],t[m]),delta_SSE_sum=total(delta[m]),ADE_delta_CV=stat(aa[m]-z['CV_last_ADE'][m],t[m]),FDE_delta_CV=stat(ff[m]-z['CV_last_FDE'][m],t[m]),ADE_delta_strongest=stat(aa[m]-baseA[m],t[m]),FDE_delta_strongest=stat(ff[m]-baseF[m],t[m]),curve=[dict(lambda_value=lam,skill=a.ratio(total(2*lam*dot[m]-lam*lam*r2[m]),total(den[m]))) for lam in Q['lambda_grid']],kept=int((cvpass&good&m).sum()),lost=int((cvpass&~good&m).sum()),rescue=int((~cvpass&good&m).sum()),both_fail=int((~cvpass&~good&m).sum()),harm_conditional=a.ci((cvpass&~good&m).astype(float)[m],cvpass[m].astype(float),t[m]),rescue_conditional=a.ci((~cvpass&good&m).astype(float)[m],(~cvpass[m]).astype(float),t[m]))
                    assert rec['kept']+rec['lost']+rec['rescue']+rec['both_fail']==n;out['D2'].append(rec)
                    if tag in ('all','P1','fallback','no_fallback','start','start_fallback','start_no_fallback'):
                        out['D1a'].append({k:rec[k] for k in ('fold','seed','method','subset','n','tracks','skill','delta_SSE_sum','ADE_delta_CV','ADE_delta_strongest','FDE_delta_strongest')})
                a.save(f'D2_{fold}_seed{seed}_{method}.npz',sample_id=d['ids'],track=t,den=den,num=num,dot=dot,r2=r2,ADE=aa,FDE=ff,delta_ADE_CV=aa-z['CV_last_ADE'],delta_ADE_strongest=aa-baseA,delta_FDE_strongest=ff-baseF,CV_pass=cvpass,model_pass=good,**{'mask_'+k:x for k,x in groups.items()})
            _,_,_,_,spread,sse=v.selection(p,GT,z['model_logits']);err=((mu-GT)**2).sum((1,2));bins=np.searchsorted(Q['spread_bins_SSE_m2'],spread,side='right')
            # D6 GT-free deterministic selectors, complete trajectory squared medoid and nearest CV.
            med=((p-mu[:,None])**2).sum((2,3)).argmin(1);near=((p-CV[:,None])**2).sum((2,3)).argmin(1);ADE,FDE=a.err(p,GT);choices={'top1':top,'nearest_meanK_squared_medoid':med,'nearest_CV_squared':near}
            for tag in ('all','P1'):
                m=groups[tag];out['D3'].append(dict(fold=fold,seed=seed,subset=tag,n=int(m.sum()),spread_over_err=a.ci(spread[m],err[m],t[m]),spread_mean=stat(spread[m],t[m]),spread_p95=float(np.quantile(spread[m],.95)) if m.any() else None,spread_discriminates_CV_fail=auc_ci(spread[m],~cvpass[m],t[m]),spread_spearman_error=float(spearmanr(spread[m],err[m]).statistic) if m.sum()>1 else None,bins=[dict(bin=j,n=int((m&(bins==j)).sum()),err_over_spread=a.ci(err[m&(bins==j)],spread[m&(bins==j)],t[m&(bins==j)])) for j in range(7)],groups={g:stat(spread[m&groups[g]],t[m&groups[g]]) for g in ('strict_CV_pass','strict_CV_fail','cue_nonCV','nonCV')}))
                for tau in Q['D6_taus']:
                    good=(ADE<=tau)&(FDE<=2*tau);ng=good.sum(1);present=ng>0;mixed=(ng>0)&(ng<p.shape[1]);aaauc=np.full(len(p),np.nan)
                    for begin in range(0,len(p),256):
                        a.guard();l=z['model_logits'][begin:begin+256];g=good[begin:begin+256];pair=g[:,:,None]&~g[:,None,:];greater=l[:,:,None]>l[:,None,:];equal=l[:,:,None]==l[:,None,:];bottom=pair.sum((1,2));aaauc[begin:begin+256]=np.divide(((greater+.5*equal)*pair).sum((1,2)),bottom,out=np.full(len(l),np.nan),where=bottom>0)
                    hits={name:stat(good[idx,ix][m&present],t[m&present]) for name,ix in choices.items()};hits['uniform_random_expectation']=stat((ng/20)[m&present],t[m&present]);mm=m&present
                    out['D6'].append(dict(fold=fold,seed=seed,subset=tag,tau=tau,n=int(m.sum()),good_present=int(mm.sum()),mixed_n=int((m&mixed).sum()),conditional_hits=hits,top1_minus_medoid=stat((good[idx,top].astype(float)-good[idx,med])[mm],t[mm]),top1_minus_nearest_CV=stat((good[idx,top].astype(float)-good[idx,near])[mm],t[mm]),within_window_logit_AUROC=stat(aaauc[m&mixed],t[m&mixed])))
            a.save(f'D3D6_{fold}_seed{seed}.npz',sample_id=d['ids'],track=t,mask_P1=groups['P1'],CV_pass=cvpass,spread=spread,error=err,spread_bin=bins,ADE_all=ADE,FDE_all=FDE,logits=z['model_logits'],top=top,medoid=med,nearest_CV=near)
            for method,pred in kn.items():
                aa,ff=a.err(pred,GT);num=((pred-GT)**2).sum((1,2));tn=((tp-GT)**2).sum((1,2))
                for tag in ('all','P1'):
                    m=groups[tag];out['D7'].append(dict(fold=fold,seed=seed,subset=tag,method=method,n=int(m.sum()),skill=a.ci(num[m],den[m],t[m],True),skill_gain_vs_top1=a.ci((tn-num)[m],den[m],t[m]),ADE_delta_top1=stat((aa-z['top1_ADE'])[m],t[m]),FDE_delta_top1=stat((ff-z['top1_FDE'])[m],t[m])))
            # saved real no-neighbor responses on previous fixed selected-ID population.
            selected=a.load(P/f'stage2_raw_interventions/data/{fold}_selected.npz');zz=a.old_job(P/f'stage2_raw_interventions/jobs/{fold}_seed{seed}/noNbr',2);assert np.array_equal(selected['ids'],zz['sample_id']);ns=a.neighborhood(selected['obs'],selected['valid']);vo=(selected['obs'][:,:,-1]-selected['obs'][:,:,-2]).astype(float);xx=selected['obs'][:,:,-1].astype(float);pr=xx[:,:,None]+vo[:,:,None]*np.arange(1,13)[None,None,:,None];legal=np.arange(1,9)[None]<selected['valid'][:,None];closest=np.where(legal[:,:,None],np.linalg.norm(pr[:,1:]-pr[:,:1],axis=-1),np.inf).min((1,2));bd=np.searchsorted(edges,closest,side='right');g=((selected['gt'][:,-1]-a.cv(selected['obs'])[:,-1])*ns['away']).sum(1);n=len(g)
            for method in ('top1','meanK'):
                before=zz['base_paths'][np.arange(n),zz['base_logits'].argmax(1)] if method=='top1' else zz['base_paths'].mean(1);after=zz['paths'][np.arange(n),zz['logits'].argmax(1)] if method=='top1' else zz['paths'].mean(1);response=((before[:,-1]-after[:,-1])*ns['away']).sum(1)
                for j in range(len(edges)+1):
                    m=ns['C3']&ns['away_valid']&(bd==j);out['D5c'].append(dict(fold=fold,seed=seed,method=method,bin=j,train_edges=edges.tolist(),n=int(m.sum()),model_response=stat(response[m],selected['track'][m]),GT_response=stat(g[m],selected['track'][m]),distance_mean=a.mean(closest[m])))
                a.save(f'D5c_{fold}_seed{seed}_{method}.npz',sample_id=selected['ids'],track=selected['track'],closest=closest,bin=bd,C3=ns['C3'],away_valid=ns['away_valid'],response=response,GTresponse=g,train_edges=edges)
            del z,p;gc.collect()
        print('COMPLETE',fold,round(time.perf_counter()-a.START,2),flush=True)
    for name,rr in out.items():rows(name,rr)
    rows('D7_SHIFT',shift)
    for p,digest in a.LOCKS.items():a.guard();assert a.sha(p)==digest,('FINAL_SHA',p)
    a.write('input_manifest.json',dict(files=a.LOCKS));a.write('RESULT.json',dict(status='PASS_T0_AND_SAVED_D5C',record_counts={k:len(rr) for k,rr in out.items()},output_sha={str(p):a.sha(p) for p in H.glob('*.npz')},summary_sha=a.OUT,input_files=len(a.LOCKS),resource=a.RES,elapsed_seconds=time.perf_counter()-a.START,no_training_or_forward=True,whole_plan_complete=False))
if __name__=='__main__':
    try:main()
    except Exception:a.write('FAILURE.json',dict(status='STOP',traceback=traceback.format_exc(),resource=a.RES));raise
