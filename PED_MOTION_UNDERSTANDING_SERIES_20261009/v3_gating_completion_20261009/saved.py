import os,sys,time,importlib.util,gc,traceback
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,rankdata
H=Path(__file__).resolve().parent;P=H.parent;D=P/'model_diagnosis_20261009'
spec=importlib.util.spec_from_file_location('v3_saved_utils',P/'claim_evidence_followup_20261009/saved_reader_recovery/analyze_saved.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
Q=a.read(H/'protocol.json');a.H=H;a.Q=Q;a.START=time.perf_counter()

def dist(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,q=np.quantile(x,[0,.05,.25,.5,.75,.95,1]).tolist() if len(x) else None)
def motion(p):
    v=np.diff(np.concatenate((np.zeros_like(p[:,:1]),p),1),axis=1)/.4;ac=np.diff(v,axis=1)/.4;j=np.diff(ac,axis=1)/.4
    v,ac,j=[np.linalg.norm(x,axis=-1).ravel() for x in (v,ac,j)]
    return dict(speed_p95=float(np.quantile(v,.95)),accel_mean=float(ac.mean()),accel_p95=float(np.quantile(ac,.95)),jerk_mean=float(j.mean()),jerk_p95=float(np.quantile(j,.95))) if len(p) else None
def point(pred,z,m,reference=None):
    gt,cv,t=z['GT'],z['CV'],z['track'];den=((gt-cv)**2).sum((1,2));num=((pred-gt)**2).sum((1,2));aa,ff=a.err(pred,gt);ca,cf=a.err(cv,gt);cp=(ca<=.2)&(cf<=.4);gp=(aa<=.2)&(ff<=.4)
    out=dict(n=int(m.sum()),tracks=len(np.unique(t[m])),skill=a.ci(num[m],den[m],t[m],True),ADE=a.mean_ci(aa[m],t[m]),FDE=a.mean_ci(ff[m],t[m]),ADE_p95=float(np.quantile(aa[m],.95)) if m.any() else None,FDE_p95=float(np.quantile(ff[m],.95)) if m.any() else None,coverage=a.mean_ci(gp[m].astype(float),t[m]),lost=int((cp&~gp&m).sum()),rescue=int((~cp&gp&m).sum()),kept=int((cp&gp&m).sum()),thresholds=[dict(tau=tau,coverage=float(((aa<=tau)&(ff<=2*tau))[m].mean()) if m.any() else None) for tau in Q['taus']],motion=motion(pred[m]))
    if reference is not None:
        rn=((reference-gt)**2).sum((1,2));ra,rf=a.err(reference,gt);rg=(ra<=.2)&(rf<=.4)
        out.update(skill_gain=a.ci((rn-num)[m],den[m],t[m]),coverage_gain=a.mean_ci((gp.astype(float)-rg)[m],t[m]),lost_reduction=int(((cp&~rg)&m).sum()-(cp&~gp&m).sum()),rescue_reduction=int(((~cp&rg)&m).sum()-(~cp&gp&m).sum()))
    return out
def groups(z):
    aa,ff=a.err(z['CV'],z['GT']);return dict(all=np.ones(len(aa),bool),P1=z['P1'],strict_CV_pass=(aa<=.2)&(ff<=.4))
def normstats(r,g,t,m):
    u=(r*r).sum((1,2));v=(g*g).sum((1,2));dot=(r*g).sum((1,2));nn=np.sqrt(u*v);valid=nn>0;c=np.divide(dot,nn,out=np.full_like(dot,np.nan),where=valid);us,vs,ds=u[m].sum(),v[m].sum(),dot[m].sum();lam=ds/us if us>0 else None;cos=ds/np.sqrt(us*vs) if us*vs>0 else None;lc=np.clip(lam,0,1) if lam is not None else None
    return dict(lambda_star_TEST_only=lam,global_signed_cosine=cos,oracle_skill_cosine_squared=cos*cos if cos is not None else None,clipped_oracle_skill=(2*lc*ds-lc*lc*us)/vs if lc is not None and vs>0 else None,original_skill=(2*ds-us)/vs if vs>0 else None,window_cosine=dist(c[m]),undefined_cosine=int((m&~valid).sum()),negative_cosine_fraction=float((c[m&valid]<0).mean()) if (m&valid).any() else None,model_residual_norm=dist(np.sqrt(u[m])),GT_residual_norm=dist(np.sqrt(v[m])),signed_cosine_mean_CI=a.mean_ci(c[m&valid],t[m&valid]))
def standard(X):
    mu=X.mean(0);sd=X.std(0);mu[0]=0;sd[0]=1;sd=np.where(sd<Q['std_floor'],1,sd);return mu,sd
def solve_gate(X,r,g):
    u=(r*r).sum((1,2));v=(r*g).sum((1,2));A=X.T@(X*u[:,None])+np.eye(X.shape[1])*Q['ridge_alpha'];b=X.T@v;return np.linalg.solve(A,b),A,b
def slope(x,y,t,m):
    m=m&np.isfinite(x)&np.isfinite(y);x=x[m];y=y[m];t=t[m]
    if len(x)<2 or np.var(x)==0:return dict(value=None,CI=None)
    _,idx=np.unique(t,return_inverse=True);T=idx.max()+1;mom=np.stack([np.bincount(idx,weights=v) for v in (np.ones(len(x)),x,y,x*x,x*y)],1)
    def calc(z):n,sx,sy,sxx,sxy=np.moveaxis(z,-1,0);bot=sxx-sx*sx/n;return np.divide(sxy-sx*sy/n,bot,out=np.full_like(bot,np.nan),where=bot>0)
    rng=np.random.default_rng(Q['stat_seed']);vals=[]
    for begin in range(0,Q['bootstrap_reps'],100):a.guard();ii=rng.integers(T,size=(min(100,Q['bootstrap_reps']-begin),T));vv=calc(mom[ii].sum(1));vals.extend(vv[np.isfinite(vv)].tolist())
    return dict(value=float(calc(mom.sum(0))),CI=np.quantile(vals,[.025,.5,.975]).tolist() if vals else None,n=len(x),tracks=T)
def auc(score,y):
    n1=int(y.sum());n0=len(y)-n1
    return float((rankdata(score)[y].sum()-n1*(n1+1)/2)/(n1*n0)) if n1*n0 else None
def main():
    a.guard()
    for manifest in (D/'input_manifest.json',D/'probes/input_manifest.json',D/'supplement_input_manifest.json'):
        a.lock(manifest)
        for p,s in a.read(manifest)['files'].items():
            if p in a.EXPECTED:assert a.EXPECTED[p]==s
            a.EXPECTED[p]=s
    for p in (Path(__file__),H/'protocol.json',H/'DESIGN.md',Path('C:/Users/linke/Downloads/ped_next_steps_v3_gating_and_completion.md')):a.lock(p)
    # Weighted contraction avoids constructing a giant (N*24,D) matrix.
    rng=np.random.default_rng(7);X=rng.normal(size=(17,9));r=rng.normal(size=(17,12,2));g=rng.normal(size=r.shape);w,A,b=solve_gate(X,r,g);expanded=(r.reshape(17,24,1)*X[:,None]).reshape(-1,9);assert np.allclose(A,expanded.T@expanded+np.eye(9),atol=1e-12,rtol=1e-12);assert np.allclose(b,expanded.T@g.ravel(),atol=1e-12,rtol=1e-12)
    a.write('PREFLIGHT.json',dict(status='PASS_WEIGHTED_RIDGE_FIXTURE',no_torch=True))
    R={k:[] for k in ('G2','G3','G7','G1','G6')};DATA={};BANK={}
    for fold in Q['folds']:
        d=a.load(P/f'data/{fold}_evaluation.npz');pr=a.load(D/f'probes/PROBES_{fold}.npz');f=a.load(D/f'FEATURES_{fold}.npz');bank=a.load(P/f'stage3_knn_memory/data/{fold}_bank.npz');calpath=P/f'calibration/{fold}.json';a.lock(calpath);cal=a.read(calpath);assert np.array_equal(d['ids'],pr['sample_id']);edges=np.array(cal['speed_edges']);trcv=np.diff(bank['obs'][:,-2:],axis=1)[:,0].astype(float)[:,None]*np.arange(1,13)[None,:,None];trnorm=np.linalg.norm((bank['gt']-trcv).reshape(len(trcv),24),axis=1);trspeed=np.linalg.norm(np.diff(bank['obs'][:,-2:],axis=1)[:,0],axis=1);BANK[fold]=(edges,trnorm,trspeed)
        for seed in Q['seeds']:
            a.guard();z0=a.old_job(P/f'jobs/{fold}_seed{seed}');paths=z0['model_paths'];logs=z0['model_logits'];top=paths[np.arange(len(paths)),logs.argmax(1)];mu=paths.mean(1);z=dict(GT=z0['GT'],CV=z0['CV'],track=z0['track'],P1=z0['mask_nonCV_cue_positive'],ids=z0['sample_id'],top1=top,meanK=mu,probe_self=pr['pred_self'],probe_social=pr['pred_social']);assert np.array_equal(z['ids'],d['ids']);rg=z['GT']-z['CV'];spread=((paths-mu[:,None])**2).sum((2,3)).mean(1);speed=np.linalg.norm((d['obs'][:,0,-1]-d['obs'][:,0,-2]).astype(float),axis=1);nbr=d['valid']-1;X=np.column_stack((np.ones(len(mu)),np.log(spread+Q['gate_log_floor']),np.sqrt(((mu-z['CV'])**2).sum((1,2))/24),speed,pr['C1'],pr['C2'],pr['C3'],nbr,logs.max(1)-logs.mean(1)));z['X']=X;DATA[fold,seed]=z
            for method in ('top1','meanK','probe_self','probe_social'):
                rm=z[method]-z['CV'];nm=np.linalg.norm(rm.reshape(len(rm),24),axis=1);ng=np.linalg.norm(rg.reshape(len(rg),24),axis=1)
                for tag,m in groups(z).items():R['G2'].append(dict(fold=fold,seed=seed,method=method,subset=tag,**normstats(rm,rg,z['track'],m)))
                for binno in range(4):
                    m=np.searchsorted(edges,speed,side='right')==binno;tm=np.searchsorted(edges,trspeed,side='right')==binno;medg=float(np.median(ng[m])) if m.any() else None;medm=float(np.median(nm[m])) if m.any() else None
                    R['G2'].append(dict(fold=fold,seed=seed,method=method,subset='speed_bin',bin=binno,train_edges=edges.tolist(),n=int(m.sum()),model_median_norm=medm,GT_median_norm=medg,model_to_GT_median_ratio=medm/medg if medg else None,train_GT_norm=dist(trnorm[tm])))
            for tag,m in dict(all=np.ones(len(mu),bool),P1=z['P1'],stationary=speed<.02,CV_pass=groups(z)['strict_CV_pass'],CV_fail=~groups(z)['strict_CV_pass'],**{f'speed{b}':np.searchsorted(edges,speed,side='right')==b for b in range(4)}).items():
                sp=spread[m];cvfail=~groups(z)['strict_CV_pass'][m];cut=float(np.quantile(sp,.95)) if len(sp) else 0
                R['G7'].append(dict(fold=fold,seed=seed,subset=tag,n=int(m.sum()),spread=dist(sp),CV_failure_fraction=float(cvfail.mean()) if len(sp) else None,spread_CV_failure_AUC=auc(sp,cvfail),stationary_fraction=float((speed[m]<.02).mean()) if m.any() else None,spread_speed_spearman=float(spearmanr(sp,speed[m]).statistic) if len(sp)>1 and np.var(sp)>0 and np.var(speed[m])>0 else None,top5percent_spread_mass=float(sp[sp>=cut].sum()/sp.sum()) if sp.sum()>0 else None))
            del paths,z0;gc.collect()
        print('READ',fold,round(time.perf_counter()-a.START,2),flush=True)
    for fold in Q['folds']:
        for seed in Q['seeds']:
            a.guard();z=DATA[fold,seed];donors=[DATA[g,seed] for g in Q['folds'] if g!=fold];X=np.concatenate([d['X'] for d in donors]);G=np.concatenate([d['GT']-d['CV'] for d in donors]);mx,sd=standard(X);ZX=(X-mx)/sd;TX=(z['X']-mx)/sd;targetmu,targetsd=standard(z['X']);trans=(z['X']-targetmu)/targetsd;out={};fit={}
            for method in ('top1','meanK'):
                rm=z[method]-z['CV'];RR=np.concatenate([d[method]-d['CV'] for d in donors]);scalar=np.clip((RR*G).sum()/(RR*RR).sum(),0,1);w,A,b=solve_gate(ZX,RR,G);ws,As,bs=solve_gate(ZX[:,:2],RR,G);fit[method]=dict(w=w,A=A,b=b,spread_w=ws,spread_A=As,spread_b=bs,scalar=float(scalar));lams=dict(identity=np.ones(len(rm)),scalar=np.full(len(rm),scalar),spread=np.clip(TX[:,:2]@ws,0,1),gate=np.clip(TX@w,0,1),transductive=np.clip(trans@w,0,1));u=(rm*rm).sum();free=float((rm*(z['GT']-z['CV'])).sum()/u) if u else 0;lams['TEST_oracle_ONLY']=np.full(len(rm),free)
                for kind,lam in lams.items():
                    pred=z['CV']+rm*lam[:,None,None];out[method+'_'+kind]=pred;out[method+'_'+kind+'_lambda']=lam
                    for tag,m in groups(z).items():R['G1'].append(dict(fold=fold,seed=seed,method=method,rule=kind,subset=tag,lambda_distribution=dist(lam[m]),**point(pred,z,m,z[method])))
                out[method+'_rm']=rm
            for probe in ('probe_self','probe_social'):
                RM=np.concatenate([d['meanK']-d['CV'] for d in donors]);RP=np.concatenate([d[probe]-d['CV'] for d in donors]);rr=np.stack((RM.ravel(),RP.ravel()),1);co=np.linalg.lstsq(rr,G.ravel(),rcond=None)[0];pred=z['CV']+co[0]*(z['meanK']-z['CV'])+co[1]*(z[probe]-z['CV']);out[probe+'_blend']=pred;fit[probe]=dict(coefficients=co,normal_matrix=rr.T@rr,normal_rhs=rr.T@G.ravel())
                for tag,m in groups(z).items():R['G3'].append(dict(fold=fold,seed=seed,probe=probe,subset=tag,a=float(co[0]),b=float(co[1]),**point(pred,z,m,z['meanK']),gain_vs_probe=a.ci((((z[probe]-z['GT'])**2).sum((1,2))-((pred-z['GT'])**2).sum((1,2)))[m],((z['GT']-z['CV'])**2).sum((1,2))[m],z['track'][m]),probe_skill=point(z[probe],z,m)['skill']))
            a.save(f'POINTS_{fold}_seed{seed}.npz',sample_id=z['ids'],track=z['track'],GT=z['GT'],CV=z['CV'],P1=z['P1'],X=z['X'],train_mu=mx,train_std=sd,target_mu=targetmu,target_std=targetsd,top1=z['top1'],meanK=z['meanK'],probe_self=z['probe_self'],probe_social=z['probe_social'],**out);a.write(f'fit_{fold}_seed{seed}.json',dict(donors=[g for g in Q['folds'] if g!=fold],donor_counts=[len(d['GT']) for d in donors],target_label_never_fit=True,upstream_not_fully_nested=True,fits=fit))
        print('FIT',fold,round(time.perf_counter()-a.START,2),flush=True)
    # G6 does not need any forward. Original selected-window with/no-neighbor outputs.
    for fold in Q['folds']:
        d=a.load(P/f'stage2_raw_interventions/data/{fold}_selected.npz');f=a.load(D/f'FEATURES_{fold}.npz');obs=d['obs'].astype(float);valid=np.arange(1,9)[None]<d['valid'][:,None];vel=np.diff(obs,axis=2)[:,:,-1];rel=obs[:,:1,-1:,]-obs[:,1:,-1:,];diff=rel[:,:,None,0]+(vel[:,:1]-vel[:,1:])[:,:,None]*np.arange(1,13)[None,None,:,None];dis=np.linalg.norm(diff,axis=-1);ti=dis.argmin(2);minimum=dis.min(2);vec=diff[np.arange(len(obs))[:,None],np.arange(8)[None],ti];unit=np.divide(vec,minimum[:,:,None],out=np.zeros_like(vec),where=minimum[:,:,None]>0);weight=np.exp(-minimum)/(1+(ti+1)/12)*valid;wu=(unit*weight[:,:,None]).sum(1);wn=np.linalg.norm(wu,axis=1);wu=np.divide(wu,wn[:,None],out=np.zeros_like(wu),where=wn[:,None]>0);nb=a.neighborhood(d['obs'],d['valid']);closest=np.where(valid,minimum,np.inf).min(1);near=nb['nearest'];edges=f['D5c_edges'];bd=np.searchsorted(edges,closest,side='right');bn=np.searchsorted(Q['nearest_now_bins'],near,side='right');cv=a.cv(d['obs']);t=d['track'];weighted_defined=wn>0;nearest_defined=nb['away_valid'];gtresp=d['gt'][:,-1]-cv[:,-1]
        for seed in Q['seeds']:
            z=a.old_job(P/f'stage2_raw_interventions/jobs/{fold}_seed{seed}/noNbr',2);assert np.array_equal(z['sample_id'],d['ids'])
            for method in ('top1','meanK'):
                bp=z['base_paths'][np.arange(len(obs)),z['base_logits'].argmax(1)] if method=='top1' else z['base_paths'].mean(1);npred=z['paths'][np.arange(len(obs)),z['logits'].argmax(1)] if method=='top1' else z['paths'].mean(1);dr=bp-npred;mag=np.linalg.norm(dr.reshape(len(obs),24),axis=1);resp=(dr[:,-1]*nb['away']).sum(1);weighted=(dr[:,-1]*wu).sum(1);gtweighted=(gtresp*wu).sum(1);gtn=(gtresp*nb['away']).sum(1)
                for name,m in dict(all=np.ones(len(obs),bool),C3=nb['C3'],**{f'closest_bin{b}':bd==b for b in range(len(edges)+1)},**{f'nearest_bin{b}':bn==b for b in range(5)},**{f'nbr{b}':d['valid']-1==b for b in range(9)}).items():
                    mm=m&nearest_defined;wm=m&weighted_defined;R['G6'].append(dict(fold=fold,seed=seed,method=method,subset=name,n=int(m.sum()),magnitude=dist(mag[m]),magnitude_mean_CI=a.mean_ci(mag[m],t[m]),signed_nearest=a.mean_ci(resp[mm],t[mm]),away_positive_nearest=a.mean_ci((resp[mm]>0).astype(float),t[mm]),signed_weighted=a.mean_ci(weighted[wm],t[wm]),away_positive_weighted=a.mean_ci((weighted[wm]>0).astype(float),t[wm]),model_slope_nearest=slope(closest,resp,t,mm&nb['C3']),GT_slope_nearest=slope(closest,gtn,t,mm&nb['C3']),model_slope_weighted=slope(closest,weighted,t,wm&nb['C3']),GT_slope_weighted=slope(closest,gtweighted,t,wm&nb['C3'])))
                a.save(f'RESPONSE_{fold}_seed{seed}_{method}.npz',sample_id=d['ids'],track=t,magnitude=mag,nearest_response=resp,weighted_response=weighted,GT_nearest=gtn,GT_weighted=gtweighted,closest=closest,nearest_now=near,weighted_direction=wu,weight=weight,nearest_defined=nearest_defined,weighted_defined=weighted_defined,C3=nb['C3'])
        print('G6',fold,round(time.perf_counter()-a.START,2),flush=True)
    for k,rows in R.items():a.emit(k,dict(records=rows,exploratory=True))
    for p,s in a.LOCKS.items():a.guard();assert a.sha(p)==s
    a.write('input_manifest.json',dict(files=a.LOCKS));a.write('RESULT_SAVED.json',dict(status='PASS_G1_G2_G3_G6_G7_SAVED',elapsed_seconds=time.perf_counter()-a.START,input_files=len(a.LOCKS),resource=a.RES,summary_sha=a.OUT,output_sha={str(p):a.sha(p) for p in H.glob('*.npz')},fit_sha={str(p):a.sha(p) for p in H.glob('fit_*.json')},no_torch_or_neural_training=True))
if __name__=='__main__':
    try:main()
    except Exception:a.write('FAILURE_SAVED.json',dict(status='STOP',traceback=traceback.format_exc(),resource=a.RES));raise
