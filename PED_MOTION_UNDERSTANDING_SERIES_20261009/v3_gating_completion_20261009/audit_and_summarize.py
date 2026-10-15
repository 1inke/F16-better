import os,sys,time,importlib.util,gc,traceback
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
H=Path(__file__).resolve().parent;P=H.parent;F=P/'model_diagnosis_20261009/forward_diagnostics'
spec=importlib.util.spec_from_file_location('v3_audit_saved',H/'saved.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);a=v.a;Q=v.Q;a.H=H;a.START=time.perf_counter();checks=0;errors=0.;max_absolute=0.
def check(x,y,atol=1e-12,rtol=1e-12):
    global checks,errors,max_absolute
    x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape,(x.shape,y.shape)
    if x.dtype.kind in 'USb' or y.dtype.kind in 'USb':assert np.array_equal(x,y)
    else:
        assert np.array_equal(np.isfinite(x),np.isfinite(y));ok=np.isfinite(x);delta=abs(x[ok]-y[ok]);scale=atol+rtol*abs(y[ok]);assert (delta<=scale).all(),('AUDIT_NUMERIC',float(delta.max()),float(scale.max()));errors=max(errors,float((delta/scale).max()) if len(delta) else 0);max_absolute=max(max_absolute,float(delta.max()) if len(delta) else 0)
    checks+=x.size
def matching(p,q):
    same=np.linalg.norm(p-q,axis=-1).mean((1,2));value=np.empty(len(p))
    for b in range(0,len(p),128):
        a.guard();C=np.linalg.norm(p[b:b+128,:,None]-q[b:b+128,None,:],axis=-1).mean(-1)
        for i,c in enumerate(C):r,s=linear_sum_assignment(c);value[b+i]=c[r,s].mean()
    return same,value
def score_auc(l,g):
    p=g[:,:,None]&~g[:,None,:];den=p.sum((1,2));num=((l[:,:,None]>l[:,None,:])+.5*(l[:,:,None]==l[:,None,:]))*p;return np.divide(num.sum((1,2)),den,out=np.full(len(l),np.nan),where=den>0)
def records_saved():
    saved=a.read(H/'RESULT_SAVED.json');a.lock(H/'RESULT_SAVED.json')
    for p,s in saved['output_sha'].items():a.lock(p,s)
    for p,s in saved['fit_sha'].items():a.lock(p,s)
    for k,s in saved['summary_sha'].items():a.lock(H/(k+'.json'),s)
    files={}
    for seed in Q['seeds']:
        for fold in Q['folds']:files[fold,seed]=a.load(H/f'POINTS_{fold}_seed{seed}.npz')
        for fold in Q['folds']:
            z=files[fold,seed];fit=a.read(H/f'fit_{fold}_seed{seed}.json');assert fold not in fit['donors'];donors=[files[g,seed] for g in Q['folds'] if g!=fold];X=np.concatenate([d['X'] for d in donors]);mu=X.mean(0);sd=X.std(0);mu[0]=0;sd[0]=1;sd=np.where(sd<1e-6,1,sd);check(z['train_mu'],mu);check(z['train_std'],sd);zx=(X-mu)/sd;G=np.concatenate([d['GT']-d['CV'] for d in donors]);tx=(z['X']-mu)/sd;tm=z['X'].mean(0);ts=z['X'].std(0);tm[0]=0;ts[0]=1;ts=np.where(ts<1e-6,1,ts);check(tm,z['target_mu']);check(ts,z['target_std'])
            for method in ('top1','meanK'):
                ff=fit['fits'][method];r=np.concatenate([d[method]-d['CV'] for d in donors]);u=np.einsum('nti,nti->n',r,r);dot=np.einsum('nti,nti->n',r,G);A=np.zeros((9,9));rhs=np.zeros(9)
                for b in range(0,len(X),256):A+=zx[b:b+256].T@(u[b:b+256,None]*zx[b:b+256]);rhs+=zx[b:b+256].T@dot[b:b+256]
                A+=np.eye(9);check(A,ff['A'],atol=1e-8,rtol=1e-10);check(rhs,ff['b'],atol=1e-8,rtol=1e-10);w=np.linalg.solve(A,rhs);check(w,ff['w'],atol=1e-9,rtol=1e-9);scalar=np.clip(dot.sum()/u.sum(),0,1);check(scalar,ff['scalar']);rm=z[method]-z['CV'];lams=dict(identity=np.ones(len(rm)),scalar=np.full(len(rm),scalar),gate=np.clip(tx@np.array(ff['w']),0,1),spread=np.clip(tx[:,:2]@np.array(ff['spread_w']),0,1),transductive=np.clip(((z['X']-tm)/ts)@np.array(ff['w']),0,1),TEST_oracle_ONLY=np.full(len(rm),(rm*(z['GT']-z['CV'])).sum()/(rm*rm).sum()))
                for name,lam in lams.items():check(lam,z[method+'_'+name+'_lambda']);check(z['CV']+rm*lam[:,None,None],z[method+'_'+name])
            for probe in ('probe_self','probe_social'):
                R=np.stack([np.concatenate([d['meanK']-d['CV'] for d in donors]).ravel(),np.concatenate([d[probe]-d['CV'] for d in donors]).ravel()],1);coeff=np.linalg.solve(R.T@R,R.T@G.ravel());check(coeff,fit['fits'][probe]['coefficients'],atol=1e-9,rtol=1e-9);cc=np.array(fit['fits'][probe]['coefficients']);check(z['CV']+cc[0]*(z['meanK']-z['CV'])+cc[1]*(z[probe]-z['CV']),z[probe+'_blend'])
    # Full point outcomes/CI/motion, rather than selected positive rows only.
    for name in ('G1','G3'):
        for row in a.read(H/(name+'.json'))['records']:
            z=files[row['fold'],row['seed']];base=dict(GT=z['GT'],CV=z['CV'],track=z['track'],P1=z['P1']);m=v.groups(base)[row['subset']];pred=z[row['method']+'_'+row['rule']] if name=='G1' else z[row['probe']+'_blend'];ref=z[row['method']] if name=='G1' else z['meanK'];r=v.point(pred,base,m,ref)
            for key in ('skill','ADE','FDE','coverage','skill_gain','coverage_gain'):
                if r[key]['value'] is not None:check(r[key]['value'],row[key]['value'])
                if r[key]['CI'] is not None:check(r[key]['CI'],row[key]['CI'])
            for key in ('lost','rescue','kept','lost_reduction','rescue_reduction'):check(r[key],row[key])
            for key in r['motion'] or {}:check(r['motion'][key],row['motion'][key])
    print('AUDIT_SAVED',checks,round(time.perf_counter()-a.START,2),flush=True)
    del files;gc.collect()
def records_forward():
    result=a.read(H/'RESULT_FORWARD.json');a.lock(H/'RESULT_FORWARD.json')
    for p,s in result['output_sha'].items():a.lock(p,s)
    G4=[];D1=[];GE=[];S=[]
    for fold in Q['folds']:
        new=a.load(H/f'RAW_{fold}.npz');raw=a.load(F/f'RAW_{fold}.npz') if fold in ('eth','hotel') else new;t=raw['track'];cv=raw['CV'];gt=raw['GT'];masks=dict(all=np.ones(len(t),bool),P1=raw['mask_P1'],strict_CV_pass=(a.err(cv,gt)[0]<=.2)&(a.err(cv,gt)[1]<=.4),fallback=raw['fallback'],no_fallback=~raw['fallback']);base=dict(GT=gt,CV=cv,track=t,P1=raw['mask_P1']);source=F if fold in ('eth','hotel') else H
        for seed in Q['seeds']:
            z=a.load(H/f'G4_{fold}_seed{seed}.npz');check(z['sample_id'],raw['sample_id']);top=z['logits'].argmax(1);idx=np.arange(len(t));preA,preF=a.err(z['pre_paths'],gt);postA,postF=a.err(z['post_paths'],gt)
            for tau in Q['taus']:
                goodpre=(preA<=tau)&(preF<=2*tau);goodpost=(postA<=tau)&(postF<=2*tau);uc=score_auc(z['logits'],goodpre);up=score_auc(z['logits'],goodpost)
                for subset in ('all','P1','strict_CV_pass'):
                    m=masks[subset];common=m&np.isfinite(uc)&np.isfinite(up);row=dict(fold=fold,seed=seed,tau=tau,subset=subset,n=int(m.sum()),has_refiner=bool(z['has_refiner']),common_mixed_n=int(common.sum()),pre_mixed_n=int((m&np.isfinite(uc)).sum()),post_mixed_n=int((m&np.isfinite(up)).sum()),pre_AUC=a.mean_ci(uc[m&np.isfinite(uc)],t[m&np.isfinite(uc)]),post_AUC=a.mean_ci(up[m&np.isfinite(up)],t[m&np.isfinite(up)]),paired_pre_minus_post_AUC=a.mean_ci((uc-up)[common],t[common]),pre_top1_hit=a.mean_ci(goodpre[idx,top][m].astype(float),t[m]),post_top1_hit=a.mean_ci(goodpost[idx,top][m].astype(float),t[m]),pre_set_coverage=a.mean_ci(goodpre.any(1)[m].astype(float),t[m]),post_set_coverage=a.mean_ci(goodpost.any(1)[m].astype(float),t[m]))
                    for tag,good in [('pre',goodpre),('post',goodpost)]:mm=m&good.any(1);row[tag+'_conditional_hit']=a.mean_ci(good[idx,top][mm].astype(float),t[mm])
                    G4.append(row)
            for name in Q['heading_variants']:
                p=a.load(source/f'{fold}_seed{seed}_{name}.npz');check(p['sample_id'],raw['sample_id']);pp=p['paths'];ll=p['logits']
                for method,pred in [('top1',pp[np.arange(len(t)),ll.argmax(1)]),('meanK',pp.mean(1))]:
                    for subset,m in masks.items():D1.append(dict(fold=fold,seed=seed,variant=name,method=method,subset=subset,**v.point(pred,base,m)))
                for tr in ('rotate','mirror'):
                    p2=a.load(source/f'{fold}_seed{seed}_{name}_{tr}.npz');same,matched=matching(pp,p2['inverse_paths']);check(same,p2['same_slot']);check(matched,p2['matched'])
                    for subset,m in masks.items():GE.append(dict(fold=fold,seed=seed,variant=name,transform=tr,subset=subset,n=int(m.sum()),matched_mean=a.mean_ci(matched[m],t[m]),matched_p95=float(np.quantile(matched[m],.95)) if m.any() else None,same_slot_mean=a.mean_ci(same[m],t[m]),same_slot_p95=float(np.quantile(same[m],.95)) if m.any() else None))
            angles=[]
            for angle in Q['angles_degrees']:
                zt=a.load(source/f'{fold}_seed{seed}_angle{angle}.npz');angles.append(zt['paths'][np.arange(len(t)),zt['logits'].argmax(1)])
            tt=a.load(source/f'{fold}_seed{seed}_TTA.npz');ens=np.mean(angles,axis=0);amp=np.linalg.norm(np.stack(angles)-ens[None],axis=-1).mean((0,2));check(ens,tt['top1_average']);check(amp,tt['angular_path_deviation'])
            for subset,m in masks.items():D1.append(dict(fold=fold,seed=seed,variant='four_fixed_orientation_TTA_OOD_not_candidate',method='top1_average',subset=subset,**v.point(ens,base,m)))
            social=a.load(source/f'{fold}_seed{seed}_SOCIAL_paths.npz');original=a.load(source/f'{fold}_seed{seed}_original.npz');normal=raw['normal'];normal_ok=raw['axis_defined'];out={}
            for method in ('top1','meanK'):
                def select(paths,logs):return paths[np.arange(len(t)),logs.argmax(1)] if method=='top1' else paths.mean(1)
                bp=select(original['paths'],original['logits']);mp=select(social['mirror_paths'],social['mirror_logits']);no=select(social['noNbr_paths'],social['noNbr_logits']);l=((bp[:,-1]-no[:,-1])*normal).sum(1);lp=((mp[:,-1]-no[:,-1])*normal).sum(1);den=abs(l)+abs(lp);per=np.divide(abs(l+lp),den,out=np.full_like(den,np.nan),where=den>0);out[method+'_l']=l;out[method+'_lprime']=lp;out[method+'_per_window_absolute_failure']=per
                if source==F:
                    old=a.load(F/f'{fold}_seed{seed}_SOCIAL_{method}.npz');check(l,old['l']);check(lp,old['lprime'])
                for subset,m in dict(all=normal_ok,P1=normal_ok&raw['mask_P1'],C3=normal_ok&raw['C3']).items():valid=m&(den>0);S.append(dict(fold=fold,seed=seed,method=method,subset=subset,n=int(m.sum()),undefined_axis=int((~normal_ok).sum()),zero_response=int((m&(den==0)).sum()),global_signed_A=a.ci((l+lp)[m],den[m],t[m]),global_absolute_failure=a.ci(abs(l+lp)[m],den[m],t[m]),per_window_absolute_failure_mean=a.mean_ci(per[valid],t[valid]),per_window_absolute_failure_distribution=v.dist(per[valid]),response_magnitude=a.mean_ci(den[m],t[m]),old_original_numeric_axis_reused=source==F))
            a.save(f'ANTISYMMETRY_{fold}_seed{seed}.npz',sample_id=raw['sample_id'],track=t,axis_defined=normal_ok,C3=raw['C3'],P1=raw['mask_P1'],**out)
            print('AUDIT_FORWARD',fold,seed,round(time.perf_counter()-a.START,2),flush=True)
    for name,rr in [('G4',G4),('G5_HEADING',D1),('G5_GEOMETRY',GE),('G5_SOCIAL',S)]:a.emit(name,dict(records=rr,exploratory=True))
def main():
    a.guard()
    for manifest in (H/'input_manifest.json',H/'forward_input_manifest.json'):
        a.lock(manifest)
        for p,s in a.read(manifest)['files'].items():
            if p in a.EXPECTED:assert a.EXPECTED[p]==s
            a.EXPECTED[p]=s
    a.lock(Path(__file__));records_saved();records_forward()
    frozen=a.read(H/'FROZEN_AUDIT.json');assert len(frozen['records'])==15 and all(r['all_parameter_bool_statbuffer_values_EXACT'] for r in frozen['records'])
    for p,digest in a.LOCKS.items():a.guard();assert a.sha(p)==digest
    a.write('audit_input_manifest.json',dict(files=a.LOCKS));a.write('REVIEW.json',dict(status='PASS_SAVED_POINT_GATES_FITS_AND_FULL_FROZEN_COMPLETION_ARITHMETIC',checks=checks,max_absolute_numeric_error=max_absolute,max_tolerance_fraction=errors,elapsed_seconds=time.perf_counter()-a.START,input_files=len(a.LOCKS),resource=a.RES,new_output_sha={str(p):a.sha(p) for p in H.glob('ANTISYMMETRY*.npz')},summary_sha=a.OUT,scope='full point fits normal equations/gated arrays/G1 G3 outcome CI/motions; all15 G4 + heading/social outcomes; all150 GTfree Hungarian matrices and summaries; TTA; original six social arithmetic; not all G2 G6 G7 CI independent second rebuild, not model replay/external review'))
if __name__=='__main__':
    try:main()
    except Exception:a.write('FAILURE_AUDIT.json',dict(status='STOP',traceback=traceback.format_exc(),checks=checks));raise
