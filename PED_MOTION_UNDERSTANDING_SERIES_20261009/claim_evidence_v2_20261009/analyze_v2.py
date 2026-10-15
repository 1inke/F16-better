"""Exploratory A1--A6. Saved observations/candidates only; no torch/training."""
import os,sys,time,json,importlib.util,gc,traceback
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent;V=P/'claim_evidence_followup_20261009/saved_reader_recovery';S=P/'stage2_raw_interventions'
spec=importlib.util.spec_from_file_location('saved_math_v1',V/'analyze_saved.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
Q=a.read(H/'protocol.json');a.H=H;a.Q=Q;a.START=time.perf_counter()
def records(name,rows):a.emit(name,dict(records=rows,exploratory=True,track_clustered_not_three_independent_populations=True))
def stat(x,t):return a.mean_ci(np.asarray(x,float),t)
def skills(nums,den,t):return {k:a.ci(x,den,t,True) for k,x in nums.items()}
def read_model(fold,seed):
    z=a.old_job(P/f'jobs/{fold}_seed{seed}');d=a.load(P/f'data/{fold}_evaluation.npz')
    assert np.array_equal(z['sample_id'],d['ids']) and np.array_equal(z['track'],d['track']) and np.array_equal(z['GT'],d['gt'])
    return z,d
def selection(p,gt,logs):
    e=p-gt[:,None];sse=(e*e).sum((2,3));top=logs.argmax(1);ix=np.arange(len(p));mu=p.mean(1)
    nums=dict(top1=sse[ix,top],meanK=((mu-gt)**2).sum((1,2)),random=sse.mean(1),oracle=sse.min(1))
    disp=((p-mu[:,None])**2).sum((2,3)).mean(1)
    assert np.allclose(nums['random'],nums['meanK']+disp,rtol=2e-13,atol=1e-10)
    strict=(sse>nums['top1'][:,None]).sum(1);tied=(sse==nums['top1'][:,None]).sum(1)-1
    rank=(strict+.5*tied)/(p.shape[1]-1)
    return nums,rank,strict/p.shape[1],top,disp,sse
def cues(obs,valid):
    v=np.diff(obs.astype(float),axis=2);speed=np.linalg.norm(v[:,0],axis=-1);first=speed[:,:3].mean(1);last=speed[:,-3:].mean(1)
    ratio=np.divide(last,first,out=np.ones_like(last),where=first>0);c1=(ratio<.75)|(ratio>1.33)|((first==0)&(last>0))
    u=v[:,0,:3].mean(1);w=v[:,0,-3:].mean(1);angle=np.degrees(np.arctan2(u[:,0]*w[:,1]-u[:,1]*w[:,0],(u*w).sum(1)))
    c2=(np.linalg.norm(u,axis=1)>=.02)&(np.linalg.norm(w,axis=1)>=.02)&(abs(angle)>15)
    pred=obs[:,:,-1,None]+v[:,:,-1,None]*np.arange(1,13)[None,None,:,None];dist=np.linalg.norm(pred[:,1:]-pred[:,:1],axis=-1)
    legal=np.arange(1,9)[None]<valid[:,None];c3=np.where(legal[:,:,None],dist,np.inf).min((1,2))<1.
    return c1,c2,c3
def gaussian(obs,valid,R):
    legal=np.arange(1,9)[None]<valid[:,None];dist=np.linalg.norm(obs[:,1:,-1].astype(float)-obs[:,:1,-1].astype(float),axis=-1)
    return (np.exp(-dist**2/(2*R**2))*legal).sum(1)
def train_calibrate():
    old=a.read(V/'CALIBRATION.json');a.lock(V/'CALIBRATION.json');out={}
    for fold in Q['folds']:
        a.guard();b=a.load(P/f'stage3_knn_memory/data/{fold}_bank.npz');ix=b['ids']
        rp=a.D/f'{fold}_data_train.npy';npth=a.D/f'{fold}_num_train.npy';a.lock(rp);a.lock(npth)
        raw=np.load(rp,mmap_mode='r');nc=np.load(npth,mmap_mode='r');obs=np.array(raw[ix,:,:8,-2:],np.float32);valid=np.minimum(nc[ix].astype(int),9)
        assert np.array_equal(obs[:,0],b['obs']);gt=(np.array(raw[ix,0,8:,-2:],np.float32)-obs[:,0,-1,None]).astype(float);assert np.array_equal(gt,b['gt'])
        nb=a.neighborhood(obs,valid);g=((gt[:,-1]-a.cv(obs)[:,-1])*nb['away']).sum(1);mask=nb['C3']&nb['away_valid']&(g>0)
        R=old[fold]['R_m'];density=gaussian(obs,valid,R);sd=float(density.std());assert R>0 and sd>0
        out[fold]=dict(R=R,density_mean=float(density.mean()),density_std=sd,positive_C3_g_n=int(mask.sum()),events={str(q):float(np.quantile(g[mask],q)) for q in Q['A5_quantiles']},training_ids_sha=a.sha(P/f'stage3_knn_memory/data/{fold}_bank.npz'))
        assert out[fold]['events']['0.75']==old[fold]['delta_g_m'];del raw,obs,b,gt;gc.collect()
    a.emit('CALIBRATION',out);return out
def a1a2():
    r1=[];r2=[]
    for fold in Q['folds']:
        for seed in Q['seeds']:
            a.guard();z,d=read_model(fold,seed);p=z['model_paths'];gt=z['GT'];nums,rank,strict,top,disp,sse=selection(p,gt,z['model_logits']);den=((gt-z['CV'])**2).sum((1,2));t=z['track'];aa,ff=a.err(p,gt);n=len(p)
            gaps=np.empty((n,len(Q['A2_taus']),3),np.uint8)
            for j,tau in enumerate(Q['A2_taus']):
                good=(aa<=tau)&(ff<=2*tau);present=good.any(1);sel=good[np.arange(n),top];gaps[:,j]=np.stack((~present,present&~sel,sel),1);assert (gaps[:,j].sum(1)==1).all()
                assert np.array_equal(present,z[f'candidate_coverage_{tau:.2f}'].astype(bool))
            for tag,m in [('all',np.ones(n,bool)),('P1',z['mask_nonCV_cue_positive'])]:
                r1.append(dict(fold=fold,seed=seed,subset=tag,n=int(m.sum()),skills=skills({k:x[m] for k,x in nums.items()},den[m],t[m]),top1_minus_random=a.ci((nums['random']-nums['top1'])[m],den[m],t[m]),percentile_minus_half=stat(rank[m]-.5,t[m]),raw_strict_percentile=stat(strict[m],t[m]),candidate_dispersion_SSE=stat(disp[m],t[m]),meanK_minus_random=a.ci(disp[m],den[m],t[m]),percentile_mean=a.mean(rank[m])))
                for j,tau in enumerate(Q['A2_taus']):r2.append(dict(fold=fold,seed=seed,subset=tag,tau=tau,n=int(m.sum()),none=stat(gaps[m,j,0],t[m]),good_not_selected=stat(gaps[m,j,1],t[m]),selected_good=stat(gaps[m,j,2],t[m])))
            a.save(f'A1A2_{fold}_seed{seed}.npz',sample_id=z['sample_id'],track=t,mask_P1=z['mask_nonCV_cue_positive'],den=den,**nums,rank=rank,raw_strict_rank=strict,dispersion_SSE=disp,gaps=gaps)
            del z,p,sse,aa,ff;gc.collect()
        print('A1A2',fold,flush=True)
    records('A1',r1);records('A2',r2)
def a3(cal):
    rows=[];extra=[]
    for seed in Q['seeds']:
        z,d=read_model('univ',seed);c1,c2,c3=cues(d['obs'],d['valid']);pattern=c1.astype(int)+2*c2.astype(int)+4*c3.astype(int);non=z['mask_nonCV'];P1=z['mask_nonCV_cue_positive'];assert np.array_equal(P1,non&(pattern>0))
        v=a.load(V/f'N2_N6_univ_seed{seed}.npz');aa=a.load(H/f'A1A2_univ_seed{seed}.npz');assert np.array_equal(aa['sample_id'],v['sample_id'])
        orig=a.read(P/'calibration/univ.json');a.lock(P/'calibration/univ.json');eb=np.searchsorted(orig['cv_fde_quantiles'],z['CV_last_FDE'],side='right');sb=v['speed_bin'];t=z['track'];nums={k:aa[k] for k in ('top1','meanK','random','oracle')}
        groups=[('nonCV',non),('P1',P1),('nonCV_no_cue',non&(pattern==0)),('C1_overlap',non&c1),('C2_overlap',non&c2),('C3_overlap',non&c3)]+[(f'pattern_{b:03b}',non&(pattern==b)) for b in range(8)]
        no=P1&(v['density']==0)
        groups += [(f'no_near_speed{s}_CVerror{e}',no&(sb==s)&(eb==e)) for s in range(4) for e in range(4)]
        for tag,m in groups:rows.append(dict(seed=seed,group=tag,n=int(m.sum()),tracks=len(np.unique(t[m])),skills=skills({k:x[m] for k,x in nums.items()},aa['den'][m],t[m]),delta_SSE_sum=float((aa['den']-aa['top1'])[m].sum()),CV_SSE_sum=float(aa['den'][m].sum())))
        kn=a.old_job(P/'stage3_knn_memory/jobs/univ_position_velocity_all_train',3);assert np.array_equal(kn['sample_id'],d['ids']);ka,_=a.err(kn['pred'],z['GT']);better=ka<z['top1_ADE'];nb=a.neighborhood(d['obs'],d['valid'],cal['univ']['R'])
        for tag,m in [('knn_better',P1&better),('not_knn_better',P1&~better)]:extra.append(dict(seed=seed,group=tag,n=int(m.sum()),tracks=len(np.unique(t[m])),speed_mps=a.mean(nb['speed'][m]/.4),density=a.mean(v['density'][m]),C1_fraction=a.mean(c1[m]),C2_fraction=a.mean(c2[m]),C3_fraction=a.mean(c3[m]),pattern_hist=np.bincount(pattern[m],minlength=8).tolist()))
        a.save(f'A3_univ_seed{seed}.npz',sample_id=d['ids'],track=t,pattern=pattern,mask_nonCV=non,mask_P1=P1,no_near=no,speed_bin=sb,CV_error_bin=eb,knn_better=better,delta_SSE=aa['den']-aa['top1'])
    a.emit('A3',dict(records=rows,knn_profiles=extra,overlapping_cues_not_disjoint_attribution=True))
def feature_stats(x,m):
    y=x[m];return dict(n=len(y),mean=float(y.mean()) if len(y) else None,median=float(np.median(y)) if len(y) else None,p95=float(np.quantile(y,.95)) if len(y) else None)
def edge_proxy(d):
    out=np.zeros(len(d['ids']))
    for r in np.unique(d['recording']):
        m=d['recording']==r;xy=d['position'][m].astype(float);low=xy.min(0);high=xy.max(0);span=high-low
        f=np.divide(xy-low,span,out=np.full_like(xy,.5),where=span>0);out[m]=np.minimum(f,1-f).min(1)
    return out
def a4(cal):
    rows=[]
    for fold in Q['folds']:
        d=a.load(S/f'data/{fold}_selected.npz');nb=a.neighborhood(d['obs'],d['valid'],cal[fold]['R']);edge=edge_proxy(d);feat=dict(neighbor_count=d['valid']-1,near_count=nb['density'],speed_mps=nb['speed']/.4,empirical_box_edge_proxy=edge)
        for seed in Q['seeds']:
            rot=a.load(V/f'N1_{fold}_seed{seed}_rotate.npz');mir=a.load(V/f'N1_{fold}_seed{seed}_mirror.npz');assert np.array_equal(rot['sample_id'],d['ids']);res=rot['residual'];feat['residual_m']=res
            if fold in ('univ','zara02'):
                threshold=float(np.quantile(rot['dT'],.95));tail=rot['dT']>threshold
                for label,m in [('rotation_tail',tail),('rotation_remainder',~tail)]:rows.append(dict(fold=fold,seed=seed,group=label,threshold=threshold,ties=int((rot['dT']==threshold).sum()),tracks=len(np.unique(d['track'][m])),features={k:feature_stats(v,m) for k,v in feat.items()},world_position_mean=d['position'][m].mean(0).tolist() if m.any() else None))
                a.save(f'A4_{fold}_seed{seed}.npz',sample_id=d['ids'],track=d['track'],recording=d['recording'],position=d['position'],tail=tail,dT=rot['dT'],**feat)
            edges=np.quantile(res,[1/3,2/3]);b=np.searchsorted(edges,res,side='right');ratio=np.divide(mir['dT'],res,out=np.full(len(res),np.nan),where=res>0)
            for j in range(3):
                m=b==j;ok=m&(res>0);rows.append(dict(fold=fold,seed=seed,group=f'mirror_residual_bin{j}',edges=edges.tolist(),n=int(m.sum()),undefined=int((m&~(res>0)).sum()),path_difference=feature_stats(mir['dT'],m),ratio=feature_stats(ratio,ok),residual=feature_stats(res,m)))
    records('A4',rows)
def a5(cal):
    rows=[]
    for fold in Q['folds']:
        d=a.load(S/f'data/{fold}_selected.npz');nb=a.neighborhood(d['obs'],d['valid']);g=((d['gt'][:,-1]-a.cv(d['obs'])[:,-1])*nb['away']).sum(1);t=d['track'];n=len(t)
        for seed in Q['seeds']:
            for control in Q['A5_controls']:
                z=a.old_job(S/f'jobs/{fold}_seed{seed}/{control}',2);assert np.array_equal(d['ids'],z['sample_id']);assert np.array_equal(z['GT'],d['gt'])
                for method in Q['A5_methods']:
                    before=z['base_paths'][np.arange(n),z['base_logits'].argmax(1)] if method=='top1' else z['base_paths'].mean(1)
                    after=z['paths'][np.arange(n),z['logits'].argmax(1)] if method=='top1' else z['paths'].mean(1)
                    m=((before[:,-1]-after[:,-1])*nb['away']).sum(1)
                    for q in Q['A5_quantiles']:
                        delta=cal[fold]['events'][str(q)];ev=nb['away_valid']&nb['C3']&(g>=delta);k=int(((m>0)&ev).sum());nn=int(ev.sum());u,inv=np.unique(t[ev],return_inverse=True);tm=np.bincount(inv,weights=m[ev])/np.bincount(inv) if len(u) else np.array([])
                        rows.append(dict(fold=fold,seed=seed,control=control,method=method,quantile=q,delta=delta,n=nn,tracks=len(u),positive=stat((m[ev]>0),t[ev]),Wilson_window_descriptive=a.wilson(k,nn),response_m=stat(m[ev],t[ev]),response_over_g=stat(m[ev]/g[ev],t[ev]),zero_response=int(((m==0)&ev).sum()),undefined_away=int((~nb['away_valid']).sum()),approx_MDE_mean=2.8*float(tm.std(ddof=1))/np.sqrt(len(u)) if len(u)>1 else None,approx_MDE_proportion=1.4/np.sqrt(len(u)) if len(u)>1 else None))
                    a.save(f'A5_{fold}_seed{seed}_{control}_{method}.npz',sample_id=d['ids'],track=t,g=g,m=m,C3=nb['C3'],away_valid=nb['away_valid'])
        print('A5',fold,flush=True)
    records('A5',rows)
def design_matrix(density,sb,record):
    cols=[np.ones(len(density)),density];names=['intercept','density_per_train_SD']
    for tag,var in [('speed',sb.astype(str)),('recording',record.astype(str))]:
        for category in sorted(np.unique(var))[1:]:cols.append((var==category).astype(float));names.append(tag+':'+category)
    return np.stack(cols,1),names
def regression(y,X,t):
    if len(t)<2:return dict(beta=None,CI=None,n=len(t),tracks=len(np.unique(t)),reason='TOO_SMALL')
    u,inv=np.unique(t,return_inverse=True);T=len(u);p=X.shape[1];XX=np.zeros((T,p,p));XY=np.zeros((T,p))
    for i in range(p):
        np.add.at(XY[:,i],inv,X[:,i]*y)
        for j in range(p):np.add.at(XX[:,i,j],inv,X[:,i]*X[:,j])
    total=XX.sum(0);rank=int(np.linalg.matrix_rank(total));beta=np.linalg.pinv(total,rcond=1e-12)@XY.sum(0)
    def identifiable(m):
        nuisance=np.delete(np.delete(m,1,axis=-1),1,axis=-2);v=np.delete(m[...,1,:],1,axis=-1)
        remaining=m[...,1,1]-np.einsum('...i,...ij,...j->...',v,np.linalg.pinv(nuisance,rcond=1e-12),v)
        return remaining>1e-10
    if not identifiable(total):return dict(beta=None,CI=None,n=len(t),tracks=T,rank=rank,reason='DENSITY_NOT_IDENTIFIABLE')
    rng=np.random.default_rng(Q['stat_seed']);vals=[]
    for begin in range(0,Q['bootstrap_reps'],50):
        a.guard();ix=rng.integers(0,T,(min(50,Q['bootstrap_reps']-begin),T));mat=XX[ix].sum(1);vec=XY[ix].sum(1);ok=identifiable(mat);b=np.einsum('bij,bj->bi',np.linalg.pinv(mat[ok],rcond=1e-12),vec[ok]);vals.extend(b[:,1].tolist())
    return dict(beta=float(beta[1]),CI=np.quantile(vals,[.025,.5,.975]).tolist() if vals else None,n=len(t),tracks=T,rank=rank,columns=p,bootstrap_identifiable=len(vals),coefficients=beta.tolist(),units='m2_SSE_improvement_per_TRAIN_density_SD')
def a6(cal):
    rows=[];pool={seed:[] for seed in Q['seeds']}
    for fold in Q['folds']:
        d=a.load(P/f'data/{fold}_evaluation.npz');raw=gaussian(d['obs'],d['valid'],cal[fold]['R']);density=(raw-cal[fold]['density_mean'])/cal[fold]['density_std']
        for seed in Q['seeds']:
            z=a.load(H/f'A1A2_{fold}_seed{seed}.npz');v=a.load(V/f'N2_N6_{fold}_seed{seed}.npz');assert np.array_equal(d['ids'],z['sample_id']);y=z['den']-z['top1'];t=d['track'];sb=v['speed_bin'];mask=z['mask_P1']
            for tag,m in [('all',np.ones(len(t),bool)),('P1',mask)]:
                X,names=design_matrix(density[m],sb[m],d['recording'][m]);rows.append(dict(fold=fold,seed=seed,subset=tag,names=names,regression=regression(y[m],X,t[m])))
            a.save(f'A6_{fold}_seed{seed}.npz',sample_id=d['ids'],track=t,mask_P1=mask,delta_SSE=y,density_raw=raw,density_standardized=density,speed_bin=sb,recording=d['recording'])
            pool[seed].append(dict(y=y,density=density,t=np.char.add(fold+':',t),sb=np.char.add(fold+':',sb.astype(str)),record=np.char.add(fold+':',d['recording']),mask=mask))
    for seed,parts in pool.items():
        c={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        for tag,m in [('all',np.ones(len(c['t']),bool)),('P1',c['mask'])]:
            X,names=design_matrix(c['density'][m],c['sb'][m],c['record'][m]);rows.append(dict(fold='pooled',seed=seed,subset=tag,names=names,regression=regression(c['y'][m],X,c['t'][m])))
    records('A6',rows)
def fixtures():
    p=np.arange(4*12*2).reshape(1,4,12,2).astype(float);gt=np.zeros((1,12,2));logs=np.zeros((1,4));nums,rank,strict,_,disp,sse=selection(p,gt,logs)
    allr=[]
    for j in range(4):
        l=np.zeros((1,4));l[:,j]=1;allr.append(selection(p,gt,l)[1][0])
    assert np.mean(allr)==.5
    ties=np.zeros_like(p);assert selection(ties,gt,logs)[1][0]==.5
    t=np.repeat(np.arange(4),4);x=np.tile(np.arange(4.),4);X=np.stack((np.ones(16),x),1);r=regression(2*x+3,X,t);assert abs(r['beta']-2)<1e-12
    a.write('PREFLIGHT.json',dict(status='PASS_NUMERIC_FIXTURES',uniform_random_rank_half=True,Jensen=True,ties_half=True,regression_known_slope=True,no_model=True))
def main():
    a.guard()
    for mf in [V/'input_manifest.json',P/'completion_review/input_manifest.json',S/'completion_review/input_manifest.json']:
        a.lock(mf)
        for p,d in a.read(mf)['files'].items():
            if p in a.EXPECTED:assert a.EXPECTED[p]==d
            a.EXPECTED[p]=d
    for p in [Path(__file__),H/'protocol.json',H/'DESIGN.md',V/'analyze_saved.py',P/'stage1.py',Path('C:/Users/linke/Downloads/ped_claim_evidence_v2_and_next_experiments.md')]:a.lock(p)
    fixtures();cal=train_calibrate();a1a2();a3(cal);a4(cal);a5(cal);a6(cal)
    for p,d in a.LOCKS.items():a.guard();assert a.sha(p)==d,('FINAL_SHA',p)
    a.write('input_manifest.json',dict(files=a.LOCKS));outs={str(p):a.sha(p) for p in H.glob('*.npz')}
    a.write('RESULT.json',dict(status='PASS_SAVED_ONLY_V2_A1_A6',completed=a.DONE,input_files=len(a.LOCKS),output_sha=outs,summary_sha=a.OUT,resource=a.RES,elapsed_seconds=time.perf_counter()-a.START,no_torch_forward_optimizer=True,whole_series_complete=False))
    print('ALL V2 COMPLETE',round(time.perf_counter()-a.START,3),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:a.write('FAILURE.json',dict(status='STOP',traceback=traceback.format_exc(),completed=a.DONE,resource=a.RES));raise
