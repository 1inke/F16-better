"""N0/N1/N2/N3/N6 saved-output analyses. CPU only; no torch/model/optimizer."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib,ctypes,traceback,gc
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import psutil
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
H=Path(__file__).resolve().parent;P=H.parent.parent;S=P/'stage2_raw_interventions';G=P/'low_cost_completion_20261009'
D=Path('C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy/processed_data_diverse')
START=time.perf_counter();LOCKS={};EXPECTED={};OUT={};DONE=[]
RES=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
Q=read(H/'protocol.json')
def write(name,x):
    path=H/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item()),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    p=psutil.Process().memory_info()
    RES['peak_private_GiB']=max(RES['peak_private_GiB'],p.peak_pagefile/2**30);RES['peak_RSS_GiB']=max(RES['peak_RSS_GiB'],p.peak_wset/2**30)
    RES['min_physical_GiB']=min(RES['min_physical_GiB'],m.avail_phys/2**30);RES['min_commit_GiB']=min(RES['min_commit_GiB'],m.avail_page/2**30);RES['checks']+=1
    if max(p.private,p.rss)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:raise MemoryError('STOP_RESOURCE')
    if time.perf_counter()-START>Q['max_compute_seconds']:raise TimeoutError('STOP_SHORT_COMPUTE_BUDGET')
def lock(p,expected=None):
    p=Path(p).resolve();k=str(p)
    if k in LOCKS:return
    guard();actual=sha(p);expected=expected if expected is not None else EXPECTED.get(k)
    if expected is not None:assert actual==expected,('INPUT_SHA',k)
    LOCKS[k]=actual
def load(p,keys=None):
    lock(p)
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def old_job(path,stage=1):
    done=path/'done.json';lock(done);r=read(done);assert r['status']=='SUCCESS'
    out=r['output_sha'] if stage==1 else r['outputs']
    for p,d in out.items():
        if p in EXPECTED:assert EXPECTED[p]==d
        EXPECTED[p]=d
    return load(path/'outputs.npz')
def emit(name,x):
    write(name+'.json',x);OUT[name]=sha(H/(name+'.json'));DONE.append(name)
    print(name,'SUCCESS',round(time.perf_counter()-START,2),flush=True)
def save(name,**x):np.savez_compressed(H/name,**x)
def mean(x):return float(np.mean(x)) if len(x) else None
def ratio(a,b):return float(a/b) if b>0 else None
def ci(num,den,track,skill=False):
    if not len(track):return dict(value=None,CI=None,tracks=0,n=0)
    _,inv=np.unique(track,return_inverse=True);a=np.bincount(inv,weights=num);b=np.bincount(inv,weights=den);value=ratio(a.sum(),b.sum())
    value=1-value if skill and value is not None else value
    if len(a)<2:return dict(value=value,CI=None,tracks=len(a),n=len(track))
    rng=np.random.default_rng(Q['stat_seed']);v=[]
    for start in range(0,Q['bootstrap_reps'],100):
        idx=rng.integers(0,len(a),(min(100,Q['bootstrap_reps']-start),len(a)));aa=a[idx].sum(1);bb=b[idx].sum(1);ok=bb>0
        vals=aa[ok]/bb[ok];v.extend((1-vals if skill else vals).tolist())
    return dict(value=value,CI=np.quantile(v,[.025,.5,.975]).tolist() if v else None,tracks=len(a),n=len(track))
def mean_ci(x,t):return ci(x,np.ones(len(x)),t)
def err(path,gt):
    e=np.linalg.norm(path-gt[:,None] if path.ndim==4 else path-gt,axis=-1);return e.mean(-1),e[...,-1]
def cv(obs):return (obs[:,0,-1]-obs[:,0,-2]).astype(float)[:,None]*np.arange(1,13)[None,:,None]
def neighborhood(obs,valid,R=None):
    v=(obs[:,:,-1]-obs[:,:,-2]).astype(float);x=obs[:,:,-1].astype(float)
    validmask=np.arange(1,9)[None]<valid[:,None]
    now=np.linalg.norm(x[:,1:]-x[:,:1],axis=-1);near=np.where(validmask,now,np.inf).min(1)
    pred=x[:,:,None]+v[:,:,None]*np.arange(1,13)[None,None,:,None]
    diff=pred[:,:1]-pred[:,1:];dist=np.where(validmask[:,:,None],np.linalg.norm(diff,axis=-1),np.inf)
    ind=dist.reshape(len(obs),-1).argmin(1);j=ind//12;t=ind%12;d=dist[np.arange(len(obs)),j,t]
    away=diff[np.arange(len(obs)),j,t];ok=np.isfinite(d)&(d>0)
    away=np.divide(away,d[:,None],out=np.zeros_like(away),where=ok[:,None])
    return dict(nearest=near,C3=d<1.,away=away,away_valid=ok,
        density=((now<=R)&validmask).sum(1) if R is not None else None,speed=np.linalg.norm(v[:,0],axis=1))
def slope(num,den,density,bins,track):
    if len(track)==0 or len(np.unique(bins))<2:return dict(value=None,CI=None,reason='LESS_THAN_TWO_NONEMPTY_BINS')
    _,inv=np.unique(track,return_inverse=True);T=inv.max()+1;B=int(bins.max())+1
    mats=[np.zeros((T,B)) for _ in range(4)]
    for mat,x in zip(mats,(num,den,density.astype(float),np.ones(len(num)))):np.add.at(mat,(inv,bins),x)
    def calc(parts):
        a,b,sd,n=parts;ok=(n>0)&(b>0);x=np.divide(sd,n,out=np.zeros_like(sd),where=n>0);y=1-np.divide(a,b,out=np.zeros_like(a),where=b>0)
        w=np.where(ok,n,0);total=w.sum(-1);xb=np.divide((w*x).sum(-1),total,out=np.zeros_like(total),where=total>0);yb=np.divide((w*y).sum(-1),total,out=np.zeros_like(total),where=total>0)
        bottom=(w*(x-xb[...,None])**2).sum(-1);top=(w*(x-xb[...,None])*(y-yb[...,None])).sum(-1)
        return np.divide(top,bottom,out=np.full_like(top,np.nan),where=bottom>0)
    value=float(calc([m.sum(0) for m in mats]));rng=np.random.default_rng(Q['stat_seed']);vals=[]
    for start in range(0,Q['bootstrap_reps'],100):
        idx=rng.integers(0,T,(min(100,Q['bootstrap_reps']-start),T));v=calc([m[idx].sum(1) for m in mats]);vals.extend(v[np.isfinite(v)].tolist())
    return dict(value=value if np.isfinite(value) else None,CI=np.quantile(vals,[.025,.5,.975]).tolist() if vals else None,tracks=T)
def calibrate():
    result={}
    for fold in Q['folds']:
        guard();bank=load(P/f'stage3_knn_memory/data/{fold}_bank.npz');ids=bank['ids']
        rawpath=D/f'{fold}_data_train.npy';countpath=D/f'{fold}_num_train.npy';lock(rawpath);lock(countpath)
        raw=np.load(rawpath,mmap_mode='r');counts=np.load(countpath,mmap_mode='r')
        obs=np.array(raw[ids,:,:8,-2:],np.float32);valid=np.minimum(np.array(counts[ids],int),9)
        assert np.array_equal(obs[:,0],bank['obs'])
        gt=(np.array(raw[ids,0,8:,-2:],np.float32)-obs[:,0,-1,None]).astype(float);assert np.array_equal(gt,bank['gt'])
        a=neighborhood(obs,valid);R=float(np.median(a['nearest'][np.isfinite(a['nearest'])]));a=neighborhood(obs,valid,R)
        edges=np.unique(np.quantile(a['density'],Q['N2_density_quantiles']))
        g=((gt[:,-1]-cv(obs)[:,-1])*a['away']).sum(1);m=a['C3']&a['away_valid']&(g>0)
        assert m.any(),('NO_POSITIVE_TRAIN_C3_g',fold)
        delta=float(np.quantile(g[m],Q['N3_positive_training_C3_g_quantile']));assert delta>0
        result[fold]=dict(n_train=len(ids),id_sha=hashlib.sha256(ids.tobytes()).hexdigest(),R_m=R,density_edges=edges.tolist(),effective_density_bins=len(edges)+1,training_density_hist=np.bincount(a['density']).tolist(),training_C3=int(a['C3'].sum()),positive_C3_g_n=int(m.sum()),delta_g_m=delta,all_train_only=True)
        del bank,obs,raw,counts;gc.collect()
    emit('CALIBRATION',result);return result
def n0():
    path=P/'completion_review/SUMMARY.json';lock(path);original=read(path)
    table=[]
    for r in original['point_records']:
        if r['method'] not in ('top1','meanK'):continue
        rr=dict(r)
        for endpoint in ('ADE','FDE'):
            d=rr.get(endpoint+'_delta_strongest_simple');base=rr[endpoint]-d if d is not None else None
            rr[endpoint+'_gain_percent_vs_strongest']=ratio(-100*d,base) if base is not None and d is not None else None
        table.append(rr)
    emit('N0',dict(point_comparisons=table,all_simple_and_model_points=original['point_records'],E7_candidates=original['candidate_records'],E7_paired_track_CIs=original['paired_E7'],P1=original['P1'],denominators=original['denominators'],all_previous_review_not_new_confirmatory=True))
def n1():
    previous=read(G/'RESULT.json');lock(G/'RESULT.json');rows=[]
    oldrecords={(r['fold'],r['seed'],r['intervention']):r for r in previous['records']}
    for fold in Q['folds']:
        data=load(S/f'data/{fold}_selected.npz');base=[]
        for seed in Q['model_seeds']:
            z=old_job(S/f'jobs/{fold}_seed{seed}/mirror',2);assert np.array_equal(data['ids'],z['sample_id']);base.append(z['base_paths'])
        values=[]
        for left,right in ((0,1),(0,2),(1,2)):
            v=np.empty((len(base[0]),2))
            for start in range(0,len(v),128):
                guard();ds=np.linalg.norm(base[left][start:start+128,:,None]-base[right][start:start+128,None,:],axis=-1).mean(-1)
                for k,d in enumerate(ds):
                    i=start+k;r,c=linear_sum_assignment(d);v[i]=(d[r,c].mean(),np.linalg.norm(base[left][i,r,-1]-base[right][i,c,-1],axis=-1).mean())
            save(f'N1_{fold}_pair{left+1}{right+1}.npz',sample_id=data['ids'],values=v);values.append(v)
        med=np.median(np.concatenate(values),axis=0);I,J=np.triu_indices(20,1)
        for i,seed in enumerate(Q['model_seeds']):
            p=base[i];spread=np.empty(len(p))
            for start in range(0,len(p),128):
                guard();spread[start:start+128]=np.linalg.norm(p[start:start+128,I]-p[start:start+128,J],axis=-1).mean((1,2))
            residual=np.linalg.norm(p-cv(data['obs'])[:,None],axis=-1).mean((1,2))
            for name in ('rotate','mirror','translate','scale08','scale12'):
                rec=oldrecords[(fold,seed,name)];path=G/rec['output'];lock(path,rec['output_sha']);z=load(path);assert np.array_equal(z['sample_id'],data['ids']);m=z['measurements']
                rs=np.divide(m[:,1],spread,out=np.full(len(m),np.nan),where=spread>0);rr=np.divide(m[:,1],residual,out=np.full(len(m),np.nan),where=residual>0)
                r=ratio(np.median(m[:,1]),med[0]);rows.append(dict(fold=fold,seed=seed,transform=name,n=len(m),seed_path_median=float(med[0]),seed_endpoint_median=float(med[1]),transform_path_median=float(np.median(m[:,1])),transform_endpoint_median=float(np.median(m[:,3])),path_seed_ratio=r,endpoint_seed_ratio=ratio(np.median(m[:,3]),med[1]),below_descriptive_1_25=r<=1.25 if r is not None else None,internal_spread_median=float(np.median(spread)),residual_median=float(np.median(residual)),dT_over_internal_spread_median=mean(np.array([np.median(rs[np.isfinite(rs)])])) if np.isfinite(rs).any() else None,dT_over_residual_median=float(np.median(rr[np.isfinite(rr)])) if np.isfinite(rr).any() else None,undefined_spread=int((spread==0).sum()),undefined_residual=int((residual==0).sum())))
                save(f'N1_{fold}_seed{seed}_{name}.npz',sample_id=data['ids'],dT=m[:,1],spread=spread,residual=residual)
        print('N1',fold,flush=True)
    emit('N1',dict(records=rows,reference_not_equivalence_test=True,not_seed_independent_windows=True))
def n2n6(cal):
    densityrows=[];oraclerows=[]
    for fold in Q['folds']:
        data=load(P/f'data/{fold}_evaluation.npz');nb=neighborhood(data['obs'],data['valid'],cal[fold]['R_m']);density=nb['density'];db=np.searchsorted(cal[fold]['density_edges'],density,side='right');lc=read(P/f'calibration/{fold}.json');lock(P/f'calibration/{fold}.json');sb=np.searchsorted(lc['speed_edges'],nb['speed'],side='right')
        knn=old_job(P/f'stage3_knn_memory/jobs/{fold}_position_velocity_all_train',3);assert np.array_equal(knn['sample_id'],data['ids']);ka,kf=err(knn['pred'],data['gt'])
        for seed in Q['model_seeds']:
            guard();z=old_job(P/f'jobs/{fold}_seed{seed}');assert np.array_equal(z['sample_id'],data['ids']);p=z['model_paths'];gt=z['GT'];cvp=z['CV'];track=z['track'];mask=z['mask_nonCV_cue_positive'];n=len(p);topslot=z['model_logits'].argmax(1);top=p[np.arange(n),topslot];mu=p.mean(1);den=((gt-cvp)**2).sum((1,2));num=((gt-top)**2).sum((1,2));sse=((p-gt[:,None])**2).sum((2,3));best=sse.argmin(1);onum=sse[np.arange(n),best];mn=((mu-gt)**2).sum((1,2));aa,ff=err(p,gt);quality=aa+.5*ff
            qualityorder=np.argsort(quality,axis=1,kind='stable');ranks=np.empty_like(qualityorder);np.put_along_axis(ranks,qualityorder,np.tile(np.arange(1,21),(n,1)),axis=1)
            logorder=np.argsort(-z['model_logits'],axis=1,kind='stable');logr=np.empty_like(logorder);np.put_along_axis(logr,logorder,np.tile(np.arange(1,21),(n,1)),axis=1)
            good=(aa<=.2)&(ff<=.4);present=good.any(1);chosen=good[np.arange(n),topslot];toprank=ranks[np.arange(n),topslot];bestrank=logr[np.arange(n),best]
            item=dict(fold=fold,seed=seed,n=int(mask.sum()),tracks=len(np.unique(track[mask])),top1=ci(num[mask],den[mask],track[mask],True),meanK=ci(mn[mask],den[mask],track[mask],True),oracle_single_complete_SSE=ci(onum[mask],den[mask],track[mask],True),selection_skill_gap=ci((num-onum)[mask],den[mask],track[mask]),good_present=mean_ci(present[mask].astype(float),track[mask]),no_good=mean_ci((~present[mask]).astype(float),track[mask]),good_present_but_not_selected=mean_ci((present&~chosen)[mask].astype(float),track[mask]),top1_quality_rank_hist=np.bincount(toprank[mask],minlength=21)[1:].tolist(),oracle_old_logit_rank_hist=np.bincount(bestrank[mask],minlength=21)[1:].tolist())
            item['knn_advantage_feature_comparison']={}
            for tag,m in (('knn_better',mask&(ka<z['top1_ADE'])),('not_knn_better',mask&(ka>=z['top1_ADE']))):
                item['knn_advantage_feature_comparison'][tag]=dict(n=int(m.sum()),speed_mps=mean(nb['speed'][m]/.4),density=mean(density[m]),world_position_mean=data['position'][m].mean(0).tolist() if m.any() else None)
            oraclerows.append(item)
            groups=[('P1',mask,None),('all',np.ones(n,bool),None)]+[(f'P1_speed{s}',mask&(sb==s),s) for s in range(4)]
            for tag,m,speedbin in groups:
                rec=dict(fold=fold,seed=seed,subset=tag,n=int(m.sum()),speed_bin=speedbin,density_slope=slope(num[m],den[m],density[m],db[m],track[m]),bins=[])
                for b in range(len(cal[fold]['density_edges'])+1):
                    ok=m&(db==b);rec['bins'].append(dict(bin=b,n=int(ok.sum()),density_mean=mean(density[ok]),skill=ci(num[ok],den[ok],track[ok],True)))
                densityrows.append(rec)
            save(f'N2_N6_{fold}_seed{seed}.npz',sample_id=z['sample_id'],track=track,mask=mask,density=density,density_bin=db,speed_bin=sb,num=num,den=den,oracle_num=onum,meanK_num=mn,good_present=present,top1_good=chosen,top_quality_rank=toprank,oracle_logit_rank=bestrank)
            del z,p,sse;gc.collect()
        print('N2/N6',fold,flush=True)
    emit('N2',dict(calibration=cal,records=densityrows,no_density_causal_claim=True));emit('N6',dict(records=oraclerows,oracle_is_unavailable_upper_bound=True))
def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [float(c-half),float(c+half)]
def n3(cal):
    rows=[];synthetic=[]
    for fold in Q['folds']:
        data=load(S/f'data/{fold}_selected.npz');n=len(data['ids']);nb=neighborhood(data['obs'],data['valid']);g=((data['gt'][:,-1]-cv(data['obs'])[:,-1])*nb['away']).sum(1);event=nb['away_valid']&nb['C3']&(g>=cal[fold]['delta_g_m']);control=nb['away_valid']&nb['C3']&~event;other=nb['away_valid']&~nb['C3']
        for seed in Q['model_seeds']:
            n=len(data['ids']);guard();z=old_job(S/f'jobs/{fold}_seed{seed}/noNbr',2);assert np.array_equal(z['sample_id'],data['ids']);t=data['track']
            for method in ('top1','meanK'):
                before=z['base_paths'][np.arange(n),z['base_logits'].argmax(1)] if method=='top1' else z['base_paths'].mean(1)
                after=z['paths'][np.arange(n),z['logits'].argmax(1)] if method=='top1' else z['paths'].mean(1);m=((before[:,-1]-after[:,-1])*nb['away']).sum(1)
                rec=dict(fold=fold,seed=seed,method=method,n=n,undefined_away=int((~nb['away_valid']).sum()),groups={})
                for label,mask in (('avoidance_event',event),('other_C3',control),('non_C3',other)):
                    rec['groups'][label]=dict(n=int(mask.sum()),positive=mean_ci((m[mask]>0).astype(float),t[mask]),response_m=mean_ci(m[mask],t[mask]),g_mean_m=mean(g[mask]),zero_response=int((m[mask]==0).sum()),response_over_g=mean_ci(m[mask]/g[mask],t[mask]) if label=='avoidance_event' else None)
                rows.append(rec)
                save(f'N3_{fold}_seed{seed}_{method}.npz',sample_id=data['ids'],track=t,g=g,m=m,away=nb['away'],valid=nb['away_valid'],event=event,other_C3=control,non_C3=other)
            for side in ('left','right'):
                for d in ('03','06','10','15','25'):
                    z=old_job(S/f'jobs/{fold}_seed{seed}/syn_{side}_{d}',2);n=len(z['paths']);t=z['track']
                    for method in ('top1','meanK'):
                        p=z['paths'][np.arange(n),z['logits'].argmax(1)] if method=='top1' else z['paths'].mean(1)
                        ref=z['noNbr_paths'][np.arange(n),z['noNbr_logits'].argmax(1)] if method=='top1' else z['noNbr_paths'].mean(1)
                        m=((p[:,-1]-ref[:,-1])*z['away']).sum(1);u,inv=np.unique(t,return_inverse=True);tm=np.bincount(inv,weights=m)/np.bincount(inv)
                        synthetic.append(dict(fold=fold,seed=seed,side=side,distance_m=float(z['distance'][0]),method=method,n=n,tracks=len(u),positive_window=mean((m>0).astype(float)),Wilson_window_descriptive=wilson(int((m>0).sum()),n),positive_track_CI=mean_ci((m>0).astype(float),t),response_m=mean_ci(m,t),zero_response_n=int((m==0).sum()),approx_MDE_mean_m=2.8*float(tm.std(ddof=1))/np.sqrt(len(tm)) if len(tm)>1 else None,approx_MDE_proportion=1.4/np.sqrt(len(u)) if len(u)>1 else None))
        print('N3',fold,flush=True)
    emit('N3',dict(real_records=rows,synthetic_records=synthetic,GT_defined_events_posthoc=True,not_necessarily_lateral=True,MDE_approx_not_actual_power=True))
def fixtures():
    assert ci(np.array([1.,2.]),np.array([2.,4.]),np.array(['a','b']),True)['value']==.5
    assert wilson(0,10)[0]>=-1e-12 and wilson(10,10)[1]<=1+1e-12
    p=np.arange(20*12*2).reshape(20,12,2).astype(float);d=np.linalg.norm(p[:,None]-p[::-1][None],axis=-1).mean(-1);r,c=linear_sum_assignment(d);assert d[r,c].sum()==0
def main():
    guard();fixtures()
    for mf in (P/'completion_review/input_manifest.json',S/'completion_review/input_manifest.json',P/'stage3_knn_memory/completion_review/input_manifest.json'):
        lock(mf)
        for p,d in read(mf)['files'].items():
            if p in EXPECTED:assert EXPECTED[p]==d
            EXPECTED[p]=d
    for p in (Path(__file__),H/'DESIGN.md',H/'protocol.json',Path('C:/Users/linke/Downloads/ped_claim_evidence_and_next_experiments.md')):lock(p)
    cal=calibrate();n0();n1();n2n6(cal);n3(cal)
    write('input_manifest.json',dict(files=LOCKS))
    for p,d in LOCKS.items():guard();assert sha(p)==d,('FINAL_INPUT_SHA',p)
    outputs={str(p):sha(p) for p in H.glob('*.npz')}
    for name,d in OUT.items():assert sha(H/(name+'.json'))==d
    write('RESULT.json',dict(status='PASS_SAVED_ONLY_N0_N1_N2_N3_N6',completed=DONE,input_files=len(LOCKS),initial_final_SHA_PASS=True,output_sha=outputs,summary_sha=OUT,resource=RES,elapsed_seconds=time.perf_counter()-START,no_torch_forward_optimizer=True,whole_original_series_complete=False))
    print('ALL COMPLETE',round(time.perf_counter()-START,2),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        write('FAILURE.json',dict(status='STOP',traceback=traceback.format_exc(),completed=DONE,resource=RES,elapsed_seconds=time.perf_counter()-START));raise
