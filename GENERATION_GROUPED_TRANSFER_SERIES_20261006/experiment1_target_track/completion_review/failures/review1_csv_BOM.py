"""CPU saved-output completion review. No torch, model import, forward or optimizer."""
from pathlib import Path
import os,sys,json,hashlib,csv,time,traceback,ctypes
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
import numpy as np
import psutil
from scipy.cluster.hierarchy import linkage,fcluster
HERE=Path(__file__).resolve().parent
EXP=HERE.parent; SERIES=EXP.parent; ROOT=SERIES.parent
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
resource=dict(peak_RSS_GiB=0.,peak_private_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
def guard():
    v=psutil.Process().memory_info();m=Mem();m.length=ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    resource['peak_RSS_GiB']=max(resource['peak_RSS_GiB'],v.peak_wset/2**30)
    resource['peak_private_GiB']=max(resource['peak_private_GiB'],v.peak_pagefile/2**30)
    resource['min_physical_GiB']=min(resource['min_physical_GiB'],m.avail_phys/2**30)
    resource['min_commit_GiB']=min(resource['min_commit_GiB'],m.avail_page/2**30)
    resource['checks']+=1
    assert max(v.rss,v.private)<6*2**30 and min(m.avail_phys,m.avail_page)>=6*2**30,'STOP_RESOURCE'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def verify(d):
    for i,(p,h) in enumerate(d.items()):
        if i%512==0:guard()
        assert sha(p)==h,('SHA',p)
def masks(g):
    return [('overall',np.ones(len(g),bool))]+[(k,g==i) for i,k in enumerate(('stable','transition','change'))]
def calc(p,gt,scores):
    guard();n=len(p);q={};a=[];f=[];early=[];middle=[];late=[];effective=[];distinct=[]
    tri=np.triu_indices(20,1)
    for start in range(0,n,256):
        guard();b=p[start:start+256];e=np.linalg.norm(b-gt[start:start+256,None].astype(np.float64),axis=-1)
        aa=e.mean(-1);ff=e[:,:,-1];ii=np.arange(len(b));slot=np.maximum(aa/.2,ff/.4).argmin(-1);ee=e[ii,slot]
        a.append(aa);f.append(ff);early.append(ee[:,:4].mean(-1));middle.append(ee[:,4:8].mean(-1));late.append(ee[:,8:].mean(-1))
        good=(aa<=.2)&(ff<=.4)
        for row in range(len(b)):
            d=np.linalg.norm(b[row,tri[0]]-b[row,tri[1]],axis=-1)
            lab=fcluster(linkage(np.maximum(d.mean(-1)/.10,d[:,-1]/.20),method='complete'),1.,criterion='distance')
            u=np.unique(lab);distinct.append(len(u));effective.append(sum(good[row,lab==x].any() for x in u))
    a=np.concatenate(a);f=np.concatenate(f);good=(a<=.2)&(f<=.4);sel=scores.argmax(-1);ii=np.arange(n)
    q.update(coverage=good.any(-1),raw_good=good.sum(-1),effective_good=np.asarray(effective,np.int16),all_distinct=np.asarray(distinct,np.int16),selected_good=good[ii,sel],selected_slot=sel,oracle_slot=np.maximum(a/.2,f/.4).argmin(-1),selected_ADE=a[ii,sel],selected_FDE=f[ii,sel],independent_min_both_but_no_samepath=(a.min(-1)<=.2)&(f.min(-1)<=.4)&~good.any(-1),loose_coverage=((a<=.3)&(f<=.5)).any(-1),minADE=a.min(-1),minFDE=f.min(-1),Q=(a+.5*f).min(-1),J=np.maximum(a/.2,f/.4).min(-1))
    q.update(early=np.concatenate(early),middle=np.concatenate(middle),late=np.concatenate(late))
    v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4;ac=np.diff(v,axis=2)/.4;je=np.diff(ac,axis=2)/.4
    motion={k:np.linalg.norm(x,axis=-1) for k,x in [('speed',v),('accel',ac),('jerk',je)]}
    return q,motion
def ci(delta,keys):
    unique,inv=np.unique(keys,return_inverse=True);rng=np.random.default_rng(20261006)
    count=np.bincount(inv);sums=np.stack([np.bincount(inv,weights=v,minlength=len(unique)) for v in delta]);out=[]
    for _ in range(2000):
        models=rng.integers(0,3,3);tracks=rng.integers(0,len(unique),len(unique));out.append(sums[models][:,tracks].sum()/(3*count[tracks].sum()))
    lo,hi=np.quantile(out,[.025,.975]);return dict(delta=float(delta.mean()),ci_low=float(lo),ci_high=float(hi))
def run():
    guard();t=time.perf_counter();reg=read(EXP/'preflight.json');old=read(SERIES/'old_results_manifest.json')['files']
    result=read(EXP/'results/RESULT.json');assert result['status']=='PASS_MATCHED_TARGET_TRACK_TRANSFER'
    assert result['jobs']==57 and len(result['outputs_SHA'])==114 and len(result['models_SHA'])==18
    verify(reg['source_hashes']);verify(old);verify(result['outputs_SHA']);verify(result['models_SHA'])
    for filename in ('ARITHMETIC_AUDIT','FROZEN_AUDIT','PRESERVATION_AUDIT'):
        assert read(EXP/('results/'+filename+'.json'))['status']=='PASS'
    split=load(SERIES/'identity_audit/splits.npz')
    lab=load(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz',['sample_id','future'])
    cache=load(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz',['sample_id','change_group','logits'])
    assert np.array_equal(lab['sample_id'],cache['sample_id']) and np.array_equal(split['sample_id'],cache['sample_id'][:25912])
    datasets={}
    for name,ix in [('outer',split['outer_eval_index']),('inner',split['inner_cal_index']),('original_CAL',np.arange(25912,28791))]:
        datasets[name]=(cache['sample_id'][ix],lab['future'][ix],cache['change_group'][ix],cache['logits'][ix])
    natives={};allq={};jobcount=0;maxerr=0.;scene_rows=[]
    for name in ['native_'+s for s in datasets]+[f'{a}_seed{seed}_{pol}_{s}' for a in ('random','grouped') for seed in (1,2,3) for pol in ('constrained','legacy','final') for s in datasets]:
        guard();s=next(s for s in datasets if name.endswith('_'+s));ids,gt,g,scores=datasets[s]
        stamp=read(EXP/('results/saved/'+name+'_done.json'));verify(stamp['outputs']);assert stamp['source_hashes']==reg['source_hashes']
        saved=load(EXP/('results/saved/'+name+'.npz'));p=saved['trajectory'];assert p.dtype==np.float64 and p.shape==(len(ids),20,12,2) and np.isfinite(p).all()
        assert np.array_equal(ids,saved['sample_id']) and len(np.unique(ids))==len(ids)
        q,motion=calc(p,gt,scores)
        for k,v in q.items():assert np.array_equal(v,saved[k]),(name,k)
        legacy=load(EXP/('results/saved/'+name+'_legacy.npz'),['minade20','minfde20'])
        for k,kk in [('minADE','minade20'),('minFDE','minfde20')]:
            err=float(np.max(np.abs(q[k]-legacy[kk])));maxerr=max(maxerr,err);assert err<=1e-6
        if name.startswith('native_'):natives[s]=(q,motion)
        else:
            parts=name.split('_');arm=parts[0];seed=int(parts[1][4:]);policy=parts[2];nq,nm=natives[s]
            for group,mask in masks(g):
                row=next(x for x in result['records'] if (x['arm'],x['seed'],x['policy'],x['split'],x['group'])==(arm,seed,policy,s,group))
                test=dict(n=int(mask.sum()),good=int(q['coverage'][mask].sum()),native_good=int(nq['coverage'][mask].sum()),rescued=int((q['coverage'][mask]&~nq['coverage'][mask]).sum()),lost=int((~q['coverage'][mask]&nq['coverage'][mask]).sum()),effective_good_total=int(q['effective_good'][mask].sum()),raw_good_total=int(q['raw_good'][mask].sum()),inherited_selected_good=int(q['selected_good'][mask].sum()))
                for k in ('minADE','minFDE','Q','J','early','middle','late','all_distinct'):test[k]=float(q[k][mask].mean());test['native_'+k]=float(nq[k][mask].mean())
                for k in ('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95'):
                    what,stat=k.split('_');x=motion[what][mask].reshape(-1);y=nm[what][mask].reshape(-1)
                    test[k]=float(x.mean() if stat=='mean' else np.quantile(x,.95));test['native_'+k]=float(y.mean() if stat=='mean' else np.quantile(y,.95))
                for k,v in test.items():assert v==row[k],(name,group,k,v,row[k])
            if s=='outer':allq[arm,seed,policy]=q
            if policy=='final' and s=='outer':
                for sc in np.unique(split['scene_family'][split['outer_eval_index']]):
                    for gr,gm in masks(g):
                        mask=(split['scene_family'][split['outer_eval_index']]==sc)&gm
                        if not mask.any():continue
                        scene_rows.append(dict(arm=arm,seed=seed,scene=str(sc),group=gr,n=int(mask.sum()),native_good=int(nq['coverage'][mask].sum()),good=int(q['coverage'][mask].sum()),minADE=float(q['minADE'][mask].mean()),minFDE=float(q['minFDE'][mask].mean()),native_minADE=float(nq['minADE'][mask].mean()),native_minFDE=float(nq['minFDE'][mask].mean())))
            if 'alpha' in saved:
                base=saved['base'].astype(np.float64);vv=np.diff(base,axis=2,prepend=np.zeros_like(base[:,:,:1]))/.4
                native_mean=np.linalg.norm(np.diff(vv,axis=2)/.4,axis=-1).mean((1,2))
                new_mean=motion['accel'].mean((1,2));assert np.all(new_mean<=1.01*native_mean)
                drift=saved['raw'].astype(np.float64).sum(2)
                assert np.max(np.abs(p[:,:,-1]-base[:,:,-1]-drift))<=2e-5
        jobcount+=1
    assert jobcount==57
    keys=np.asarray([str(s)+'|'+str(p) for s,p in zip(split['sequence'][split['outer_eval_index']],split['pedestrian_id'][split['outer_eval_index']])])
    cimax=0.
    for b in result['bootstrap']:
        mask=dict(masks(datasets['outer'][2]))[b['group']];metric=b['metric'];policy=b['policy']
        if b['reference']=='random':delta=np.stack([allq['grouped',seed,policy][metric][mask].astype(float)-allq['random',seed,policy][metric][mask].astype(float) for seed in (1,2,3)])
        else:delta=np.stack([allq[b['arm'],seed,policy][metric][mask].astype(float)-natives['outer'][0][metric][mask].astype(float) for seed in (1,2,3)])
        check=ci(delta,keys[mask])
        for k,v in check.items():cimax=max(cimax,abs(v-b[k]));assert v==b[k],(b,k)
    histories=[]
    for a in ('random','grouped'):
        for seed in (1,2,3):
            d=EXP/f'results/models/{a}/seed_{seed}/balanced_protected';complete=read(d/'complete.json');assert complete['status']=='COMPLETE' and complete['epochs']==80
            verify(complete['model_hashes'])
            with (d/'history.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
            trace=read(d/'objective_history.json')['rows'];assert len(rows)==len(trace)==80 and [int(x['epoch']) for x in rows]==list(range(1,81))
            for x in trace:assert abs(x['total']-x['original']-.1*x['P'])<=1e-12
            histories.append(dict(arm=a,seed=seed,epochs=80,constrained_epoch=complete['best_feasible_epoch'],legacy_epoch=complete['legacy_epoch'],initial_loss=float(rows[0]['train_loss']),final_loss=float(rows[-1]['train_loss'])))
    verify(reg['source_hashes']);verify(old);verify(result['outputs_SHA']);verify(result['models_SHA']);guard()
    write('PER_SCENE.json',dict(status='DESCRIPTIVE_NOT_SCENE_INDEPENDENT_CI',rows=scene_rows))
    review=dict(status='PASS',scope='CPU saved outputs and labels only; no torch/model forward/backward/optimizer; same assistant not external reviewer',jobs=57,outputs=114,models=18,records_exact=216,full_complete_link_reconstructed=True,all_quality_columns_EXACT=True,all_group_pooled_motion_EXACT=True,bootstrap_reconstructed=180,bootstrap_max_error=cimax,legacy_minADE_FDE_max_error=maxerr,strict_perwindow_accel_and_endpoint_PASS=True,sources=len(reg['source_hashes']),old_files=len(old),old_source_output_model_SHA_before_after=True,histories=histories,resource=resource,elapsed_seconds=time.perf_counter()-t)
    write('REVIEW.json',review);print(json.dumps(review,ensure_ascii=False),flush=True)
try:run()
except Exception:
    write('FAILURE.json',dict(status='FAILURE',traceback=traceback.format_exc(),resource=resource));raise

