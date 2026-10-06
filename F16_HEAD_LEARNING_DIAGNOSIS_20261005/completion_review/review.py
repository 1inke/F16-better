"""Saved-data completion audit and summaries; no torch/model/optimizer/DEV."""
from pathlib import Path
from datetime import datetime
import ast,ctypes,hashlib,json,os,time
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import psutil
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;ROOT=PARENT.parent
SNAPS=(0,1,5,10,20,40,60,80,100,120)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def verify(files):
    for p,h in files.items():assert sha(p)==h,('SHA mismatch',p)
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (keys or z.files)}
def dump(p,v):
    assert HERE in p.parents;p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    p=psutil.Process().memory_info();assert max(p.rss,p.private)<6*2**30 and min(m.avail_phys,m.avail_page)>=6*2**30
    return dict(peak_private_GiB=p.peak_pagefile/2**30,peak_RSS_GiB=p.peak_wset/2**30,physical_available_GiB=m.avail_phys/2**30,commit_available_GiB=m.avail_page/2**30)
def extract(path,name,env):
    nodes=[n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name==name];assert len(nodes)==1
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env);return env[name]
def metrics(z,include_clusters=True):
    x=z['trajectory'].astype(np.float64);y=z['future'].astype(np.float64)
    delta=x-y[:,None];err=np.sqrt(np.einsum('nktd,nktd->nkt',delta,delta));ade=np.sum(err,axis=2)/12;fde=err[:,:,11]
    good=(ade<=.2)&(fde<=.4)
    v=np.empty_like(x);v[:,:,0]=x[:,:,0]/.4;v[:,:,1:]=(x[:,:,1:]-x[:,:,:-1])/.4
    a=(v[:,:,1:]-v[:,:,:-1])/.4;j=(a[:,:,1:]-a[:,:,:-1])/.4
    out=dict(minade=ade.min(-1),minfde=fde.min(-1),joint=(ade+.5*fde).min(-1),good_count=np.count_nonzero(good,axis=1),coverage=np.any(good,axis=1),loose_coverage=np.any((ade<=.3)&(fde<=.5),axis=1))
    for k,value in [('speed',v),('accel',a),('jerk',j)]:
        mag=np.sqrt(np.einsum('nktd,nktd->nkt',value,value)).reshape(len(x),-1)
        out[k+'_mean']=np.sum(mag,axis=1)/mag.shape[1];out[k+'_window_p95']=np.quantile(mag,.95,axis=1)
    # Original complete-link reclustering was performed by completed worker;
    # supplement verifies bounds and group aggregation, not a new cluster algorithm.
    if include_clusters:
        for k in ('effective_good','all_distinct'):
            out[k]=z[k];assert np.all((z[k]>=0)&(z[k]<=20))
        assert np.all(out['effective_good']<=out['good_count']) and np.array_equal(out['effective_good']>0,out['coverage'])
    return out
def main():
    t=time.perf_counter();guard();pre=read(PARENT/'preflight.json');result=read(PARENT/'RESULT.json')
    assert result['status']=='PASS_DIAGNOSTIC_ONLY_HEAD_LEARNING' and len(result['runs'])==6
    worker=read(PARENT/'independent_worker_state.json');assert worker['status']==read(PARENT/'exit_status.json')['status']=='SUCCESS'
    assert read(PARENT/'ARITHMETIC_AUDIT.json')['status']==read(PARENT/'RESOURCE_USAGE.json')['status']=='PASS'
    old=read(PARENT/'old_results_manifest.json')['files'];locks={**old,**pre['source_hashes']};verify(locks)
    extra={str(p):sha(p) for p in [Path(__file__)]+[PARENT/p for p in ('RESULT.json','ARITHMETIC_AUDIT.json','RESOURCE_USAGE.json','preflight.json','split_ids.npz','split_indices.npz','independent_worker_state.json','exit_status.json')]+list(PARENT.glob('gradient_existing_seed*.json'))}
    verify(extra);indices=load(PARENT/'split_indices.npz');split=load(PARENT/'split_ids.npz')
    assert len(split['FIT'])==256 and len(split['CAL'])==393 and not set(split['FIT'])&set(split['CAL'])
    cache=load(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz',['sample_id','motion','change_group'])
    meta={};groupdir=ROOT/'F16_CUE_RESPONSE_CONCORDANCE_20261004/recovery_memory_20261004/jobs'
    for pop in ('FIT','CAL'):
        ix=indices[pop];local=ix if pop=='FIT' else ix-25912
        assert np.array_equal(cache['sample_id'][ix],split[pop])
        g=load(groupdir/(pop+'_native.npz'),['sample_id','cue_first_compatible','class_0','GT_event','GT_uncertain'])
        assert np.array_equal(g['sample_id'][local],split[pop]);clear=g['GT_event'][local].astype(bool)&~g['GT_uncertain'][local].astype(bool)
        compatible=clear&g['cue_first_compatible'][local].astype(bool);reg=cache['change_group'][ix]
        past=cache['motion'][ix,:16].reshape(-1,8,2);vel=(past[:,-1]-past[:,-2]).astype(np.float32)
        nt=128 if pop=='FIT' else 137
        masks=dict(all=np.ones(len(ix),bool),targeted=np.arange(len(ix))<nt,reference=np.arange(len(ix))>=nt,compatible=compatible,coarse=compatible&(g['class_0'][local]==7),clear=clear)
        masks.update({n:reg==i for i,n in enumerate(('stable','transition','change'))});meta[pop]=dict(masks=masks,reg=reg,vel=vel)
    del cache
    prior=read(ROOT/'F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/preflight.json')
    cfg=read(ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/config.json')
    classifier=extract(ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/directional_regime_pilot.py','classify_velocity',dict(np=np,CONTINUE=0,DECELERATE=1,STOP=2,ACCELERATE=3,LEFT=4,RIGHT=5,REVERSE=6))
    objective=extract(ROOT/'F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/diagnosis.py','numpy_terms',dict(np=np))
    objdata=dict(weights=np.array(prior['weights']),scale=np.array(prior['fit_protection_scales']))
    rows=[];gradients=[];max_metric=0.;max_group=0.;max_objective=0.;snapshots=0;outputs=0;curves=[]
    for run in result['runs']:
        arm,seed=run['arm'],run['seed'];directory=PARENT/'runs'/arm/f'seed{seed}';done=read(directory/'done.json')
        assert done['status']=='SUCCESS' and done['source_hashes']==pre['source_hashes'];verify(done['outputs']);outputs+=len(done['outputs'])
        hist=read(directory/'history.json');assert hist==run['history'] and [r['epoch'] for r in hist]==list(SNAPS)
        for item in hist:
            epoch=item['epoch']
            if 'gradients' in item:gradients.append(dict(arm=arm,seed=seed,epoch=epoch,report=item['gradients']))
            for pop in ('FIT','CAL'):
                guard();z=load(directory/f'{pop}_epoch{epoch}.npz');m=meta[pop]
                assert np.array_equal(z['sample_id'],split[pop]) and z['trajectory'].shape==(len(split[pop]),20,12,2) and np.isfinite(z['trajectory']).all()
                for name,mask in m['masks'].items():assert np.array_equal(z['mask_'+name],mask)
                if epoch==0:assert np.array_equal(z['trajectory'],z['native'])
                metric=metrics(z)
                for name,value in metric.items():
                    error=float(np.max(np.abs(value.astype(float)-z[name].astype(float))));max_metric=max(max_metric,error);assert error<1e-12,(arm,seed,pop,epoch,name,error)
                x=z['trajectory'].astype(np.float64);base=z['native'].astype(np.float64);future=z['future'].astype(np.float64)
                target=classifier(m['vel'],z['future'][:,-1]/12,cfg)
                family=classifier(np.repeat(m['vel'],20,axis=0),z['native'][:,:,-1].reshape(-1,2)/12,cfg).reshape(-1,20)
                ctx=dict(n=len(x),cross=int((family[:,:,None]!=family[:,None,:]).sum()),protected=int((m['reg']!=2).sum()))
                terms=objective(x,base,future,m['reg'],family,target,objdata,ctx)
                # Preserve saved float32 increments used by the actual objective;
                # avoid reconstructing their bitpatterns from rounded cumulative paths.
                res=z['residual'].astype(np.float64);terms['residual']=np.square(res).mean((1,2,3))*.01;terms['smoothness']=np.square(np.diff(res,axis=2)).mean((1,2,3))*.01
                obj={k:float(v.mean()) for k,v in terms.items()};obj['total']=sum(obj.values())
                diff=abs(obj['total']-item[pop]['objective']);max_objective=max(max_objective,diff)
                assert diff<=64*np.finfo(np.float32).eps*max(1,abs(obj['total'])),('float64 objective reference mismatch',diff)
                native_z={**z,'trajectory':z['native']};native=metrics(native_z,include_clusters=False)
                for group,mask in m['masks'].items():
                    n=int(mask.sum())
                    if not n:continue
                    priorgroup=item[pop]['groups'][group];assert n==priorgroup['n']
                    out={k:float(v[mask].mean()) for k,v in metric.items()}
                    for k,v in out.items():
                        e=abs(v-priorgroup[k]);max_group=max(max_group,e);assert e<1e-12,(arm,seed,pop,epoch,group,k,e)
                    covered=metric['coverage'];original=native['coverage']
                    detail=dict(arm=arm,seed=seed,pop=pop,epoch=epoch,group=group,n=n,good_windows=int(covered[mask].sum()),native_good_windows=int(original[mask].sum()),rescued=int((mask&covered&~original).sum()),lost=int((mask&~covered&original).sum()),raw_good_total=int(metric['good_count'][mask].sum()),effective_good_total=int(metric['effective_good'][mask].sum()),**out)
                    for k in ('minade','minfde','accel_mean','jerk_mean'):
                        bv=float(native[k][mask].mean());detail[k+'_change_percent']=100*(out[k]/bv-1) if bv else None
                    rows.append(detail)
                curves.append(dict(arm=arm,seed=seed,pop=pop,epoch=epoch,objective=item[pop]['objective'],numpy_objective=obj,reference_difference=diff,cal_feasible=item[pop]['cal_feasible_description']))
                snapshots+=1
    assert snapshots==120
    existing=[read(PARENT/f'gradient_existing_seed{s}.json') for s in (1,2,3)]
    verify(locks);verify(extra)
    report=dict(status='PASS',no_torch=True,no_model_forward=True,no_optimizer=True,saved_snapshots=120,six_registered120epochs_and10snapshots=True,done_outputs_SHA=outputs,old_files=len(old),sources=len(pre['source_hashes']),full_old_sources_before_after_SHA=True,all_saved_group_aggregates_and_masks_independently_recomputed=True,metric_max_error=max_metric,group_max_error=max_group,original_objective_float64_reference_max_error=max_objective,objective_check='NumPy reference with original terms/denominators and saved increments; bounded float32 rounding, not bitwise torch replay.',cluster_check='Completed worker recluster evidence; supplement validates per-row cluster bounds and group sums, not fresh all-cluster replay.',frozen_check='Completed worker original frozen_audit plus immutable checkpoint/source hashes, not fresh model forward.',resource=guard(),elapsed_seconds=time.perf_counter()-t,finished_at=worker['finished_at'],actual_worker_seconds=(datetime.fromisoformat(worker['finished_at'])-datetime.fromisoformat(worker['started_at'])).total_seconds(),direct_inputs_before_after_SHA=extra)
    dump(HERE/'REVIEW.json',report);dump(HERE/'GROUP_ROWS.json',rows);dump(HERE/'CURVES.json',curves);dump(HERE/'GRADIENTS.json',dict(fresh=gradients,existing=existing))
    key=[r for r in rows if r['epoch'] in (0,120) and r['group'] in ('all','coarse','reference')]
    dump(HERE/'KEY_RESULTS.json',key)
    lines=['# Fixed diagnostic sample results','','SAME candidate ADE<=.2m AND FDE<=.4m. Three seeds use SAME IDs, never sum as independent people.','', '| Arm | Seed | Population/group | Epoch | N | Good windows | Rescue | Lost | minADE | minFDE | Effective good total |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in key:lines.append(f"| {r['arm']} | {r['seed']} | {r['pop']}/{r['group']} | {r['epoch']} | {r['n']} | {r['good_windows']} | {r['rescued']} | {r['lost']} | {r['minade']:.6f} | {r['minfde']:.6f} | {r['effective_good_total']} |")
    (HERE/'GROUP_TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='direct_inputs_before_after_SHA'},ensure_ascii=False),flush=True)
    for r in key:
        if r['epoch']==120:print(json.dumps({k:r[k] for k in ('arm','seed','pop','group','n','good_windows','rescued','lost','minade','minfde','effective_good_total','accel_mean_change_percent')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
