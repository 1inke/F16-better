"""Frozen matched FIT/CAL diagnosis; no model training or legacy top-level imports."""
from pathlib import Path
from datetime import datetime
import argparse, csv, gc, hashlib, json, os, sys, time, traceback
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from resource_guard import guard
from frozen_loader import build, exact, FAMILIES, SOURCES, NATIVE
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
TRAIN=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
ENC=ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'
LABEL=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'
SPLITS={'FIT':(0,25912),'CAL':(25912,28791)}
PAIR=np.triu_indices(20,1)
START=time.perf_counter()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    for attempt in range(6):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==5:raise
            time.sleep(.02*2**attempt)
def stage(name,**info):
    write(HERE/'progress.json',dict(stage=name,at=datetime.now().astimezone().isoformat(),**info))
    print(name,json.dumps(info),flush=True)
def verify(files,label):
    for i,(p,h) in enumerate(files.items()):
        if i%1000==0:guard.check();stage(label,files=i,total=len(files))
        assert sha(p)==h,(label,p)
    return len(files)
def old_snapshot():
    target=HERE/'old_results_manifest.json'
    if target.exists():return read(target)['files']
    files=dict(read(ROOT/'GENERATION_DISCREPANCY_NORMALIZATION_SERIES_20261005/old_results_manifest.json')['files'])
    verify(files,'INHERITED_OLD_SHA')
    for folder in ROOT.iterdir():
        if folder==HERE or not folder.is_dir() or not folder.name.startswith(('GENERATION_','F16_','HEAD_','PULSE_')):continue
        for path in folder.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.lock','.tmp','.pending') and path.name!='runtime_state.json':
                if str(path) not in files:files[str(path)]=sha(path)
    write(target,dict(files=files,prior_digests_preserved=True,excluded_controls=['.memory','AUTONOMOUS_GENERATION_CYCLE.json','runtime_state.json','locks/tmp/bytecode']))
    return files
def checkpoints():
    result=[]
    for family,folder in FAMILIES.items():
        locks=read(ROOT/folder/'results/model_lock.json')['models']
        for seed in (1,2,3):
            for policy in ('constrained','final'):
                path=ROOT/folder/f'results/models/mean_std/seed_{seed}/balanced_protected/{policy}.pth'
                assert str(path) in locks and sha(path)==locks[str(path)]
                result.append(dict(family=family,seed=seed,policy=policy,path=str(path),sha256=locks[str(path)]))
    assert len(result)==18;return result
def source_hashes(models):
    inherited=dict(read(ROOT/'GENERATION_DISCREPANCY_NORMALIZATION_SERIES_20261005/preflight.json')['source_hashes'])
    verify(inherited,'PRIOR_SOURCE_SHA')
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/'README.md',HERE/'DESIGN.md',HERE/'protocol.json',HERE/'old_results_manifest.json',TRAIN,ENC,LABEL,NATIVE]
    for folder in FAMILIES.values():
        paths += [ROOT/folder/'run_head.py',ROOT/folder/'state_features.py',ROOT/folder/'state_scalers.npz',ROOT/folder/'results/model_lock.json']
    paths += [ROOT/s for s in SOURCES]
    return {**inherited,**{str(p):sha(p) for p in paths if p.is_file()},**{x['path']:x['sha256'] for x in models}}
def load_npz(path,keys):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in keys}
def load_data():
    guard.check();cache=load_npz(TRAIN,['sample_id','candidate','motion','logits','change_group','ade','fde'])
    encoded=load_npz(ENC,['sample_id','past','tokens'])
    labels=load_npz(LABEL,['sample_id','future'])
    ids=cache['sample_id']
    assert len(ids)==len(np.unique(ids))==28791 and np.array_equal(ids,encoded['sample_id']) and np.array_equal(ids,labels['sample_id'])
    assert not set(ids[:25912])&set(ids[25912:])
    obs=np.zeros((28791,940),np.float32);obs[:,:16]=encoded['past'].reshape(-1,16);obs[:,16:784]=encoded['tokens'].reshape(-1,768)
    assert np.count_nonzero(obs[:,784:])==0
    past=cache['motion'][:,:16].reshape(-1,8,2);v=(past[:,-1]-past[:,-2]).astype(np.float32)
    speed=np.concatenate((np.linalg.norm(v,axis=-1)[:,None],v),1).astype(np.float32)
    base=cache['candidate'][...,:24].reshape(-1,20,12,2).copy();future=labels['future']
    assert base.dtype==obs.dtype==future.dtype==speed.dtype==np.float32
    e=np.linalg.norm(base-future[:,None],axis=-1)
    assert np.allclose(e.mean(-1),cache['ade'],atol=2e-5,rtol=0) and np.allclose(e[:,:,-1],cache['fde'],atol=2e-5,rtol=0)
    assert set(np.unique(cache['change_group']))=={0,1,2}
    # Validate saved FIT-only scaler identities without refitting.
    for folder in FAMILIES.values():
        with np.load(ROOT/folder/'state_scalers.npz',allow_pickle=False) as s:
            assert np.array_equal(s['fit_id'],ids[:25912]) and np.isfinite(s['mean']).all() and (s['std']>=.001).all()
    with np.load(ROOT/FAMILIES['centered_own']/'centered_scalers.npz',allow_pickle=False) as s:assert np.array_equal(s['fit_id'],ids[:25912])
    result=dict(ids=ids,obs=obs,base=base,future=future,speed=speed,logits=cache['logits'],group=cache['change_group'])
    del cache,encoded,labels,e;gc.collect();guard.check();return result
def masks(group):
    return [('overall',np.ones(len(group),bool))]+[(key,group==i) for i,key in enumerate(('stable','transition','change'))]
def metrics(paths,future,logits):
    guard.check();n=len(paths);a=np.empty((n,20),np.float64);f=a.copy()
    q={key:np.empty(n,dtype=dtype) for key,dtype in [('coverage',bool),('loose_coverage',bool),('raw_good',np.int16),('effective_good',np.int16),('all_distinct',np.int16),('selected_good',bool),('selected_slot',np.int16),('oracle_slot',np.int16),('early',np.float64),('middle',np.float64),('late',np.float64),('selected_ADE',np.float64),('selected_FDE',np.float64),('disjoint_minima',bool)]}
    motion={k:np.empty((n,20,t),np.float64) for k,t in [('speed',12),('accel',11),('jerk',10)]}
    for begin in range(0,n,256):
        guard.check();sl=slice(begin,min(n,begin+256));x=paths[sl].astype(np.float64);truth=future[sl].astype(np.float64)
        error=np.linalg.norm(x-truth[:,None],axis=-1);aa=error.mean(-1);ff=error[:,:,-1];a[sl]=aa;f[sl]=ff
        good=(aa<=.2)&(ff<=.4);slot=np.maximum(aa/.2,ff/.4).argmin(-1);row=np.arange(len(x));selected=logits[sl].argmax(-1)
        q['coverage'][sl]=good.any(-1);q['loose_coverage'][sl]=((aa<=.3)&(ff<=.5)).any(-1);q['raw_good'][sl]=good.sum(-1)
        q['selected_slot'][sl]=selected;q['selected_good'][sl]=good[row,selected];q['oracle_slot'][sl]=slot
        q['selected_ADE'][sl]=aa[row,selected];q['selected_FDE'][sl]=ff[row,selected]
        q['disjoint_minima'][sl]=(aa.min(-1)<=.2)&(ff.min(-1)<=.4)&~good.any(-1)
        for key,ts in [('early',slice(0,4)),('middle',slice(4,8)),('late',slice(8,12))]:q[key][sl]=error[row,slot,ts].mean(-1)
        for r in range(len(x)):
            dist=np.linalg.norm(x[r,PAIR[0]]-x[r,PAIR[1]],axis=-1)
            lab=fcluster(linkage(np.maximum(dist.mean(-1)/.10,dist[:,-1]/.20),method='complete'),1.,criterion='distance')
            q['all_distinct'][begin+r]=len(np.unique(lab));q['effective_good'][begin+r]=len(np.unique(lab[good[r]]))
        vel=np.diff(x,axis=2,prepend=np.zeros_like(x[:,:,:1]))/.4;acc=np.diff(vel,axis=2)/.4;jerk=np.diff(acc,axis=2)/.4
        for k,z in [('speed',vel),('accel',acc),('jerk',jerk)]:motion[k][sl]=np.linalg.norm(z,axis=-1)
    q['minADE']=a.min(-1);q['minFDE']=f.min(-1)
    return q,motion
def grouped(q,motion,group,native=None):
    rows=[]
    for name,mask in masks(group):
        row=dict(group=name,n=int(mask.sum()))
        for k in ('coverage','loose_coverage','raw_good','effective_good','selected_good','disjoint_minima'):row[k+'_total']=int(q[k][mask].sum())
        for k in ('minADE','minFDE','early','middle','late','selected_ADE','selected_FDE'):row[k]=float(q[k][mask].mean())
        row['distinct_histogram']=np.bincount(q['all_distinct'][mask],minlength=21).tolist()
        row['effective_good_histogram']=np.bincount(q['effective_good'][mask],minlength=21).tolist()
        row['selection_miss']=int((q['coverage'][mask]&~q['selected_good'][mask]).sum())
        for k,z in motion.items():
            arr=z[mask].reshape(-1);row[k+'_mean']=float(arr.mean());row[k+'_p95']=float(np.quantile(arr,.95))
        if native is not None:
            nq,nm=native;row['rescued']=int((~nq['coverage'][mask]&q['coverage'][mask]).sum());row['lost']=int((nq['coverage'][mask]&~q['coverage'][mask]).sum())
            for k in ('minADE','minFDE'):row[k+'_native_gain_pct']=float(100*(1-row[k]/nq[k][mask].mean()))
            for k,stat in [('speed','p95'),('accel','mean'),('accel','p95'),('jerk','mean'),('jerk','p95')]:
                arr=nm[k][mask].reshape(-1);old=float(arr.mean() if stat=='mean' else np.quantile(arr,.95))
                row[k+'_'+stat+'_native_harm_pct']=100*(row[k+'_'+stat]/max(old,1e-12)-1)
            row['motion_1pct_each_gate']=all(row[k+'_'+stat+'_native_harm_pct']<=1+1e-10 for k,stat in [('speed','p95'),('accel','mean'),('accel','p95'),('jerk','mean'),('jerk','p95')])
        rows.append(row)
    return rows
def arithmetic(saved,future,logits):
    paths=saved['trajectory'];q,mot=metrics(paths,future,logits);maximum=0.
    for k,v in q.items():
        actual=saved[k];assert v.dtype==actual.dtype
        diff=float(np.max(np.abs(v.astype(float)-actual.astype(float))));maximum=max(maximum,diff)
        assert np.array_equal(v,actual),(k,diff)
    for k,v in mot.items():assert np.array_equal(v,saved['motion_'+k]),k
    # Independent coordinate sums and scalar temporal/kinematic sanity on boundary rows.
    for r in sorted(set([0,len(paths)//2,len(paths)-1])):
        err=np.sqrt(np.sum((paths[r].astype(float)-future[r].astype(float))**2,axis=-1))
        assert np.max(np.abs(err.mean(-1).min()-q['minADE'][r]))<1e-12
        assert abs(err[:,-1].min()-q['minFDE'][r])<1e-12
    return dict(status='PASS',all_metric_cluster_motion_columns_exact=True,max_error=maximum,independent_coordinate_scalar_boundary_rows=3,saved_arithmetic_not_all_models_replayed=True)
def synthetic():
    truth=np.zeros((1,12,2),np.float32);paths=np.zeros((1,20,12,2),np.float32);q,_=metrics(paths,truth,np.zeros((1,20),np.float32))
    assert q['coverage'][0] and q['raw_good'][0]==20 and q['effective_good'][0]==q['all_distinct'][0]==1
    paths.fill(0)
    paths[:,:,:,:]=2.;paths[0,0,:,:]=0.;paths[0,0,-1]=[1.,0.];paths[0,1,:,:]=[1.,0.];paths[0,1,-1]=[0.,0.]
    q,_=metrics(paths,truth,np.zeros((1,20),np.float32))
    assert not q['coverage'][0] and q['disjoint_minima'][0]
    return dict(status='PASS',twenty_copies_one_effective=True,disjoint_minima_not_same_path=True)
def history_alignment(entry,meta,q,group):
    directory=Path(entry['path']).parent
    with (directory/'history.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert [int(x['epoch']) for x in rows]==list(range(1,81))
    if meta['epoch']==0:assert meta['fallback'];return dict(status='PASS_EXPLICIT_NATIVE_FALLBACK')
    row=rows[meta['epoch']-1];maximum=0.
    for name,mask in masks(group):
        for key,oldkey in [('minADE','minade20'),('minFDE','minfde20')]:
            diff=abs(float(q[key][mask].mean())-float(row[name+'_'+oldkey]));maximum=max(maximum,diff);assert diff<2e-5,(entry,name,key,diff)
    return dict(status='PASS_CAL_OUTPUT_HISTORY_ALIGNMENT',max_error=maximum,epoch=meta['epoch'])
def predict(model,data,a,b,device):
    result=np.empty((b-a,20,12,2),np.float32)
    with torch.inference_mode():
        for begin in range(a,b,128):
            guard.check();end=min(b,begin+128)
            obs,base,speed=[torch.from_numpy(data[k][begin:end]).to(device) for k in ('obs','base','speed')]
            out,res=model(obs,base,speed);assert out.shape==(end-begin,20,12,2) and torch.isfinite(out).all()
            limit=1.5*(speed[:,0,None,None,None]+.03);assert (res.abs()<=limit+1e-6).all()
            result[begin-a:end-a]=out.cpu().numpy()
    return result
def preflight():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);guard.check()
    old=old_snapshot();models=checkpoints();sources=source_hashes(models)
    verify(old,'OLD_BEFORE');verify(sources,'SOURCE_BEFORE')
    data=load_data();mathaudit=synthetic();device=torch.device('cuda:0');assert torch.cuda.is_available()
    checks=[];benchmark=[]
    for family in FAMILIES:
        entry=next(x for x in models if x['family']==family and x['seed']==1 and x['policy']=='constrained')
        model,meta=build(family,entry['path']);state={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.to(device);indices=np.arange(128);path=predict(model,data,0,128,device);repeat=predict(model,data,0,128,device)
        assert np.array_equal(path,repeat)
        with torch.inference_mode():
            local=torch.from_numpy(data['obs'][:128,:16].reshape(-1,1,8,2)).to(device)
            tokens=torch.from_numpy(data['obs'][:128,16:784].reshape(-1,1,8,96)).to(device)
            nativepath,logit=model.decoder(tokens,local);nativepath=(nativepath[:,0]-local[:,0,-1,None,None]).cpu().numpy()
        native_error=float(np.max(np.abs(nativepath-data['base'][:128])));assert native_error<2e-5,native_error
        native_score_error=float(np.max(np.abs(logit[:,0].cpu().numpy()-data['logits'][:128])));assert native_score_error<2e-5,native_score_error
        q,mot=metrics(path,data['future'][:128],data['logits'][:128]);fixture=dict(trajectory=path,**q,**{'motion_'+k:v for k,v in mot.items()})
        ar=arithmetic(fixture,data['future'][:128],data['logits'][:128]);exact(model,state)
        checks.append(dict(family=family,meta=meta,repeat_path_exact=True,native_path_error=native_error,native_logits_error=native_score_error,all_parameters_buffers_frozen_exact=True,arithmetic=ar))
        torch.cuda.synchronize();t=time.perf_counter()
        for i in range(8):predict(model,data,i*128,(i+1)*128,device)
        torch.cuda.synchronize();sec=(time.perf_counter()-t)/8
        t=time.perf_counter();metrics(data['base'][:256],data['future'][:256],data['logits'][:256]);clustersec=time.perf_counter()-t
        benchmark.append(dict(family=family,seconds_per128_frozen_forward=sec,seconds_per256_metrics_cluster=clustersec))
        exact(model,state);del model,state;gc.collect();torch.cuda.empty_cache()
    try:verify({str(HERE/'protocol.json'):'0'*64},'INTENTIONAL_BAD_SOURCE_FIXTURE')
    except AssertionError:bad_source_rejected=True
    else:raise AssertionError('bad source accepted')
    # Read-only all18 metadata and canonical decoder lock checked, one checkpoint at a time.
    for entry in models:
        model,meta=build(entry['family'],entry['path']);del model;gc.collect()
    verify(sources,'SOURCE_AFTER');verify(old,'OLD_AFTER')
    forward_sec=max(x['seconds_per128_frozen_forward'] for x in benchmark)*np.ceil(28791/128)*18
    metrics_sec=max(x['seconds_per256_metrics_cluster'] for x in benchmark)*np.ceil(28791/256)*19*2
    estimated=forward_sec+metrics_sec+180
    report=dict(status='PASS',checkpoints=models,source_hashes=sources,old_files=len(old),math=mathaudit,real_GPU128_checks=checks,benchmarks=benchmark,
        planning_compute_io_seconds=estimated,scope='FIT_CAL_FROZEN_DIAGNOSTIC_ONLY_NO_OPTIMIZER_NO_DEV',memory=guard.check(),
        no_training_gradient_or_optimizer_resume_claim=True,source_mismatch_rejected=bad_source_rejected)
    write(HERE/'preflight.json',report);guard.save('PREFLIGHT_PASS');print('PREFLIGHT_PASS',json.dumps(dict(old=len(old),sources=len(sources),estimated_seconds=estimated,memory=guard.values())),flush=True)
def job_name(entry,split):
    return 'native_'+split if entry is None else f"{entry['family']}_seed{entry['seed']}_{entry['policy']}_{split}"
def run():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);audit=read(HERE/'preflight.json');assert audit['status']=='PASS'
    sources=audit['source_hashes'];old=read(HERE/'old_results_manifest.json')['files'];verify(sources,'INITIAL_SOURCE_SHA');verify(old,'INITIAL_OLD_SHA')
    data=load_data();device=torch.device('cuda:0');done=0;records=[];joblocks={};frozen=[];native={}
    output=HERE/'jobs';output.mkdir(exist_ok=True)
    for entry in [None]+audit['checkpoints']:
        model=state=meta=None
        if entry is not None:
            assert sha(entry['path'])==entry['sha256']
            model,meta=build(entry['family'],entry['path']);state={k:v.detach().clone() for k,v in model.state_dict().items()};model.to(device)
        for split,(a,b) in SPLITS.items():
            key=job_name(entry,split);dst=output/(key+'.npz');marker=output/(key+'.done.json')
            stage('FROZEN_FORWARD_AND_SAVED_AUDIT',active_job=key,completed_jobs=done,total_jobs=38)
            if marker.exists():
                record=read(marker);assert record['status']=='SUCCESS' and record['source_signature']==sha(HERE/'preflight.json')
                assert sha(dst)==record['outputs'][str(dst)]
                z=load_npz(dst,record['columns']);assert np.array_equal(z['sample_id'],data['ids'][a:b])
            else:
                path=data['base'][a:b].copy() if entry is None else predict(model,data,a,b,device)
                q,mot=metrics(path,data['future'][a:b],data['logits'][a:b])
                z=dict(sample_id=data['ids'][a:b],trajectory=path,**q,**{'motion_'+k:v for k,v in mot.items()})
                np.savez_compressed(dst,**z)
                columns=list(z);del z;gc.collect();z=load_npz(dst,columns)
            q={k:z[k] for k in z if k not in ('sample_id','trajectory') and not k.startswith('motion_')}
            mot={k:z['motion_'+k] for k in ('speed','accel','jerk')}
            independent=arithmetic(z,data['future'][a:b],data['logits'][a:b])
            if entry is None:native[split]=(q,mot)
            groups=grouped(q,mot,data['group'][a:b],native[split])
            align=None if entry is None or split!='CAL' else history_alignment(entry,meta,q,data['group'][a:b])
            record=dict(status='SUCCESS',source_signature=sha(HERE/'preflight.json'),columns=list(z),outputs={str(dst):sha(dst)},arithmetic=independent,groups=groups,model=entry,metadata=meta,cal_history_alignment=align)
            write(marker,record);records.append(dict(job=key,split=split,model=entry,metadata=meta,groups=groups,cal_history_alignment=align))
            joblocks[str(dst)]=sha(dst);joblocks[str(marker)]=sha(marker);done+=1
            del z,q,mot;gc.collect();guard.check()
        if model is not None:exact(model,state);frozen.append(dict(checkpoint=entry['path'],all_weights_buffers_exact=True,no_parameter_grad=True));del model,state;gc.collect();torch.cuda.empty_cache()
    assert done==38 and len(frozen)==18
    verify(joblocks,'FINAL_76_OUTPUT_SHA');verify(sources,'FINAL_SOURCE_SHA');verify(old,'FINAL_OLD_SHA')
    write(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=38,full_saved_metric_cluster_motion_recomputed=True,coordinate_scalar_boundary_check=True,all_saved_output_SHA=joblocks))
    write(HERE/'FROZEN_REPLAY_AUDIT.json',dict(status='PASS',records=frozen,no_optimizer=True,GT_forward=False,original_encoder_scorer_immutable_SHA=True))
    write(HERE/'RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY',records=records,jobs=38,checkpoint_count=18,FIT=25912,CAL=2879,source_files=len(sources),old_files=len(old),no_DEV=True,no_new_training=True,
        limitations=['CAL used for checkpoint selection; not independent generalization','3scene and overlapping windows','Frozen diagnostic not causal mapping/intent/SOTA proof'],seconds=time.perf_counter()-START))
    guard.save('SUCCESS');write(HERE/'exit_status.json',dict(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat()))
if __name__=='__main__':
    try:
        parser=argparse.ArgumentParser();parser.add_argument('--action',choices=['preflight','run'],required=True);args=parser.parse_args()
        (preflight if args.action=='preflight' else run)()
    except BaseException:
        guard.save('FAILURE');write(HERE/'failure.json',dict(status='FAILURE',at=datetime.now().astimezone().isoformat(),traceback=traceback.format_exc()))
        raise

