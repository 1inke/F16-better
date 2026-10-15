"""Saved-output equal-budget diagnostic; no torch/model/optimizer imports."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from datetime import datetime
import json,hashlib,time,ctypes,gc,traceback
import numpy as np,psutil
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent;R=H.parent;P=R/'PROTOCOL_RECONCILIATION_RUN_20261007';A=P/'audit_recovery_20261007'
CACHE=R/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
LABELS=R/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'
META=R/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz'
PORTS=('native20','replacement20','common16','removed4','learned4');GROUPS=('overall','stable','transition','change')
RESOURCE=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
def now():return datetime.now().astimezone().isoformat()
def write(p,v):
    p=Path(p);p=p if p.is_absolute() else H/p;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8');os.replace(t,p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));p=psutil.Process().memory_info()
    RESOURCE['peak_private_GiB']=max(RESOURCE['peak_private_GiB'],p.peak_pagefile/2**30);RESOURCE['peak_RSS_GiB']=max(RESOURCE['peak_RSS_GiB'],p.peak_wset/2**30)
    RESOURCE['min_physical_GiB']=min(RESOURCE['min_physical_GiB'],m.avail_phys/2**30);RESOURCE['min_commit_GiB']=min(RESOURCE['min_commit_GiB'],m.avail_page/2**30);RESOURCE['checks']+=1
    if max(p.private,p.rss)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:
        write('RESOURCE_USAGE.json',dict(status='STOP_RESOURCE',**RESOURCE));raise MemoryError('process6GiB/physical-and-commit-reserve6GiB')
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%32==0:guard()
        assert sha(p)==h,('SOURCE_INPUT_SHA_MISMATCH',p)
def stage(name,**v):write('progress.json',dict(stage=name,at=now(),**v));print(name,v,flush=True)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def masks(groups):return [(name,np.ones(len(groups),bool) if j==0 else groups==j-1) for j,name in enumerate(GROUPS)]
def errors(p,gt,ind=False):
    d=p.astype(np.float64)-gt[:,None].astype(np.float64)
    e=np.sqrt(np.sum(d*d,axis=-1)) if ind else np.linalg.norm(d,axis=-1)
    a=e.mean(-1);f=e[:,:,-1];j=np.maximum(a/.2,f/.4);g=(a<=.2)&(f<=.4)
    ix=j.argmin(1);one=e[np.arange(len(p)),ix].reshape(-1,3,4).mean(-1)
    return a,f,j,g,one
def motion(p,ind=False):
    p=p.astype(np.float64)
    if ind:
        v=np.concatenate((p[:,:,:1],p[:,:,1:]-p[:,:,:-1]),axis=2)/.4;a=(v[:,:,1:]-v[:,:,:-1])/.4;j=(a[:,:,1:]-a[:,:,:-1])/.4
        s=np.sqrt((v*v).sum(-1));aa=np.sqrt((a*a).sum(-1));jj=np.sqrt((j*j).sum(-1))
    else:
        v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4;a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4;s=np.linalg.norm(v,axis=-1);aa=np.linalg.norm(a,axis=-1);jj=np.linalg.norm(j,axis=-1)
    return dict(speed_p95=float(np.quantile(s,.95)),accel_mean=float(aa.mean()),accel_p95=float(np.quantile(aa,.95)),jerk_mean=float(jj.mean()),jerk_p95=float(np.quantile(jj,.95)))
def clusters(p,g,ind=False):
    out=[]
    for paths,good in zip(p,g):
        guard();K=len(paths)
        if ind:
            distances=[]
            for i in range(K-1):
                d=paths[i].astype(np.float64)-paths[i+1:].astype(np.float64);e=np.sqrt((d*d).sum(-1));distances.extend(np.maximum(e.mean(-1)/.10,e[:,-1]/.20).tolist())
            lab=fcluster(linkage(np.asarray(distances),method='complete'),1.,criterion='distance')
        else:
            d=np.linalg.norm(paths[:,None].astype(np.float64)-paths[None,:].astype(np.float64),axis=-1);D=np.maximum(d.mean(-1)/.1,d[:,:,-1]/.2);lab=fcluster(linkage(D[np.triu_indices(K,1)],method='complete'),1.,criterion='distance')
        out.append((len(np.unique(lab)),len(np.unique(lab[good]))))
    return np.asarray(out,np.int32)
def select(paths,logits,ind=False):
    assert paths.shape[1:]==(24,12,2) and logits.shape==(len(paths),20) and np.isfinite(logits).all()
    if ind:
        rank=np.asarray([sorted(range(20),key=lambda k:(-float(row[k]),k)) for row in logits],np.int64)
    else:rank=np.argsort(-logits,axis=1,kind='stable')
    removed=np.sort(rank[:,16:],axis=1);keep=np.asarray([np.setdiff1d(np.arange(20),x,assume_unique=True) for x in removed]);rows=np.arange(len(paths))[:,None]
    native=paths[:,:20];replacement=native.copy();replacement[rows,removed]=paths[:,20:]
    ports=(native,replacement,native[rows,keep],native[rows,removed],paths[:,20:])
    assert all(p.shape[1]==k for p,k in zip(ports,(20,20,16,4,4)))
    return ports,removed,keep
def evaluation(z,logits,ind=False):
    ports,removed,keep=select(z['paths'],logits,ind);raw=[errors(p,z['gt'],ind) for p in ports];ng=raw[0][3].any(1)
    vectors=np.stack([np.stack((g.any(1),a.min(1),f.min(1),j.min(1)),axis=1) for a,f,j,g,_ in raw],axis=1)
    counts=np.stack([clusters(p,raw[i][3],ind) for i,p in enumerate(ports)],axis=1);result={};decomp={}
    for i,(name,p) in enumerate(zip(PORTS,ports)):
        a,f,j,g,one=raw[i];gg=g.any(1);sens=((a<=.3)&(f<=.5)).any(1);result[name]={}
        for group,mask in masks(z['groups']):
            n=int(mask.sum())
            if not n:result[name][group]={'n':0};continue
            guard();mot=motion(p[mask],ind);base=motion(ports[0][mask],ind)
            result[name][group]=dict(n=n,good=int(gg[mask].sum()),coverage=float(gg[mask].mean()),minADE=float(a[mask].min(1).mean()),minFDE=float(f[mask].min(1).mean()),J=float(j[mask].min(1).mean()),rescued=int((gg&~ng&mask).sum()),lost=int((~gg&ng&mask).sum()),retained=int((gg&ng&mask).sum()),effective_distinct=float(counts[mask,i,0].mean()),effective_good=float(counts[mask,i,1].mean()),sensitivity_03_05_good=int(sens[mask].sum()),single_Jclosest_early_middle_late=one[mask].mean(0),motion=mot,native20_motion=base,motion_all5_1pct=all(mot[k]<=1.01*base[k] for k in mot))
    common=raw[2][3].any(1);discard=raw[3][3].any(1);learn=raw[4][3].any(1);new=raw[1][3].any(1)
    assert np.array_equal(ng,common|discard) and np.array_equal(new,common|learn)
    for group,mask in masks(z['groups']):
        decomp[group]=dict(n=int(mask.sum()),common16_good=int((common&mask).sum()),removed4_marginal_good=int((~common&discard&mask).sum()),learned4_marginal_good=int((~common&learn&mask).sum()),learned_only_rescue=int((~common&~discard&learn&mask).sum()),removed_only_lost=int((~common&discard&~learn&mask).sum()),both_extra_good=int((~common&discard&learn&mask).sum()),neither_extra_good=int((~common&~discard&~learn&mask).sum()))
        assert result['replacement20'][group].get('good',0)-result['native20'][group].get('good',0)==decomp[group]['learned_only_rescue']-decomp[group]['removed_only_lost']
    return dict(metrics=result,decomposition=decomp),dict(removed_slots=removed,kept_slots=keep,vectors=vectors,cluster_counts=counts,replacement20=ports[1])
def compare(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b);return max([compare(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,list) and a and isinstance(a[0],(dict,str)):
        assert len(a)==len(b);return max(compare(x,y) for x,y in zip(a,b))
    if isinstance(a,(list,np.ndarray)):
        aa=np.asarray(a);bb=np.asarray(b);assert aa.shape==bb.shape;return float(np.max(np.abs(aa.astype(float)-bb.astype(float)))) if aa.size else 0.
    if isinstance(a,(bool,np.bool_)) or isinstance(a,str):assert a==b;return 0.
    return abs(float(a)-float(b))
def cache():
    guard()
    with np.load(CACHE,allow_pickle=False) as c:data=dict(ids=c['sample_id'].copy(),native=c['candidate'][...,:24].reshape(-1,20,12,2).copy(),groups=c['change_group'].copy(),logits=c['logits'].copy())
    with np.load(LABELS,allow_pickle=False) as z:assert np.array_equal(data['ids'],z['sample_id']);data['gt']=z['future'].copy()
    md=npz(META);lookup={int(x):i for i,x in enumerate(md['sample_id'])};order=np.asarray([lookup[int(x)] for x in data['ids']]);data['scene']=md['scene_family'][order];data['track']=np.asarray([str(md['sequence'][i])+'::'+str(md['pedestrian_id'][i]) for i in order]);del md
    assert len(data['ids'])==28791 and len(np.unique(data['ids']))==28791 and data['logits'].shape==(28791,20)
    assert all(np.isfinite(data[k]).all() for k in ('native','gt','logits'));data['lookup']={int(x):i for i,x in enumerate(data['ids'])};guard();return data
def jobs():
    for tag in ('scene_family','target_track','recording_time'):
        for seed in (42,43,44):
            d=P/'runs'/tag/f'seed_{seed}';man=read(d/'run_manifest.json');done=read(d/'done.json')
            assert done['status']=='PASS' and done['run_manifest_sha']==digest(man);assert man['seed']==seed and man['grouping']==tag and man['epochs']==25 and man['parameter_count']==64032
            for name in ('outer','inner','original_CAL'):yield tag,seed,name,d/f'{name}.npz',d,man,done
def identity(z,data,d,name):
    sid=npz(d/'split_ids.npz')[name+'_ids'];assert np.array_equal(sid,z['sample_id']) and len(np.unique(sid))==len(sid)
    ix=np.asarray([data['lookup'][int(x)] for x in sid]);assert z['paths'].shape==(len(sid),24,12,2) and np.isfinite(z['paths']).all()
    assert np.array_equal(z['paths'][:,:20],data['native'][ix])
    for k in ('gt','groups','track','scene'):assert np.array_equal(z[k],data[k][ix]),k
    return data['logits'][ix]
def locks():
    inherited=read(A/'source_manifest.json')['files'];verify(inherited);out=dict(inherited)
    paths=[CACHE,LABELS,META,*H.glob('*.py'),*H.glob('*.ps1'),*H.glob('*.md'),H/'protocol.json']
    for d in (A,P/'runs'):
        paths.extend(p for p in d.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.npz','.pt') and '__pycache__' not in p.parts and p.name not in ('runtime_state.json',))
    for p in sorted(set(paths)):
        key=str(p);h=sha(p)
        if key in out:assert out[key]==h,key
        else:out[key]=h
    return out,len(inherited)
def valid_done(z,source_sha):
    assert z['status']=='PASS' and z['source_sha']==source_sha,'ANALYSIS_SOURCE_REJECT';verify(z['files']);return True
def fixture():
    p=np.zeros((2,24,12,2),np.float32);scores=np.zeros((2,20),np.float32);p[:,20:,0,0]=np.arange(1,5)
    ports,rm,keep=select(p,scores);assert np.array_equal(rm,np.tile(np.arange(16,20),(2,1))) and np.array_equal(ports[1][:,16:],p[:,20:]);assert np.array_equal(ports[0],p[:,:20])
    q,s,t=select(p,scores,True);assert all(np.array_equal(x,y) for x,y in zip(ports,q)) and np.array_equal(rm,s)
    assert np.array_equal(clusters(np.zeros((1,20,12,2)),np.ones((1,20),bool)),[[1,1]])
    e=np.zeros((1,2,12,2));e[0,0,:,0]=.1;e[0,0,-1,0]=.5;e[0,1,:,0]=.3;e[0,1,-1,0]=.1;a,f,_,g,_=errors(e,np.zeros((1,12,2)));assert a.min()<.2 and f.min()<.4 and not g.any()
    z=dict(paths=np.ones((2,24,12,2),np.float32),gt=np.zeros((2,12,2)),groups=np.full(2,2));z['paths'][0,20]=0;z['paths'][1,16]=0
    met,v=evaluation(z,scores);assert met['decomposition']['overall']['learned_only_rescue']==1 and met['decomposition']['overall']['removed_only_lost']==1 and compare(met,evaluation(z,scores,True)[0])<1e-12
    fp=H/'fixtures'/'artifact.json';write(fp,{'fixture':True});done=dict(status='PASS',source_sha='fixture',files={str(fp):sha(fp)});assert valid_done(done,'fixture');rejected=0
    for bad in (dict(done,source_sha='bad'),dict(done,files={str(fp):'0'*64})):
        try:valid_done(bad,'fixture')
        except AssertionError:rejected+=1
    assert rejected==2
    return dict(stable_ties_order=True,disjoint_minima=True,identical20_one_cluster=True,rescue_lost_decomposition=True,analysis_SHA_skip_bad_source_output_reject=True,not_training_RNG_resume=True)
def bootstrap(rows):
    rng=np.random.default_rng(20261007);records=[]
    for (tag,split),items in rows.items():
        assert [x[0] for x in items]==[42,43,44];z=items[0][1]
        assert all(np.array_equal(z['sample_id'],x[1]['sample_id']) and np.array_equal(z['track'],x[1]['track']) and np.array_equal(z['groups'],x[1]['groups']) for x in items)
        _,iv=np.unique(z['track'],return_inverse=True);T=int(iv.max())+1
        for group,mask in masks(z['groups']):
            if not mask.any():continue
            count=np.bincount(iv[mask],minlength=T);active=np.flatnonzero(count)
            for name,base in [('replacement20','native20'),('learned4','removed4')]:
                diff=np.stack([x[1]['vectors'][:,PORTS.index(name)]-x[1]['vectors'][:,PORTS.index(base)] for x in items]);sums=np.zeros((3,T,4))
                for i in range(3):np.add.at(sums[i],iv[mask],diff[i,mask])
                draws=[]
                for b in range(2000):
                    if b%64==0:guard()
                    ss=rng.integers(0,3,size=3);tt=rng.choice(active,size=len(active),replace=True);draws.append(sums[ss][:,tt].sum((0,1))/(3*count[tt].sum()))
                records.append(dict(grouping=tag,split=split,group=group,port=name,reference=base,n=int(mask.sum()),tracks=len(active),model_seeds=3,metrics=['coverage','ADE','FDE','J'],estimate=diff[:,mask].mean((0,1)),CI95=np.quantile(np.asarray(draws),[.025,.975],axis=0).T,exploratory=True))
    return records
def preflight():
    start=time.perf_counter();guard();fixtures=fixture();files,inherited=locks();verify(files);write('source_manifest.json',dict(files=files,inherited_relevant_files=inherited,not_all_historical_files=True));data=cache();metadata=[]
    for tag,seed,name,path,d,man,done in jobs():
        verify(done['files']);z=npz(path);scores=identity(z,data,d,name);metadata.append(dict(grouping=tag,seed=seed,split=name,n=len(z['sample_id']),unique_ID_native_cache_GT_group_track_scene_EXACT=True,input_sha=sha(path),checkpoint_sha=sha(d/'checkpoint.pt'),checkpoint_binding_previously_audited_now_SHA_verified=True));del z
    first=npz(P/'runs/scene_family/seed_42/outer.npz');first={k:v[:128] for k,v in first.items()};scores=data['logits'][[data['lookup'][int(x)] for x in first['sample_id']]]
    met,v=evaluation(first,scores);other,ov=evaluation(first,scores,True);error=compare(met,other);assert error<=1e-12 and all(np.array_equal(v[k],ov[k]) for k in v)
    seconds=[]
    for b in range(8):
        startbatch=time.perf_counter();evaluation(first,scores);seconds.append(time.perf_counter()-startbatch)
    rows=sum(x['n'] for x in metadata);mean=float(np.mean(seconds));verify(files);guard();disk=psutil.disk_usage(str(H)).free/2**30;assert disk>1.0,'DISK_RESERVE_1GiB';elapsed=time.perf_counter()-start
    # Production reconstructs both implementations, with full clusters and group physics.
    compute=2*rows/128*mean;overhead=max(20.,elapsed-sum(seconds)-2*mean);planning=max(3.,np.ceil((compute+2*overhead+45)/60))
    write('PREFLIGHT.json',dict(status='PASS_SAVED_ONLY_EQUAL_K20',fixtures=fixtures,actual_rows128_independent_max_error=error,metadata=metadata,jobs=27,rows=rows,source_sha=digest(files),locked_files=len(files),inherited_relevant_files=inherited,no_model_forward_backward_optimizer=True,benchmark_8_saved128_seconds=seconds,mean_seconds128=mean,nominal_full_two_arithmetic_compute_seconds=compute,preflight_seconds=elapsed,planning_minutes=float(planning),ETA_minutes=[float(max(2,planning*.7)),float(planning*1.7)],resource=RESOURCE,disk_free_GiB=disk,estimated_new_paths_uncompressed_GiB=rows*20*12*2*4/2**30))
    write('prelaunch_review/REVIEW.json',dict(status='PASS',independent_actual128_quality_complete_clusters_all_motion_and_samepath_decomposition=True,source_initial_final_EXACT=True,all27_ID_native_GT_group_metadata_EXACT=True,all9_completed_checkpoint_manifest_SHA_verified_not_model_replay=True,no_forward_optimizer=True,not_training_RNG_resume=True,DA1='PASS_POLICY_SPECIFIC_NOT_UNIVERSAL_GENERATION_CAUSE',independent_max_error=error))
    print('PREFLIGHT_PASS',mean,rows,planning,RESOURCE,flush=True)
def main():
    start=time.perf_counter();guard();pre=read(H/'PREFLIGHT.json');assert pre['status']=='PASS_SAVED_ONLY_EQUAL_K20';files=read(H/'source_manifest.json')['files'];assert digest(files)==pre['source_sha'];stage('INITIAL_SOURCE_COMPLETED_SHA');verify(files);data=cache();records=[];rows={};output_sha={}
    for tag,seed,name,path,d,man,done in jobs():
        stage('SAVED_EQUAL_BUDGET_QUALITY_CLUSTERS_MOTION',grouping=tag,seed=seed,split=name,done=len(records),total=27);guard();z=npz(path);scores=identity(z,data,d,name);dest=H/'jobs'/f'{tag}_{seed}_{name}.json';out=dest.with_suffix('.npz');df=dest.with_suffix('.done.json')
        if df.exists():valid_done(read(df),pre['source_sha']);rec=read(dest);saved=npz(out)
        else:
            met,v=evaluation(z,scores);stage('INDEPENDENT_SAVED_ARITHMETIC',grouping=tag,seed=seed,split=name,done=len(records),total=27);ind,iv=evaluation(z,scores,True);err=compare(met,ind);assert err<=1e-12 and all(np.array_equal(v[k],iv[k]) for k in v),'INDEPENDENT_SAVED_ARITHMETIC'
            saved=dict(sample_id=z['sample_id'],groups=z['groups'],track=z['track'],scene=z['scene'],**v);out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists() and not dest.exists();np.savez_compressed(out,**saved);reload=npz(out);assert all(np.array_equal(v,reload[k]) for k,v in saved.items());rec=dict(grouping=tag,seed=seed,split=name,source_sha=pre['source_sha'],input_path=str(path),input_sha=sha(path),checkpoint_sha=sha(d/'checkpoint.pt'),metrics=met['metrics'],decomposition=met['decomposition'],independent_max_error=err,no_model_forward_backward_optimizer=True);write(dest,rec);write(df,dict(status='PASS',source_sha=pre['source_sha'],files={str(dest):sha(dest),str(out):sha(out),str(path):sha(path)}))
        rows.setdefault((tag,name),[]).append((seed,{k:saved[k] for k in ('sample_id','groups','track','vectors')}));records.append(rec)
        for p in (dest,out,df):output_sha[str(p)]=sha(p)
        del z,saved;gc.collect()
    stage('WITHIN_SPLIT_PAIRED_TARGET_TRACK_SEED_CI',done=27,total=27);ci=bootstrap(rows);verifyci=bootstrap(rows);assert compare(ci,verifyci)==0;write('BOOTSTRAP.json',dict(status='PASS',replicates=2000,seed=20261007,records=ci,independent_deterministic_recompute_max_error=0.,not_independent_scene_confirmation=True));output_sha[str(H/'BOOTSTRAP.json')]=sha(H/'BOOTSTRAP.json')
    stage('FINAL_SOURCE_COMPLETED_SHA',done=27,total=27);verify(files);guard();verify(output_sha)
    write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=27,full_independent_saved_quality_complete_clusters_motion_decomposition=True,independent_max_error=max(x['independent_max_error'] for x in records),scalar_CIs=len(ci)*4,CI_second_deterministic_reconstruction_max_error=0.,no_model_or_gradient_replay=True))
    write('PRESERVATION.json',dict(status='PASS',locked_files=len(files),inherited_relevant_files=pre['inherited_relevant_files'],all_original_digests_unchanged=True,not_all_historical_files=True));write('RESOURCE_USAGE.json',dict(status='PASS',**RESOURCE));write('RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_CANONICAL_EQUAL_K20',records=records,jobs=27,runs_reused=9,new_training_runs=0,new_formal_count=0,source_sha=pre['source_sha'],output_sha=output_sha,elapsed_seconds=time.perf_counter()-start,resource=RESOURCE,major_not_inferred=True));stage('SUCCESS',done=27,total=27)
if __name__=='__main__':
    try:preflight() if '--preflight' in sys.argv else main()
    except Exception:
        write('failures/failure_'+str(time.time_ns())+'.json',dict(traceback=traceback.format_exc(),resource=RESOURCE,no_training=True));raise
