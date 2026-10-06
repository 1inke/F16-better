"""Endpoint-preserving projection: frozen FIT/CAL saved-coordinate diagnostic."""
from real_preflight import *
from scipy.cluster.hierarchy import linkage,fcluster
from datetime import datetime
PAIR=np.triu_indices(20,1)
namespace=dict(np=np,guard=guard,PAIR=PAIR,linkage=linkage,fcluster=fcluster)
MATH_SOURCE=ROOT/'F16_MATCHED_FIT_CAL_TRANSFER_DIAGNOSIS_20261005/diagnosis.py'
nodes(MATH_SOURCE,['metrics','masks','arithmetic','synthetic'],namespace)
metrics,masks,arithmetic,synthetic=[namespace[k] for k in ('metrics','masks','arithmetic','synthetic')]
SPLITS={'FIT':(0,25912),'CAL':(25912,28791)}
def progress(stage,**kw):
    write(HERE/'progress.json',dict(stage=stage,at=datetime.now().astimezone().isoformat(),**kw));print(stage,kw,flush=True)
def joint(path,future):
    out=np.empty(len(path),np.float64)
    for i in range(0,len(path),256):
        e=np.linalg.norm(path[i:i+256].astype(np.float64)-future[i:i+256,None].astype(np.float64),axis=-1)
        out[i:i+256]=np.maximum(e.mean(-1)/.2,e[:,:,-1]/.4).min(-1)
    return out
def grouped(q,motion,group,native=None,original=None):
    out=[]
    for name,mask in masks(group):
        row=dict(group=name,n=int(mask.sum()))
        for k in ('coverage','loose_coverage','raw_good','effective_good','selected_good','disjoint_minima'):row[k+'_total']=int(q[k][mask].sum())
        for k in ('minADE','minFDE','joint_J','early','middle','late','selected_ADE','selected_FDE'):row[k]=float(q[k][mask].mean())
        row['distinct_histogram']=np.bincount(q['all_distinct'][mask],minlength=21).tolist()
        row['effective_good_histogram']=np.bincount(q['effective_good'][mask],minlength=21).tolist()
        row['selection_miss']=int((q['coverage'][mask]&~q['selected_good'][mask]).sum())
        for k,v in motion.items():
            x=v[mask].reshape(-1);row[k+'_mean']=float(x.mean());row[k+'_p95']=float(np.quantile(x,.95))
        for label,reference in [('native',native),('original',original)]:
            if reference is None:continue
            rq,rm=reference
            row[label+'_rescued']=int((~rq['coverage'][mask]&q['coverage'][mask]).sum())
            row[label+'_lost']=int((rq['coverage'][mask]&~q['coverage'][mask]).sum())
            row[label+'_retained']=int((rq['coverage'][mask]&q['coverage'][mask]).sum())
            gates=[]
            for k,stat in [('speed','p95'),('accel','mean'),('accel','p95'),('jerk','mean'),('jerk','p95')]:
                arr=rm[k][mask].reshape(-1);old=float(arr.mean() if stat=='mean' else np.quantile(arr,.95));new=row[k+'_'+stat]
                row[k+'_'+stat+'_'+label+'_harm_pct']=100*(new/old-1) if old!=0 else (0. if new==0 else None)
                gates.append(new<=1.01*old)
            row[label+'_motion_1pct_all_five']=all(gates)
        out.append(row)
    return out
def package(path,truth,logits):
    q,m=metrics(path,truth,logits);q['joint_J']=joint(path,truth)
    return dict(trajectory=path,**q,**{'motion_'+k:v for k,v in m.items()})
def unpack(z,prefix):
    plain={k[len(prefix):]:v for k,v in z.items() if k.startswith(prefix)}
    q={k:v for k,v in plain.items() if k!='trajectory' and not k.startswith('motion_')}
    m={k:plain['motion_'+k] for k in ('speed','accel','jerk')}
    return plain,q,m
def saved_audit(z,d,a,b,projected):
    for prefix in (['original_','projected_'] if projected else ['native_']):
        plain,q,m=unpack(z,prefix);arithmetic(plain,d['future'][a:b],d['logits'][a:b])
        assert np.array_equal(plain['joint_J'],joint(plain['trajectory'],d['future'][a:b]))
    assert np.array_equal(z['sample_id'],d['ids'][a:b])
    if projected:
        for i in range(0,b-a,128):
            sl=slice(i,min(b-a,i+128));audit(d['base'][a:b][sl],z['increments'][sl],z['projected_trajectory'][sl],z['projected_increments'][sl],z['limit'][sl])
            with torch.inference_mode():
                pp,up,stats=project(torch.from_numpy(d['base'][a:b][sl]).cuda(),torch.from_numpy(z['increments'][sl]).cuda(),torch.from_numpy(z['limit'][sl]).cuda())
            assert np.array_equal(pp.cpu().numpy(),z['projected_trajectory'][sl]) and np.array_equal(up.cpu().numpy(),z['projected_increments'][sl])
            for k,v in stats.items():assert np.array_equal(v.cpu().numpy(),z['projection_'+k][sl])
    return dict(status='PASS',full_saved_quality_cluster_motion_exact=True,projection_recomputed_no_model_forward=True)
def prelaunch():
    prior=read(HERE/'real_preflight.json');assert prior['status']=='PASS_REAL_SIX_HEAD_NUMERIC_PREFLIGHT'
    verify(prior['source_hashes']);assert read(HERE/'numeric_preflight/REPORT.json')['status']=='PASS_NUMERIC_REPAIR_SYNTHETIC_ONLY'
    math=synthetic();d=data();t=time.perf_counter()
    fixture=package(d['base'][:256],d['future'][:256],d['logits'][:256]);arithmetic(fixture,d['future'][:256],d['logits'][:256])
    assert np.array_equal(fixture['joint_J'],joint(fixture['trajectory'],d['future'][:256]))
    seconds=(time.perf_counter()-t)/2
    src=dict(prior['source_hashes'])
    for p in list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/'protocol.json',HERE/'old_results_manifest.json',MATH_SOURCE,TRAIN,ENC,LABEL]:
        if str(p) in src:assert sha(p)==src[str(p)]
        else:src[str(p)]=sha(p)
    signature=hashlib.sha256(json.dumps(src,sort_keys=True).encode()).hexdigest()
    tmp=HERE/'prelaunch_review/resume_fixture.npz';tmp.parent.mkdir(exist_ok=True);np.savez_compressed(tmp,sample_id=d['ids'][:2])
    done=dict(status='SUCCESS',source_signature=signature,outputs={str(tmp):sha(tmp)})
    assert all(sha(p)==h for p,h in done['outputs'].items())
    bad_source=done['source_signature']!='bad';bad_output=sha(tmp)!='0'*64;assert bad_source and bad_output
    verify(src);guard.check()
    compute=max(x['seconds_per128'] for x in prior['checks'])*np.ceil(28791/128)*6
    cluster=seconds*np.ceil(28791/256)*26 # original+projected+native, compute+saved audit
    estimate=compute+cluster+max(180,2*prior['initial_SHA_seconds'])+240
    report=dict(status='PASS',source_hashes=src,source_signature=signature,checkpoints=models(),real_preflight_sha256=sha(HERE/'real_preflight.json'),math=math,seconds_per256_geometry=seconds,planning_seconds=estimate,memory=guard.check(),analysis_done_SHA_skip_fixture=True,bad_source_output_rejected=True,no_optimizer_resume_claim=True)
    write(HERE/'prelaunch_review/REVIEW.json',report);print(json.dumps({k:report[k] for k in ('status','seconds_per256_geometry','planning_seconds','memory')}))
def run():
    begin=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    review=read(HERE/'prelaunch_review/REVIEW.json');assert review['status']=='PASS'
    sources=review['source_hashes'];old=read(HERE/'old_results_manifest.json')['files']
    progress('INITIAL_SOURCE_OLD_SHA',completed_jobs=0);verify(sources);verify(old)
    d=data();native={};records=[];outputs={};frozen=[];done=0;folder=HERE/'jobs';folder.mkdir(exist_ok=True)
    for entry in [None]+review['checkpoints']:
        model=state=None
        if entry:
            model,meta=build('raw',entry['path']);state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.cuda()
        for split,(a,b) in SPLITS.items():
            key=('native' if entry is None else entry['arm']+'_seed'+str(entry['seed']))+'_'+split
            dst=folder/(key+'.npz');marker=folder/(key+'.done.json');progress('FROZEN_FORWARD_METRIC_SAVED_AUDIT',active_job=key,completed_jobs=done,total_jobs=14)
            if marker.exists():
                record=read(marker);assert record['status']=='SUCCESS' and record['source_signature']==review['source_signature'];verify(record['outputs'])
                with np.load(dst,allow_pickle=False) as f:z={k:f[k] for k in f.files}
            else:
                if entry is None:
                    z=dict(sample_id=d['ids'][a:b],**{'native_'+k:v for k,v in package(d['base'][a:b],d['future'][a:b],d['logits'][a:b]).items()})
                else:
                    batches=predict(model,d,a,b)
                    arr=[np.concatenate([x[0][j] for x in batches],0) for j in range(5)]
                    stats={k:np.concatenate([x[1][k] for x in batches],0) for k in batches[0][1]}
                    z=dict(sample_id=d['ids'][a:b],increments=arr[1],projected_increments=arr[3],limit=arr[4],**{'projection_'+k:v for k,v in stats.items()})
                    for prefix,path in [('original_',arr[0]),('projected_',arr[2])]:z.update({prefix+k:v for k,v in package(path,d['future'][a:b],d['logits'][a:b]).items()})
                    del batches,arr,stats;gc.collect();guard.check()
                np.savez_compressed(dst,**z);del z;gc.collect()
                with np.load(dst,allow_pickle=False) as f:z={k:f[k] for k in f.files}
            ar=saved_audit(z,d,a,b,entry is not None);groups={}
            if entry is None:
                _,q,m=unpack(z,'native_');native[split]=(q,m);groups['native']=grouped(q,m,d['group'][a:b])
            else:
                _,oq,om=unpack(z,'original_');_,pq,pm=unpack(z,'projected_')
                groups['original']=grouped(oq,om,d['group'][a:b],native[split]);groups['projected']=grouped(pq,pm,d['group'][a:b],native[split],(oq,om))
                groups['alpha_summary']={k:np.quantile(z['projection_'+k],[0,.05,.5,.95,1]).tolist() for k in ('alpha','alpha_nominal')}
                groups['max_alpha_reservation_difference']=float(np.max(z['projection_alpha_nominal']-z['projection_alpha']))
            record=dict(status='SUCCESS',source_signature=review['source_signature'],outputs={str(dst):sha(dst)},arithmetic=ar,groups=groups,model=entry,split=split)
            write(marker,record);records.append(record);outputs[str(dst)]=sha(dst);outputs[str(marker)]=sha(marker);done+=1
            del z;gc.collect();guard.check()
        if model:
            exact(model,state);frozen.append(dict(checkpoint=entry['path'],all_weights_bool_scalers_exact=True));del model,state;gc.collect();torch.cuda.empty_cache()
    assert done==14 and len(frozen)==6
    progress('FINAL_OUTPUT_SOURCE_OLD_SHA',completed_jobs=done);verify(outputs);verify(sources);verify(old)
    write(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=done,outputs=outputs,full_saved_quality_geometry_projection_recomputed=True,no_all_models_second_replay=True))
    write(HERE/'FROZEN_REPLAY_AUDIT.json',dict(status='PASS',records=frozen,no_optimizer=True,GT_forward=False))
    write(HERE/'RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_ENDPOINT_ACCEL_PROJECTION',records=records,jobs=done,source_count=len(sources),old_count=len(old),seconds=time.perf_counter()-begin,no_DEV=True,formal_counter_unchanged='2/3'))
    guard.save('SUCCESS');write(HERE/'exit_status.json',dict(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat()))
if __name__=='__main__':prelaunch()
