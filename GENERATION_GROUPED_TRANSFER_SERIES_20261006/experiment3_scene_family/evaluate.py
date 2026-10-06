"""Fixed outer epoch80; shared-inner secondary; originalCAL descriptive only."""
import adapter as h
import numpy as np,torch,gc
from scipy.cluster.hierarchy import linkage,fcluster
from preflight import independent_projection
quality=h.module('saved_quality',h.OLD/'quality_audit.py',['quantities'],dict(linkage=linkage,fcluster=fcluster,guard=h.guard))
KEYS=('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95')
def original_cal_data():
    raw=h.RAW_LOAD(h.ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz')
    enc=h.RAW_LOAD(h.ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz');lab=h.RAW_LOAD(h.ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz')
    assert np.array_equal(raw['sample_id'],enc['sample_id']) and np.array_equal(raw['sample_id'],lab['sample_id'])
    cache={k:v[25912:].copy() for k,v in raw.items()};n=2879;assert len(cache['sample_id'])==n
    obs=np.zeros((n,940),np.float32);obs[:,:16]=enc['past'][25912:].reshape(n,16);obs[:,16:784]=enc['tokens'][25912:].reshape(n,768)
    base=cache['candidate'][...,:24].reshape(n,20,12,2).copy();v=np.diff(cache['motion'][:,:16].reshape(n,8,2)[:,-2:],axis=1)[:,0];speed=np.c_[np.linalg.norm(v,axis=-1),v].astype(np.float32)
    gt=lab['future'][25912:].copy();error=np.linalg.norm(base-gt[:,None],axis=-1)
    assert np.allclose(error.mean(-1),cache['ade'],atol=2e-5,rtol=0) and np.allclose(error[:,:,-1],cache['fde'],atol=2e-5,rtol=0)
    return cache,[obs,base,speed,gt]
def quantities(path,gt,scores):
    q,motion=quality.quantities(path,gt,scores)
    err=np.linalg.norm(path.astype(np.float64)-gt[:,None].astype(np.float64),axis=-1);ade=err.mean(-1);fde=err[:,:,-1]
    q.update(minADE=ade.min(-1),minFDE=fde.min(-1),Q=(ade+.5*fde).min(-1),J=np.maximum(ade/.2,fde/.4).min(-1))
    assert np.array_equal(q['J']<=1,q['coverage'])
    return q,motion
def summary(q,motion,groups,nativeq,nativemotion,arm,seed,policy,split):
    rows=[]
    for name,mask in h.m.masks(groups):
        assert mask.any(),('EMPTY_GROUP',split,name)
        row=dict(arm=arm,seed=seed,policy=policy,split=split,group=name,n=int(mask.sum()),good=int(q['coverage'][mask].sum()),native_good=int(nativeq['coverage'][mask].sum()),rescued=int((q['coverage'][mask]&~nativeq['coverage'][mask]).sum()),lost=int((~q['coverage'][mask]&nativeq['coverage'][mask]).sum()),effective_good_total=int(q['effective_good'][mask].sum()),raw_good_total=int(q['raw_good'][mask].sum()),inherited_selected_good=int(q['selected_good'][mask].sum()))
        for key in ('minADE','minFDE','Q','J','early','middle','late','all_distinct'):
            row[key]=float(q[key][mask].mean());row['native_'+key]=float(nativeq[key][mask].mean())
        for key in KEYS:
            what,stat=key.split('_');array=motion[what][mask].reshape(-1);ref=nativemotion[what][mask].reshape(-1)
            row[key]=float(np.mean(array) if stat=='mean' else np.quantile(array,.95));row['native_'+key]=float(np.mean(ref) if stat=='mean' else np.quantile(ref,.95))
        row['motion_allfive_1pct']=all(row[k]<=row['native_'+k]*1.01 for k in KEYS);rows.append(row)
    return rows
def saved_job(name,model,cache,values,device):
    ids=cache['sample_id'];directory=h.HERE/'results/saved';directory.mkdir(parents=True,exist_ok=True)
    done=directory/(name+'_done.json')
    if done.exists():
        stamp=h.read(done);assert stamp['source_hashes']==h.read(h.HERE/'preflight.json')['source_hashes'];h.verify(stamp['outputs']);return h.RAW_LOAD(directory/(name+'.npz')),stamp
    before=None if model is None else {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    pieces={};physics=[]
    if model is None:path=values[1].astype(np.float64)
    else:
        model.eval()
        with torch.no_grad():
            for start in range(0,len(ids),128):
                h.guard.check();o,b,s=[torch.from_numpy(v[start:start+128]).to(device) for v in values[:3]]
                output,residual=model(o,b,s);raw,parts=model.routed_components(o,b,s)
                if bool(model.native_fallback):raw=torch.zeros_like(raw)
                projected,up,info=h.projection.project(b,raw,parts['limit']);assert torch.equal(output,projected) and torch.equal(residual,up)
                physics.append(independent_projection(b,raw,output,up,parts['limit'],info))
                for key,val in dict(trajectory=output,raw=raw,base=b,up=up,limit=parts['limit'],**info).items():pieces.setdefault(key,[]).append(val.cpu().numpy())
        path=np.concatenate(pieces.pop('trajectory'))
        for key,val in model.state_dict().items():assert torch.equal(val.detach().cpu(),before[key]),key
        h.frozen_audit(model)
    assert path.shape==(len(ids),20,12,2) and np.isfinite(path).all() and len(np.unique(ids))==len(ids)
    q,motion=quantities(path,values[3],cache['logits'])
    filename=directory/(name+'.npz');np.savez_compressed(filename,sample_id=ids,trajectory=path,**q,**{k:np.concatenate(v) for k,v in pieces.items()})
    saved=h.RAW_LOAD(filename);assert np.array_equal(saved['sample_id'],ids)
    # No model forward here: all samepath metrics, complete-link and physics from saved output.
    check,mm=quantities(saved['trajectory'],values[3],cache['logits']);assert check.keys()==q.keys()
    for key,val in q.items():assert np.array_equal(check[key],val),key
    for key,val in motion.items():assert np.array_equal(mm[key],val),key
    stats=h.r.generation_metrics(path,values[1],values[2],values[3]);statfile=directory/(name+'_legacy.npz');np.savez_compressed(statfile,**stats)
    stamp=dict(status='PASS_SAVED_ARITHMETIC_FROZEN',rows=len(ids),samepath_and_complete_link_EXACT=True,physical=physics,source_hashes=h.read(h.HERE/'preflight.json')['source_hashes'],outputs={str(filename):h.sha(filename),str(statfile):h.sha(statfile)})
    h.write(done,stamp);return saved,stamp
def cluster_ci(delta,keys,seed,repetitions=2000):
    # Hierarchical3-model-seed/registered-cluster bootstrap. Exploratory, not3 new datasets.
    assert delta.shape[1]==len(keys);unique,inverse=np.unique(keys,return_inverse=True);rng=np.random.default_rng(seed)
    counts=np.bincount(inverse);sums=np.stack([np.bincount(inverse,weights=v,minlength=len(unique)) for v in delta]);values=[]
    for _ in range(repetitions):
        models=rng.integers(0,len(delta),len(delta));tracks=rng.integers(0,len(unique),len(unique))
        values.append(float(sums[models][:,tracks].sum()/(len(models)*counts[tracks].sum())))
    lo,hi=np.quantile(values,[.025,.975]);return dict(delta=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),clusters=len(unique),models=len(delta),repetitions=repetitions,exploratory=True)
def run(cache,values,splits,device):
    Class=h.initialize_classes();records=[];outputs={};jobs=[];allq={}
    outer=splits['outer_eval_index'];inner=splits['inner_cal_index'];calcache,calvalues=original_cal_data()
    datasets={name:({k:v[idx] for k,v in cache.items()},[v[idx] for v in values[:4]]) for name,idx in [('outer',outer),('inner',inner)]};datasets['original_CAL']=(calcache,calvalues)
    natives={}
    for name,(cc,vv) in datasets.items():
        saved,stamp=saved_job('native_'+name,None,cc,vv,device);natives[name]=quantities(saved['trajectory'],vv[3],cc['logits']);outputs.update(stamp['outputs']);jobs.append(stamp)
    frozen={};modelSHA={}
    for arm in ('random','grouped'):
        for seed in (1,2,3):
            for policy in ('constrained','legacy','final'):
                modelpath=h.HERE/f'results/models/{arm}/seed_{seed}/balanced_protected/{policy}.pth';modelSHA[str(modelpath)]=h.sha(modelpath)
                ck=torch.load(modelpath,map_location='cpu',weights_only=False);model=Class(940,ck['config']).to(device);model.load_state_dict(ck['state_dict']);h.frozen_audit(model)
                assert ck['config']['adaptation_arm']==arm
                for split,(cc,vv) in datasets.items():
                    h.stage('SAVED_EVAL_AUDIT',arm=arm,seed=seed,policy=policy,split=split)
                    saved,stamp=saved_job(f'{arm}_seed{seed}_{policy}_{split}',model,cc,vv,device);q,motion=quantities(saved['trajectory'],vv[3],cc['logits']);native,nmotion=natives[split]
                    records.extend(summary(q,motion,cc['change_group'],native,nmotion,arm,seed,policy,split));outputs.update(stamp['outputs']);jobs.append(stamp)
                    if split=='outer':allq[arm,seed,policy]=q
                frozen[str(modelpath)]=dict(status='PASS_ALLPARAMS_BOOL_SCALERS_EXACT',no_optimizer=True)
                del model,ck,saved,q,motion;gc.collect();torch.cuda.empty_cache()
    boots=[];targetboots=[]
    timekeys=np.asarray([str(seq)+'|block'+str(block) for seq,block in zip(splits['sequence'][outer],splits['time_block'][outer])])
    targetkeys=np.asarray([str(seq)+'|'+str(ped) for seq,ped in zip(splits['sequence'][outer],splits['pedestrian_id'][outer])])
    for unit,keys,destination in (('target_track',targetkeys,boots),):
        for policy in ('final','constrained','legacy'):
            for group,mask in h.m.masks(cache['change_group'][outer]):
                for metric in ('minADE','minFDE','Q','J','coverage'):
                    delta=np.stack([allq['grouped',seed,policy][metric][mask].astype(np.float64)-allq['random',seed,policy][metric][mask].astype(np.float64) for seed in (1,2,3)])
                    destination.append(dict(cluster_unit=unit,policy=policy,group=group,metric=metric,reference='random',**cluster_ci(delta,keys[mask],20261006)))
                    for arm in ('random','grouped'):
                        delta=np.stack([allq[arm,seed,policy][metric][mask].astype(np.float64)-natives['outer'][0][metric][mask].astype(np.float64) for seed in (1,2,3)])
                        destination.append(dict(cluster_unit=unit,arm=arm,policy=policy,group=group,metric=metric,reference='native',**cluster_ci(delta,keys[mask],20261006)))
    # Full saved SHA/output arithmetic audit already executed per job, not new model replay.
    h.verify(outputs);h.verify(modelSHA)
    h.write(h.HERE/'results/ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=len(jobs),outputs=len(outputs),saved_allquality_cluster_motion_exact=True,outputs_SHA=outputs,no_gradient_second_replay=True))
    h.write(h.HERE/'results/FROZEN_AUDIT.json',dict(status='PASS',checkpoints=frozen,models_SHA=modelSHA))
    result=dict(status='PASS_MATCHED_SCENE_FAMILY_TRANSFER',primary='outer_fixed_epoch80_final',original_CAL_selection=False,records=records,bootstrap=boots,target_track_bootstrap=targetboots,jobs=len(jobs),outputs_SHA=outputs,models_SHA=modelSHA,split_audit=h.read(h.HERE/'identity_audit/RESULT.json'),major_verdict='NO_BASELINE_UPGRADE_REQUIRES_ORIGINAL_ALL_GATES_AND_CORRELATION_REVIEW',limitations=['frozenF16alreadyseendata','one_held_zara_family_newhead_only_not_wholemodel_unseen','3scenes','3modelseeds_not3datasets','innerCALselection_secondary','populationandfiniteoptimization_confound','no_unique_sharedhead_causality'])
    h.write(h.HERE/'results/RESULT.json',result);return result
