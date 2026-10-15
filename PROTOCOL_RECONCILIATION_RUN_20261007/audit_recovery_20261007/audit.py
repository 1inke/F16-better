"""Saved-only recovery. Never call legacy model/training entry points."""
import sys,os
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import importlib.util,copy,time,gc,traceback
import numpy as np,torch
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent;P=H.parent
spec=importlib.util.spec_from_file_location('canonical_saved_helpers',P/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
r.H=H
ns=dict(np=np,linkage=linkage,fcluster=fcluster)
r.definitions(P/'prelaunch_review.py',['independently_check'],ns)
independent=ns['independently_check']

def locks():
    out=dict(r.read(P/'PREFLIGHT.json')['files'])
    for d in (P/'runs').glob('*/seed_*'):
        done=r.read(d/'done.json');assert done['status']=='PASS'
        for p,h in done['files'].items():
            if p in out:assert out[p]==h
            else:out[p]=h
        out[str(d/'done.json')]=r.sha(d/'done.json')
    for p in [P/'worker_state.json',P/'exit_status.json',P/'worker_stderr.log',P/'RESOURCE_USAGE.json',P/'completion_check.json',P/'STOP_RESOURCE_AND_RECOVERY_PLAN.md',*P.glob('FAILURE_*.json'),*H.glob('*.py'),*H.glob('*.ps1'),*H.glob('*.md'),H/'protocol.json']:
        key=str(p);actual=r.sha(p)
        if key in out:assert out[key]==actual
        else:out[key]=actual
    return out

def cached():
    r.guard()
    with np.load(r.INPUTS[0],allow_pickle=False) as c:
        data=dict(ids=c['sample_id'].copy(),native=c['candidate'][...,:24].reshape(-1,20,12,2).astype(np.float32),groups=c['change_group'].copy())
    with np.load(r.INPUTS[1],allow_pickle=False) as l:
        assert np.array_equal(data['ids'],l['sample_id']);data['gt']=l['future'].astype(np.float32)
    with np.load(r.INPUTS[2],allow_pickle=False) as e:
        assert np.array_equal(data['ids'],e['sample_id']);data['p2']=e['tokens'].mean(1).astype(np.float32)
    md=r.npz(r.INPUTS[3]);lookup={int(x):i for i,x in enumerate(md['sample_id'])};order=np.array([lookup[int(x)] for x in data['ids']])
    data['track']=np.array([str(md['sequence'][i])+'::'+str(md['pedestrian_id'][i]) for i in order]);data['scene']=md['scene_family'][order]
    assert len(data['ids'])==28791 and len(np.unique(data['ids']))==28791
    r.guard();return data

def binding(data,tag,split,seed,source):
    d=P/'runs'/tag/f'seed_{seed}';man=r.read(d/'run_manifest.json');protocol=r.read(P/'protocol.json')
    assert all(man[k]==v for k,v in protocol.items())
    assert man['source_files']==source and man['source_sha']==r.digest(source)
    assert man['seed']==seed and man['grouping']==tag and man['parameter_count']==64032
    assert man['split_sha']==r.sha(split) and man['split_file']==str(split)
    sp=r.validate_split(data,split);sid=r.npz(d/'split_ids.npz');st=r.npz(d/'statistics.npz');mu,sd=r.stats(data,sp['grouped_train_index'])
    for k,ix in [('train_ids',sp['grouped_train_index']),('outer_ids',sp['outer_eval_index']),('inner_ids',sp['inner_cal_index']),('original_CAL_ids',np.arange(25912,28791))]:assert np.array_equal(sid[k],data['ids'][ix])
    assert np.array_equal(st['normalization_ids'],sid['train_ids']) and np.array_equal(st['mean'],mu) and np.array_equal(st['std'],sd)
    assert man['statistics_sha']==r.sha(d/'statistics.npz') and man['split_ids_sha']==r.sha(d/'split_ids.npz')
    assert man['statistics_mean']==mu.tolist() and man['statistics_std']==sd.tolist()
    history=r.read(d/'history.json');assign=r.read(d/'assignment_stats.json');ck=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert ck['run_manifest']==man and ck['run_manifest_sha']==r.digest(man) and ck['epoch']==25 and ck['history']==history and len(history)==25
    assert torch.equal(ck['statistics']['mean'],torch.from_numpy(mu)) and torch.equal(ck['statistics']['std'],torch.from_numpy(sd))
    assert ck['parameter_names']==man['parameter_names'] and sum(v.numel() for v in ck['state_dict'].values())==64032
    assert all(torch.isfinite(v).all() for v in ck['state_dict'].values())
    assert set(ck['rng'])=={'python','numpy','torch','cuda','loader'}
    for group in ck['optimizer']['param_groups']:assert group['lr']==.001 and group['weight_decay']==.0001
    for ep,z in enumerate(history,1):
        assert z['epoch']==ep and z['n']==len(sp['grouped_train_index'])
        cm=np.array(z['confusion']);wc=np.array(z['winner_counts']);assert cm.shape==(4,4) and int(cm.sum())==z['n'] and np.array_equal(cm.sum(1),wc)
        assert z['switch_count']==int(cm.sum()-np.trace(cm)) and abs(z['switch_rate']-z['switch_count']/z['n'])<1e-12
        assert abs(z['total']-z['quality']-.05*z['diversity'])<1e-6
    assert assign['training_preupdate_history']==history and assign['four_generated_slots_only'] and assign['first_index_tie_rule']
    done=r.read(d/'done.json');assert done['run_manifest_sha']==r.digest(man);r.verify(done['files'])
    del ck;gc.collect();return d,sp,man

def assignment_independent(z):
    err=np.sqrt(np.square(z['paths'][:,20:].astype(np.float64)-z['gt'][:,None].astype(np.float64)).sum(-1));a=err.mean(-1);f=err[:,:,-1]
    q=a+.5*f;j=np.maximum(a/.2,f/.4);wa=np.argmin(q,axis=1);wb=np.argmin(j,axis=1);sq=np.sort(q,axis=1);sj=np.sort(j,axis=1);out={}
    for name,mask in [('overall',np.ones(len(a),bool)),('stable',z['groups']==0),('transition',z['groups']==1),('change',z['groups']==2)]:
        n=int(mask.sum())
        if not n:out[name]={'n':0};continue
        cm=np.zeros((4,4),np.int64);np.add.at(cm,(wa[mask],wb[mask]),1)
        out[name]=dict(n=n,switched=int((wa[mask]!=wb[mask]).sum()),switch_rate=float((wa[mask]!=wb[mask]).mean()),scene_counts=np.bincount(wa[mask],minlength=4),track_counts=np.bincount(wb[mask],minlength=4),confusion=cm,scene_exact_ties=int((sq[mask,1]==sq[mask,0]).sum()),track_exact_ties=int((sj[mask,1]==sj[mask,0]).sum()),scene_near_ties=int((sq[mask,1]-sq[mask,0]<=1e-6).sum()),track_near_ties=int((sj[mask,1]-sj[mask,0]<=1e-6).sum()),scene_gap_mean=float((sq[mask,1]-sq[mask,0]).mean()),track_gap_mean=float((sj[mask,1]-sj[mask,0]).mean()))
    return out

def geometry(z):
    out={};vectors={}
    for name,slots in [('native20',slice(0,20)),('full24',slice(0,24)),('learned4',slice(20,24))]:
        p=z['paths'][:,slots];a,f,j,g=r.errors(p,z['gt']);ix=j.argmin(1);e=np.linalg.norm(p.astype(np.float64)-z['gt'][:,None],axis=-1);one=e[np.arange(len(p)),ix];out[name]={}
        vectors[name]=np.stack([g.any(1).astype(float),a.min(1),f.min(1),j.min(1)],1)
        for group,mask in [('overall',np.ones(len(p),bool)),('stable',z['groups']==0),('transition',z['groups']==1),('change',z['groups']==2)]:
            out[name][group]={'n':int(mask.sum()),'single_Jclosest_slot_early_middle_late':one[mask].reshape(-1,3,4).mean((0,2)).tolist()} if mask.any() else {'n':0}
    return out,vectors

def identity(z,data,ix):
    assert np.array_equal(z['sample_id'],data['ids'][ix]) and len(np.unique(z['sample_id']))==len(ix)
    assert z['paths'].shape==(len(ix),24,12,2) and np.isfinite(z['paths']).all()
    for k in ('gt','groups','track','scene'):assert np.array_equal(z[k],data[k][ix]),k
    assert np.array_equal(z['paths'][:,:20],data['native'][ix])

def valid_done(done,source_sha):
    assert done['status']=='PASS' and done['source_sha']==source_sha,'ANALYSIS_SOURCE_REJECT'
    r.verify(done['files']);return True

def fixture():
    f=H/'fixtures'/'output.json';r.write(f,{'finite':True});good=dict(status='PASS',source_sha='fixture',files={str(f):r.sha(f)});assert valid_done(good,'fixture')
    n=0
    for bad,expected in [(dict(good,source_sha='bad'),'fixture'),(dict(good,files={str(f):'0'*64}),'fixture')]:
        try:valid_done(bad,expected)
        except AssertionError:n+=1
    assert n==2;return {'completed_analysis_SHA_skip':True,'bad_source_and_bad_output_rejected':True,'not_training_resume':True}

def bootstrap(jobs):
    rng=np.random.default_rng(20261007);out=[]
    for (tag,split),rows in jobs.items():
        assert len(rows)==3
        z=rows[0][0];tracks=z['track'];_,inverse=np.unique(tracks,return_inverse=True);T=int(inverse.max())+1
        for group,mask in [('overall',np.ones(len(tracks),bool)),('stable',z['groups']==0),('transition',z['groups']==1),('change',z['groups']==2)]:
            if not mask.any():continue
            counts=np.bincount(inverse[mask],minlength=T);active=np.flatnonzero(counts)
            for port in ('full24','learned4'):
                diff=np.stack([v[port]-v['native20'] for _,v in rows]);sums=np.zeros((3,T,4))
                for s in range(3):np.add.at(sums[s],inverse[mask],diff[s,mask])
                samples=[]
                for b in range(2000):
                    if b%64==0:r.guard()
                    seeds=rng.integers(0,3,size=3);tt=rng.choice(active,size=len(active),replace=True);samples.append(sums[seeds][:,tt].sum((0,1))/(3*counts[tt].sum()))
                samples=np.asarray(samples);out.append(dict(grouping=tag,split=split,group=group,port=port,reference='native20',n=int(mask.sum()),tracks=len(active),seeds=3,metrics=['coverage','ADE','FDE','J'],estimate=diff[:,mask].mean((0,1)).tolist(),CI95=np.quantile(samples,[.025,.975],axis=0).T.tolist(),exploratory=True))
    return out

def preflight():
    start=time.perf_counter();r.guard();source=locks();r.verify(source);r.write('source_manifest.json',dict(files=source,scope='273 inherited relevant files plus completed parent artifacts and child sources; not all historical files'))
    data=cached();bindings=[];init={};old=r.read(P/'PREFLIGHT.json')['files']
    for tag,split,_ in r.PLAN:
        for seed in r.SEEDS:
            d,sp,man=binding(data,tag,split,seed,old);init.setdefault(seed,set()).add(man['initial_parameters_sha']);bindings.append({'grouping':tag,'seed':seed,'checkpoint_manifest_stats_history_PASS':True})
    assert all(len(v)==1 for v in init.values())
    d=P/'runs'/'scene_family'/'seed_42';saved=r.npz(d/'outer.npz');z={k:v[:256] for k,v in saved.items()};t=time.perf_counter();met=r.evaluate(z);error=independent(z,met);assert r.digest(assignment_independent(z))==r.digest(met['assignment']);seconds=time.perf_counter()-t
    restore=fixture();r.verify(source);r.guard();rows=sum((len(r.npz(split)['outer_eval_index'])+len(r.npz(split)['inner_cal_index'])+2879)*3 for _,split,_ in r.PLAN)
    nominal=rows/256*seconds;elapsed=time.perf_counter()-start
    r.write('PREFLIGHT.json',dict(status='PASS_SAVED_ONLY',bindings=bindings,source_sha=r.digest(source),locked_files=len(source),inherited_relevant_files=len(old),actual_saved_rows=256,full_original_quality_clusters_motion_independent_max_error=error,assignment_independent=True,analysis_resume=restore,no_forward_backward_optimizer=True,seconds256=seconds,rows27=rows,nominal_arithmetic_seconds=nominal,preflight_seconds_including_SHA_checkpoint_IO=elapsed,resource=r.RESOURCE))
    print('PREFLIGHT_PASS',seconds,rows,nominal,elapsed,flush=True)

def main():
    start=time.perf_counter();r.guard();pre=r.read(H/'PREFLIGHT.json');assert pre['status']=='PASS_SAVED_ONLY';source=r.read(H/'source_manifest.json')['files'];assert r.digest(source)==pre['source_sha'];r.stage('INITIAL_SOURCE_COMPLETED_SHA');r.verify(source);data=cached();old=r.read(P/'PREFLIGHT.json')['files'];records=[];jobs={};bindings=[]
    for tag,split,_ in r.PLAN:
        for seed in r.SEEDS:
            d,sp,man=binding(data,tag,split,seed,old);bindings.append({'grouping':tag,'seed':seed,'checkpoint_manifest_statistics_history_PASS':True})
            metrics=r.read(d/'metrics.json');assign=r.read(d/'assignment_stats.json')
            for name,ix in [('outer',sp['outer_eval_index']),('inner',sp['inner_cal_index']),('original_CAL',np.arange(25912,28791))]:
                r.stage('SAVED_QUALITY_CLUSTER_MOTION_ASSIGNMENT_AUDIT',grouping=tag,seed=seed,split=name,done=len(records),total=27);z=r.npz(d/f'{name}.npz');identity(z,data,ix);dest=H/'jobs'/f'{tag}_{seed}_{name}.json';donefile=dest.with_suffix('.done.json')
                if donefile.exists():valid_done(r.read(donefile),pre['source_sha']);rec=r.read(dest)
                else:
                    met=r.evaluate(z);assert r.digest(met)==r.digest(metrics['splits'][name]);error=independent(z,met);r.guard();asgn=assignment_independent(z);assert r.digest(asgn)==r.digest(met['assignment'])==r.digest(assign['final_posthoc'][name]);geo,_=geometry(z)
                    rec=dict(grouping=tag,seed=seed,split=name,metrics=met,single_slot_temporal=geo,independent_max_error=error,input_sha=r.sha(d/f'{name}.npz'),source_sha=pre['source_sha'],no_model_forward=True);r.write(dest,rec);r.write(donefile,dict(status='PASS',source_sha=pre['source_sha'],files={str(dest):r.sha(dest),str(d/f'{name}.npz'):rec['input_sha']}))
                _,vectors=geometry(z);jobs.setdefault((tag,name),[]).append(({'track':z['track'],'groups':z['groups']},vectors));records.append(rec);del z;gc.collect()
    r.stage('WITHIN_GROUP_TARGET_TRACK_SEED_BOOTSTRAP',done=27);ci=bootstrap(jobs);r.write('BOOTSTRAP.json',dict(status='PASS',records=ci,replicates=2000,seed=20261007,exploratory=True,no_across_grouping_pairing=True));r.stage('FINAL_SOURCE_COMPLETED_SHA');r.verify(source);r.guard()
    outputs={str(p):r.sha(p) for p in (H/'jobs').glob('*.json')};outputs[str(H/'BOOTSTRAP.json')]=r.sha(H/'BOOTSTRAP.json')
    r.write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=27,records=bindings,all27_saved_quality_clusters_motion_assignment_recomputed=True,independent_max_error=max(v['independent_max_error'] for v in records),no_forward_backward_optimizer=True));r.write('PRESERVATION.json',dict(status='PASS',files=len(source),all_inherited_273_preserved=True,all_completed_parent_artifacts_preserved=True,not_full_historical_SHA=True));r.write('RESOURCE_USAGE.json',dict(status='PASS',**r.RESOURCE));r.write('RESULT.json',dict(status='PASS_SAVED_AUDIT_RECOVERY',records=records,runs=9,jobs=27,epoch_rows=225,source_sha=pre['source_sha'],output_sha=outputs,elapsed_seconds=time.perf_counter()-start,no_retraining=True,no_model_forward=True,no_optimizer=True,scientific_major_not_inferred=True,resource=r.RESOURCE));r.stage('SUCCESS',jobs=27)

if __name__=='__main__':
    try:preflight() if '--preflight' in sys.argv else main()
    except Exception:
        r.write('FAILURE_'+str(time.time_ns())+'.json',dict(traceback=traceback.format_exc(),resource=r.RESOURCE,no_training=True));raise
