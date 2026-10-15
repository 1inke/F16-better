"""One GT-free fixed-K20 policy on saved paths. No torch/models/optimizer."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import importlib.util,json,time,gc,traceback
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;OLD=R/'F16_REMOVED_TAIL_UTILITY_AUDIT_20261007';E=R/'F16_CANONICAL_EQUAL_K20_REPLACEMENT_DIAGNOSIS_20261007'
spec=importlib.util.spec_from_file_location('saved_k20_helper',E/'run.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q);q.H=H
write=q.write;read=q.read;sha=q.sha;digest=q.digest;guard=q.guard;stage=q.stage
PORTS=('native20','protected20','bottom_logit20')
def norm(x,ind=False):return np.sqrt((x*x).sum(-1)) if ind else np.linalg.norm(x,axis=-1)
def distance(a,b,ind=False):
    e=norm(a.astype(float)[:,None]-b.astype(float)[None],ind)
    return np.maximum(e.mean(-1)/.10,e[:,:,-1]/.20)
def physical_arrays(p,ind=False):
    p=p.astype(float)
    if ind:
        v=np.concatenate((p[:,:1],p[:,1:]-p[:,:-1]),1)/.4;a=(v[:,1:]-v[:,:-1])/.4;j=(a[:,1:]-a[:,:-1])/.4
    else:
        v=np.diff(p,axis=1,prepend=np.zeros_like(p[:,:1]))/.4;a=np.diff(v,axis=1)/.4;j=np.diff(a,axis=1)/.4
    return tuple(np.sort(norm(x,ind).reshape(-1)) for x in (v,a,j))
def native_labels(p,ind=False):
    D=distance(p,p,ind);return q.fcluster(q.linkage(D[np.triu_indices(20,1)],method='complete'),1.,criterion='distance').astype(np.int32)
def envelope(p,base,ind=False):return all(np.all(x<=1.01*y) for x,y in zip(physical_arrays(p,ind),base))
def choose_one(paths,logits,ind=False):
    assert paths.shape==(24,12,2) and logits.shape==(20,) and np.isfinite(paths).all() and np.isfinite(logits).all()
    original=paths[:20];labels=native_labels(original,ind);source=np.arange(20,dtype=np.int64);current=original.copy();decisions=np.zeros(4,np.int8);attempts=np.zeros(4,np.int16)
    base=physical_arrays(original,ind);order=sorted(range(20),key=lambda k:(float(logits[k]),k))
    for learned in range(4):
        new=paths[20+learned:21+learned]
        if distance(new,current,ind).min()<=1:decisions[learned]=0;continue
        present=np.flatnonzero(source<20);eligible=[k for k in order if source[k]<20 and np.sum(labels[source[present]]==labels[source[k]])>1]
        if not eligible:decisions[learned]=1;continue
        decisions[learned]=2
        for k in eligible:
            attempts[learned]+=1;proposal=current.copy();proposal[k]=new[0]
            if not envelope(proposal,base,ind):continue
            current=proposal;source[k]=20+learned;decisions[learned]=3;break
    assert envelope(current,base,ind)
    retained=source[source<20];assert set(labels[retained])==set(labels)
    singletons=[i for i in range(20) if np.sum(labels==labels[i])==1];assert all(i in retained for i in singletons)
    assert np.array_equal(current,paths[source]) and len(np.unique(source))==20
    return source,labels,decisions,attempts
def select(paths,logits,ind=False):
    out=[choose_one(p,l,ind) for p,l in zip(paths,logits)];cols=tuple(np.stack([x[j] for x in out]) for j in range(4));src,lab,dec,attempt=cols
    p=paths[np.arange(len(paths))[:,None],src];assert p.shape==(len(paths),20,12,2) and np.isfinite(p).all()
    return p,dict(source_slots=src,native_cluster_labels=lab,decisions=dec,attempted_removals=attempt)
def batchselect(paths,logits,ind=False):
    p=[];v=[]
    for first in range(0,len(paths),128):guard();a,b=select(paths[first:first+128],logits[first:first+128],ind);p.append(a);v.append(b)
    return np.concatenate(p),{k:np.concatenate([x[k] for x in v]) for k in v[0]}
def evaluate(z,logits,ind=False):
    protected,sel=batchselect(z['paths'],logits,ind);bottom=q.select(z['paths'],logits,ind)[0][1];paths=(z['paths'][:,:20],protected,bottom)
    raw=[q.errors(p,z['gt'],ind) for p in paths];nativegood=raw[0][3].any(1);counts=np.stack([q.clusters(p,raw[i][3],ind) for i,p in enumerate(paths)],1)
    vectors=np.stack([np.stack((g.any(1),a.min(1),f.min(1),j.min(1)),1) for a,f,j,g,one in raw],1);metrics={};admission={}
    for group,mask in q.masks(z['groups']):
        n=int(mask.sum());assert n>0;base=q.motion(paths[0][mask],ind);admission[group]=dict(n=n,accepted_total=int((sel['decisions'][mask]==3).sum()),accepted_windows=int((sel['decisions'][mask]==3).any(1).sum()),accepted_per_window_hist=np.bincount((sel['decisions'][mask]==3).sum(1),minlength=5),decision_counts=np.bincount(sel['decisions'][mask].reshape(-1),minlength=4),attempted_removals=int(sel['attempted_removals'][mask].sum()),geometric_clusters_preserved=True,singleton_native_slots_preserved=True)
        for i,name in enumerate(PORTS):
            a,f,j,g,one=raw[i];good=g.any(1);motion=q.motion(paths[i][mask],ind)
            metrics.setdefault(name,{})[group]=dict(n=n,good=int(good[mask].sum()),coverage=float(good[mask].mean()),minADE=float(a[mask].min(1).mean()),minFDE=float(f[mask].min(1).mean()),J=float(j[mask].min(1).mean()),rescued=int((good&~nativegood&mask).sum()),lost=int((~good&nativegood&mask).sum()),retained=int((good&nativegood&mask).sum()),effective_distinct=float(counts[mask,i,0].mean()),effective_good=float(counts[mask,i,1].mean()),sensitivity_03_05_good=int(((a<=.3)&(f<=.5)).any(1)[mask].sum()),single_Jclosest_early_middle_late=one[mask].mean(0),motion=motion,native20_motion=base,motion_all5_1pct=all(motion[k]<=1.01*base[k] for k in motion))
        assert metrics['protected20'][group]['motion_all5_1pct'],'STRICT_POOLED_MOTION_INVARIANT'
    saved=dict(sample_id=z['sample_id'],groups=z['groups'],track=z['track'],scene=z['scene'],protected20=protected,vectors=vectors,cluster_counts=counts,**sel)
    return dict(metrics=metrics,admission=admission),saved
def fixtures():
    t=np.arange(1,13,dtype=float);p=np.zeros((24,12,2));p[:20,:,0]=t;logits=np.zeros(20)
    # A genuinely different stationary path can replace one of20 identical moving paths without increasing any order statistic.
    src,lab,d,a=choose_one(p,logits);assert d.tolist()==[3,0,0,0] and src[0]==20 and len(set(lab))==1
    assert all(np.array_equal(x,y) for x,y in zip((src,lab,d,a),choose_one(p,logits,True)))
    zero=np.zeros((24,12,2));assert (choose_one(zero,logits)[2]==0).all();assert np.array_equal(q.clusters(zero[None,:20],np.ones((1,20),bool)),[[1,1]])
    p[:20,:,0]=(np.arange(20)[:,None]+1)*t;src,lab,d,a=choose_one(p,logits);assert len(set(lab))==20 and (d==1).all() and np.array_equal(src,np.arange(20))
    p[:20,:,0]=t;p[20:,:,0]=10*t;assert (choose_one(p,logits)[2]==2).all()
    # Geometric nearness DOES NOT guarantee matching GT threshold status.
    pp=np.zeros((1,2,12,2));pp[0,0,:,0]=.19;pp[0,1,:,0]=.21;e=q.errors(pp,np.zeros((1,12,2)));assert e[3][0].tolist()==[True,False] and distance(pp[0,:1],pp[0,1:]).item()<1
    fp=H/'fixtures/saved.json';write(fp,dict(test=True));done=dict(status='PASS',source_sha='fixture',files={str(fp):sha(fp)});q.valid_done(done,'fixture');bad=0
    for x in (dict(done,source_sha='bad'),dict(done,files={str(fp):'0'*64})):
        try:q.valid_done(x,'fixture')
        except AssertionError:bad+=1
    assert bad==2
    return dict(singletons_exact_preserved=True,all_original_clusters_retain_native=True,novel_stationary_admission=True,duplicate_rejection=True,motion_violation_rejection=True,stable_ties_originalslot=True,near_paths_quality_straddle=True,selector_signature_paths_logits_only=True,completed_SHA_skip_badsource_badoutput_reject=True,not_training_RNG_resume=True)
def locks():
    inherited=read(OLD/'source_manifest.json')['files'];out=dict(inherited);q.verify(inherited)
    prior=[p for p in OLD.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.tmp','.lock')]
    own=[*H.glob('*.py'),*H.glob('*.ps1'),H/'README.md',H/'DESIGN.md',H/'protocol.json']
    for p in sorted(set(prior+own)):
        k=str(p);d=sha(p)
        if k in out:assert out[k]==d,('INHERITED_DIGEST_CHANGED',k)
        else:out[k]=d
    return out,len(inherited)
def bootstrap(items):
    rng=np.random.default_rng(20261008);out=[]
    for (tag,split),seq in items.items():
        assert [x[0] for x in seq]==[42,43,44];z=seq[0][1]
        for _,x in seq:assert all(np.array_equal(z[k],x[k]) for k in ('sample_id','groups','track'))
        _,iv=np.unique(z['track'],return_inverse=True);T=int(iv.max())+1
        for group,mask in q.masks(z['groups']):
            count=np.bincount(iv[mask],minlength=T);active=np.flatnonzero(count)
            for reference in ('native20','bottom_logit20'):
                diff=np.stack([x[1]['vectors'][:,1]-x[1]['vectors'][:,PORTS.index(reference)] for x in seq]);sums=np.zeros((3,T,4))
                for i in range(3):np.add.at(sums[i],iv[mask],diff[i,mask])
                draws=[]
                for b in range(2000):
                    if b%64==0:guard()
                    ss=rng.integers(0,3,size=3);tt=rng.choice(active,size=len(active),replace=True);draws.append(sums[ss][:,tt].sum((0,1))/(3*count[tt].sum()))
                out.append(dict(grouping=tag,split=split,group=group,port='protected20',reference=reference,n=int(mask.sum()),tracks=len(active),model_seeds=3,metrics=['coverage','ADE','FDE','J'],estimate=diff[:,mask].mean((0,1)),CI95=np.quantile(draws,[.025,.975],axis=0).T,exploratory=True))
    return out
def preflight():
    start=time.perf_counter();guard();fx=fixtures();files,inherited=locks();q.verify(files);data=q.cache();info=[]
    for tag,seed,split,path,d,man,done in q.jobs():
        guard();q.verify(done['files']);z=q.npz(path);q.identity(z,data,d,split);info.append(dict(grouping=tag,seed=seed,split=split,n=len(z['sample_id']),input_sha=sha(path)));del z
    z=q.npz(q.P/'runs/scene_family/seed_42/outer.npz');z={k:v[:128] for k,v in z.items()};logits=data['logits'][[data['lookup'][int(x)] for x in z['sample_id']]]
    met,v=evaluate(z,logits);im,iv=evaluate(z,logits,True);err=q.compare(met,im);assert err<=1e-12 and all(np.array_equal(v[k],iv[k]) for k in v)
    tmp=H/'fixtures/actual128.npz';np.savez_compressed(tmp,**v);reload=q.npz(tmp);assert all(np.array_equal(v[k],reload[k]) for k in v)
    sec=[]
    for b in range(8):t=time.perf_counter();evaluate(z,logits);sec.append(time.perf_counter()-t)
    rows=sum(x['n'] for x in info);q.verify(files);guard();free=q.psutil.disk_usage(str(H)).free/2**30;assert free>2;elapsed=time.perf_counter()-start;compute=2*rows/128*np.mean(sec);planning=max(4,int(np.ceil((compute+2*elapsed+45)/60)))
    write('source_manifest.json',dict(files=files,inherited_relevant_files=inherited,not_all_historical_files=True));write('PREFLIGHT.json',dict(status='PASS_SAVED_ONLY_PROTECTED_INSERTION',fixtures=fx,actual128_independent_max_error=err,actual128_accepted_windows=met['admission']['overall']['accepted_windows'],all27_ID_native_GT_group_scene_track_exact=True,source_sha=digest(files),locked_files=len(files),inherited_relevant_files=inherited,jobs=27,rows=rows,metadata=info,benchmark8_saved128_seconds=sec,compute_seconds=compute,preflight_seconds=elapsed,planning_minutes=planning,ETA_minutes=[max(3,int(planning*.7)),int(np.ceil(planning*1.7))],resource=q.RESOURCE,disk_free_GiB=free,estimated_saved_paths_GiB=rows*20*12*2*z['paths'].dtype.itemsize/2**30,no_forward_backward_optimizer=True))
    write('prelaunch_review/REVIEW.json',dict(status='PASS',actual128_full_quality_clusters_motion_independent_error=err,actual_selector_source_labels_decisions_attempts_paths_exact=True,all27_ID_identity_native_cache_exact=True,initial_final_source_SHA_PASS=True,no_torch_or_model_or_optimizer=True,DA1='PASS_GEOMETRIC_SUPPORT_NOT_GT_COVERAGE_GUARANTEE',not_external_review=True));print('PREFLIGHT_PASS',planning,met['admission']['overall'],q.RESOURCE,flush=True)
def main():
    start=time.perf_counter();guard();pre=read(H/'PREFLIGHT.json');assert pre['status']=='PASS_SAVED_ONLY_PROTECTED_INSERTION';files=read(H/'source_manifest.json')['files'];assert digest(files)==pre['source_sha'];stage('INITIAL_SOURCE_COMPLETED_SHA');q.verify(files);data=q.cache();records=[];outsha={};cirows={}
    for tag,seed,split,path,d,man,done in q.jobs():
        stage('SAVED_GT_FREE_PROTECTED_INSERTION',grouping=tag,seed=seed,split=split,done=len(records),total=27);guard();z=q.npz(path);logits=q.identity(z,data,d,split);dest=H/'jobs'/f'{tag}_{seed}_{split}.json';vf=dest.with_suffix('.npz');df=dest.with_suffix('.done.json')
        if df.exists():q.valid_done(read(df),pre['source_sha']);rec=read(dest);v=q.npz(vf)
        else:
            met,v=evaluate(z,logits);stage('INDEPENDENT_SAVED_ARITHMETIC',grouping=tag,seed=seed,split=split,done=len(records),total=27);im,iv=evaluate(z,logits,True);err=q.compare(met,im);assert err<=1e-12 and all(np.array_equal(v[k],iv[k]) for k in v)
            # Old negative-control metrics MUST reproduce original fixed K20 results, rather than redefining its failures.
            previous=read(E/'jobs'/f'{tag}_{seed}_{split}.json');assert q.compare(met['metrics']['bottom_logit20'],previous['metrics']['replacement20'])<=1e-12
            vf.parent.mkdir(parents=True,exist_ok=True);assert not vf.exists();np.savez_compressed(vf,**v);reload=q.npz(vf);assert all(np.array_equal(v[k],reload[k]) for k in v)
            rec=dict(grouping=tag,seed=seed,split=split,source_sha=pre['source_sha'],input_sha=sha(path),metrics=met['metrics'],admission=met['admission'],independent_max_error=err);write(dest,rec);write(df,dict(status='PASS',source_sha=pre['source_sha'],files={str(p):sha(p) for p in (dest,vf,path)}));del iv,reload
        cirows.setdefault((tag,split),[]).append((seed,{k:v[k] for k in ('sample_id','groups','track','vectors')}));records.append(rec)
        for p in (dest,vf,df):outsha[str(p)]=sha(p)
        del z,v;gc.collect()
    stage('PAIRED_TRACK_SEED_BOOTSTRAP',done=27,total=27);ci=bootstrap(cirows);ci2=bootstrap(cirows);assert q.compare(ci,ci2)==0;write('BOOTSTRAP.json',dict(status='PASS',records=ci,replicates=2000,seed=20261008,deterministic_second_reconstruction_error=0));outsha[str(H/'BOOTSTRAP.json')]=sha(H/'BOOTSTRAP.json')
    stage('FINAL_SOURCE_COMPLETED_SHA',done=27,total=27);q.verify(files);q.verify(outsha);guard();write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=27,all_saved_selector_maps_paths_and_geometry_invariants_exact=True,full_independent_samepath_quality_complete_clusters_all5_motion=True,old_bottom_replacement_metrics_reproduced=True,independent_max_error=max(x['independent_max_error'] for x in records),bootstrap_scalar_CIs=len(ci)*4,not_model_or_gradient_replay=True));write('PRESERVATION.json',dict(status='PASS',locked_files=len(files),inherited_relevant_files=pre['inherited_relevant_files'],original_digests_unchanged=True,not_all_historical_files=True));write('RESOURCE_USAGE.json',dict(status='PASS',**q.RESOURCE));write('RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_NATIVE_SUPPORT_PROTECTED_INSERTION',records=records,jobs=27,output_sha=outsha,source_sha=pre['source_sha'],resource=q.RESOURCE,elapsed_seconds=time.perf_counter()-start,new_training_runs=0,new_formal_count=0));stage('SUCCESS',done=27,total=27)
if __name__=='__main__':
    try:preflight() if '--preflight' in sys.argv else main()
    except Exception:
        write('failures/failure_'+str(time.time_ns())+'.json',dict(traceback=traceback.format_exc(),resource=q.RESOURCE,no_training=True));raise
