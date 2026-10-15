"""Saved paths only. No torch, model, optimizer, candidate policy search."""
import sys,os
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import importlib.util,json,time,gc,traceback
import numpy as np
H=Path(__file__).resolve().parent;R=H.parent;E=R/'F16_CANONICAL_EQUAL_K20_REPLACEMENT_DIAGNOSIS_20261007'
spec=importlib.util.spec_from_file_location('saved_equal_k_audit',E/'run.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
q.H=H
write=q.write;read=q.read;sha=q.sha;digest=q.digest;guard=q.guard;stage=q.stage
GROUPS=q.GROUPS;EVENTS=('lost','rescue','retained','neither');FEATURES=('v_max','v_p95','arc','displacement','terminal_speed','signed_turn','abs_turn','turn_valid_count')
def norm(v,ind=False):return np.sqrt(np.sum(v*v,axis=-1)) if ind else np.linalg.norm(v,axis=-1)
def compare(a,b):
    if a is None or b is None:assert a is None and b is None;return 0.
    if isinstance(a,dict):assert set(a)==set(b);return max([compare(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,list) and a and isinstance(a[0],dict):assert len(a)==len(b);return max(compare(x,y) for x,y in zip(a,b))
    return q.compare(a,b)
def features(p,ind=False):
    p=p.astype(np.float64);v=np.diff(p,axis=-2,prepend=np.zeros_like(p[...,:1,:]))/.4;s=norm(v,ind)
    cross=v[..., :-1,0]*v[...,1:,1]-v[...,:-1,1]*v[...,1:,0];dot=(v[...,:-1,:]*v[...,1:,:]).sum(-1)
    valid=(s[...,:-1]>.03)&(s[...,1:]>.03);turn=np.where(valid,np.arctan2(cross,dot),0.)
    return np.stack((s.max(-1),np.quantile(s,.95,axis=-1),s.sum(-1)*.4,norm(p[...,-1,:],ind),s[...,-3:].mean(-1),turn.sum(-1),np.abs(turn).sum(-1),valid.sum(-1)),axis=-1),s
def utility(a,f,j,g,ind=False):
    n,K=a.shape;out=np.zeros((n,K,4),np.float64);base=g.any(1)
    if ind:
        for k in range(K):
            keep=[l for l in range(K) if l!=k]
            out[:,k,0]=base-g[:,keep].any(1).astype(np.int8)
            for c,x in enumerate((a,f,j),1):out[:,k,c]=x[:,keep].min(1)-x.min(1)
    else:
        out[:,:,0]=g&(g.sum(1,keepdims=True)==1)
        for c,x in enumerate((a,f,j),1):
            two=np.partition(x,1,axis=1)[:,:2];one=x.argmin(1)
            out[np.arange(n),one,c]=two[:,1]-two[:,0]
    assert np.isfinite(out).all() and out.min()>=0
    return out
def calc(z,logits,ind=False):
    ports,removed,keep=q.select(z['paths'],logits,ind);na,nf,nj,ng,_=q.errors(ports[0],z['gt'],ind);ra,rf,rj,rg,_=q.errors(ports[1],z['gt'],ind);la,lf,lj,lg,_=q.errors(ports[4],z['gt'],ind)
    n=len(na);rows=np.arange(n);ns=nj.argmin(1);ls=lj.argmin(1);qs=(na+.5*nf).argmin(1)
    good=ng.any(1);new=rg.any(1);ev=np.stack((good&~new,~good&new,good&new,~good&~new),1)
    removed_mask=(np.arange(20)[None,None,:]==removed[:,:,None]).any(1)
    assert np.all((~ng|removed_mask)[ev[:,0]])
    assert np.all(np.isin(ns[ev[:,0]],np.arange(20)))
    inremoved=(ns[:,None]==removed).any(1);qinremoved=(qs[:,None]==removed).any(1)
    assert inremoved[ev[:,0]].all() and (nj.min(1)[ev[:,0]]<=1).all()
    # One normalized pair-nearest common slot, not separate mean/end minima.
    d=norm(ports[0][rows,ns][:,None]-ports[2],ind);D=np.maximum(d.mean(-1)/.1,d[:,:,-1]/.2);nearest=D.argmin(1)
    redundancy=np.stack((D.min(1),d.mean(-1)[rows,nearest],d[:,:,-1][rows,nearest],keep[rows,nearest]),1)
    rank=np.argsort(-logits,axis=1,kind='stable');slotrank=np.empty_like(rank);slotrank[rows[:,None],rank]=np.arange(1,21)
    un=utility(na,nf,nj,ng,ind);ur=utility(ra,rf,rj,rg,ind)
    fp,speed=features(z['paths'],ind);fg,gt_speed=features(z['gt'],ind)
    single=fp[rows,20+ls];origin=z['paths'][rows,20+ls,-1].astype(float);gte=z['gt'][:,-1].astype(float);gtn=norm(gte,ind);pn=norm(origin,ind)
    valid=(gtn>.03)&(pn>.03);axis=gte/np.maximum(gtn[:,None],.03);along=(origin*axis).sum(-1)-gtn;cross=origin[:,0]*axis[:,1]-origin[:,1]*axis[:,0]
    cos=(origin*gte).sum(-1)/np.maximum(pn*gtn,.03**2);angle=np.where(valid,np.arccos(np.clip(cos,-1,1)),0)
    turnvalid=(single[:,7]>0)&(fg[:,7]>0);tdiff=np.abs(single[:,6]-fg[:,6])
    geometry=np.stack((along,cross,angle,valid,single[:,2]-fg[:,2],tdiff,turnvalid),1)
    flags=np.stack((fg[:,2]/4.8<=.1,fg[:,4]<=.1,(angle>np.pi/6)&valid,(along>.2)&(gtn>.03),single[:,2]-fg[:,2]>.2,(fg[:,4]<=.1)&(single[:,4]>.1),(tdiff>np.pi/6)&turnvalid),1)
    # Independent native individual utility can be zero when removed group has >=2 good paths.
    common_good=ng[rows[:,None],keep].any(1);removed_joint=good&~common_good
    return dict(events=ev,native_errors=np.stack((na,nf,nj),-1),learned_errors=np.stack((la,lf,lj),-1),native_good=ng,replacement_good=rg,native_Jslot=ns,native_Qslot=qs,learned_Jslot=ls,native_minADEslot=na.argmin(1),native_minFDEslot=nf.argmin(1),removed_slots=removed,kept_slots=keep,old_rank_by_slot=slotrank,native_utility=un,replacement_utility=ur,redundancy=redundancy,native_J_in_removed=inremoved,native_Q_in_removed=qinremoved,removed_joint_coverage_utility=removed_joint,path_features=fp,GT_features=fg,learned_J_geometry=geometry,behavior_flags=flags,GT_speed=gt_speed,path_speed=speed)
def batchcalc(z,scores,ind=False):
    allparts={}
    for start in range(0,len(scores),128):
        guard();part=calc({k:v[start:start+128] for k,v in z.items()},scores[start:start+128],ind)
        for k,v in part.items():allparts.setdefault(k,[]).append(v)
    return {k:np.concatenate(v,axis=0) for k,v in allparts.items()}
def descript(x):
    if not len(x):return dict(n=0,mean=None,p50=None,p95=None)
    return dict(n=len(x),mean=np.asarray(x).mean(0),p50=np.quantile(x,.5,axis=0),p95=np.quantile(x,.95,axis=0))
def summarize(z,v):
    out={};n=len(z['sample_id']);rr=np.arange(n)
    for group,mask in q.masks(z['groups']):
        if not mask.any():continue
        res=dict(n=int(mask.sum()),unique_ID=int(np.unique(z['sample_id'][mask]).size),tracks=int(np.unique(z['track'][mask]).size),events={},rank_utility={})
        for e,name in enumerate(EVENTS):
            m=mask&v['events'][:,e];ix=np.flatnonzero(m);ns=v['native_Jslot'][ix];ls=v['learned_Jslot'][ix]
            res['events'][name]=dict(n=len(ix),unique_ID=int(np.unique(z['sample_id'][m]).size),tracks=int(np.unique(z['track'][m]).size),native_Jwinner_removed_count=int(v['native_J_in_removed'][m].sum()),native_Qwinner_removed_count=int(v['native_Q_in_removed'][m].sum()),native_Qwinner_good_count=int(v['native_good'][ix,v['native_Qslot'][ix]].sum()),native_Jslot_hist=np.bincount(ns,minlength=20),native_Jrank_hist=np.bincount(v['old_rank_by_slot'][ix,ns],minlength=21)[1:],native_good_count_hist=np.bincount(v['native_good'][m].sum(1),minlength=21),removed_joint_needed_count=int(v['removed_joint_coverage_utility'][m].sum()),any_native_individual_coverage_utility=int((v['native_utility'][m,:,0]>0).any(1).sum()),redundancy=descript(v['redundancy'][m,:3]),pair_near_D_le1_count=int((v['redundancy'][m,0]<=1).sum()),GT_features=descript(v['GT_features'][m]),native_J_features=descript(v['path_features'][ix,ns]),learned_J_features=descript(v['path_features'][ix,20+ls]),all_learned_features=descript(v['path_features'][m,20:].reshape(-1,8)),native_J_errors=descript(v['native_errors'][ix,ns]),learned_J_errors=descript(v['learned_errors'][ix,ls]),learned_independent_minADE=float(v['learned_errors'][m,:,0].min(1).mean()) if len(ix) else None,learned_independent_minFDE=float(v['learned_errors'][m,:,1].min(1).mean()) if len(ix) else None,learned_J_geometry=descript(v['learned_J_geometry'][m]),flags_count=v['behavior_flags'][m].sum(0))
        for rank in range(1,21):
            slots=(v['old_rank_by_slot']==rank);u=v['native_utility'][mask][slots[mask]]
            res['rank_utility'][str(rank)]=dict(n=len(u),mean=u.mean(0),sum=u.sum(0),positive_count=(u>0).sum(0))
        speed=v['path_speed'][mask];rm=v['removed_slots'][mask];keep=v['kept_slots'][mask];ar=np.arange(len(speed))[:,None]
        parts={'native20':speed[:,:20],'replacement20':np.concatenate((speed[ar,keep],speed[:,20:]),axis=1),'common16':speed[ar,keep],'removed4':speed[ar,rm],'learned4':speed[:,20:],'native24':speed}
        cutoff=float(np.quantile(parts['native20'],.95));ex={k:int((x>cutoff).sum()) for k,x in parts.items()}
        assert ex['replacement20']-ex['native20']==ex['learned4']-ex['removed4']
        res['speed_tail']=dict(native_p95_cutoff=cutoff,pooled_p95={k:float(np.quantile(x,.95)) for k,x in parts.items()},exceedance_counts=ex,exceedance_delta=ex['replacement20']-ex['native20'],attribution_identity_PASS=True)
        res['GT_behavior_prevalence']=dict(low_average_speed=int(v['behavior_flags'][mask,0].sum()),terminal_stop=int(v['behavior_flags'][mask,1].sum()),allwindows=n)
        out[group]=res
    return out
def cases(z,v):
    out=[]
    for i in np.flatnonzero(v['events'][:,:2].any(1)):
        out.append(dict(sample_id=int(z['sample_id'][i]),track=str(z['track'][i]),scene=str(z['scene'][i]),group=int(z['groups'][i]),event=EVENTS[int(v['events'][i].argmax())],native_Jslot=int(v['native_Jslot'][i]),native_Qslot=int(v['native_Qslot'][i]),native_minADEslot=int(v['native_minADEslot'][i]),native_minFDEslot=int(v['native_minFDEslot'][i]),learned_Jslot=int(v['learned_Jslot'][i]),native_good_slots=np.flatnonzero(v['native_good'][i]),removed_slots=v['removed_slots'][i],old_rank_by_slot=v['old_rank_by_slot'][i],native_error=v['native_errors'][i,v['native_Jslot'][i]],learned_errors_all4=v['learned_errors'][i],redundancy=v['redundancy'][i],GT_features=v['GT_features'][i],native_J_features=v['path_features'][i,v['native_Jslot'][i]],learned_features_all4=v['path_features'][i,20:],learned_J_geometry=v['learned_J_geometry'][i],flags=v['behavior_flags'][i],native_utility=v['native_utility'][i]))
    return out
def manifests():
    inherited=read(E/'source_manifest.json')['files'];q.verify(inherited);files=dict(inherited)
    paths=[*H.glob('*.py'),*H.glob('*.ps1'),*H.glob('*.md'),H/'protocol.json',E/'RESULT.json',E/'ARITHMETIC_AUDIT.json',E/'PRESERVATION.json',E/'source_manifest.json']
    paths.extend(p for p in E.rglob('*') if p.is_file() and ('jobs' in p.parts or 'completion_review' in p.parts or p.name in ('RESULT_ANALYSIS.md','RESULT_ANALYSIS.json','CONTROL_REPORT_PATCH_PROVENANCE.md')) and p.suffix in ('.json','.npz','.py','.md') and '__pycache__' not in p.parts)
    for p in sorted(set(paths)):
        key=str(p);s=sha(p)
        if key in files:assert files[key]==s
        else:files[key]=s
    return files,len(inherited)
def fixture():
    a=np.array([[.1,.1,.3],[.1,.2,.3]]);f=np.array([[.1,.1,.3],[.5,.1,.3]]);j=np.maximum(a/.2,f/.4);g=(a<=.2)&(f<=.4)
    assert np.array_equal(utility(a,f,j,g),utility(a,f,j,g,True)) and utility(a,f,j,g)[0,:,0].sum()==0
    p=np.zeros((3,24,12,2));p[:,0:20,:,0]=1.;p[0,16,:,0]=0.;p[1,20,:,0]=0.;p[2,16:18,:,0]=0.
    z=dict(paths=p,gt=np.zeros((3,12,2)),groups=np.full(3,2));s=np.zeros((3,20));x=calc(z,s);y=calc(z,s,True)
    assert q.compare(x,y)<1e-12 and x['events'][:,0].tolist()==[True,False,True] and x['events'][1,1]
    assert x['native_utility'][2,:,0].sum()==0 and x['removed_joint_coverage_utility'][2]
    # Close geometry can straddle the original quality threshold.
    p=np.full((1,24,12,2),0.);p[:,:,:,0]=.201;p[:,16,:,0]=.199;p[:,20:,:,0]=.202
    x=calc(dict(paths=p,gt=np.zeros((1,12,2)),groups=np.full(1,2)),np.zeros((1,20)))
    assert x['events'][0,0] and x['redundancy'][0,0]<1
    t=np.arange(1,13)*.4;p=np.stack((t,np.zeros_like(t)),1)[None];f,s=features(p);assert np.allclose(s,1) and abs(f[0,2]-4.8)<1e-12 and abs(f[0,4]-1)<1e-12
    stopped=p.copy();stopped[:,9:]=stopped[:,8:9];ff,_=features(stopped);assert ff[0,4]==0 and ff[0,2]<f[0,2]
    pp=np.repeat(p[:,None],24,axis=1);pp[:,20:]=-p[:,None]
    xx=calc(dict(paths=pp,gt=p,groups=np.full(1,2)),np.zeros((1,20)))
    assert xx['learned_J_geometry'][0,3] and abs(xx['learned_J_geometry'][0,2]-np.pi)<1e-12 and xx['behavior_flags'][0,2]
    pp[:,20:]=2*p[:,None];xx=calc(dict(paths=pp,gt=p,groups=np.full(1,2)),np.zeros((1,20)));assert xx['behavior_flags'][0,3] and xx['behavior_flags'][0,4]
    fp=H/'fixtures/artifact.json';write(fp,{'test':True});done=dict(status='PASS',source_sha='fixture',files={str(fp):sha(fp)});q.valid_done(done,'fixture');reject=0
    for bad in (dict(done,source_sha='bad'),dict(done,files={str(fp):'0'*64})):
        try:q.valid_done(bad,'fixture')
        except AssertionError:reject+=1
    assert reject==2
    return dict(tied_LOO_exact=True,multi_good_individual_zero_group_needed=True,lost_rescue_identity=True,geometry_close_quality_straddle=True,constant_speed_stop_direction_overshoot_exact=True,analysis_SHA_skip_badsource_output_reject=True,no_training_resume=True)
def aligned(z,data,d,name,tag,seed):
    logits=q.identity(z,data,d,name);prev=read(E/'jobs'/f'{tag}_{seed}_{name}.json');pz=q.npz(E/'jobs'/f'{tag}_{seed}_{name}.npz')
    assert np.array_equal(pz['sample_id'],z['sample_id']);ports,rm,keep=q.select(z['paths'],logits);assert np.array_equal(ports[1],pz['replacement20']) and np.array_equal(rm,pz['removed_slots'])
    return logits,prev
def preflight():
    start=time.perf_counter();guard();fx=fixture();files,inherit=manifests();q.verify(files);data=q.cache();info=[]
    for tag,seed,name,path,d,man,done in q.jobs():
        q.verify(done['files']);z=q.npz(path);scores,_=aligned(z,data,d,name,tag,seed);info.append(dict(grouping=tag,seed=seed,split=name,n=len(scores),input_sha=sha(path)));del z
    z=q.npz(q.P/'runs/scene_family/seed_42/outer.npz');z={k:v[:128] for k,v in z.items()};scores=data['logits'][[data['lookup'][int(x)] for x in z['sample_id']]]
    a=calc(z,scores);b=calc(z,scores,True);error=compare(a,b);assert error<=1e-12;assert compare(summarize(z,a),summarize(z,b))<=1e-12
    seconds=[]
    for _ in range(8):
        t=time.perf_counter();calc(z,scores);seconds.append(time.perf_counter()-t)
    import matplotlib
    assert matplotlib.__version__
    rows=sum(x['n'] for x in info);q.verify(files);guard();free=q.psutil.disk_usage(str(H)).free/2**30;assert free>1
    elapsed=time.perf_counter()-start;compute=2*rows/128*np.mean(seconds);planning=max(3,int(np.ceil((compute+2*elapsed+45)/60)))
    write('source_manifest.json',dict(files=files,inherited_relevant_files=inherit,not_all_historical_files=True));write('PREFLIGHT.json',dict(status='PASS_SAVED_ONLY_TAIL_UTILITY',fixtures=fx,actual128_independent_error=error,source_sha=digest(files),locked_files=len(files),inherited_relevant_files=inherit,jobs=27,rows=rows,metadata=info,benchmark_saved128_seconds=seconds,compute_seconds=compute,elapsed_seconds=elapsed,planning_minutes=planning,ETA_minutes=[max(2,int(planning*.7)),int(np.ceil(planning*1.8))],estimated_vectors_GiB=rows*20*4*8*2/2**30,disk_free_GiB=free,resource=q.RESOURCE,no_model_forward_backward_optimizer=True))
    write('prelaunch_review/REVIEW.json',dict(status='PASS',all27_ID_prior_replacement_native_GT_group_alignment=True,fixture_and_actual128_independent_utility_kinematic_error=error,all_source_initial_final_SHA_PASS=True,DA1='PASS_WITH_ALGEBRAIC_WINNER_AND_GROUP_UTILITY_LIMITS',not_external_review=True,no_model_forward=True));print('PREFLIGHT_PASS',planning,q.RESOURCE,flush=True)
def main():
    start=time.perf_counter();guard();pre=read(H/'PREFLIGHT.json');assert pre['status']=='PASS_SAVED_ONLY_TAIL_UTILITY';files=read(H/'source_manifest.json')['files'];assert digest(files)==pre['source_sha'];stage('INITIAL_SOURCE_COMPLETED_SHA');q.verify(files);data=q.cache();records=[];outsha={};native_by_split={}
    for tag,seed,name,path,d,man,done in q.jobs():
        stage('SAVED_TAIL_UTILITY',grouping=tag,seed=seed,split=name,done=len(records),total=27);guard();z=q.npz(path);scores,prev=aligned(z,data,d,name,tag,seed);dest=H/'jobs'/f'{tag}_{seed}_{name}.json';vf=dest.with_suffix('.npz');cf=dest.with_suffix('.cases.json');df=dest.with_suffix('.done.json')
        if df.exists():q.valid_done(read(df),pre['source_sha']);rec=read(dest);saved=q.npz(vf)
        else:
            v=batchcalc(z,scores);iv=batchcalc(z,scores,True);err=compare(v,iv);assert err<=1e-12;summary=summarize(z,v);isum=summarize(z,iv);assert compare(summary,isum)<=1e-12
            for group,mask in q.masks(z['groups']):
                assert summary[group]['events']['lost']['n']==prev['metrics']['replacement20'][group]['lost'];assert summary[group]['events']['rescue']['n']==prev['metrics']['replacement20'][group]['rescued']
            saved=dict(sample_id=z['sample_id'],groups=z['groups'],track=z['track'],scene=z['scene'],**v);vf.parent.mkdir(parents=True,exist_ok=True);assert not vf.exists();np.savez_compressed(vf,**saved);reload=q.npz(vf);assert all(np.array_equal(v,reload[k]) for k,v in saved.items());write(cf,dict(grouping=tag,seed=seed,split=name,cases=cases(z,v),units=dict(speed='m/s',distance='m',angle='radian')));rec=dict(grouping=tag,seed=seed,split=name,source_sha=pre['source_sha'],input_sha=sha(path),summary=summary,independent_max_error=err);write(dest,rec);write(df,dict(status='PASS',source_sha=pre['source_sha'],files={str(p):sha(p) for p in (dest,vf,cf,path)}));del v,iv,reload
        key=(tag,name)
        sig=digest({k:saved[k] for k in ('sample_id','native_errors','native_good','native_utility','old_rank_by_slot')})
        if key in native_by_split:assert native_by_split[key]==sig,'Native repeated seed must agree'
        else:native_by_split[key]=sig
        records.append(rec)
        for p in (dest,vf,cf,df):outsha[str(p)]=sha(p)
        del saved,z;gc.collect()
    stage('FINAL_SOURCE_OUTPUT_SHA',done=27,total=27);q.verify(files);q.verify(outsha);guard();write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=27,all_saved_LOO_20_deletions_independently_enumerated=True,all_lost_rescue_geometry_kinematics_and_rank_tail_reconstructed=True,independent_max_error=max(x['independent_max_error'] for x in records),all_prior_lost_rescue_exact=True,native_repeated_seed_utility_exact=True,no_model_or_gradient_replay=True));write('PRESERVATION.json',dict(status='PASS',locked_files=len(files),inherited_relevant_files=pre['inherited_relevant_files'],original_digests_unchanged=True,not_all_historical_files=True));write('RESOURCE_USAGE.json',dict(status='PASS',**q.RESOURCE));write('RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_REMOVED_TAIL_UTILITY',records=records,jobs=27,output_sha=outsha,new_training_runs=0,new_formal_count=0,resource=q.RESOURCE,source_sha=pre['source_sha'],elapsed_seconds=time.perf_counter()-start));stage('SUCCESS',done=27,total=27)
if __name__=='__main__':
    try:preflight() if '--preflight' in sys.argv else main()
    except Exception:
        write('failures/failure_'+str(time.time_ns())+'.json',dict(traceback=traceback.format_exc(),resource=q.RESOURCE,no_training=True));raise
