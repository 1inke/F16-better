import sys
sys.dont_write_bytecode=True
import stage1 as s
import time, gc, traceback, json
import numpy as np
from pathlib import Path

def fixtures():
    obs=np.zeros((4,9,8,2),np.float32);obs[:,0,:,0]=np.arange(8)[None]*.1;gt=s.cv(obs)
    assert np.max(abs(s.cv(obs,3)-gt))<1e-6
    aa,ff=s.errors(gt,gt);assert aa.max()==0 and ff.max()==0
    cs=s.candidate_summary(np.repeat(gt[:,None],20,1),gt,gt,.1);assert np.all(cs['coverage']==1) and np.all(cs['nonCV_hit']==0)
    altered=gt.copy();altered[:,:,0]+=.5;assert (s.errors(altered,gt)[0]>.49).all()
    rm=np.array([[[1.,0.]],[[2.,0.]],[[3.,0.]]]);rg=rm.copy();null=s.permute(rm,rg,np.zeros(3));assert null is not None and np.max(null)<=1.;assert s.bootstrap_ratio(np.zeros(3),np.ones(3),np.array(['a','b','c']))==[1.,1.,1.]
    assert s.permute(np.zeros_like(rm),np.zeros_like(rm),np.zeros(3)) is None
    # Joint coverage never splices ADE-best with FDE-best.
    p=np.zeros((1,2,12,2));g=np.zeros((1,12,2));p[0,0,-1,0]=.5;p[0,1,:,0]=.3
    a,f=s.errors(p,g);assert (a.min(1)<=.2).all() and (f.min(1)<=.4).all() and not ((a<=.2)&(f<=.4)).any()
    c1,c2,c3=s.cues(obs,np.ones(4,dtype=int));assert not c3.any()
    return dict(status='PASS',CV_and_samepath_and_zero_denominator_and_ratio_bootstrap=True)

def main():
    t=time.perf_counter();s.guard();initial={str(p):s.sha(p) for p in s.inputs()};s.verify(initial);fix=fixtures();calibration={}
    # No model or test-output evaluation before all train-only thresholds frozen.
    for fold in s.FOLDS:
        s.stage('TRAIN_ONLY_CALIBRATION',fold=fold);s.calibrate(fold)
        for p in (s.H/'calibration').glob(fold+'*'):calibration[str(p)]=s.sha(p)
    identities=[]
    for fold in s.FOLDS:
        data=s.load_raw(fold,'test');md=s.metadata(fold,data);s.save(f'preflight/{fold}_identity.npz',sample_id=data['ids'],**{k:v for k,v in md.items() if isinstance(v,np.ndarray)});identities.append(dict(fold=fold,n=len(data['ids']),tracks=len(np.unique(md['track'])),ambiguous_time_n=md['ambiguous_time_n']));del data,md;gc.collect()
    times=[];frozen=[];backendfiles=set();cache=s.npz(s.R/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz');ids=cache['sample_id'][:128];data=s.load_raw('eth','train',ids)
    for fold in s.FOLDS:
        # One actual GPU batch for every locked seed/checkpoint.
        batch=data if fold=='eth' else s.load_raw(fold,'train',s.npz(s.R/f'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/{fold}_train.npz')['sample_id'][:128])
        for seed in s.SEEDS:
            s.stage('PREFLIGHT_FROZEN_GPU',fold=fold,seed=seed);m,tr,cfg=s.build(fold,seed);old=s.snapshot(m);p,l=s.forward(m,tr,batch['obs'],batch['valid']);pp,ll=s.forward(m,tr,batch['obs'],batch['valid']);assert np.array_equal(p,pp) and np.array_equal(l,ll);s.exact(m,old)
            if fold=='eth' and seed==1:
                u=batch['obs'][:,0,-1]-batch['obs'][:,0,-4];norm=np.linalg.norm(u,axis=1);unit=np.tile([1.,0.],(len(u),1));ok=norm>.03;unit[ok]=u[ok]/norm[ok,None];loc=np.stack((p[...,0]*unit[:,None,None,0]+p[...,1]*unit[:,None,None,1],-p[...,0]*unit[:,None,None,1]+p[...,1]*unit[:,None,None,0]),axis=-1);err=float(abs(loc-cache['candidate'][:128,:,:24].reshape(-1,20,12,2)).max());logerr=float(abs(l-cache['logits'][:128]).max());assert max(err,logerr)<2e-5,(err,logerr)
                import torch
                for _ in range(4):s.forward(m,tr,batch['obs'],batch['valid'])
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(8):s.forward(m,tr,batch['obs'],batch['valid'])
                torch.cuda.synchronize();times.append((time.perf_counter()-start)/8)
            frozen.append(dict(fold=fold,seed=seed,parameters=sum(x.numel() for x in m.parameters()),all_weights_bool_buffers_EXACT=True,repeat_forward_bitwise=True))
            for mod in tuple(sys.modules.values()):
                fn=getattr(mod,'__file__',None)
                if fn:
                    path=Path(fn).resolve()
                    if path.suffix=='.py' and (path.is_relative_to(s.R) or path.is_relative_to(s.A)):backendfiles.add(path)
            del m,old,p,l,pp,ll;gc.collect()
    # Same saved-stat code benchmark on training rows, not outcome selection.
    md=np.array(['fixture_track_'+str(i//4) for i in range(len(data['ids']))]);data['track']=md;m,tr,cfg=s.build('eth',1);p,l=s.forward(m,tr,data['obs'],data['valid']);start=time.perf_counter();summary,vectors,nulls=s.summarize(data,s.calibrate('eth'),p,l);statsecs=time.perf_counter()-start;s.save('preflight/saved_math_fixture.npz',**vectors);s.save('preflight/null_fixture.npz',**nulls);s.write('preflight/saved_math_fixture.json',summary)
    # Check roundtrip numbers directly without a new model replay.
    reread=s.npz(s.H/'preflight/saved_math_fixture.npz');assert all(np.array_equal(v,reread[k]) for k,v in vectors.items());del m,p,l,vectors,nulls;gc.collect()
    files=dict(initial)
    for p in sorted(backendfiles):
        actual=s.sha(p)
        if str(p) in files:assert files[str(p)]==actual
        else:files[str(p)]=actual
    s.verify(files);s.verify(calibration);s.write('source_manifest.json',dict(files=files,scope='plan, current code, all15 model pairs, all5 raw archives, actual imported backend sources, labels and train-ID caches; not all historical results'))
    totalrows=sum(x['n'] for x in identities);forwardestimate=totalrows*3/128*times[0];statestimate=15*statsecs;planning=max(600.,forwardestimate+statestimate+180.)
    s.write('PREFLIGHT.json',dict(status='PASS',fixtures=fix,identities=identities,actual15models=frozen,cache_alignment_ETH_seed1=dict(path_error=err,logit_error=logerr),forward_seconds_128=times[0],training128_stat_benchmark_seconds=statsecs,stat_benchmark_not_linear_N_guarantee=True,planning_seconds=planning,ETA_range_minutes=[planning/60,planning/60*2],calibration_sha=calibration,source_files=len(files),elapsed_seconds=time.perf_counter()-t,resource=s.RESOURCE,not_training=True,not_optimizer_rng_resume_test=True))
    s.write('prelaunch_review/REVIEW.json',dict(status='PASS',actual_all15_frozen_finite_repeat=True,all_train_thresholds_before_test_outputs=True,identity_all_test_tracks_verified=True,saved_math_roundtrip_exact=True,all_source_SHA_unchanged=True,scientific_limitations=['Repeated old ETHUCY test benchmark, not fresh confirmatory test','Permutation windows within bins, track bootstrap separate','No new training or understanding assertion','E3-E9 still pending'],inline_DA1='PASS_WITH_LIMITATIONS, same assistant not external reviewer'))
    s.stage('PREFLIGHT_PASS',planning_seconds=planning)

if __name__=='__main__':
    try:main()
    except BaseException:
        s.write(f'failures/preflight_{time.time_ns()}/FAILURE.json',dict(status='FAILURE',traceback=traceback.format_exc(),resource=s.RESOURCE,at=time.time(),source_sha={str(p):s.sha(p) for p in s.H.glob('*.py')}));raise
