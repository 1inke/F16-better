import sys,copy,time,traceback,gc
sys.dont_write_bytecode=True
import run as r
import numpy as np,torch
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def numpy_objective(p,gt):
    e=np.linalg.norm(p.astype(np.float64)-gt[:,None].astype(np.float64),axis=-1);a=e.mean(-1);f=e[:,:,-1];quality=(a+.5*f).min(1).mean();div=0.
    for i in range(4):
        for j in range(i+1,4):div+=(np.maximum(.3-np.linalg.norm(p[:,i].astype(np.float64)-p[:,j].astype(np.float64),axis=-1).mean(-1),0)**2).mean()
    return quality+.05*div/6
def resume_fixture(ns,data,tag,sp,st):
    tr=sp['grouped_train_index'][:513];s=42;directory=r.H/'resume_smoke'/tag;directory.mkdir(parents=True,exist_ok=True)
    man=dict(test='real_two_epoch_small_subset',grouping=tag,source_sha=r.sha(r.H/'run.py'),training_ids=data['ids'][tr].tolist(),normalization_full_training_ids_sha=r.digest(data['ids'][sp['grouped_train_index']].tolist()),protocol=r.read(r.H/'protocol.json'))
    def build():
        r.seed_all(s);m=ns['BestOf4Generator']().to(r.DEVICE);o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);ldr,g=r.loader(data,tr,st,s);return m,o,ldr,g
    m,o,ldr,g=build();h=[r.epoch(ns,m,o,ldr,1),r.epoch(ns,m,o,ldr,2)];r.save_checkpoint(directory/'continuous.pt',m,o,g,h,st,man);a=torch.load(directory/'continuous.pt',map_location='cpu',weights_only=False);del m,o,ldr
    m,o,ldr,g=build();h=[r.epoch(ns,m,o,ldr,1)];r.save_checkpoint(directory/'interrupted.pt',m,o,g,h,st,man);del m,o,ldr
    # New model, optimizer and loader, perturb all RNG, then restore actual saved states.
    m,o,ldr,g=build();r.seed_all(991);torch.rand(17,device=r.DEVICE);np.random.rand(7);h=r.resume(directory/'interrupted.pt',m,o,g,st,man);h.append(r.epoch(ns,m,o,ldr,2));r.save_checkpoint(directory/'resumed.pt',m,o,g,h,st,man);b=torch.load(directory/'resumed.pt',map_location='cpu',weights_only=False)
    assert equal(a,b),'ALL_PARAMS_OPTIMIZER_RNG_HISTORY_STATISTICS_NOT_EXACT'
    before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};small=sp['inner_cal_index'][:12];p1=r.generated(ns,m,data,small,st);p2=r.generated(ns,m,data,small,st);assert np.array_equal(p1,p2) and all(torch.equal(v.cpu(),before[k]) for k,v in m.state_dict().items())
    r.write(directory/'calibration_smoke.json',dict(posthoc_only=True,deterministic_saved_eval_EXACT=True,ids=data['ids'][small],subset_n=12))
    rejected=False
    try:r.resume(directory/'interrupted.pt',m,o,g,st,dict(man,source_sha='bad'))
    except AssertionError:rejected=True
    assert rejected
    r.write(directory/'done.json',dict(status='PASS',run_manifest_sha=r.digest(man),files={str(directory/'resumed.pt'):r.sha(directory/'resumed.pt')}));r.completed_valid(directory,man)
    original=r.read(directory/'done.json');r.write(directory/'done.json',dict(original,run_manifest_sha='bad'));rejected=False
    try:r.completed_valid(directory,man)
    except AssertionError:rejected=True
    assert rejected;r.write(directory/'done.json',original)
    bad=copy.deepcopy(original);bad['files'][str(directory/'resumed.pt')]='0'*64;r.write(directory/'done.json',bad);rejected=False
    try:r.completed_valid(directory,man)
    except AssertionError:rejected=True
    assert rejected;r.write(directory/'done.json',original)
    del m,o,ldr,a,b;gc.collect();torch.cuda.empty_cache();return dict(grouping=tag,train_subset=513,inner_posthoc_subset=12,epochs=2,all_params_optimizer_allRNG_statistics_history_EXACT=True,bad_source_bad_output_rejected=True,completed_SHA_skip_PASS=True)
def main():
    start=time.perf_counter();r.guard();assert torch.cuda.is_available();ns=r.setup();r.stage('PREFLIGHT_INITIAL_SOURCE_OLD_SHA');locks=r.manifest();r.write('source_manifest.json',dict(files=locks,scope='inherited previous relevant replay SHA extended, not every historical file'));data=r.load_data(ns);checks=[];initial={}
    for tag,split,_ in r.PLAN:
        r.stage('PREFLIGHT_GPU',grouping=tag);sp=r.validate_split(data,split);st=r.stats(data,sp['grouped_train_index']);assert np.isfinite(st[0]).all() and np.isfinite(st[1]).all() and (st[1]>0).all();r.seed_all(42);m=ns['BestOf4Generator']().to(r.DEVICE);initial[tag]=r.tensor_hash(m.state_dict());o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);tr=sp['grouped_train_index'];x=torch.from_numpy(((data['p2'][tr[:256]]-st[0])/st[1]).astype(np.float32)).to(r.DEVICE);p=torch.from_numpy(data['past'][tr[:256]]).to(r.DEVICE);gt=torch.from_numpy(data['gt'][tr[:256]]).to(r.DEVICE)
        native_before=data['native'][tr[:256]].copy();out=r.trajectory(ns,m,x,p);loss=r.objective(out,gt)[0];reference=ns['reference_loss'](out,gt);assert torch.equal(loss,reference),'Canonical AST scalar mismatch'
        grads=torch.autograd.grad(loss,list(m.parameters()),retain_graph=True);refgrads=torch.autograd.grad(reference,list(m.parameters()));assert all(torch.equal(a,b) for a,b in zip(grads,refgrads)),'Canonical AST gradient mismatch'
        error=abs(float(loss)-numpy_objective(out.detach().cpu().numpy(),gt.cpu().numpy()));assert error<2e-5,('Numpy objective',error)
        norms={};
        for _ in range(8):
            r.guard();out=r.trajectory(ns,m,x,p);val=r.objective(out,gt)[0];o.zero_grad(set_to_none=True);val.backward()
            assert torch.isfinite(out).all() and torch.isfinite(val);norms={name:float(v.grad.norm()) for name,v in m.named_parameters()};assert all(np.isfinite(n) and n>0 for n in norms.values()),norms;o.step()
        assert np.array_equal(native_before,data['native'][tr[:256]]) and out.shape==(256,4,12,2)
        # Bounds from approved general profile: steps <=1.85*base step speed.
        last=p[:,-1]-p[:,-2];smooth=(p[:,-1]-p[:,-4])/3;base=torch.where((torch.linalg.vector_norm(last,dim=-1)>.03)[:,None],last,smooth);fallback=torch.zeros_like(base);fallback[:,0]=.05;base=torch.where((torch.linalg.vector_norm(base,dim=-1)>.015)[:,None],base,fallback);speed=torch.linalg.vector_norm(base,dim=-1)
        vel=torch.diff(out,dim=2,prepend=torch.zeros_like(out[:,:,:1]));assert (torch.linalg.vector_norm(vel,dim=-1)<=1.85*speed[:,None,None]+2e-5).all()
        checks.append(dict(grouping=tag,train=len(tr),outer=len(sp['outer_eval_index']),inner=len(sp['inner_cal_index']),canonical_AST_loss_and_all_gradients_EXACT=True,numpy_objective_error=error,actual_GPU8_FIT256_all_parameter_grad_positive_finite=True,trainable_parameters=sum(v.numel() for v in m.parameters()),native20_cache_EXACT=True,bounds_PASS=True,normalization_training_only=True));del m,o,out;gc.collect();torch.cuda.empty_cache()
    assert len(set(initial.values()))==1,'Across-group paired seed initialization mismatch'
    # Same-path/disjoint minima, duplicate cluster and rule-switch fixtures.
    zero=np.zeros((1,20,12,2));assert r.clusters(zero,np.ones((1,20),bool)).tolist()==[[1,1]]
    a=np.array([[.05,.3,.8,1.]],float);f=np.array([[.8,.1,1.,2.]],float);assert not ((a<=.2)&(f<=.4)).any() and np.maximum(a/.2,f/.4).min()>1
    sw,wa,wb=r.assignment(np.array([[.01,.19,.8,1.]]),np.array([[.39,.01,1.,2.]]),np.array([2]));assert wa[0]==1 and wb[0]==1
    # Both rules disagree in a known case: Q prefers (.01,.39), J prefers (.15,.25).
    sw,wa,wb=r.assignment(np.array([[.01,.15,.8,1.]]),np.array([[.39,.25,1.,2.]]),np.array([2]));assert wa[0]==0 and wb[0]==1 and sw['overall']['switch_rate']==1
    resumes=[]
    for tag,split,_ in r.PLAN:
        r.stage('REAL_TWO_EPOCH_RESUME',grouping=tag);sp=r.validate_split(data,split);resumes.append(resume_fixture(ns,data,tag,sp,r.stats(data,sp['grouped_train_index'])))
    # Real output fixture: scalar finite difference in float64 output-space, not parameter training.
    p=torch.randn(7,4,12,2,device=r.DEVICE,dtype=torch.float64,requires_grad=True);gt=torch.randn(7,12,2,device=r.DEVICE,dtype=torch.float64);d=torch.randn_like(p);d=d/d.norm();l=r.objective(p,gt)[0];analytic=float((torch.autograd.grad(l,p)[0]*d).sum());fds=[]
    for eps in (1e-4,1e-5):
        fd=float((r.objective(p.detach()+eps*d,gt)[0]-r.objective(p.detach()-eps*d,gt)[0])/(2*eps));assert abs(fd-analytic)<=1e-7+1e-3*abs(analytic);fds.append(dict(eps=eps,derivative=fd,error=abs(fd-analytic)))
    tag,split,_=r.PLAN[0];sp=r.validate_split(data,split);st=r.stats(data,sp['grouped_train_index']);r.seed_all(42);m=ns['BestOf4Generator']().to(r.DEVICE);o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);ix=sp['grouped_train_index'][:256];x=torch.from_numpy(((data['p2'][ix]-st[0])/st[1]).astype(np.float32)).to(r.DEVICE);p=torch.from_numpy(data['past'][ix]).to(r.DEVICE);gt=torch.from_numpy(data['gt'][ix]).to(r.DEVICE);times=[]
    for i in range(36):
        r.guard();torch.cuda.synchronize();t=time.perf_counter();out=r.trajectory(ns,m,x,p);v=r.objective(out,gt)[0];o.zero_grad(set_to_none=True);v.backward();o.step();torch.cuda.synchronize()
        if i>=4:times.append(time.perf_counter()-t)
    del m,o;torch.cuda.empty_cache();compute_seconds=sum(25*((n+255)//256)*3 for n in (16337,18909,16104))*float(np.mean(times));r.stage('OLD_FROZEN_FULL4_SWITCH_DIAGNOSTIC');switch=r.frozen_winner_switch(ns,data)
    r.stage('PREFLIGHT_FINAL_SOURCE_OLD_SHA');r.verify(locks);r.guard();r.write('PREFLIGHT.json',dict(status='PASS',files=locks,source_sha=r.digest(locks),checks=checks,paired_seed_initial_parameters_sha=initial,resume=resumes,finite_difference=fds,benchmark=dict(throwaway_steps=36,warm=4,measured=32,batch256_seconds=float(np.mean(times)),training9runs_compute_minutes=compute_seconds/60,not_production_guarantee=True,no_saved_benchmark_model=True),winner_switch_jobs=27,old_full4_models=9,resource=r.RESOURCE,elapsed_seconds=time.perf_counter()-start,old_F16_decoder_scorer_not_instantiated_or_updated=True,no_oldmodule_top_level_execution=True))
    r.stage('PREFLIGHT_PASS',compute_minutes=compute_seconds/60)
if __name__=='__main__':
    try:main()
    except Exception:
        r.write('PREFLIGHT_FAILURE_'+str(time.time_ns())+'.json',dict(status='STOP',traceback=traceback.format_exc(),resource=r.RESOURCE));raise
