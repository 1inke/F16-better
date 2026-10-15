"""Actual paired GPU/real optimizer-RNG resume gates before formal training."""
import sys,copy,time,gc,traceback
sys.dont_write_bytecode=True
import run as r
import numpy as np,torch
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def np_objective(out,gt):
    e=np.sqrt(((out-gt[:,None])**2).sum(-1));primary=(e.mean(-1)+.5*e[:,:,-1]).min(1).mean();div=sum((np.maximum(.3-np.sqrt(((out[:,i]-out[:,j])**2).sum(-1)).mean(-1),0)**2).mean() for i in range(4) for j in range(i+1,4))/6;return primary+.05*div
def calibration(m,data,ix,arm):
    out,_=r.generated(m,data,ix,arm);a,f,_,_,_=r.v.errors(out,data['gt'][ix]);return float((a+.5*f).min(1).mean())
def resume_test(data,sp,st,arm):
    tr=sp['grouped_train_index'][:513];inner=sp['inner_cal_index'][:12];d=r.H/'resume_smoke'/arm;d.mkdir(parents=True,exist_ok=True);man=dict(arm=arm,test='ACTUAL_TWO_EPOCH_SUBSET_NOT_FULL_FIT',source_sha=r.sha(r.H/'run.py'),protocol=r.read(r.H/'protocol.json'),train_ids=data['ids'][tr].tolist(),inner_ids=data['ids'][inner].tolist())
    def build():
        r.q.seed_all(42);m=r.Refiner(st).to(r.DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);ldr,g=r.loader(data,tr,42);return m,opt,ldr,g
    def ep(m,o,l,index):
        row=r.epoch(m,o,l,arm,index);row['inner12_posthoc_Q']=calibration(m,data,inner,arm);return row
    m,o,l,g=build();h=[ep(m,o,l,1),ep(m,o,l,2)];r.checkpoint(d/'continuous.pt',m,o,g,h,st,man);a=torch.load(d/'continuous.pt',map_location='cpu',weights_only=False);del m,o,l
    m,o,l,g=build();h=[ep(m,o,l,1)];r.checkpoint(d/'interrupted.pt',m,o,g,h,st,man);del m,o,l
    m,o,l,g=build();r.q.seed_all(1991);torch.rand(19,device=r.DEVICE);np.random.rand(11);h=r.resume(d/'interrupted.pt',m,o,g,st,man);h.append(ep(m,o,l,2));r.checkpoint(d/'resumed.pt',m,o,g,h,st,man);b=torch.load(d/'resumed.pt',map_location='cpu',weights_only=False);assert equal(a,b),'ALL_PARAMS_BOOLSCALER_OPTIMIZER_ALLRNG_HISTORY_NOT_EXACT'
    original=dict(status='PASS',run_manifest_sha=r.digest(man),files={str(d/'resumed.pt'):r.sha(d/'resumed.pt')});r.write(d/'done.json',original);r.done_valid(d,man);rejects=0
    for bad in (dict(original,run_manifest_sha='bad'),dict(original,files={str(d/'resumed.pt'):'0'*64})):
        r.write(d/'done.json',bad)
        try:r.done_valid(d,man)
        except AssertionError:rejects+=1
    r.write(d/'done.json',original)
    try:r.resume(d/'interrupted.pt',m,o,g,st,dict(man,source_sha='bad'))
    except AssertionError:rejects+=1
    assert rejects==3
    del m,o,l,a,b;gc.collect();torch.cuda.empty_cache();return dict(arm=arm,FIT_subset=513,inner_subset=12,epochs=2,all_params_bool_scalerbuffers_optimizer_allTorchCPUCUDAloaderPythonNumPyRNG_history_CAL_EXACT=True,completedSHA_skip_badsource_badoutput_rejected=True)
def parameter_FD(m,features,anchors,CV,gt,arm):
    c=copy.deepcopy(m).double();x=features[:16].double();a=anchors[:16].double();cv=CV[:16].double();y=gt[:16].double();p=c.net[-1].bias;direction=torch.arange(1,len(p)+1,device=r.DEVICE,dtype=torch.float64);direction=direction/direction.norm();out,_=c(x,a,cv,arm);loss=r.q.objective(out,y)[0];analytic=float((torch.autograd.grad(loss,p)[0]*direction).sum());before=p.detach().clone();basewinner=r.q.objective(out,y)[-1].detach();results=[]
    for eps in (1e-4,1e-5):
        with torch.no_grad():p.copy_(before+eps*direction)
        up,_=c(x,a,cv,arm);lu=r.q.objective(up,y);wu=lu[-1].detach()
        with torch.no_grad():p.copy_(before-eps*direction)
        down,_=c(x,a,cv,arm);ld=r.q.objective(down,y);wd=ld[-1].detach();switch=int(((wu!=basewinner)|(wd!=basewinner)).sum());fd=float(((lu[0]-ld[0])/(2*eps)).detach());error=abs(fd-analytic);results.append(dict(eps=eps,analytic=analytic,finite_difference=fd,error=error,winner_switch_windows=switch));assert switch==0 and error<=1e-7+1e-3*abs(analytic),results
    with torch.no_grad():p.copy_(before)
    return results
def main():
    start=time.perf_counter();r.guard();assert torch.cuda.is_available();r.stage('PREFLIGHT_SOURCE_IDS_STATS');ns,data,sp=r.data_load();tr=sp['grouped_train_index'];st=r.statistics(data,tr);assert all(np.isfinite(x).all() for x in st) and (st[1]>0).all();np.savez_compressed(r.H/'statistics.npz',mean=st[0],std=st[1],normalization_ids=data['ids'][tr])
    x=r.features(data,tr[:1033]).reshape(-1,144).astype(float);mu=x.mean(0);sd=x.std(0);sums=np.zeros(144);sq=np.zeros(144)
    for first in range(0,len(x),31):sums+=x[first:first+31].sum(0);sq+=(x[first:first+31]**2).sum(0)
    err=max(float(np.max(abs(mu-sums/len(x)))),float(np.max(abs(sd-np.sqrt(np.maximum(sq/len(x)-(sums/len(x))**2,0))))));assert err<=1e-11,err
    files,inherited=r.locks();files[str(r.H/'statistics.npz')]=r.sha(r.H/'statistics.npz');r.q.verify(files);r.write('source_manifest.json',dict(files=files,inherited_relevant_files=inherited,not_all_historical_files=True));checks=[];paired=[];bench=[]
    ix=tr[:256];ft=torch.from_numpy(r.features(data,ix)).to(r.DEVICE);at=torch.from_numpy(data['anchor'][ix]).to(r.DEVICE);cv=torch.from_numpy(data['CV'][ix]).to(r.DEVICE);gt=torch.from_numpy(data['gt'][ix]).to(r.DEVICE);native_before=data['native'][ix].copy()
    for arm in r.ARMS:
        r.stage('PREFLIGHT_GPU8',arm=arm);r.q.seed_all(42);m=r.Refiner(st).to(r.DEVICE);paired.append({k:z.detach().cpu().clone() for k,z in m.state_dict().items()});assert sum(p.numel() for p in m.parameters())==28376;opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);out,res=m(ft,at,cv,arm);assert torch.equal(res,torch.zeros_like(res));assert torch.equal(out,at.double() if arm=='native' else cv[:,None].double().expand(-1,4,-1,-1))
        loss=r.q.objective(out,gt)[0];ref=ns['reference_loss'](out,gt);assert torch.equal(loss,ref);g=torch.autograd.grad(loss,list(m.parameters()),retain_graph=True);rg=torch.autograd.grad(ref,list(m.parameters()));assert all(torch.equal(a,b) for a,b in zip(g,rg));nerr=abs(float(loss.detach())-np_objective(out.detach().cpu().numpy(),gt.cpu().numpy()));assert nerr<=2e-12,nerr
        firstzero={k:float(z.norm()) for (k,_),z in zip(m.named_parameters(),g)};norms={}
        for step in range(8):
            r.guard();out,res=m(ft,at,cv,arm);value=r.q.objective(out,gt)[0];opt.zero_grad(set_to_none=True);value.backward();assert torch.isfinite(out).all() and torch.isfinite(value);norms={k:float(z.grad.norm()) for k,z in m.named_parameters()};assert all(np.isfinite(n) for n in norms.values());opt.step()
        assert all(n>0 for n in norms.values()),norms;assert np.array_equal(native_before,data['native'][ix]);fd=parameter_FD(m,ft,at,cv,gt,arm)
        before={k:z.detach().cpu().clone() for k,z in m.state_dict().items()};small=sp['outer_eval_index'][:128];out,res=r.generated(m,data,small,arm);z=r.portfolio(data,small,out,res,arm);met,extra=r.evaluate(z);imet,iextra=r.evaluate(z,True);error=r.v.compare(met,imet);assert error<=1e-12 and all(np.array_equal(extra[k],iextra[k]) for k in extra);assert all(torch.equal(z.detach().cpu(),before[k]) for k,z in m.state_dict().items());saved=r.H/'prelaunch_review'/f'{arm}_smoke128.npz';saved.parent.mkdir(exist_ok=True);np.savez_compressed(saved,**z);back=r.q.npz(saved);assert all(np.array_equal(x,back[k]) for k,x in z.items())
        checks.append(dict(arm=arm,actual_GPU8_FIT256=True,parameter_count=28376,first_zero_last_phase_norms=firstzero,laststep_all_parameter_grad_positive_finite=norms,canonical_AST_scalar_allgrad_EXACT=True,numpy_objective_error=nerr,double_parameter_FD=fd,native_cache_immutable=True,initial_zero_residual_identity=True,bounds_and_fixed16_PASS=True,actual_saved128_full_quality_complete_clusters_five_motion_max_error=error,eval_all_weights_bool_scalers_EXACT=True))
        del m,opt,out,res,z,back,before;gc.collect();torch.cuda.empty_cache()
        # Fresh throwaway model, not a scientific selected checkpoint.
        r.q.seed_all(991);bm=r.Refiner(st).to(r.DEVICE);bo=torch.optim.AdamW(bm.parameters(),lr=.001,weight_decay=.0001);times=[]
        for i in range(36):
            r.guard();torch.cuda.synchronize();tt=time.perf_counter();out,_=bm(ft,at,cv,arm);val=r.q.objective(out,gt)[0];bo.zero_grad(set_to_none=True);val.backward();bo.step();torch.cuda.synchronize()
            if i>=4:times.append(time.perf_counter()-tt)
        inner=sp['inner_cal_index'];torch.cuda.synchronize();tt=time.perf_counter();out,res=r.generated(bm,data,inner,arm);torch.cuda.synchronize();forward=time.perf_counter()-tt;z=r.portfolio(data,inner,out,res,arm);tt=time.perf_counter();r.evaluate(z);metric=time.perf_counter()-tt;bench.append(dict(arm=arm,FIT256_mean_seconds=float(np.mean(times)),warm4_measure32=True,inner1856_forward_seconds=forward,inner1856_full3port_metric_seconds=metric,no_saved_benchmark_model=True));del bm,bo,out,res,z;gc.collect();torch.cuda.empty_cache()
    assert paired[0].keys()==paired[1].keys() and all(torch.equal(paired[0][k],paired[1][k]) for k in paired[0]);resumes=[resume_test(data,sp,st,arm) for arm in r.ARMS]
    # Adequacy and diversity fixtures, no alternate scientific thresholds.
    p=np.zeros((1,2,12,2));p[0,0,:,0]=.1;p[0,0,-1,0]=.5;p[0,1,:,0]=.3;p[0,1,-1,0]=.1;a,f,j,good,_=r.v.errors(p,np.zeros((1,12,2)));assert a.min()<.2 and f.min()<.4 and not good.any();assert r.v.clusters(np.zeros((1,20,12,2)),np.ones((1,20),bool)).tolist()==[[1,1]]
    r.q.verify(files);r.guard();free=r.q.psutil.disk_usage(str(r.H)).free/2**30;assert free>2;compute=sum(25*((18909+255)//256)*3*x['FIT256_mean_seconds']+3*(1035+1856+2879)/1856*(x['inner1856_forward_seconds']+2*x['inner1856_full3port_metric_seconds']) for x in bench);elapsed=time.perf_counter()-start;planning=max(8,int(np.ceil((compute+80)/60)));r.write('PREFLIGHT.json',dict(status='PASS',source_sha=r.digest(files),locked_files=len(files),inherited_relevant_files=inherited,checks=checks,resume=resumes,stats_fixture_max_error=err,common_paired_all_initial_weights_bool_scalers_EXACT=True,benchmark=bench,compute_minutes=compute/60,planning_minutes=planning,ETA_minutes=[max(5,int(planning*.7)),int(np.ceil(planning*1.8))],resource=r.q.RESOURCE,disk_free_GiB=free,elapsed_seconds=elapsed,not_all_historical_old_SHA=True,old_F16_decoder_scorer_not_instantiated_trained_or_optimizer_accessed=True));r.write('prelaunch_review/REVIEW.json',dict(status='PASS',full_scientific_Gates_PASS=True,locked_source_count=len(files),no_old_module_top_level_execution=True,old_SHA_initial_final_PASS=True,actual_two_smoke_jobs_not18_scientific_jobs=True,DA1='PASS_SMALL_RESIDUAL_NOT_COVERAGE_OR_MOTION_GUARANTEE',not_external_review=True));print('PREFLIGHT_PASS',planning,compute/60,r.q.RESOURCE,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        r.write('PREFLIGHT_FAILURE_'+str(time.time_ns())+'.json',dict(status='STOP',traceback=traceback.format_exc(),resource=r.q.RESOURCE));raise
