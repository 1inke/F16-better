"""Actual paired GPU/real optimizer-RNG resume gates before formal training."""
import sys,copy,time,gc,traceback,importlib.util,math
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
    c=copy.deepcopy(m).double();before={k:v.detach().clone() for k,v in c.state_dict().items()}
    with torch.no_grad():
        full,_,u=c(features.double(),anchors.double(),CV.double(),arm,True)
        proposal=torch.einsum('ti,bkid->bktd',c.time_basis,u)*(.20*(torch.arange(1,13,device=r.DEVICE,dtype=torch.float64)/12)**2)[None,None,:,None]
        win=r.q.objective(full,gt.double())[-1]
        if arm=='ray':
            _,alpha=r.pr.project_torch(anchors.double(),proposal);aw=alpha.gather(1,win[:,None]).squeeze(1);eligible=torch.where(aw>0)[0];chosen=eligible[torch.argsort(aw[eligible],descending=True)[:16]]
        else:chosen=torch.arange(16,device=r.DEVICE)
    assert len(chosen)>0
    x=features[chosen].double();a=anchors[chosen].double();cv=CV[chosen].double();y=gt[chosen].double();p=c.net[-1].bias
    def calc():
        out,_,u=c(x,a,cv,arm,True);proposal=torch.einsum('ti,bkid->bktd',c.time_basis,u)*(.20*(torch.arange(1,13,device=r.DEVICE,dtype=torch.float64)/12)**2)[None,None,:,None]
        if arm=='ray':
            face=torch.cat([r.pr.root_torch(b,d) for b,d in zip(r.pr.differences_torch(a),r.pr.differences_torch(proposal))],-1).argmin(-1)
        else:
            _,g=r.sim.project_torch(a,proposal);Jp=torch.stack((-a[...,1],a[...,0]),-1);fit=(g['capped_a']-1)[...,None,None]*a+g['capped_b'][...,None,None]*Jp;norm=torch.linalg.vector_norm(fit,dim=-1);bounds=.20*(torch.arange(1,13,device=r.DEVICE,dtype=torch.float64)/12)**2
            ratio=bounds/torch.where(norm>0,norm,torch.ones_like(norm));face=ratio.argmin(-1)+(g['scale_before']>1.01).long()*12+(g['position_lambda']<1).long()*24
        return r.q.objective(out,y),face
    base,faces=calc();grad=torch.autograd.grad(base[0],p)[0];assert torch.isfinite(grad).all() and grad.norm()>1e-12
    direction=grad/grad.norm();analytic=float((grad*direction).sum());saved=p.detach().clone();results=[]
    for eps in (1e-4,1e-5,1e-6,1e-7):
        with torch.no_grad():p.copy_(saved+eps*direction)
        up,uf=calc()
        with torch.no_grad():p.copy_(saved-eps*direction)
        down,df=calc()
        value=float(((up[0]-down[0])/(2*eps)).detach());error=abs(value-analytic);switch=int(((up[-1]!=base[-1])|(down[-1]!=base[-1])).sum());fs=int(((uf!=faces)|(df!=faces)).sum())
        results.append(dict(eps=eps,analytic=analytic,finite_difference=value,error=error,winner_switch_windows=switch,geometry_face_switch=fs,original_error_rule_pass=error<=1e-7+1e-3*abs(analytic)))
    assert all(z['original_error_rule_pass'] and z['winner_switch_windows']==0 and z['geometry_face_switch']==0 for z in results[-2:]),results
    with torch.no_grad():p.copy_(saved)
    assert all(torch.equal(v,before[k]) for k,v in c.state_dict().items())
    return results

def basis_fixtures():
    B=r.bernstein_matrix();ref=np.zeros((12,12))
    for t in range(12):
        x=t/11.
        for i in range(12):ref[t,i]=math.factorial(11)/math.factorial(i)/math.factorial(11-i)*x**i*(1-x)**(11-i)
    err=float(abs(B-ref).max());assert err==0
    assert np.array_equal(B[0],np.eye(12)[0]) and np.array_equal(B[-1],np.eye(12)[-1])
    rng=np.random.default_rng(20261008);u=rng.normal(size=(17,4,12,2));u/=np.maximum(1,np.linalg.norm(u,axis=-1,keepdims=True));w=np.einsum('ti,bkid->bktd',B,u);manual=np.zeros_like(w)
    for t in range(12):
        for i in range(12):manual[:,:,t]+=B[t,i]*u[:,:,i]
    assert np.max(abs(w-manual))<=1e-15 and (np.linalg.norm(w,axis=-1)<=1+2e-15).all()
    assert np.array_equal(w[:,:,0],u[:,:,0]) and np.array_equal(w[:,:,-1],u[:,:,-1])
    const=np.broadcast_to(u[:,:,:1],u.shape);fixed=np.einsum('ti,bkid->bktd',B,const);assert np.max(abs(fixed-const))<=1e-15
    alt=np.zeros((1,4,12,2));alt[:,:,:,0]=(-1.)**np.arange(12);wa=np.einsum('ti,bkid->bktd',B,alt);assert (abs(wa)<=1+2e-15).all() and np.max(abs(wa[:,:,1:-1,0]))<1
    tu=torch.from_numpy(u).to(r.DEVICE).requires_grad_(True);bw=torch.einsum('ti,bkid->bktd',torch.from_numpy(B).to(r.DEVICE),tu);gpu_error=float(abs(bw.detach().cpu().numpy()-w).max());assert gpu_error<=1e-15
    g=torch.autograd.grad(bw.sum(),tu)[0];assert torch.isfinite(g).all() and (g>0).all()
    return dict(status='PASS',matrix_rank=12,matrix_condition_number=float(np.linalg.cond(B)),independent_matrix_error=err,GPU_NumPy_error=gpu_error,convex_norm_bound=True,first_last_given_controls_EXACT=True,constant_fixture=True,alternating_fixture_not_physical_success=True,all12_control_connection_positive=True)

def prior_control_alignment(m,ft,at,cv,arm):
    if arm!='ray':return dict(status='TREATMENT',not_preassumed_repeat=True)
    spec=importlib.util.spec_from_file_location('prior_ray_definitions_readonly',r.R/'GENERATION_NATIVE_ANCHORED_PHYSICAL_RAY_SERIES_20261008/roundoff_certified_recovery_20261008/approved_launch_20261008/run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    om=old.Refiner((m.mean.detach().cpu().numpy(),m.std.detach().cpu().numpy())).to(r.DEVICE);om.load_state_dict(m.state_dict(),strict=True);om.eval();m.eval()
    with torch.no_grad():a,b=m(ft,at,cv,arm);c,d=om(ft,at,cv,'physical')
    assert torch.equal(a,c) and torch.equal(b,d)
    del om,old
    return dict(status='PASS',same_nonzero_actual_weights_paths_residuals_bitwise_EXACT=True,no_old_model_optimizer_or_parameter_update=True)

def nondegeneracy(data,sp,st):
    r.q.seed_all(20261008);m=r.Refiner(st).to(r.DEVICE)
    with torch.no_grad():m.net[-1].weight.normal_(0,.03);m.net[-1].bias.normal_(0,.03)
    ix=sp['grouped_train_index'][:128];f=torch.from_numpy(r.features(data,ix)).to(r.DEVICE);a=torch.from_numpy(data['anchor'][ix]).to(r.DEVICE);cv=torch.from_numpy(data['CV'][ix]).to(r.DEVICE)
    with torch.no_grad():out,res,u=m(f,a,cv,'similarity',True)
    norms=torch.linalg.vector_norm(res,dim=-1).amax(-1);fraction=float((norms>=.001).double().mean());maximum=float(norms.max())
    assert maximum>=.001 and fraction>=.05,('STOP_SCIENTIFIC_DEGENERATE_GEOMETRY',maximum,fraction)
    proposed=torch.einsum('ti,bkid->bktd',m.time_basis,u)*(.20*(torch.arange(1,13,device=r.DEVICE,dtype=torch.float64)/12)**2)[None,None,:,None]
    nr,g=r.sim.project_numpy(a.double().cpu().numpy(),proposed.cpu().numpy());error=float(abs(nr-res.cpu().numpy()).max());assert error<=2e-12
    r.sim.audit_numpy(a.double().cpu().numpy(),out.cpu().numpy(),g)
    _,rayalpha=r.pr.project_torch(a.double(),proposed)
    value=dict(status='PASS',fixed_seed=20261008,FIT128_only_no_GT_selection=True,max_residual=maximum,candidate_ge_1mm_fraction=fraction,torch_NumPy_max=error,ray_alpha_zero_fraction=float((rayalpha==0).double().mean()),not_scientific_trained_checkpoint=True)
    del m;return value

def main():
    start=time.perf_counter();r.guard();assert torch.cuda.is_available();basis=basis_fixtures();physical_fixtures=r.pr.fixtures();similarity_fixtures=r.sim.fixtures();r.stage('PREFLIGHT_SOURCE_IDS_STATS');ns,data,sp=r.data_load();tr=sp['grouped_train_index'];st=r.statistics(data,tr);assert all(np.isfinite(x).all() for x in st) and (st[1]>0).all();np.savez_compressed(r.H/'statistics.npz',mean=st[0],std=st[1],normalization_ids=data['ids'][tr])
    x=r.features(data,tr[:1033]).reshape(-1,120).astype(float);mu=x.mean(0);sd=x.std(0);sums=np.zeros(120);sq=np.zeros(120)
    for first in range(0,len(x),31):sums+=x[first:first+31].sum(0);sq+=(x[first:first+31]**2).sum(0)
    err=max(float(np.max(abs(mu-sums/len(x)))),float(np.max(abs(sd-np.sqrt(np.maximum(sq/len(x)-(sums/len(x))**2,0))))));assert err<=1e-11,err
    files,inherited=r.locks();files[str(r.H/'statistics.npz')]=r.sha(r.H/'statistics.npz');r.q.verify(files);r.write('source_manifest.json',dict(files=files,inherited_relevant_files=inherited,not_all_historical_files=True));checks=[];paired=[];bench=[];nd=nondegeneracy(data,sp,st)
    ix=tr[:256];ft=torch.from_numpy(r.features(data,ix)).to(r.DEVICE);at=torch.from_numpy(data['anchor'][ix]).to(r.DEVICE);cv=torch.from_numpy(data['CV'][ix]).to(r.DEVICE);gt=torch.from_numpy(data['gt'][ix]).to(r.DEVICE);native_before=data['native'][ix].copy()
    for arm in r.ARMS:
        r.stage('PREFLIGHT_GPU8',arm=arm);r.q.seed_all(42);m=r.Refiner(st).to(r.DEVICE);paired.append({k:z.detach().cpu().clone() for k,z in m.state_dict().items()});assert sum(p.numel() for p in m.parameters())==25304;opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);out,res=m(ft,at,cv,arm);assert torch.equal(res,torch.zeros_like(res));assert torch.equal(out,at.double())
        loss=r.q.objective(out,gt)[0];ref=ns['reference_loss'](out,gt);assert torch.equal(loss,ref);g=torch.autograd.grad(loss,list(m.parameters()),retain_graph=True);rg=torch.autograd.grad(ref,list(m.parameters()));assert all(torch.equal(a,b) for a,b in zip(g,rg));nerr=abs(float(loss.detach())-np_objective(out.detach().cpu().numpy(),gt.cpu().numpy()));assert nerr<=2e-12,nerr
        firstzero={k:float(z.norm()) for (k,_),z in zip(m.named_parameters(),g)};norms={}
        for step in range(8):
            r.guard();out,res=m(ft,at,cv,arm);value=r.q.objective(out,gt)[0];opt.zero_grad(set_to_none=True);value.backward();assert torch.isfinite(out).all() and torch.isfinite(value);norms={k:float(z.grad.norm()) for k,z in m.named_parameters()};assert all(np.isfinite(n) for n in norms.values());opt.step()
        assert all(n>0 for n in norms.values()),norms;assert np.array_equal(native_before,data['native'][ix]);fd=parameter_FD(m,ft,at,cv,gt,arm)
        with torch.no_grad():po,pres,pc=m(ft,at,cv,arm,True)
        proposed=torch.einsum('ti,bkid->bktd',m.time_basis,pc)*(.20*(torch.arange(1,13,device=r.DEVICE,dtype=torch.float64)/12.)**2)[None,None,:,None]
        _,alpha=r.pr.project_torch(at.double(),proposed)
        projection_active=dict(alpha_min=float(alpha.min()),alpha_mean=float(alpha.mean()),alpha_nonzero_candidates=int((alpha>0).sum()),candidate_count=alpha.numel(),native_zero_budget_described_not_deleted=True)
        if arm=='ray':
            r.pr.audit_numpy(at.double().cpu().numpy(),po.cpu().numpy())
            assert int((alpha>0).sum())>0,'STOP_SCIENTIFIC_ALL_DIRECTIONS_COLLAPSED'
        control=prior_control_alignment(m,ft,at,cv,arm)
        before={k:z.detach().cpu().clone() for k,z in m.state_dict().items()};small=sp['outer_eval_index'][:128];out,res,u=r.generated(m,data,small,arm,True);z=r.portfolio(data,small,out,res,arm,u);met,extra=r.evaluate(z);imet,iextra=r.evaluate(z,True);error=r.v.compare(met,imet);assert error<=1e-12 and all(np.array_equal(extra[k],iextra[k]) for k in extra);assert all(torch.equal(z.detach().cpu(),before[k]) for k,z in m.state_dict().items());saved=r.H/'prelaunch_review'/f'{arm}_smoke128.npz';saved.parent.mkdir(exist_ok=True);np.savez_compressed(saved,**z);back=r.q.npz(saved);assert all(np.array_equal(x,back[k]) for k,x in z.items())
        checks.append(dict(arm=arm,actual_GPU8_FIT256=True,parameter_count=25304,first_zero_last_phase_norms=firstzero,laststep_all_parameter_grad_positive_finite=norms,canonical_AST_scalar_allgrad_EXACT=True,numpy_objective_error=nerr,double_parameter_FD=fd,prior_nonzero_control_alignment=control,physical_ray_actual=projection_active,native_cache_immutable=True,initial_zero_residual_identity=True,bounds_and_fixed16_PASS=True,actual_saved128_full_quality_complete_clusters_five_motion_max_error=error,eval_all_weights_bool_scalers_EXACT=True))
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
    r.q.verify(files);r.guard();free=r.q.psutil.disk_usage(str(r.H)).free/2**30;assert free>2;compute=sum(25*((18909+255)//256)*3*x['FIT256_mean_seconds']+3*(1035+1856+2879)/1856*(x['inner1856_forward_seconds']+2*x['inner1856_full3port_metric_seconds']) for x in bench);elapsed=time.perf_counter()-start;planning=max(8,int(np.ceil((compute+80)/60)));r.write('PREFLIGHT.json',dict(status='PASS',source_sha=r.digest(files),locked_files=len(files),inherited_relevant_files=inherited,checks=checks,resume=resumes,basis_fixtures=basis,physical_ray_fixtures=physical_fixtures,similarity_fixtures=similarity_fixtures,actual_nondegeneracy=nd,stats_fixture_max_error=err,common_paired_all_initial_weights_bool_scalers_EXACT=True,benchmark=bench,compute_minutes=compute/60,planning_minutes=planning,ETA_minutes=[max(5,int(planning*.7)),int(np.ceil(planning*1.8))],resource=r.q.RESOURCE,disk_free_GiB=free,elapsed_seconds=elapsed,not_all_historical_old_SHA=True,old_F16_decoder_scorer_not_instantiated_trained_or_optimizer_accessed=True));r.write('prelaunch_review/REVIEW.json',dict(status='PASS',full_scientific_Gates_PASS=True,locked_source_count=len(files),no_old_module_top_level_execution=True,old_SHA_initial_final_PASS=True,actual_two_smoke_jobs_not18_scientific_jobs=True,DA1='PASS_NONDEGENERATE_GLOBAL_SIMILARITY_NOT_NEW_TEMPORAL_SHAPES_OR_GENERAL_FEASIBILITY_PROOF',not_external_review=True));print('PREFLIGHT_PASS',planning,compute/60,r.q.RESOURCE,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        r.write('PREFLIGHT_FAILURE_'+str(time.time_ns())+'.json',dict(status='STOP',traceback=traceback.format_exc(),resource=r.q.RESOURCE));raise
