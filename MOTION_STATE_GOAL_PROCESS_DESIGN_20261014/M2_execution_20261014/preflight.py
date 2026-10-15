"""M2 real gates. No production launch or old model updates here."""
import common as c,model as m,metrics as v,run
import numpy as np,torch,time,copy,inspect,ast,traceback,gc,shutil
def exact(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b),"TENSOR_NOT_EXACT"
 elif isinstance(a,np.ndarray):assert np.array_equal(a,b),"ARRAY_NOT_EXACT"
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:exact(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):exact(x,y)
 else:assert a==b,(a,b)
def reject(call):
 try:call()
 except AssertionError:return True
 raise AssertionError("BAD_CONTRACT_NOT_REJECTED")
def fixture():
 rng=np.random.default_rng(20261014);b=rng.normal(0,.07,(4,20,12,2));g=rng.normal(0,2,(4,20,2));vel=rng.normal(0,.4,(4,2))
 tb=torch.tensor(b,device=m.DEVICE,dtype=torch.float64,requires_grad=True);tg=torch.tensor(g,device=m.DEVICE,dtype=torch.float64,requires_grad=True);tv=torch.tensor(vel,device=m.DEVICE,dtype=torch.float64)
 z=m.project(tb,tg,tv);nz=m.project_numpy(b,g,vel)
 err=max(np.max(abs(z[k].detach().cpu().numpy()-nz[k])) for k in m.FIELDS);assert err<1e-10
 w=np.arange(12,0,-1,dtype=float);assert int(w@w)==650
 assert np.max(abs(nz["paths"][:,:,-1]-g))<1e-10
 assert np.max(abs(np.sum(w[None,None,:,None]*nz["null_term"],2)))<1e-10
 zero=m.project_numpy(np.zeros_like(b),4.8*vel[:,None]+np.zeros_like(g),vel)
 cv=np.arange(1,13)[None,None,:,None]*.4*vel[:,None,None]
 assert np.max(abs(zero["paths"]-cv))<1e-10
 wb=b+w[None,None,:,None]*rng.normal(size=(4,20,1,2))
 assert np.max(abs(m.project_numpy(wb,g,vel)["paths"]-nz["paths"]))<1e-10
 coeff=torch.tensor(rng.normal(size=b.shape),device=m.DEVICE,dtype=torch.float64);loss=(z["paths"]*coeff).sum()
 grads=torch.autograd.grad(loss,(tb,tg));f=[]
 for x,grad,idx in ((tb,grads[0],(0,0,4,0)),(tg,grads[1],(0,0,0))):
  val=x[idx].item();analytic=grad[idx].item()
  for eps in (1e-4,1e-5):
   scores=[]
   for sign in (1,-1):
    with torch.no_grad():x[idx]=val+sign*eps
    scores.append((m.project(tb,tg,tv)["paths"]*coeff).sum().item())
   with torch.no_grad():x[idx]=val
   error=abs((scores[0]-scores[1])/(2*eps)-analytic)/(1+abs(analytic));assert error<1e-8
   f.append(dict(eps=eps,error=error,analytic=analytic))
 return dict(independent_projection_max=float(err),S=650,effective_rank=22,CV_and_nullspace_PASS=True,FD_b_g=f)
def resume(data,arm):
 ix=data["split"]["FIT"][:513];inner=data["split"]["inner"][:12];man=dict(fixture=True,arm=arm,source_sha="fixture",ids=data["ids"][ix].tolist())
 def fresh():net=m.fresh(42,arm);return net,m.optimizer(net),torch.Generator().manual_seed(20261014)
 def one(net,opt,g,ep):
  row=m.epoch(net,opt,data,ix,g,ep);z=m.generated(net,data,inner);row["inner12_Q_posthoc"]=m.objective_numpy(z["paths"],data["gt"][inner])[0];return row
 net,opt,g=fresh();h=[one(net,opt,g,k) for k in (1,2)];ref=copy.deepcopy(dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=m.rng_state(g),history=h))
 del net,opt;net,opt,g=fresh();h=[one(net,opt,g,1)];path=c.H/f"preflight_smoke/{arm}_resume.pt";path.parent.mkdir(parents=True,exist_ok=True);m.checkpoint(path,net,opt,g,h,man)
 del net,opt;net,opt,g=fresh();h=m.restore(path,net,opt,g,man);h.append(one(net,opt,g,2))
 exact(ref,dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=m.rng_state(g),history=h));reject(lambda:m.restore(path,net,opt,g,{**man,"source_sha":"bad"}))
 return dict(FIT513_inner12_SUBSET_twoepoch_ALLparams_buffers_optimizer_TorchCPUCUDA_loader_numpy_pythonRNG_history_EXACT=True,not_fullFIT2epochs_or_analysis_resume=True)
def parameter_fd(data,arm):
 net=m.fresh(42,arm).double()
 # Deterministic isolated arithmetic fixture: no scientific checkpoint or outcome selection.
 with torch.no_grad():
  net.goal.centers.copy_(torch.tensor(np.stack((np.arange(20)*.21,np.arange(20)**2*.017),1),device=m.DEVICE))
  net.process[-1].weight.normal_(0,.0003)
 args=tuple(x.double() for x in m.inputs(data,data["split"]["FIT"][:4]))
 with torch.no_grad():gt=net(*args)["paths"][:,0].clone()+torch.tensor([.017,.031],device=m.DEVICE,dtype=torch.float64)
 names=["process.2.weight"]+(["goal.head.2.weight"] if arm=="joint_goal" else [])
 result=[]
 for name in names:
  p=dict(net.named_parameters())[name];idx=(0,0);val=p[idx].item();grad=torch.autograd.grad(m.objective(net(*args)["paths"],gt)[0],p)[0][idx].item()
  assert abs(grad)>0
  for eps in (1e-4,1e-5):
   scores=[];faces=[]
   for sign in (1,-1):
    with torch.no_grad():p[idx]=val+sign*eps
    out=net(*args)["paths"];loss=m.objective(out,gt);scores.append(loss[0].item());pair=torch.triu_indices(20,20,1,device=m.DEVICE)
    faces.append((loss[-1].detach().cpu().numpy(),(torch.linalg.vector_norm(out[:,pair[0]]-out[:,pair[1]],dim=-1).mean(-1)<.3).detach().cpu().numpy()))
   with torch.no_grad():p[idx]=val
   assert all(np.array_equal(a,b) for a,b in zip(*faces)),("FD_ACTIVE_FACE_SWITCH",name,eps)
   err=abs((scores[0]-scores[1])/(2*eps)-grad)/(1+abs(grad));assert err<1e-8,("PARAMETER_FD",name,eps,err)
   result.append(dict(parameter=name,eps=eps,analytic=grad,error=err))
 return result
def main():
 started=time.perf_counter();c.guard();assert torch.cuda.is_available()
 sha_start=time.perf_counter();files,meta=c.locks();sha_seconds=time.perf_counter()-sha_start
 data=c.load();tr=data["split"]["FIT"];st=c.npz(c.G1/"statistics.npz");assert np.array_equal(st["training_ids"],data["ids"][tr])
 c.save("split_ids.npz",**{k:data["ids"][ix] for k,ix in data["split"].items()})
 geometry_fixture=fixture();stats_error=0.
 for s in (42,43,44):
  original=c.npz(c.G1/f"runs/seed_{s}/FIT.npz");assert np.array_equal(original["sample_id"],data["ids"][tr]);goals=original["goals"]
  gm=goals.reshape(-1,2).mean(0);gs=goals.reshape(-1,2).std(0);gs=np.where(gs<1e-6,1.,gs)
  vm=data["vel"][tr].mean(0);vs=data["vel"][tr].std(0);vs=np.where(vs<1e-6,1.,vs)
  for chunk in (31,256):
   for array,mu,sd in ((goals.reshape(-1,2),gm,gs),(data["vel"][tr],vm,vs)):
    sm=np.zeros(2);sq=np.zeros(2);n=0
    for start in range(0,len(array),chunk):
     a=array[start:start+chunk];sm+=a.sum(0);sq+=(a*a).sum(0);n+=len(a)
    mean=sm/n;std=np.sqrt(np.maximum(0,sq/n-mean*mean));std=np.where(std<1e-6,1.,std)
    stats_error=max(stats_error,float(np.max(abs(mean-mu))),float(np.max(abs(std-sd))))
  assert stats_error<1e-11,("STAT_CHUNK",stats_error)
  c.save(f"goal_stats_{s}.npz",goal_mean=gm,goal_std=gs,v0_mean=vm,v0_std=vs,training_ids=data["ids"][tr],initial_parent_sha=np.array(c.sha(c.G1/f"runs/seed_{s}/checkpoint.pt")))
  del original,goals;gc.collect()
 parent_checks=[]
 for s in (42,43,44):
  aa=m.fresh(s,m.ARMS[0]);bb=m.fresh(s,m.ARMS[1]);exact(aa.state_dict(),bb.state_dict())
  with torch.inference_mode():
   za=aa(*m.inputs(data,tr[:128]));zb=bb(*m.inputs(data,tr[:128]))
  for k in m.FIELDS:assert torch.equal(za[k],zb[k]),("PAIRED_INITIAL",s,k)
  parent_saved=c.npz(c.G1/f"runs/seed_{s}/FIT.npz")
  assert np.array_equal(za["goals"].cpu().numpy(),parent_saved["goals"][:128]),("PARENT_REPLAY",s)
  parent_checks.append(dict(seed=s,paired_all_params_buffers_and_initial_outputs_EXACT=True,actual_GPU128_parent_saved_goals_EXACT=True))
  del aa,bb,za,zb,parent_saved;gc.collect();torch.cuda.empty_cache()
 for name in ("split_ids.npz","goal_stats_42.npz","goal_stats_43.npz","goal_stats_44.npz"):files[str(c.H/name)]=c.sha(c.H/name)
 args=m.inputs(data,tr[:256]);gt=torch.from_numpy(data["gt"][tr[:256]]).to(m.DEVICE);report={};initials=[];times=[];dual=[];aud=[];gen=[];saved_bytes=[]
 for arm in m.ARMS:
  net=m.fresh(42,arm);initial=copy.deepcopy({k:v.cpu() for k,v in net.state_dict().items()});initials.append(initial)
  assert sum(p.numel() for p in net.parameters())==33306
  assert sum(p.numel() for p in net.parameters() if p.requires_grad)==(7000 if arm=="frozen_goal" else 12314)
  assert list(inspect.signature(net.forward).parameters)==["p2","velocity"]
  z=net(*args);assert not torch.count_nonzero(z["raw_b"])
  # Actual G1 replay agreement for the current GPU batch.
  parent=m.Goal20(st);parent.load_state_dict(m.parent_state(42));parent=parent.to(m.DEVICE)
  with torch.no_grad():pg=parent(*args)[1]
  assert torch.equal(z["goals"],pg);del parent
  source=inspect.getsource(m.objective);scope={"torch":torch};exec(compile(ast.parse(source),"canonical_readonly_AST","exec"),scope)
  p=z["paths"];a=m.objective(p,gt)[0];b=scope["objective"](p,gt)[0]
  assert torch.equal(a,b) and torch.equal(torch.autograd.grad(a,p,retain_graph=True)[0],torch.autograd.grad(b,p,retain_graph=True)[0])
  numeric,winner=m.objective_numpy(p.detach().cpu().numpy(),data["gt"][tr[:256]])
  assert abs(numeric-a.item())<1e-12 and np.array_equal(winner,m.objective(p,gt)[-1].detach().cpu().numpy())
  opt=m.optimizer(net);opt.zero_grad(set_to_none=True);a.backward();first={k:float(x.grad.norm().item()) if x.grad is not None else None for k,x in net.named_parameters()};opt.step()
  connected={}
  for step in range(8):
   ix=tr[step*256:(step+1)*256];opt.zero_grad(set_to_none=True);loss=m.objective(net(*m.inputs(data,ix))["paths"],torch.from_numpy(data["gt"][ix]).to(m.DEVICE))[0];loss.backward()
   for k,x in net.named_parameters():
    if x.requires_grad:
     assert x.grad is not None and torch.isfinite(x.grad).all();connected[k]=max(connected.get(k,0),float(x.grad.norm().item()))
    else:assert x.grad is None,k
   opt.step()
  assert all(x>0 for x in connected.values()),("DISCONNECTED_PARAMETER",arm,connected)
  m.frozen_check(net,initial)
  # Independent differentiation through goal-input and projected-endpoint routes.
  routes=None
  if arm=="joint_goal":
   z=net(*args);h=net.goal.context((args[0]-net.goal.mean)/net.goal.std).detach();q=net.goal.query.detach()[None].expand(256,-1,-1)
   X=torch.cat((h[:,None].expand(-1,20,-1),q),2);gg=z["goals"].detach().requires_grad_(True)
   vz=((args[1]-net.v0_mean)/net.v0_std).float()[:,None].expand(-1,20,-1)
   raw=net.process(torch.cat((X,((gg-net.goal_mean)/net.goal_std).float(),vz),2)).double().reshape(-1,20,12,2)
   input_grad=torch.autograd.grad(raw.square().sum(),gg)[0]
   endpoint_grad=torch.autograd.grad(m.project(z["raw_b"].detach(),gg,args[1])["paths"].square().sum(),gg)[0]
   assert input_grad.norm()>0 and endpoint_grad.norm()>0
   routes=dict(input_norm=float(input_grad.norm().item()),endpoint_norm=float(endpoint_grad.norm().item()),both_connected=True)
  ix=data["split"]["outer"][:128];t=time.perf_counter();out=m.generated(net,data,ix);gen.append((time.perf_counter()-t)/128)
  geometry_error=run.geometry(out)
  # Initial goals for this actual smoke are the exact original prediction, not GT.
  p0=m.generated(m.fresh(42,arm),data,ix)["goals"]
  t=time.perf_counter();met,ge=run.job(data,ix,out,f"preflight_smoke/{arm}.npz",p0);dual.append((time.perf_counter()-t)/128)
  smoke=c.npz(c.H/f"preflight_smoke/{arm}.npz")
  t=time.perf_counter();v.evaluate(smoke["paths"],smoke["gt"],smoke["groups"],smoke["native20"],True);run.geometry(smoke);aud.append((time.perf_counter()-t)/128)
  file=str(c.H/f"preflight_smoke/{arm}.npz");saved_bytes.append((c.H/f"preflight_smoke/{arm}.npz").stat().st_size/128)
  done=dict(status="COMPLETE",source_sha="fixture",files={file:c.sha(file)});assert c.done_valid(done,"fixture")
  reject(lambda:c.done_valid(done,"bad"));reject(lambda:c.done_valid({**done,"files":{file:"bad"}},"fixture"))
  del out,smoke;gc.collect()
  fds=parameter_fd(data,arm);res=resume(data,arm);bench=m.fresh(42,arm);op=m.optimizer(bench);bt=[]
  for step in range(36):
   c.guard();torch.cuda.synchronize();t=time.perf_counter();op.zero_grad(set_to_none=True);loss=m.objective(bench(*args)["paths"],gt)[0];loss.backward();op.step();torch.cuda.synchronize()
   if step>=4:bt.append(time.perf_counter()-t)
  times.append(float(np.mean(bt)))
  report[arm]=dict(total_parameters=33306,trainable=7000 if arm=="frozen_goal" else 12314,initial_zero_last_gradients=first,actual_GPU8FIT256_positive_finite=connected,canonical_AST_scalar_outputgrad_EXACT=True,numpy_scalar_error=abs(numeric-a.item()),G1_actual_goal_replay_EXACT=True,frozen_parent_EXACT=True,goal_gradient_routes=routes,geometry_saved_error=geometry_error,FD=fds,resume=res,bad_source_output_resume_rejected=True)
  del net,opt,bench,op;gc.collect();torch.cuda.empty_cache()
 exact(initials[0],initials[1])
 import statistics_ci as ci
 r=np.random.default_rng(20261014);T=len(np.unique(data["track"][tr]));t=time.perf_counter()
 for _ in range(512):ss=r.integers(0,3,3);tt=r.integers(0,T,T);ci.draw_hash(ss,tt)
 draw_seconds=(time.perf_counter()-t)/512
 rows=sum(len(ix) for ix in data["split"].values());train=3*1850*sum(times)
 quality=9*rows*(max(dual)+max(aud)+max(gen)) # 24 final + 12 P0 jobs, includes independent full CPU audit.
 boot=16*20000*draw_seconds*3+60;nominal=train+quality+boot+2*sha_seconds;planning=(1.25*nominal+150)/60
 disk=9*rows*max(saved_bytes)/2**30*1.5+2.;free=shutil.disk_usage(c.H).free/2**30
 c.write("BENCHMARK.json",dict(FIT256_seconds_per_arm=times,dual_saved_seconds_perwindow=dual,independent_audit_seconds_perwindow=aud,generation_seconds_perwindow=gen,source_initial_SHA_seconds=sha_seconds,bootstrap_dual_estimate_seconds=boot,complete36_savedjobs_included=True,training_seconds=train,save_recompute_audit_seconds=quality,nominal_seconds=nominal,planning_minutes=planning,ETA_minutes=[.7*planning,1.5*planning],disk_required_GiB=disk,disk_free_GiB=free,no_saved_benchmarkmodel_or_outcome_selection=True))
 assert free>disk+2,("STOP_DISK",free,disk)
 c.verify(files);c.guard()
 c.write("PREFLIGHT.json",dict(status="PASS" if planning<=30 else "STOP_BUDGET",source_count=len(files),inheritance=meta,geometry=geometry_fixture,arms=report,parent_checks=parent_checks,paired_all_initial_state_EXACT=True,stats_chunk_error=stats_error,planning_minutes=planning,ETA_minutes=[.7*planning,1.5*planning],resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-started,old_F16_not_instantiated_or_optimized=True))
 c.write("preflight_source_snapshot.json",dict(files=files,source_sha=c.digest(files)))
 assert planning<=30,("STOP_PRODUCTION_PLANNING_OVER30MIN",planning)
 c.write("source_manifest.json",dict(files=files,source_sha=c.digest(files),inheritance=meta));print("PREFLIGHT_PASS",planning,flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:
  traceback.print_exc();c.write("failures/preflight1/FAILURE.json",dict(status="STOP",error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
