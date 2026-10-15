"""Actual GPU, resume, saved-pipeline and budget gates; no production outcome selection."""
import common as c,model as m,metrics as v
import numpy as np,torch,time,copy,inspect,gc,traceback,shutil,ast
def exact(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b),"TENSOR_NOT_EXACT"
 elif isinstance(a,np.ndarray):assert np.array_equal(a,b),"ARRAY_NOT_EXACT"
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for key in a:exact(a[key],b[key])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):exact(x,y)
 else:assert a==b,(a,b)
def reject(call):
 try:call()
 except AssertionError:return True
 raise AssertionError("REJECTION_GATE_ACCEPTED_BAD_INPUT")
def resume(data,st,arm):
 ix=data["split"]["FIT"][:513];inner=data["split"]["inner"][:12];man={"fixture":True,"source_sha":"fixture","arm":arm,"IDs":data["ids"][ix].tolist()}
 def fresh():
  m.seed(20261014);net=m.MotionGoal(st,arm).to(m.DEVICE);return net,m.optimizer(net),torch.Generator().manual_seed(20261014)
 def one(net,opt,g,ep):
  row=m.epoch(net,opt,data,ix,g,ep);paths=m.generated(net,data,inner)[0];row["inner12_Q_posthoc"]=m.objective_numpy(paths,data["gt"][inner])[0];return row
 net,opt,g=fresh();h=[one(net,opt,g,i) for i in (1,2)]
 ref=dict(model=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=m.rng_state(g),history=copy.deepcopy(h));del net,opt
 net,opt,g=fresh();h=[one(net,opt,g,1)];p=c.H/f"preflight_smoke/{arm}_resume.pt";m.checkpoint(p,net,opt,g,h,man);del net,opt
 net,opt,g=fresh();h=m.restore(p,net,opt,g,man);h.append(one(net,opt,g,2))
 exact(ref,dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=m.rng_state(g),history=h))
 reject(lambda:m.restore(p,net,opt,g,{**man,"source_sha":"bad"}))
 return dict(actualFIT513_inner12_SUBSET_twoepoch_ALLparams_buffers_optimizer_TorchCPU_CUDA_loader_numpy_pythonRNG_history_EXACT=True,not_fullFIT2epochs_or_analysis_resume=True)
def objective_ast_check(path,gt):
 source=inspect.getsource(m.objective);tree=ast.parse(source);scope={"torch":torch};exec(compile(tree,"canonical_objective_readonly_AST","exec"),scope)
 x=path.detach().clone().requires_grad_(True);a=m.objective(x,gt)[0];b=scope["objective"](x,gt)[0]
 assert torch.equal(a,b) and torch.equal(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(b,x)[0])
 return c.sha(c.G1/"model.py")
def fd(st,data,arm):
 local={k:x.copy() for k,x in st.items()};local["centers"]=np.stack((np.arange(20)/10.,np.arange(20)**2/200.),1)
 m.seed(20261014);net=m.MotionGoal(local,arm).to(m.DEVICE).double()
 with torch.no_grad():net.head[-1].weight.normal_(0,.003)
 ix=data["split"]["FIT"][:4];p2,past,vel=m.inputs(data,ix);p2=p2.double();past=past.double();vel=torch.zeros_like(vel)
 gt=(net.s[None,:,None]*torch.tensor([.313,.025],device=m.DEVICE,dtype=torch.float64)[None,None]).expand(4,-1,-1)
 result=[]
 for name,idx in (("head.2.weight",(0,0)),("context.0.weight",(0,0)),("motion.weight_ih_l0",(0,0))):
  param=dict(net.named_parameters())[name];loss=m.objective(net(p2,past,vel)[0],gt)[0];grad=torch.autograd.grad(loss,param)[0][idx].item();before=param[idx].item()
  for eps in (1e-4,1e-5):
   values=[];faces=[]
   for sign in (1,-1):
    with torch.no_grad():param[idx]=before+sign*eps
    p=net(p2,past,vel)[0];z=m.objective(p,gt);values.append(z[0].item())
    pair=torch.triu_indices(20,20,1,device=m.DEVICE);hinge=torch.linalg.vector_norm(p[:,pair[0]]-p[:,pair[1]],dim=-1).mean(-1)<.3
    faces.append((z[-1].detach().cpu().numpy(),hinge.detach().cpu().numpy()))
   assert all(np.array_equal(a,b) for a,b in zip(*faces)),("FD_ACTIVE_FACE_SWITCH",arm,name,eps)
   error=abs((values[0]-values[1])/(2*eps)-grad)/(1+abs(grad));assert error<=1e-8,("FD",name,eps,error)
   result.append(dict(parameter=name,eps=eps,error=error,analytic=grad,no_active_face_switch=True))
  with torch.no_grad():param[idx]=before
 return result
def main():
 started=time.perf_counter();c.guard();assert torch.cuda.is_available();files,meta=c.locks();data=c.load();st=c.stats(data)
 staterr=c.stat_chunk_error(data,st);c.save("statistics.npz",**st);c.save("split_ids.npz",**{k:data["ids"][ix] for k,ix in data["split"].items()})
 tr=data["split"]["FIT"];fixed=sorted(tr,key=lambda i:__import__("hashlib").sha256(("m1-fixed-raw128-v1:"+str(data["ids"][i])).encode()).digest())[:128];fixed=np.array(fixed)
 c.save("fixed_feature_fixture.npz",indices=fixed,sample_id=data["ids"][fixed],raw_past=data["past"][fixed],velocity=data["raw_velocity"][fixed],acceleration=data["raw_acceleration"][fixed],p2=data["p2"][fixed],v0=data["vel"][fixed])
 for name in ("statistics.npz","split_ids.npz","fixed_feature_fixture.npz"):files[str(c.H/name)]=c.sha(c.H/name)
 first=tr[:256];args=m.inputs(data,first);gt=torch.from_numpy(data["gt"][first]).to(m.DEVICE)
 reports={};initials=[];steps=[];dual=[];audit=[];gens=[];smoke=None
 for arm in m.ARMS:
  m.seed(42);net=m.MotionGoal(st,arm).to(m.DEVICE);count=sum(x.numel() for x in net.parameters());assert count==34146
  initials.append({k:x.detach().cpu().clone() for k,x in net.state_dict().items()});assert list(inspect.signature(net.forward).parameters)==["p2","past","velocity"]
  token=net.tokens(args[1]).detach().cpu().numpy();assert token.shape==(256,7,5) and np.isfinite(token).all() and not np.count_nonzero(token[:,0,2:5]) and np.all(token[:,1:,4]==1)
  with torch.inference_mode():
   fx=m.inputs(data,fixed);fp,fg,_,_=net(*fx);ft=net.tokens(fx[1]).cpu().numpy()
  c.save(f"preflight_smoke/{arm}_fixed_raw_normalized.npz",sample_id=data["ids"][fixed],raw_past=data["past"][fixed],tokens=ft,paths=fp.cpu().numpy(),goals=fg.cpu().numpy())
  rv=data["raw_velocity"][first].copy();ra=data["raw_acceleration"][first].copy()
  if arm==m.ARMS[0]:rv[:]=rv[:,-1:];ra[:]=0
  expected=np.concatenate((((rv.astype(float)-st["velocity_mean"])/st["velocity_std"]).astype(np.float32),
   np.concatenate((np.zeros((len(first),1,2),np.float32),((ra.astype(float)-st["acceleration_mean"])/st["acceleration_std"]).astype(np.float32)),1),
   np.broadcast_to(np.array([0,1,1,1,1,1,1],np.float32)[None,:,None],(len(first),7,1))),2)
  assert np.array_equal(token,expected),("TOKEN_PHYSICAL_ZERO_MASK",arm)
  path,goals,_,delta=net(*args);proto=(np.arange(1,13)/12.)[None,None,:,None]*(4.8*data["vel"][first,None]+st["centers"][None])[:,:,None]
  assert np.array_equal(path.detach().cpu().numpy(),proto) and torch.count_nonzero(delta)==0
  canonical=objective_ast_check(path,gt);loss=m.objective(path,gt)[0];nl,nw=m.objective_numpy(proto,data["gt"][first]);scalar_error=abs(loss.item()-nl);assert scalar_error<=1e-12
  assert np.array_equal(m.objective(path,gt)[-1].detach().cpu().numpy(),nw)
  loss.backward();firstgrads={k:float(p.grad.norm().item()) for k,p in net.named_parameters()};assert firstgrads["head.2.weight"]>0 and firstgrads["head.2.bias"]>0
  opt=m.optimizer(net);opt.step();connected={}
  for i in range(8):
   ind=tr[i*256:(i+1)*256];opt.zero_grad(set_to_none=True);out=net(*m.inputs(data,ind))[0];z=m.objective(out,torch.from_numpy(data["gt"][ind]).to(m.DEVICE))[0];z.backward()
   for k,p in net.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all();connected[k]=max(connected.get(k,0),float(p.grad.norm().item()))
   opt.step()
  assert all(x>0 for x in connected.values()),("DISCONNECTED_LEGAL_PARAMETER",arm,connected)
  # No GT/group/ID fields exist in forward signature; changing only those labels cannot enter computation.
  frozen=copy.deepcopy(net.state_dict());ix=data["split"]["outer"][:128];t=time.perf_counter();p,g,delta=m.generated(net,data,ix);gens.append((time.perf_counter()-t)/128)
  t=time.perf_counter();met,aux=v.evaluate(p,data["gt"][ix],data["groups"][ix],data["native"][ix].astype(float));imet,ia=v.evaluate(p,data["gt"][ix],data["groups"][ix],data["native"][ix].astype(float),True);dual.append((time.perf_counter()-t)/128);c.eq(met,imet,2e-11)
  for k,a in aux.items():
   if a.dtype.kind in "biu":assert np.array_equal(a,ia[k])
   else:assert np.max(abs(a-ia[k]))<=2e-11
  t=time.perf_counter();v.evaluate(p,data["gt"][ix],data["groups"][ix],data["native"][ix].astype(float),True);audit.append((time.perf_counter()-t)/128)
  assert np.array_equal(v.legacy_clusters(p,aux["good_candidate"],True),ia["clusters"]),"VECTORIZED_VS_ORIGINAL_SCIPY_COMPLETE_LINK_NOT_EXACT"
  exact(frozen,net.state_dict());smoke=v.sample(data,ix,p,goals=g,goal_delta=delta,velocity=data["vel"][ix]);c.save(f"preflight_smoke/{arm}.npz",**smoke);back=c.npz(c.H/f"preflight_smoke/{arm}.npz");exact(smoke,back)
  path0=str(c.H/f"preflight_smoke/{arm}.npz");done=dict(status="COMPLETE",source_sha="fixture",files={path0:c.sha(path0)});assert c.done_valid(done,"fixture")
  reject(lambda:c.done_valid(done,"bad"));reject(lambda:c.done_valid({**done,"files":{path0:"bad"}},"fixture"))
  fds=fd(st,data,arm);res=resume(data,st,arm)
  m.seed(42);bench=m.MotionGoal(st,arm).to(m.DEVICE);op=m.optimizer(bench);times=[]
  for step in range(36):
   torch.cuda.synchronize();t=time.perf_counter();op.zero_grad(set_to_none=True);loss=m.objective(bench(*args)[0],gt)[0];loss.backward();op.step();torch.cuda.synchronize()
   if step>=4:times.append(time.perf_counter()-t)
  steps.append(float(np.mean(times)));reports[arm]=dict(parameters=count,initial_zero_last_gradient=firstgrads,actual_GPU8FIT256_all_legal_gradients_positive_finite=connected,canonical_AST_all_output_gradient_EXACT=canonical,numpy_scalar_error=scalar_error,tokens_physical_zero_normalization_mask_EXACT=True,saved128_full_quality_SciPyclusters_motion_frozen_EXACT=True,done_bad_source_output_rejected=True,double_parameter_FD=fds,resume=res)
  del net,opt,bench,op;gc.collect();torch.cuda.empty_cache()
 exact(initials[0],initials[1])
 # Check inherited physical convention with explicit independent discrete calculation.
 check=smoke["paths"];vv=np.concatenate((check[:,:,:1],check[:,:,1:]-check[:,:,:-1]),2)/.4;aa=np.diff(vv,axis=2)/.4;jj=np.diff(aa,axis=2)/.4
 assert vv.shape[2]==12 and aa.shape[2]==11 and jj.shape[2]==10
 # Benchmark full registered bootstrap arithmetic including hashing, with simulated fixed-size vectors only.
 import statistics_ci as ci
 T=len(np.unique(data["track"][data["split"]["FIT"]]));r=np.random.default_rng(20261014);t=time.perf_counter()
 for _ in range(512):ss=r.integers(0,3,3);tt=r.integers(0,T,T);ci.draw_hash(ss,tt)
 draw_hash_seconds=(time.perf_counter()-t)/512
 totalrows=sum(map(len,data["split"].values()));training=3*1850*sum(steps)
 quality=6*totalrows*(max(dual)+max(audit)+max(gens))
 bootstrap_seconds=16*20000*draw_hash_seconds*3+60
 nominal=training+quality+bootstrap_seconds;planning=(1.25*nominal+150)/60
 diskfree=shutil.disk_usage(c.H).free/2**30;diskbudget=6*totalrows*20*12*2*8*2/2**30+2.
 c.write("BENCHMARK.json",dict(FIT256_seconds_per_arm=steps,dualquality_seconds_perwindow=dual,auditquality_seconds_perwindow=audit,generation_seconds_perwindow=gens,
  bootstrap_including_independent_audit_estimate_seconds=bootstrap_seconds,full24_sciencejobs_included=True,nominal_compute_save_audit_seconds=nominal,planning_minutes=planning,disk_free_GiB=diskfree,conservative_new_artifact_GiB=diskbudget,no_saved_benchmarkmodel_or_scientific_outcome_choice=True))
 assert diskfree>=diskbudget+2,("STOP_DISK_BUDGET",diskfree,diskbudget)
 c.verify(files);c.guard()
 c.write("PREFLIGHT.json",dict(status="PASS" if planning<=30 else "STOP_BUDGET",source_count=len(files),inheritance=meta,arms=reports,paired_all_initial_params_buffers_EXACT=True,stats_chunk_error=staterr,raw_vlast_G1_EXACT=True,physics_count_12_11_10_EXACT=True,planning_minutes=planning,ETA_minutes=[planning*.7,planning*1.5],resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-started,old_F16_not_instantiated_or_optimized=True))
 assert planning<=30,("STOP_PRODUCTION_PLANNING_OVER30MIN",planning)
 c.write("source_manifest.json",dict(source_sha=c.digest(files),files=files,inheritance=meta));print("PREFLIGHT_PASS",planning,flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:
  traceback.print_exc();c.write("failures/preflight2/FAILURE.json",dict(status="STOP",error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
