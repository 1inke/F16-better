import common as c,model as m,metrics as v,run as r
import numpy as np,torch,time,copy,inspect,traceback,shutil
def exact(a,b):
 if torch.is_tensor(a):assert a.device==b.device and torch.equal(a,b),"TORCH_RESUME_NOT_EXACT"
 elif isinstance(a,np.ndarray):assert np.array_equal(a,b),"NP_RESUME_NOT_EXACT"
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:exact(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):exact(x,y)
 else:assert a==b,(a,b)
def resume_fixture(data):
 tr=data["split"]["FIT"][:513];inner=data["split"]["inner"][:12];manifest=dict(source_sha="subsetfixture",training_ids=data["ids"][tr].tolist(),epoch=2)
 def fresh():
  net=m.fresh(42);opt=r.optimizer(net);g=torch.Generator().manual_seed(42);return net,opt,g
 def epoch(net,opt,g,n):
  row=m.epoch(net,opt,data,tr,g,n);p=m.generated(net,data,inner)[0];row["inner12_Q"]=m.objective_numpy(p,data["gt"][inner])[0];return row
 def snapshot(net,opt,g,h):return dict(model=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=m.rng_state(g),history=copy.deepcopy(h))
 net,opt,g=fresh();hist=[epoch(net,opt,g,k) for k in (1,2)];reference=snapshot(net,opt,g,hist);del net,opt
 net,opt,g=fresh();hist=[epoch(net,opt,g,1)];p=c.H/"preflight_smoke/resume.pt";p.parent.mkdir(parents=True,exist_ok=True);m.checkpoint(p,net,opt,g,hist,manifest);del net,opt
 net,opt,g=fresh();hist=m.restore(p,net,opt,g,manifest);hist.append(epoch(net,opt,g,2));exact(reference,snapshot(net,opt,g,hist))
 try:m.restore(p,net,opt,g,{**manifest,"source_sha":"bad"})
 except AssertionError:pass
 else:raise AssertionError("BAD_RESUME_SOURCE_ACCEPTED")
 return dict(actualFIT513_inner12_twoepoch_ALLparams_boolbuffers_basis_optimizer_CPU_CUDA_loader_numpy_pythonRNG_history_EXACT=True,not_fullFIT2epochs_or_analysisresume=True)
def fd_fixture(data):
 net=copy.deepcopy(m.fresh(42)).double()
 with torch.no_grad():net.goal.centers.copy_(torch.stack((torch.arange(20,dtype=torch.float64,device=m.DEVICE)*2,torch.zeros(20,dtype=torch.float64,device=m.DEVICE)),1))
 ix=data["split"]["FIT"][:4];p2=torch.from_numpy(data["p2"][ix]).to(m.DEVICE).double();vel=torch.from_numpy(data["vel"][ix]).to(m.DEVICE)
 with torch.no_grad():base=net.goal(p2,vel)[0];gt=base[:,0]+torch.tensor([.0313,.0025],device=m.DEVICE)
 param=net.process[-1].weight;z=m.objective(net(p2,vel)[0],gt);gradient=torch.autograd.grad(z[0],param)[0][0,0].item();assert np.isfinite(gradient) and abs(gradient)>1e-12;original=param[0,0].item();result=[]
 for eps in (1e-4,1e-5):
  val=[];winner=[];face=[]
  for direction in (1,-1):
   with torch.no_grad():param[0,0]=original+direction*eps
   path=net(p2,vel)[0];loss=m.objective(path,gt);val.append(loss[0].item());winner.append(loss[-1].detach().cpu().numpy());pairs=torch.triu_indices(20,20,1,device=m.DEVICE);face.append((torch.linalg.vector_norm(path[:,pairs[0]]-path[:,pairs[1]],dim=-1).mean(-1)<.3).detach().cpu().numpy())
  assert np.array_equal(winner[0],winner[1]) and np.array_equal(face[0],face[1]),"FD_PIECEWISE_FACE_SWITCH";finite=(val[0]-val[1])/(2*eps);error=abs(finite-gradient)/(1+abs(gradient));assert error<=1e-8,(eps,error);result.append(dict(eps=eps,error=error,analytic=gradient,no_winner_hinge_face_switch=True))
 return dict(fixture="isolated_double_copy_fixed_synthetic_centres_2k0_not_saved_scientific_model_or_float32quantization_proof",values=result)
def main():
 started=time.perf_counter();c.guard();assert torch.cuda.is_available();files,inheritance=c.locks();data=c.load();tr=data["split"]["FIT"];ix=data["split"]["outer"][:128];stats_checks=[]
 for seed in r.SEEDS:
  parent=c.npz(c.OLD/f"runs/seed_{seed}/FIT.npz");assert np.array_equal(parent["sample_id"],data["ids"][tr]);g=parent["goals"];mean=g.mean((0,1));std=g.std((0,1));stats=dict(mean=mean,std=np.where(std<1e-6,1.,std),training_ids=parent["sample_id"])
  flat=g.reshape(-1,2);ss=np.zeros(2);square=np.zeros(2)
  for first in range(0,len(flat),256):z=flat[first:first+256];ss+=z.sum(0);square+=(z*z).sum(0)
  e=max(float(np.max(abs(ss/len(flat)-mean))),float(np.max(abs(np.sqrt(np.maximum(0,square/len(flat)-(ss/len(flat))**2))-std))));assert e<=1e-11
  name=f"goal_stats_{seed}.npz";c.save(name,**stats);files[str(c.H/name)]=c.sha(c.H/name);stats_checks.append(dict(seed=seed,n_candidates=len(flat),chunk256_independent_mean_std_max_error=e,training_only=True));del parent,g,flat
 c.save("split_ids.npz",**c.npz(c.OLD/"split_ids.npz"));files[str(c.H/"split_ids.npz")]=c.sha(c.H/"split_ids.npz")
 tests=[];gradient_rows=[]
 for seed in r.SEEDS:
  net=m.fresh(seed);assert sum(x.numel() for x in net.parameters() if x.requires_grad)==5572;assert list(inspect.signature(net.forward).parameters)==["p2","velocity"];frozen=m.tensor_hash(net.goal.state_dict());p,goal,ab,base=m.generated(net,data,ix);a=c.npz(c.OLD/f"runs/seed_{seed}/outer.npz");expected={k:z[:128] if z.ndim and len(z)==len(data["split"]["outer"]) else z for k,z in a.items()};assert np.array_equal(p,base);r.geometry(p,goal,ab,base,expected)
  first=tr[:256];p2=torch.from_numpy(data["p2"][first]).to(m.DEVICE);vel=torch.from_numpy(data["vel"][first]).to(m.DEVICE);gt=torch.from_numpy(data["gt"][first]).to(m.DEVICE);path=net(p2,vel)[0];loss=m.objective(path,gt);scalar,w=m.objective_numpy(path.detach().cpu().numpy(),data["gt"][first]);assert abs(loss[0].item()-scalar)<=1e-12 and np.array_equal(w,loss[-1].cpu().numpy());opt=r.optimizer(net);opt.zero_grad();loss[0].backward();initial={k:float(torch.linalg.vector_norm(x.grad).item()) for k,x in net.process.named_parameters()};assert initial["2.weight"]>0 and initial["2.bias"]>0;opt.step();grads={}
  for step in range(8):
   ind=tr[step*256:(step+1)*256];opt.zero_grad(set_to_none=True);z=net(torch.from_numpy(data["p2"][ind]).to(m.DEVICE),torch.from_numpy(data["vel"][ind]).to(m.DEVICE))[0];zz=m.objective(z,torch.from_numpy(data["gt"][ind]).to(m.DEVICE))[0];zz.backward()
   for k,x in net.process.named_parameters():assert x.grad is not None and torch.isfinite(x.grad).all();grads[k]=max(grads.get(k,0),float(torch.linalg.vector_norm(x.grad).item()))
   assert all(x.grad is None for x in net.goal.parameters());opt.step()
  assert all(x>0 for x in grads.values()) and m.tensor_hash(net.goal.state_dict())==frozen;state=m.tensor_hash(net.state_dict());p,goal,ab,base=m.generated(net,data,ix);r.geometry(p,goal,ab,base,expected);assert m.tensor_hash(net.state_dict())==state
  met,aux=v.evaluate(p,data["gt"][ix],data["groups"][ix],data["native"][ix].astype(float));ind,ia=v.evaluate(p,data["gt"][ix],data["groups"][ix],data["native"][ix].astype(float),True);c.eq(met,ind,2e-11)
  for k in aux:
   if aux[k].dtype.kind in "biu":assert np.array_equal(aux[k],ia[k])
   else:assert np.max(abs(aux[k]-ia[k]))<=2e-11
  c.save(f"preflight_smoke/seed_{seed}.npz",**v.sample(data,ix,p,goals=goal,controls=ab,base_paths=base));tests.append(dict(seed=seed,zero_A_B_EXACT=True,all128_endpoints_FDE_EXACT=True,G1_all_parameters_buffers_and_no_grad_EXACT=True,scalar_error=abs(loss[0].item()-scalar),saved_full_clusters_motion_independent=True));gradient_rows.append(dict(seed=seed,zero_last_phase=initial,actual8FIT256_all_process_gradients_positive_finite=grads))
 fd=fd_fixture(data);resume=resume_fixture(data);path=str(c.H/"preflight_smoke/seed_42.npz");done=dict(status="COMPLETE",source_sha="fixture",files={path:c.sha(path)});assert c.done_valid(done,"fixture")
 for bad in (dict(done,source_sha="bad"),dict(done,files={path:"bad"})):
  try:c.done_valid(bad,"fixture")
  except AssertionError:pass
  else:raise AssertionError("BAD_SOURCE_OUTPUT_ACCEPTED")
 net=m.fresh(42);opt=r.optimizer(net);times=[];first=tr[:256];p2=torch.from_numpy(data["p2"][first]).to(m.DEVICE);vel=torch.from_numpy(data["vel"][first]).to(m.DEVICE);gt=torch.from_numpy(data["gt"][first]).to(m.DEVICE)
 for step in range(36):
  torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True);loss=m.objective(net(p2,vel)[0],gt)[0];loss.backward();opt.step();torch.cuda.synchronize()
  if step>=4:times.append(time.perf_counter()-t)
 p,goal,ab,base=m.generated(net,data,ix);t=time.perf_counter();v.evaluate(p,data["gt"][ix],data["groups"][ix],data["native"][ix].astype(float));v.evaluate(p,data["gt"][ix],data["groups"][ix],data["native"][ix].astype(float),True);dual=(time.perf_counter()-t)/128;t=time.perf_counter();v.evaluate(p,data["gt"][ix],data["groups"][ix],data["native"][ix].astype(float),True);audit=(time.perf_counter()-t)/128;rows=sum(len(z) for z in data["split"].values())*3;nominal=5550*np.mean(times)+rows*(dual+audit);planning=(nominal*1.25+150)/60;assert planning<=30,("STOP_BUDGET",planning);c.verify(files);c.guard()
 c.write("PREFLIGHT.json",dict(status="PASS",source_sha=c.digest(files),source_count=len(files),inheritance=inheritance,all3_actual_GPU=tests,all3_actual_gradient=gradient_rows,perseed_FIT_goal_stats=stats_checks,fixed_double_FD=fd,real_twoepoch_resume=resume,done_badsource_output_reject=True,trainable_parameters=5572,optimization_steps_total=5550,FIT36_4warm32measure_seconds=float(np.mean(times)),dualquality_seconds_perwindow=dual,auditquality_seconds_perwindow=audit,nominal_compute_saved_audit_seconds=nominal,planning_minutes=planning,ETA_minutes=[max(3,planning*.7),planning*1.5],disk_free_GiB=shutil.disk_usage(c.H).free/2**30,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-started,no_scientific_benchmark_checkpoint_or_outcome_choice=True,preflight_not_fullFIT2epochs_or_allold_GPUreplay=True));print("PREFLIGHT_PASS",planning,flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:traceback.print_exc();c.write("failures/preflight1/FAILURE.json",dict(status="STOP",error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise

