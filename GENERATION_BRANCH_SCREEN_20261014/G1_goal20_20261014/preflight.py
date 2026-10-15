import common as c,model as m,metrics as v,baselines as b
import numpy as np,torch,time,copy,inspect,gc,traceback,shutil
from pathlib import Path
def exact(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b),"TENSOR_RESUME_NOT_EXACT"
 elif isinstance(a,np.ndarray):assert np.array_equal(a,b),"NP_RESUME_NOT_EXACT"
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:exact(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):exact(x,y)
 else:assert a==b,(a,b)
def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,foreach=False)
def resume_fixture(data,st):
 indices=data["split"]["FIT"][:513];inner=data["split"]["inner"][:12];man=dict(source_sha="preflightfixture",train_ids=data["ids"][indices].tolist(),epochs=2)
 def fresh():
  m.seed(20261014);net=m.Goal20(st).to(m.DEVICE);opt=optimizer(net);g=torch.Generator().manual_seed(20261014);return net,opt,g
 def one(net,opt,g,index):
  row=m.epoch(net,opt,data,indices,g,index);paths=m.generated(net,data,inner)[0];row["inner12_posthoc_Q"]=m.objective_numpy(paths,data["gt"][inner])[0];return row
 net,opt,g=fresh();h=[one(net,opt,g,i) for i in (1,2)];reference=dict(model=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=m.rng_state(g),history=copy.deepcopy(h));del net,opt
 net,opt,g=fresh();hist=[one(net,opt,g,1)];path=c.H/"preflight_smoke/resume.pt";path.parent.mkdir(parents=True,exist_ok=True);m.checkpoint(path,net,opt,g,hist,man);del net,opt
 net,opt,g=fresh();hist=m.restore(path,net,opt,g,man);hist.append(one(net,opt,g,2));exact(reference,dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=m.rng_state(g),history=hist))
 try:m.restore(path,net,opt,g,{**man,"source_sha":"bad"})
 except AssertionError:pass
 else:raise AssertionError("BAD_RESUME_SOURCE_ACCEPTED")
 return dict(actualFIT513_inner12_twoepoch_ALLparams_buffers_optimizer_CPU_CUDA_loader_numpy_pythonRNG_history_EXACT=True,not_fullFIT2epochs=True)
def fixture_fd(st,data):
 local={k:v.copy() for k,v in st.items()};local["centers"]=np.stack((np.arange(20)/10.,np.arange(20)**2/200.),1)
 m.seed(20261014);net=m.Goal20(local).to(m.DEVICE).double();p2=torch.from_numpy(data["p2"][data["split"]["FIT"][:4]]).to(m.DEVICE).double();velocity=torch.zeros((4,2),dtype=torch.float64,device=m.DEVICE);s=torch.arange(1,13,dtype=torch.float64,device=m.DEVICE)/12;gt=s[None,:,None]*torch.tensor([.313,.025],dtype=torch.float64,device=m.DEVICE)[None,None]
 gt=gt.expand(4,-1,-1);param=net.head[-1].weight
 loss=m.objective(net(p2,velocity)[0],gt)[0];grad=torch.autograd.grad(loss,param)[0][0,0].item();assert np.isfinite(grad);before=param[0,0].item();results=[]
 for eps in (1e-4,1e-5):
  values=[];winners=[]
  for direction in (1,-1):
   with torch.no_grad():param[0,0]=before+direction*eps
   p=net(p2,velocity)[0];z=m.objective(p,gt);values.append(z[0].item());winners.append(z[-1].detach().cpu().numpy())
  assert np.array_equal(winners[0],winners[1]),"FD_WINNER_SWITCH"
  fd=(values[0]-values[1])/(2*eps);err=abs(fd-grad)/(1+abs(grad));assert err<=1e-8,("FD_ERROR",eps,err);results.append(dict(eps=eps,relative_error=err,winner_switch=False))
 with torch.no_grad():param[0,0]=before
 return results
def main():
 started=time.perf_counter();c.guard();assert torch.cuda.is_available(),"STOP_NO_GPU";files,meta=c.locks();d=c.load();st=c.stats(d);st2=c.stats(d);exact(st,st2);c.save("statistics.npz",**st);c.save("split_ids.npz",**{k:d["ids"][ix] for k,ix in d["split"].items()})
 for name in ("statistics.npz","split_ids.npz"):files[str(c.H/name)]=c.sha(c.H/name)
 m.seed(42);net=m.Goal20(st).to(m.DEVICE);params=sum(x.numel() for x in net.parameters());assert params==26306
 tr=d["split"]["FIT"];first=tr[:256];p2=torch.from_numpy(d["p2"][first]).to(m.DEVICE);vel=torch.from_numpy(d["vel"][first]).to(m.DEVICE);gt=torch.from_numpy(d["gt"][first]).to(m.DEVICE)
 zero=net(p2,vel)[0].detach().cpu().numpy();proto=b.build(d,st,first,"prototype20")[0];assert np.array_equal(zero,proto),"INITIAL_PROTO_IDENTITY"
 assert list(inspect.signature(net.forward).parameters)==["p2","velocity"]
 loss,q,div,w=m.objective(net(p2,vel)[0],gt);nl,nw=m.objective_numpy(zero,d["gt"][first]);assert abs(loss.item()-nl)<=1e-12 and np.array_equal(w.cpu().numpy(),nw)
 scalar_error=abs(loss.item()-nl)
 loss.backward();firstgrads={k:float(torch.linalg.vector_norm(x.grad).item()) for k,x in net.named_parameters()};assert firstgrads["head.2.weight"]>0 and firstgrads["head.2.bias"]>0
 opt=optimizer(net);opt.step();grads={}
 for step in range(1,9):
  ix=tr[(step-1)*256:step*256];opt.zero_grad(set_to_none=True);out=net(torch.from_numpy(d["p2"][ix]).to(m.DEVICE),torch.from_numpy(d["vel"][ix]).to(m.DEVICE))[0];loss=m.objective(out,torch.from_numpy(d["gt"][ix]).to(m.DEVICE))[0];loss.backward()
  for k,x in net.named_parameters():
   assert x.grad is not None and torch.isfinite(x.grad).all();grads[k]=max(grads.get(k,0),float(torch.linalg.vector_norm(x.grad).item()))
  opt.step()
 assert all(value>0 for value in grads.values()),("GRAD_CONNECTION",grads)
 frozen={k:x.detach().cpu().clone() for k,x in net.state_dict().items()};evalix=d["split"]["outer"][:128];paths,goals,delta=m.generated(net,d,evalix);assert all(torch.equal(x.detach().cpu(),frozen[k]) for k,x in net.state_dict().items())
 met,aux=v.evaluate(paths,d["gt"][evalix],d["groups"][evalix],d["native"][evalix].astype(float));imet,iaux=v.evaluate(paths,d["gt"][evalix],d["groups"][evalix],d["native"][evalix].astype(float),True);c.eq(met,imet,2e-11);assert np.array_equal(aux["clusters"],iaux["clusters"])
 c.save("preflight_smoke/saved.npz",**v.sample(d,evalix,paths,goals=goals,goal_delta=delta));p=str(c.H/"preflight_smoke/saved.npz");done=dict(status="COMPLETE",source_sha="fixture",files={p:c.sha(p)});assert c.done_valid(done,"fixture")
 for bad in (dict(done,source_sha="bad"),dict(done,files={p:"bad"})):
  try:c.done_valid(bad,"fixture")
  except AssertionError:pass
  else:raise AssertionError("BAD_DONE_ACCEPTED")
 fd=fixture_fd(st,d);resume=resume_fixture(d,st)
 m.seed(42);net=m.Goal20(st).to(m.DEVICE);opt=optimizer(net);times=[]
 for step in range(36):
  torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True);loss=m.objective(net(p2,vel)[0],gt)[0];loss.backward();opt.step();torch.cuda.synchronize()
  if step>=4:times.append(time.perf_counter()-t)
 # Full current dualsaved-quality and single audit math throughput, not scientific outcome selection.
 t=time.perf_counter();v.evaluate(paths,d["gt"][evalix],d["groups"][evalix],d["native"][evalix].astype(float));v.evaluate(paths,d["gt"][evalix],d["groups"][evalix],d["native"][evalix].astype(float),True);metric_seconds=(time.perf_counter()-t)/128
 t=time.perf_counter();v.evaluate(paths,d["gt"][evalix],d["groups"][evalix],d["native"][evalix].astype(float),True);audit_seconds=(time.perf_counter()-t)/128
 t=time.perf_counter();retrieved,donors=b.build(d,st,tr[:8],"retrieval20");retrieval_seconds=(time.perf_counter()-t)/8
 assert all(len(np.unique(row))==20 for row in donors)
 lut={int(sid):i for i,sid in enumerate(d["ids"])}
 for i,row in zip(tr[:8],donors):
  donorix=np.array([lut[int(sid)] for sid in row]);assert np.isin(donorix,tr).all() and len(np.unique(d["track"][donorix]))==20 and not np.any(d["track"][donorix]==d["track"][i]);ref=d["gt"][donorix].astype(float)-d["vel"][donorix,None]*(.4*np.arange(1,13))[None,:,None]+d["vel"][i,None]*(.4*np.arange(1,13))[None,:,None];assert np.array_equal(ref,retrieved[np.where(tr[:8]==i)[0][0]])
 from audit_saved import retrieval_certificate
 t=time.perf_counter();retrieval_certificate(d,st,tr[:8],donors,retrieved);retrieval_audit_seconds=(time.perf_counter()-t)/8
 steps=75*int(np.ceil(18909/256));rows=sum(len(ix) for ix in d["split"].values());nominal=steps*np.mean(times)+rows*6*(metric_seconds+audit_seconds)+rows*(retrieval_seconds+retrieval_audit_seconds)
 planning=(nominal*1.25+150)/60;assert planning<=30,("STOP_BUDGET",planning)
 c.write("BENCHMARK.json",dict(status="PASS",independent_retrieval_audit_seconds_perwindow=retrieval_audit_seconds,full_saved_FIT_included=True,production_steps=steps,unique_rows_across_splits=rows,no_saved_benchmark_model_or_scientific_outcome_choice=True))
 c.verify(files);c.guard();c.write("PREFLIGHT.json",dict(status="PASS",source_sha=c.digest(files),source_count=len(files),inheritance=meta,actual_parameter_count=params,train_only_stats_kmeans_repeat_EXACT=True,initial_prototype20_identity_EXACT=True,initial_center_distinct=20,initial_zero_last_gradients=firstgrads,actual8GPUFIT256_alllegal_gradients_positive_finite=grads,canonical20pair190_numpy_scalar_error=scalar_error,fixed_double_parameter_FD=fd,resume=resume,saved128_full_clusters_motion_independent_PASS=True,done_source_output_reject_PASS=True,FIT_step36_4warm32measure_seconds=float(np.mean(times)),dualquality_seconds_perwindow=metric_seconds,auditquality_seconds_perwindow=audit_seconds,retrieval_seconds_perwindow=retrieval_seconds,nominal_compute_saved_audit_seconds=nominal,planning_minutes=planning,ETA_minutes=[max(3,planning*.7),planning*1.5],disk_free_GiB=shutil.disk_usage(c.H).free/2**30,elapsed_seconds=time.perf_counter()-started,resource=c.RESOURCE,benchmark_checkpoint_or_outcome_choice=False,not_all_old_GPU_replay=True));print("PREFLIGHT_PASS",planning,flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:traceback.print_exc();c.write("failures/preflight2/FAILURE.json",dict(status="STOP",error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
