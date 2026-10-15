"""T1 numerical/source/resume/resource gates before any production launch."""
import common as c,model as m,metrics as v,prepare as prep
import numpy as np,torch,time,copy,inspect,ast,gc,shutil,traceback,hashlib

def exact(a,b):
 if torch.is_tensor(a):assert torch.equal(a,b),'TENSOR_NOT_EXACT'
 elif isinstance(a,np.ndarray):assert np.array_equal(a,b),'ARRAY_NOT_EXACT'
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:exact(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):exact(x,y)
 else:assert a==b,(a,b)

def reject(call):
 try:call()
 except AssertionError:return True
 raise AssertionError('BAD_INPUT_ACCEPTED')

def resume(data,st,arm):
 ix=data['split']['FIT'][:513];inner=data['split']['inner'][:12]
 man=dict(source_sha='fixture',arm=arm,FIT_ids=data['ids'][ix].tolist(),label_sha=c.digest(data['fixed_labels'][ix]))
 def fresh():
  m.seed(20261015);net=m.Pattern20(st).to(m.DEVICE)
  return net,m.optimizer(net),torch.Generator().manual_seed(20261015)
 def one(net,opt,g,ep):
  row=m.epoch(net,opt,data,ix,g,ep,arm);p,_=m.generated(net,data,inner)
  row['inner12_Q_posthoc']=m.objective_numpy(p,data['gt'][inner],m.ARMS[0])[0];return row
 net,opt,g=fresh();h=[one(net,opt,g,i) for i in (1,2)]
 ref=dict(model=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=m.rng_state(g),history=copy.deepcopy(h));del net,opt
 net,opt,g=fresh();h=[one(net,opt,g,1)];path=c.H/f'preflight_smoke/{arm}_resume.pt';m.checkpoint(path,net,opt,g,h,man);del net,opt
 net,opt,g=fresh();h=m.restore(path,net,opt,g,man);h.append(one(net,opt,g,2))
 exact(ref,dict(model=net.state_dict(),optimizer=opt.state_dict(),rng=m.rng_state(g),history=h))
 reject(lambda:m.restore(path,net,opt,g,{**man,'source_sha':'bad'}))
 return dict(status='PASS',actual_FIT513_inner12_SUBSET_twoepoch_ALLparams_boolscalers_centers_optimizer_TorchCPU_CUDA_loader_NumPy_PythonRNG_history_EXACT=True,not_fullFIT_or_analysis_resume=True)

def fd(st,data,arm):
 # Fixed isolated synthetic centers/GT, not altered scientific prototypes.
 local={k:x.copy() for k,x in st.items()};s=np.arange(1,13)/12.
 local['centers']=np.stack([s[:,None]*np.array([3.*k,.17*k*k])[None] for k in range(20)])
 m.seed(20261015);net=m.Pattern20(local).to(m.DEVICE).double()
 with torch.no_grad():net.head[-1].weight.normal_(0,.0005)
 ix=data['split']['FIT'][:4];p2,vel=m.inputs(data,ix);p2=p2.double();vel=torch.zeros_like(vel)
 gt=torch.from_numpy(np.broadcast_to((s[:,None]*np.array([.173,.051])[None])[None],(4,12,2)).copy()).to(m.DEVICE)
 labels=torch.zeros(4,device=m.DEVICE,dtype=torch.int64);answer=[]
 for name,index in (('head.4.weight',(0,0)),('head.2.weight',(0,0)),('head.0.weight',(0,0))):
  parameter=dict(net.named_parameters())[name];out=net(p2,vel)[0];loss=m.objective(out,gt,arm,labels)[0]
  analytical=torch.autograd.grad(loss,parameter)[0][index].item();before=parameter[index].item()
  for eps in (1e-4,1e-5):
   values=[];faces=[]
   for sign in (1,-1):
    with torch.no_grad():parameter[index]=before+sign*eps
    p=net(p2,vel)[0];obj=m.objective(p,gt,arm,labels);values.append(obj[0].item())
    pairs=torch.triu_indices(20,20,1,device=m.DEVICE);hinge=torch.linalg.vector_norm(p[:,pairs[0]]-p[:,pairs[1]],dim=-1).mean(-1)<.3
    faces.append((obj[3].detach().cpu().numpy(),hinge.detach().cpu().numpy()))
   assert all(np.array_equal(a,b) for a,b in zip(*faces)),('FD_ACTIVE_FACE_SWITCH',arm,name,eps)
   error=abs((values[0]-values[1])/(2*eps)-analytical)/(1+abs(analytical))
   assert error<=1e-8,('FD_ERROR',arm,name,eps,error)
   answer.append(dict(parameter=name,eps=eps,error=error,analytic=analytical,no_active_face_switch=True))
  with torch.no_grad():parameter[index]=before
 return answer

def job(data,ix,paths,correction,name):
 z=v.sample(data,ix,paths,correction=correction,velocity=data['vel'][ix]);c.save(name,**z);z=c.npz(c.H/name)
 a,aa=v.evaluate(z['paths'],z['gt'],z['groups'],z['native20']);b,bb=v.evaluate(z['paths'],z['gt'],z['groups'],z['native20'],True)
 c.eq(c.safe(a),c.safe(b),2e-11)
 for k,value in aa.items():
  if value.dtype.kind in 'bui':assert np.array_equal(value,bb[k])
  else:assert np.max(abs(value-bb[k]))<=2e-11
 a,aa=v.enrich(z,a,aa);b,bb=v.enrich(z,b,bb);c.eq(c.safe(a),c.safe(b),2e-11)
 c.save(name.replace('.npz','_metrics.npz'),sample_id=z['sample_id'],groups=z['groups'],track=z['track'],**aa)
 return z,a,aa

def main():
 started=time.perf_counter();c.guard();assert torch.cuda.is_available()
 assert c.read(c.H/'dictionary_review/REVIEW.json')['status']=='PASS'
 assert c.read(c.H/'DICTIONARY_PREFLIGHT.json')['status']=='PASS_DICTIONARY_SUPPORT'
 t=time.perf_counter();files,meta=c.locks()
 for name in ('dictionary.npz','dictionary_trace.json','DICTIONARY_PREFLIGHT.json','dictionary_source_manifest.json','dictionary_review/REVIEW.json','split_ids.npz'):
  path=c.H/name;files[str(path)]=c.sha(path)
 sha_seconds=time.perf_counter()-t
 data=c.load();dictionary=c.npz(c.H/'dictionary.npz');prep.attach(data,dictionary);st=prep.stats(data,dictionary);staterr=prep.chunk_error(data,dictionary,st)
 c.save('statistics.npz',**st);files[str(c.H/'statistics.npz')]=c.sha(c.H/'statistics.npz')
 tr=data['split']['FIT'];ix=tr[:256];args=m.inputs(data,ix);gt=torch.from_numpy(data['gt'][ix]).to(m.DEVICE);labels=torch.from_numpy(data['fixed_labels'][ix]).to(m.DEVICE)
 fixed=np.array(sorted(tr,key=lambda i:hashlib.sha256(('t1-fixed-input128-v1:'+str(data['ids'][i])).encode()).digest())[:128])
 c.save('fixed_feature_fixture.npz',indices=fixed,sample_id=data['ids'][fixed],p2=data['p2'][fixed],velocity=data['vel'][fixed],labels=data['fixed_labels'][fixed])
 files[str(c.H/'fixed_feature_fixture.npz')]=c.sha(c.H/'fixed_feature_fixture.npz')
 reports={};initials=[];times=[];dual=[];audits=[];generation=[];saved_bytes=[]
 for arm in m.ARMS:
  m.seed(42);net=m.Pattern20(st).to(m.DEVICE);assert sum(p.numel() for p in net.parameters())==25560
  assert list(inspect.signature(net.forward).parameters)==['p2','velocity']
  initials.append(copy.deepcopy(net.state_dict()));out,delta=net(*args)
  P0=np.arange(1,13)[None,None,:,None]*.4*data['vel'][ix,None,None]+st['centers'][None]
  assert np.array_equal(out.detach().cpu().numpy(),P0) and not torch.count_nonzero(delta)
  obj=m.objective(out,gt,arm,labels);numeric,qw,zz=m.objective_numpy(P0,data['gt'][ix],arm,data['fixed_labels'][ix])
  scalar_error=abs(numeric-obj[0].item());assert scalar_error<=1e-12
  assert np.array_equal(qw,obj[3].detach().cpu().numpy()) and np.array_equal(zz,obj[4].detach().cpu().numpy())
  # Same registered dynamic objective AST, all path-coordinate gradients EXACT.
  x=out.detach().clone().requires_grad_(True);canonical=m.legacy.objective(x,gt)[0];control=m.objective(x,gt,m.ARMS[0])[0]
  assert torch.equal(canonical,control) and torch.equal(torch.autograd.grad(canonical,x,retain_graph=True)[0],torch.autograd.grad(control,x)[0])
  # Fixed loss reference uses NumPy-selected indices, not production label selection.
  x=out.detach().clone().requires_grad_(True);e=torch.linalg.vector_norm(x-gt.double()[:,None],dim=-1);Q=e.mean(-1)+.5*e[:,:,-1]
  ref=Q[torch.arange(len(x),device=m.DEVICE),torch.as_tensor(zz,device=m.DEVICE)].mean()
  pair=torch.triu_indices(20,20,1,device=m.DEVICE);dist=torch.linalg.vector_norm(x[:,pair[0]]-x[:,pair[1]],dim=-1).mean(-1)
  ref=ref+.05*torch.relu(.3-dist).square().mean();actual=m.objective(x,gt,arm,labels)[0]
  assert torch.equal(ref,actual) and torch.equal(torch.autograd.grad(ref,x,retain_graph=True)[0],torch.autograd.grad(actual,x)[0])
  obj[0].backward();first={k:float(p.grad.norm().item()) for k,p in net.named_parameters()};assert first['head.4.weight']>0 and first['head.4.bias']>0
  assert all(v==0 for k,v in first.items() if not k.startswith('head.4'))
  opt=m.optimizer(net);opt.step();connected={}
  for i in range(8):
   sub=tr[i*256:(i+1)*256];opt.zero_grad(set_to_none=True);p=net(*m.inputs(data,sub))[0];fixedlabels=torch.from_numpy(data['fixed_labels'][sub]).to(m.DEVICE)
   loss=m.objective(p,torch.from_numpy(data['gt'][sub]).to(m.DEVICE),arm,fixedlabels)[0];loss.backward()
   for k,param in net.named_parameters():
    assert param.grad is not None and torch.isfinite(param.grad).all();connected[k]=max(connected.get(k,0),float(param.grad.norm().item()))
   opt.step()
  assert all(n>0 for n in connected.values()),('DISCONNECTED_PARAMETER',connected)
  frozen=copy.deepcopy(net.state_dict());outer=data['split']['outer'][:128]
  t=time.perf_counter();p,delta=m.generated(net,data,outer);generation.append((time.perf_counter()-t)/128)
  t=time.perf_counter();z,met,aux=job(data,outer,p,delta,f'preflight_smoke/{arm}.npz');dual.append((time.perf_counter()-t)/128)
  t=time.perf_counter();v.evaluate(z['paths'],z['gt'],z['groups'],z['native20'],True);audits.append((time.perf_counter()-t)/128)
  saved_bytes.append((c.H/f'preflight_smoke/{arm}.npz').stat().st_size/128)
  assert np.array_equal(v.fast.legacy_clusters(p,aux['good_candidate'],True),aux['clusters'])
  exact(frozen,net.state_dict())
  file=str(c.H/f'preflight_smoke/{arm}.npz');done=dict(status='COMPLETE',source_sha='fixture',files={file:c.sha(file)})
  assert c.base.done_valid(done,'fixture');reject(lambda:c.base.done_valid(done,'bad'));reject(lambda:c.base.done_valid({**done,'files':{file:'bad'}},'fixture'))
  fds=fd(st,data,arm);res=resume(data,st,arm)
  m.seed(42);bench=m.Pattern20(st).to(m.DEVICE);op=m.optimizer(bench);bt=[]
  for j in range(36):
   c.guard();torch.cuda.synchronize();t=time.perf_counter();m.step(bench,op,data,ix,arm);torch.cuda.synchronize()
   if j>=4:bt.append(time.perf_counter()-t)
  times.append(float(np.mean(bt)))
  reports[arm]=dict(parameters=25560,initial_all_buffers_weights_P0_paired_EXACT=True,zero_last_first_gradient=first,GPU8FIT256_phased_legal_gradients_positive_finite=connected,canonical_A_all_scalar_and_output_gradient_EXACT=True,fixed_reference_scalar_and_output_gradient_EXACT=True,numpy_scalar_error=scalar_error,forward_has_no_GT_label_ID_native_logits=True,actual_saved128_full_quality_original_SciPy_clusters_motion_independent=True,all_weights_buffers_frozen_evaluation_EXACT=True,double_parameter_FD=fds,resume=res,bad_source_output_resume_contract_rejected=True)
  del net,opt,bench,op;gc.collect();torch.cuda.empty_cache()
 exact(initials[0],initials[1])
 # Full registered-size draws/hashes/aggregation timing, no scientific outcomes.
 T=len(np.unique(data['track'][tr]));rng=np.random.Generator(np.random.PCG64(20261015));fixture=np.arange(3*T*4,dtype=float).reshape(3,T,4)/1000.;cnt=np.ones(T,int)
 t=time.perf_counter()
 for j in range(512):
  ss=rng.integers(0,3,3);tt=rng.integers(0,T,T);hashlib.sha256(ss.astype('<i8').tobytes()+tt.astype('<i8').tobytes()).hexdigest();weights=np.bincount(tt,minlength=T)
  values=(fixture[:,np.nonzero(weights)[0]]*weights[weights>0][None,:,None]).sum(1)[ss].sum(0)/(3*np.sum(weights*cnt));assert np.isfinite(values).all()
 draw_seconds=(time.perf_counter()-t)/512
 rows=sum(map(len,data['split'].values()));training=3*1850*sum(times)
 # Six final populations + one deterministic P0 population. Both production
 # independent saved checks and the later full CPU audit are explicitly costed.
 quality=7*rows*(max(dual)+max(audits)+max(generation))
 boot=16*20000*draw_seconds*3*2+60.;nominal=training+quality+boot+2*sha_seconds;planning=(1.25*nominal+150)/60.
 disk=7*rows*max(saved_bytes)/2**30*1.5+2.;free=shutil.disk_usage(c.H).free/2**30
 benchmark=dict(FIT256_seconds_per_arm=times,per_batch_diagnostics_and_postupdate_forward_included=True,dual_saved_seconds_perwindow=dual,full_CPUaudit_seconds_perwindow=audits,generation_seconds_perwindow=generation,source_SHA_seconds=sha_seconds,bootstrap_and_full_rebuild_seconds=boot,training_seconds=training,full28_savedjobs_save_recompute_audit_seconds=quality,nominal_seconds=nominal,planning_minutes=planning,ETA_minutes=[.7*planning,1.5*planning],disk_required_GiB=disk,disk_free_GiB=free,no_saved_benchmarkmodel_or_scientific_outcome_choice=True)
 c.write('BENCHMARK.json',benchmark);assert free>disk+2,('STOP_DISK_BUDGET',free,disk)
 c.verify(files);c.guard()
 c.write('PREFLIGHT.json',dict(status='PASS_NUMERIC_AND_BUDGET' if planning<=30 else 'STOP_BUDGET',at=c.now(),numeric_status='PASS',dictionary_gate='PASS',dictionary_full_independent_saved_review='PASS',source_count=len(files),inheritance=meta,arms=reports,paired_initial_EXACT=True,stats_chunk_error=staterr,planning_minutes=planning,ETA_minutes=[.7*planning,1.5*planning],source_initial_final='PASS',resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-started,old_F16_decoder_scorer_not_instantiated_or_optimized=True,not_full_historical34k_or_alloldGPUreplay=True))
 c.write('preflight_source_snapshot.json',dict(files=files,source_sha=c.digest(files),not_production_lock=True))
 c.write('runtime_state.json',dict(status='NUMERIC_BUDGET_PASS_IMPLEMENT_FINAL_RUNNER' if planning<=30 else 'STOP_BUDGET_WAIT_HUMAN',at=c.now(),production_started=False,active_worker=False,numeric_preflight='PASS',dictionary_gate='PASS',completed_runs=0,planned_runs=6,planning_minutes=planning,ETA_minutes=[.7*planning,1.5*planning],registered_budget_minutes=30,T2_started=False,T3_started=False,whole_series_complete=False,next_action='IMPLEMENT_AND_REVIEW_FINAL_RUNNER' if planning<=30 else 'NO_LAUNCH_WITHOUT_THIS_ITEM_BUDGET_DECISION'))
 print(dict(status='PASS_NUMERIC_AND_BUDGET' if planning<=30 else 'STOP_BUDGET',planning_minutes=planning,resource=c.RESOURCE),flush=True)

if __name__=='__main__':
 try:main()
 except Exception:
  c.write('failures/preflight1/FAILURE.json',dict(status='STOP',at=c.now(),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
