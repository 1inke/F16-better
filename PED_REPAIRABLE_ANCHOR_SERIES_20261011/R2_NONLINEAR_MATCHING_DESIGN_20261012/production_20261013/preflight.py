import common as c,saved_math as sm
import numpy as np,time,importlib.util,traceback
def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);return mod
def certificates():
 # No legacy common/run/selector imported; these two pure modules have no top-level I/O.
 n=module("old_pure_numerics",c.OLD/"numerics.py");lab=module("old_pure_labels",c.OLD/"labels.py")
 d=c.npz(c.OLD/"sample_design.npz");s=c.npz(c.OLD/"solutions.npz");recs=c.read(c.OLD/"records.json");safe=np.zeros((384,20),bool);err=0.;disagreements=0
 for k in range(1022):
  if k%64==0:c.guard()
  i=int(s["window"][k]);j=int(s["slot"][k]);prod,p,b=n.validate(d["native"][i],d["gt"][i],j,s["raw32"][k]);ind,q,bank=n.independent_verify(d["native"][i],d["gt"][i],j,s["raw32"][k]);valid=bool(prod["witness"] and ind["witness"])
  assert valid==recs[k]["safe_witness"];safe[i,j]|=valid;err=max(err,float(abs(q-s["paths"][k]).max()));assert np.array_equal(bank[np.arange(20)!=j],d["native"][i].astype(float)[np.arange(20)!=j])
  disagreements+=int(prod["witness"]!=ind["witness"])
 y,cert=lab.build(d["native"],d["gt"],d["envelope"],safe);z=c.npz(c.OLD/"labels.npz");assert np.array_equal(y,z["labels"]) and np.array_equal(safe,z["verified_safe"]);assert err<=2e-12 and disagreements==1
 return dict(all1022_saved_paths_PNU_certificates_PASS=True,reconstruction_max=err,known_conservative_boundary_disagreements=disagreements,no_solver_rerun=True)
def reference_weights(d,tr):
 y=d["labels"][tr];tracks=d["track"][tr];pos=y==1;neg=y==0;_,inv,ct=np.unique(tracks,return_inverse=True,return_counts=True)
 w=(.5*pos/pos.sum(1,keepdims=True)+.5*neg/neg.sum(1,keepdims=True))/ct[inv,None];w*=pos.size/w.sum();return w
def check_linear(model,d,f):
 tr,va=c.ic.split_masks(d,f);old=c.npz(c.OLD/("witness/models/fold_"+str(f)+".npz"));target,known,w=c.m.witness_supervision(d["labels"][tr],d["track"][tr])
 assert np.array_equal(model.mapping.mean,old["mean"]) and np.array_equal(model.mapping.std,old["std"]) and np.array_equal(w,old["weight"]) and np.array_equal(known,old["known"])
 score=c.m.predict(d["X"][va],model,c.guard);err=float(abs(score-d["old_witness_scores"][va]).max());assert err<=1e-10 and np.array_equal(c.m.rank4(score),d["reference_slots"]["old_witness"][va])
 return dict(fold=f,score_max_error=err,stats_weights_EXACT=True,top4_EXACT=True)
def math_fixture(d,tr,arm):
 ix=np.flatnonzero(tr)[:21];x=d["X"][ix];y=d["labels"][ix];tracks=d["track"][ix];ids=d["ids"][ix]
 start=time.perf_counter();model,diag=c.m.fit_selector(x,y,tracks,ids,arm,guard=c.guard);again,_=c.m.fit_selector(x,y,tracks,ids,arm,guard=c.guard)
 assert all(np.array_equal(v,c.m.model_arrays(again)[k]) for k,v in c.m.model_arrays(model).items())
 a=c.m.model_arrays(model);scores=c.m.predict(x,model,c.guard);restored=c.m.restore_model(a);assert np.array_equal(scores,c.m.predict(x,restored,c.guard))
 target,known,w=c.m.witness_supervision(y,tracks);assert np.array_equal(w,reference_weights(d,np.isin(np.arange(384),ix)))
 assert (w[y==-1]==0).all();assert abs(w[y==1].sum()-w[y==0].sum())<1e-8
 sums=[w[tracks==v].sum() for v in np.unique(tracks)];assert max(sums)-min(sums)<1e-10
 f=c.m.transform(x,model.mapping,c.guard)[known];yy=target[known];sw=w[known];v=np.sin(np.arange(f.shape[1]+1)*.17)*.03
 def fg(vec):return c.m.objective_grad(f,yy,sw,vec[:-1],vec[-1])
 score=f@v[:-1]+v[-1];prob=np.exp(-np.logaddexp(0.,-score));aug=np.c_[f,np.ones(len(f))];h=aug.T@((sw*prob*(1-prob))[:,None]*aug)/sw.sum();h[:-1,:-1]+=np.eye(f.shape[1])/sw.sum()
 value,grad=fg(v);errors=[]
 for k in range(3):
  u=np.cos(np.arange(len(v))*(.11+k*.07));u/=np.linalg.norm(u);eps=1e-5
  ge=abs((fg(v+eps*u)[0]-fg(v-eps*u)[0])/(2*eps)-grad@u);he=float(abs((fg(v+eps*u)[1]-fg(v-eps*u)[1])/(2*eps)-h@u).max());assert ge<=1e-8 and he<=1e-8;errors.append([float(ge),he])
 if arm=="rbf64":
  z=((x.astype(float)-a["mean"])/a["std"]).reshape(-1,136)
  err=float(abs(c.m.rbf_values(z[:64],a["centers"],c.guard)-sm.rbf(z[:64],a["centers"])).max());assert err<=1e-12
  assert np.array_equal(c.m.rbf_values(z[:64],a["centers"],chunk=1),c.m.rbf_values(z[:64],a["centers"],chunk=31))
  _,keys=c.m.landmark_rows(ids[::-1]);assert np.array_equal(keys,a["center_keys"])
  before=c.digest(c.m.model_arrays(model));c.m.predict(x+1000,model,c.guard);assert before==c.digest(c.m.model_arrays(model))
 bad=a.copy();bad["coef"]=a["coef"].copy();bad["coef"][0]=np.nan
 try:c.m.restore_model(bad)
 except ValueError:pass
 else:raise AssertionError("bad model accepted")
 c.save("preflight_smoke/"+arm+".npz",**a);read=c.m.restore_model(c.npz(c.H/("preflight_smoke/"+arm+".npz")));assert np.array_equal(scores,c.m.predict(x,read,c.guard))
 files={str(c.H/("preflight_smoke/"+arm+".npz")):c.sha(c.H/("preflight_smoke/"+arm+".npz"))};done=dict(status="COMPLETE",source_sha="fixture",files=files);assert c.done_valid(done,"fixture",files)
 for bad_done in (dict(done,source_sha="bad"),dict(done,files={**files,"not_real":"bad"})):
  try:c.done_valid(bad_done,"fixture",files)
  except (AssertionError,FileNotFoundError):pass
  else:raise AssertionError("bad done accepted")
 return dict(arm=arm,repeat_fit_EXACT=True,saved_model_scores_EXACT=True,unknown_weights_zero=True,FD=errors,fit=diag,elapsed_seconds=time.perf_counter()-start)
def main():
 t=time.perf_counter();c.guard();start=time.perf_counter();files,counts=c.original_sources();sha_seconds=time.perf_counter()-start;cert=certificates();d=c.inputs();actual=[];times=[]
 for fold in range(3):
  tr,va=c.ic.split_masks(d,fold);start=time.perf_counter();model,diag=c.m.fit_selector(d["X"][tr],d["labels"][tr],d["track"][tr],d["ids"][tr],"linear",guard=c.guard);times.append(time.perf_counter()-start);actual.append(dict(**check_linear(model,d,fold),fit=diag))
 tr,_=c.ic.split_masks(d,0);fixtures=[math_fixture(d,tr,arm) for arm in ("linear","rbf64")]
 start=time.perf_counter();bench,diagnostic=c.m.fit_selector(d["X"][tr],d["labels"][tr],d["track"][tr],d["ids"][tr],"rbf64",guard=c.guard);c.m.predict(d["X"],bench,c.guard);rbf_seconds=time.perf_counter()-start
 a=c.m.model_arrays(bench);features=c.m.transform(d["X"][tr],bench.mapping,c.guard);raw=c.m.rbf_values(((d["X"][tr].astype(float)-a["mean"])/a["std"]).reshape(-1,136),a["centers"],c.guard)
 distribution=dict(phi_min=float(raw.min()),phi_max=float(raw.max()),phi_std_min=float(raw.std(0).min()),phi_std_max=float(raw.std(0).max()),constant_columns=int((raw.std(0)<1e-6).sum()),landmark_keys=a["center_keys"],not_outcome_selection=True)
 c.verify(files);c.guard();planning=(6*max(max(times),rbf_seconds)+4*sha_seconds+60)*1.5/60;assert planning<=30,"STOP_BUDGET"
 # No production fitted checkpoint/outcome kept from benchmark or preflight full controls.
 c.write("PREFLIGHT.json",dict(status="PASS",at=c.now(),counts=counts,source_files_before_lock=len(files),certificates=cert,linear_reproduction=actual,subset_fixtures=fixtures,full_rbf_throughput_seconds=rbf_seconds,full_linear_throughput_seconds=times,rbf_distribution=distribution,sha_seconds=sha_seconds,planning_minutes=planning,ETA_minutes=[max(2.,planning*.6),planning*1.8],elapsed_seconds=time.perf_counter()-t,versions=c.version(),resource=c.RESOURCE,no_solver_or_neural_training=True,no_presaved_scientific_checkpoint=True))
 print("PREFLIGHT_PASS",planning,flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:
  c.write("PREFLIGHT_FAILURE_"+str(time.time_ns())+".json",dict(status="FAILURE",error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
