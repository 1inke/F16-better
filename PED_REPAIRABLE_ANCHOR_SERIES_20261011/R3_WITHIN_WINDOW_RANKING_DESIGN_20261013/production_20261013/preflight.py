import common as c,saved_math as sm
import numpy as np,time,traceback
def reproduce(model,ztr,zva,d,train,valid,fold):
 a=c.npz(c.OLD/("rbf64/fold_"+str(fold)+"/model.npz"));pred=c.npz(c.OLD/("rbf64/fold_"+str(fold)+"/predictions.npz"))
 assert np.array_equal(model.coef,a["coef"]) and model.intercept==float(a["intercept"][0]),"STOP_POINTWISE_COEF_REPRODUCTION"
 s=c.k.score(zva,model);ts=c.k.score(ztr,model);assert np.array_equal(s,d["old_rbf_scores"][valid]) and np.array_equal(ts,pred["train_scores"]) and np.array_equal(s,pred["validation_scores"]),"STOP_POINTWISE_SCORE_REPRODUCTION"
 assert np.array_equal(c.k.rank4(s),d["reference_slots"]["old_rbf"][valid])
 contract=c.npz(c.OLD/("rbf64/fold_"+str(fold)+"/fit_contract.npz"));y,known,w=c.k.pointwise_weights(d["labels"][train],d["track"][train]);assert np.array_equal(w,contract["weight"]) and np.array_equal(y,contract["target"]) and np.array_equal(known,contract["known"]);return dict(fold=fold,coef_intercept_scores_top4_weights_EXACT=True,score_max_error=0.)
def fixture(z,y,tracks,arm):
 model,diag=c.k.fit_objective(z,y,tracks,arm,guard=c.guard);again,_=c.k.fit_objective(z,y,tracks,arm,guard=c.guard);assert np.array_equal(model.coef,again.coef) and model.intercept==again.intercept
 pairs=c.k.build_pairs(z,y,tracks,guard=c.guard);x,yy,sw=c.k.symmetric_rows(pairs);target,known,w=c.k.pointwise_weights(y,tracks);assert np.array_equal(w,sm.weights(y,tracks));pd,pw,wi,pp,nn=sm.pair_arrays(z,y,tracks)
 assert all(np.array_equal(a,b) for a,b in zip((pd,pw,wi,pp,nn),(pairs.delta,pairs.weight,pairs.window,pairs.positive,pairs.negative)))
 assert np.count_nonzero(pd[:,sm.SHARED])==0;assert np.array_equal(np.bincount(wi,weights=pw),w.sum(1)) or abs(np.bincount(wi,weights=pw)-w.sum(1)).max()<1e-10
 theta=np.sin(np.arange(201 if arm=="pointwise" else 200)*.17)*.03
 def fun(v):
  if arm=="pointwise":return c.k.logistic_objective(z[known],target[known],w[known],v[:-1],v[-1])
  return c.k.pair_objective(pairs,v)
 value,g,H=fun(theta);errors=[]
 if arm=="pairwise":
  direct=c.k.pair_objective(pairs,theta);symmetric=c.k.logistic_objective(x,yy,sw,theta,fit_intercept=False)
  assert all(np.max(np.abs(a-b))<=1e-12 for a,b in zip(direct,symmetric));mut=z.copy();mut[y==-1]+=300;qq=c.k.build_pairs(mut,y,tracks,guard=c.guard);assert np.array_equal(qq.delta,pd)
 for k in range(3):
  direction=np.cos(np.arange(len(theta))*(.11+k*.07));direction/=np.linalg.norm(direction)
  for eps in (1e-4,1e-5):
   plus=fun(theta+eps*direction);minus=fun(theta-eps*direction);ge=float(abs((plus[0]-minus[0])/(2*eps)-g@direction));he=float(abs((plus[1]-minus[1])/(2*eps)-H@direction).max());assert ge<=1e-8 and he<=1e-8,(arm,eps,ge,he);errors.append(dict(eps=eps,gradient_error=ge,hessian_error=he))
 independent,gr=sm.objective(z,y,tracks,arm,model.coef,model.intercept);assert abs(independent-diag["loss"])<=1e-12 and abs(gr).max()<=1e-6
 c.save("preflight_smoke/"+arm+".npz",coef=model.coef,intercept=np.array([model.intercept]),features=z,labels=y,tracks=tracks)
 a=c.npz(c.H/("preflight_smoke/"+arm+".npz"));rest=c.k.RankModel(arm,a["coef"],float(a["intercept"][0]));assert np.array_equal(c.k.score(z,model),c.k.score(z,rest))
 path=str(c.H/("preflight_smoke/"+arm+".npz"));done=dict(status="COMPLETE",source_sha="fixture",files={path:c.sha(path)});assert c.done_valid(done,"fixture")
 for bad in (dict(done,source_sha="bad"),dict(done,files={path:"0"*64})):
  try:c.done_valid(bad,"fixture")
  except AssertionError:pass
  else:raise AssertionError("bad source/output accepted")
 return dict(arm=arm,repeat_fit_EXACT=True,pair_contract_EXACT=True,shared_history_delta_ZERO=True,independent_loss_and_gradient_PASS=True,FD=errors,saved_score_EXACT=True,fit=diag)
def main():
 t=time.perf_counter();c.guard();ts=time.perf_counter();files,counts=c.sources();sha_seconds=time.perf_counter()-ts
 d,maps,core,contract=c.inputs();linear_checks=[];times=[];arrays=[]
 for fold in range(3):
  rebuilt=core.build_map(d["X"][contract.split_masks(d,fold)[0]],d["ids"][contract.split_masks(d,fold)[0]],"rbf64",c.guard);assert all(np.array_equal(getattr(rebuilt,k),getattr(maps[fold],k)) for k in c.maps_arrays(rebuilt))
  ztr,zva,tr,va=c.fi.fold_features(d,maps,core,contract,fold,guard=c.guard);a=c.maps_arrays(maps[fold]);ind,itr,iva=sm.rebuilt_features(d,fold,a);assert np.array_equal(tr,itr) and np.array_equal(va,iva);assert abs(ind[tr]-ztr).max()<=1e-12 and abs(ind[va]-zva).max()<=1e-12
  assert np.array_equal(ztr[:,:,sm.SHARED],np.broadcast_to(ztr[:,:1,sm.SHARED],(len(ztr),20,100)))
  start=time.perf_counter();model,diag=c.k.fit_objective(ztr,d["labels"][tr],d["track"][tr],"pointwise",guard=c.guard);times.append(time.perf_counter()-start);linear_checks.append(dict(**reproduce(model,ztr,zva,d,tr,va,fold),fit=diag));arrays.append((ztr,zva,tr,va))
 ztr,zva,tr,va=arrays[0];fixtures=[fixture(ztr[:21],d["labels"][tr][:21],d["track"][tr][:21],arm) for arm in ("pointwise","pairwise")]
 start=time.perf_counter();bench,diag=c.k.fit_objective(ztr,d["labels"][tr],d["track"][tr],"pairwise",guard=c.guard);c.k.score(zva,bench);pair_seconds=time.perf_counter()-start
 td={k:v[tr] for k,v in d.items() if isinstance(v,np.ndarray) and len(v)==384};report=c.reports(td,c.k.rank4(c.k.score(ztr,bench)),c.k.score(ztr,bench));assert report["fold_0"]["windows"]==0 and report["fold_0"]["Hit4"] is None;json_check=c.digest(report)
 # No scientific outcome/selector from throughput retained; no hyperparameter selection.
 c.verify(files);c.guard();planning=(6*max(pair_seconds,max(times))+4*sha_seconds+90)*1.5/60;assert planning<=30,"STOP_BUDGET"
 c.write("PREFLIGHT.json",dict(status="PASS",at=c.now(),source_count_before_final_lock=len(files),counts=counts,map_rebuild_all_arrays_EXACT=True,shared_history_delta_ZERO=True,pointwise_R2_reproduction=linear_checks,subset_numeric_fixtures=fixtures,full_pair_throwaway_seconds=pair_seconds,pair_diagnostic_only_no_outcome_selection=diag,pointwise_seconds=times,sha_seconds=sha_seconds,empty_group_JSON_N0_null=True,planning_minutes=planning,ETA_minutes=[max(2,planning*.6),planning*1.8],resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-t,versions=c.versions(),no_solver_neural_training=True))
 print("PREFLIGHT_PASS",planning,flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:c.write("PREFLIGHT_FAILURE_"+str(time.time_ns())+".json",dict(error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
