import common as c,independent_math as im
import numpy as np,time,traceback
def main():
 t=time.perf_counter();c.guard();files,meta=c.sources();before=c.digest(files);c.verify(files)
 d,m,core,con,fn=c.inputs();assert len(d["ids"])==384 and len(np.unique(d["ids"]))==384
 old=c.k.best_positive(d["pairwise_scores"],d["labels"]);assert np.array_equal(np.bincount(old["bin"][old["hasP"]],minlength=3),[82,75,106])
 change=old["hasP"]&(d["groups"]==2);assert np.array_equal(np.bincount(old["bin"][change],minlength=3),[20,26,31])
 timings=[];fixture=[]
 for fold in range(3):
  ztr,zva,tr,va=c.fold(fold);z2,t2,v2=im.features(d,fold,m[fold]);assert np.array_equal(tr,t2) and np.array_equal(va,v2)
  error=float(np.max(abs(np.r_[ztr.reshape(-1,200),zva.reshape(-1,200)]-np.r_[z2[tr].reshape(-1,200),z2[va].reshape(-1,200)])));assert error<=1e-12
  bank=c.k.make_bank(ztr,d["ids"][tr],d["track"][tr]);q=c.k.make_bank(zva,d["ids"][va],d["track"][va])
  for calibration in (True,False):
   query=bank if calibration else q;count=32;ts=time.perf_counter();ix,dist=c.k.retrieve(query.z[:count],query.tracks[:count],bank,guard=c.guard);production=(time.perf_counter()-ts)/count
   ts=time.perf_counter();ix2,dist2=im.retrieve(query.z[:count],query.tracks[:count],bank);independent=(time.perf_counter()-ts)/count
   assert np.array_equal(ix,ix2);c.eq(dist,dist2);c.eq(c.k.distance_components(query.z[:count],bank,ix).sum(-1),dist**2)
   assert all(query.tracks[i] not in bank.tracks[row] and len(np.unique(bank.tracks[row]))==5 for i,row in enumerate(ix))
   timings.append(dict(fold=fold,unit="train_LOTO" if calibration else "validation",total_queries=len(query.z),measured_queries=count,production_seconds_per_query=production,independent_seconds_per_query=independent,feature_reconstruction_max_error=error))
   fixture.append((ix,dist,bank,d["labels"][tr]))
  subset=c.k.label_support(ix,bank,d["labels"][tr]);c.eq(subset,im.labels(ix,bank,d["labels"][tr]))
 c.save("preflight_fixture.npz",indices=fixture[-1][0],distances=fixture[-1][1])
 fp=str(c.H/"preflight_fixture.npz");good=dict(status="COMPLETE",source_sha=before,files={fp:c.sha(fp)});assert c.done_valid(good,before)
 bads=[{**good,"source_sha":"bad"},{**good,"files":{fp:"bad"}}]
 for bad in bads:
  try:c.done_valid(bad,before)
  except AssertionError:pass
  else:raise AssertionError("bad source/output not rejected")
 c.verify(files);after,meta2=c.sources();assert files==after
 compute=sum(x["total_queries"]*(x["production_seconds_per_query"]+x["independent_seconds_per_query"]) for x in timings)
 planning=(compute*1.35+90)/60
 assert planning<=30,("STOP_BUDGET",planning)
 c.write("PREFLIGHT.json",dict(status="PASS",source_sha=before,source_count=len(files),inheritance=meta,timings=timings,nominal_geometry_and_full_independent_seconds=compute,planning_minutes=planning,ETA_minutes=[max(1,planning*.7),planning*1.6],elapsed_seconds=time.perf_counter()-t,all_three_map_ID_geometry_independent_PASS=True,done_source_output_reject_PASS=True,zero_fits=True,preflight_queries_not_scientific_population=True,resource=c.RESOURCE))
 print("PREFLIGHT_PASS",planning,flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:
  traceback.print_exc();c.write("failures/preflight1/FAILURE.json",dict(status="STOP",error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
