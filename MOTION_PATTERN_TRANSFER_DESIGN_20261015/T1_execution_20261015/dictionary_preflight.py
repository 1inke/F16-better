"""One preregistered dictionary, no neural imports/evaluation or search."""
import common as c
import numpy as np,time,traceback

def assign(x,centers,chunk=256):
 labels=np.empty(len(x),np.int64);error=np.empty(len(x))
 for start in range(0,len(x),chunk):
  c.guard();a=x[start:start+chunk];dist=np.square(a[:,None]-centers[None]).sum(2)
  z=dist.argmin(1);labels[start:start+len(a)]=z;error[start:start+len(a)]=dist[np.arange(len(a)),z]
 return labels,error

def fit(x):
 rng=np.random.Generator(np.random.PCG64(20261015));initial_state=rng.bit_generator.state
 first=int(rng.integers(len(x)));selected=[first];centers=[x[first].copy()]
 nearest=np.square(x-centers[0]).sum(1);sampling=[]
 for k in range(1,20):
  c.guard();nearest[selected]=0.;total=float(nearest.sum());assert total>0,'STOP_ZERO_KMEANS_PROBABILITY'
  j=int(rng.choice(len(x),p=nearest/total));assert j not in selected
  sampling.append(dict(mode=k,index=j,total_squared_distance=total))
  selected.append(j);centers.append(x[j].copy());nearest=np.minimum(nearest,np.square(x-x[j]).sum(1))
 centers=np.array(centers);all_centers=[centers.copy()];rounds=[];last=None;converged=False
 for it in range(1,101):
  labels,error=assign(x,centers);counts=np.bincount(labels,minlength=20)
  same=last is not None and np.array_equal(labels,last)
  row=dict(iteration=it,objective=float(error.sum()),counts=counts.tolist(),unchanged_labels=bool(same),empty_reinitializations=[])
  if same:
   rounds.append(row);converged=True;break
  updated=np.empty_like(centers);used=set()
  for k in range(20):
   if counts[k]:updated[k]=x[labels==k].mean(0)
   else:
    candidates=np.argsort(-error,kind='stable');j=next(int(v) for v in candidates if int(v) not in used);used.add(j);updated[k]=x[j]
    row['empty_reinitializations'].append(dict(mode=k,index=j))
  rounds.append(row);last=labels.copy();centers=updated;all_centers.append(centers.copy())
 order=np.array(sorted(range(20),key=lambda k:tuple(centers[k].tolist())+(k,)),dtype=np.int64)
 centers=centers[order];labels,error=assign(x,centers)
 return centers,labels,error,np.array(all_centers),dict(initial_rng=initial_state,final_rng=rng.bit_generator.state,kmeanspp_selected=selected,kmeanspp_sampling=sampling,rounds=rounds,converged=converged,final_sort_order=order.tolist())

def main():
 started=time.perf_counter();c.guard();assert not (c.H/'dictionary.npz').exists(),'NO_OVERWRITE_DICTIONARY'
 files,inherit=c.locks();source=c.digest(files)
 c.write('dictionary_source_manifest.json',dict(source_sha=source,files=files,inheritance=inherit,phase='PRE_GPU_DICTIONARY_GATE_ONLY'))
 data=c.load();tr=data['split']['FIT'];ids=data['ids'];track=data['track'][tr];scene=data['scene'][tr]
 # Only current training GT used. No outer/inner/CAL outcomes are computed.
 x=(data['gt'][tr].astype(np.float64)-np.arange(1,13)[None,:,None]*.4*data['vel'][tr,None]).reshape(len(tr),24)
 assert x.shape==(18909,24) and np.isfinite(x).all()
 centers,z,error,iterations,trace=fit(x)
 rows=[]
 for k in range(20):
  mask=z==k;families,counts=np.unique(scene[mask],return_counts=True)
  rows.append(dict(mode=k,windows=int(mask.sum()),tracks=len(np.unique(track[mask])),scenes=dict(zip(families.tolist(),counts.tolist())),center_rms_m=float(np.sqrt(np.square(centers[k]).mean())),squared_reconstruction_error=float(error[mask].sum())))
 passed=all(r['windows']>=50 and r['tracks']>=5 for r in rows)
 c.save('dictionary.npz',centers=centers.reshape(20,12,2),labels=z,fit_indices=tr,fit_ids=ids[tr],fit_tracks=track,fit_scenes=scene,residual=x,final_squared_errors=error,lloyd_centers=iterations,initial_indices=np.array(trace['kmeanspp_selected']),final_sort_order=np.array(trace['final_sort_order']))
 c.save('split_ids.npz',**{k:ids[v] for k,v in data['split'].items()})
 c.write('dictionary_trace.json',trace)
 output={str(c.H/n):c.sha(c.H/n) for n in ('dictionary.npz','dictionary_trace.json','split_ids.npz')}
 c.verify(files);c.verify(output);c.guard()
 report=dict(status='PASS_DICTIONARY_SUPPORT' if passed else 'STOP_DICTIONARY_SUPPORT',at=c.now(),source_sha=source,source_count=len(files),source_preservation='PASS_INITIAL_FINAL',inheritance=inherit,output_sha=output,training_windows=len(tr),training_tracks=len(np.unique(track)),fit_only=True,converged=trace['converged'],lloyd_iterations=len(trace['rounds']),final_objective=float(error.sum()),minimum_mode_windows=min(r['windows'] for r in rows),minimum_mode_tracks=min(r['tracks'] for r in rows),required_windows=50,required_tracks=5,modes=rows,failed_modes=[r for r in rows if r['windows']<50 or r['tracks']<5],GPU_preflight=False,neural_training_started=False,worker_started=False,elapsed_seconds=time.perf_counter()-started,resource=c.RESOURCE)
 c.write('DICTIONARY_PREFLIGHT.json',report)
 c.write('runtime_state.json',dict(status=report['status'],at=c.now(),production_started=False,active_worker=False,scientific_gate_passed=passed,completed_runs=0,planned_runs=6,T2_started=False,T3_started=False,whole_series_complete=False,next_action='IMPLEMENT_GPU_RESUME_BUDGET_PREFLIGHT' if passed else 'STOP_FIXED_DICTIONARY_DESIGN_NO_AUTORETRY',source_sha=source))
 print(c.safe(dict(status=report['status'],modes=rows,elapsed_seconds=report['elapsed_seconds'],resource=c.RESOURCE)),flush=True)

if __name__=='__main__':
 try:main()
 except Exception:
  c.write('DICTIONARY_FAILURE.json',dict(status='FAILURE',at=c.now(),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
