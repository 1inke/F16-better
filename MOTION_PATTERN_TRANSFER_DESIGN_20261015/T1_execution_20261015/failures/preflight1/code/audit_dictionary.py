"""Independent CPU saved dictionary review; no Torch/model/optimizer/search."""
import common as c
import numpy as np,time,traceback

def nearest(x,centers):
 # Independent per-mode accumulation rather than the producer's [chunk,K,24].
 distances=np.empty((len(x),20))
 for k in range(20):
  c.guard();distances[:,k]=np.sum((x-centers[k])**2,axis=1)
 z=np.argmin(distances,axis=1)
 return z,distances[np.arange(len(x)),z]

def main():
 t=time.perf_counter();c.guard();manifest=c.read(c.H/'dictionary_source_manifest.json');c.verify(manifest['files'])
 report=c.read(c.H/'DICTIONARY_PREFLIGHT.json');c.verify(report['output_sha'])
 before={str(p):c.sha(p) for p in (c.H/'audit_dictionary.py',c.H/'DICTIONARY_PREFLIGHT.json',c.H/'dictionary_source_manifest.json')}
 d=c.npz(c.H/'dictionary.npz');trace=c.read(c.H/'dictionary_trace.json');data=c.load();tr=data['split']['FIT']
 assert np.array_equal(tr,d['fit_indices']) and np.array_equal(data['ids'][tr],d['fit_ids'])
 assert np.array_equal(data['track'][tr],d['fit_tracks']) and np.array_equal(data['scene'][tr],d['fit_scenes'])
 x=np.empty((len(tr),24))
 for i in range(12):x[:,2*i:2*i+2]=data['gt'][tr,i].astype(float)-(i+1)*.4*data['vel'][tr]
 assert np.array_equal(x,d['residual'])
 rng=np.random.Generator(np.random.PCG64(20261015));assert rng.bit_generator.state==trace['initial_rng']
 chosen=[int(rng.integers(len(x)))];mind=np.sum((x-x[chosen[0]])**2,axis=1)
 for j in range(1,20):
  mind[chosen]=0.;total=float(mind.sum());pick=int(rng.choice(len(x),p=mind/total));chosen.append(pick)
  assert pick==trace['kmeanspp_sampling'][j-1]['index'] and total==trace['kmeanspp_sampling'][j-1]['total_squared_distance']
  mind=np.minimum(mind,np.sum((x-x[pick])**2,axis=1))
 assert chosen==d['initial_indices'].tolist()==trace['kmeanspp_selected'] and rng.bit_generator.state==trace['final_rng']
 history=d['lloyd_centers'];assert np.array_equal(history[0],x[chosen]);last=None;maxerror=0.
 for it,row in enumerate(trace['rounds']):
  z,e=nearest(x,history[it]);counts=np.bincount(z,minlength=20)
  assert counts.tolist()==row['counts'] and float(e.sum())==row['objective']
  same=last is not None and np.array_equal(z,last);assert same==row['unchanged_labels']
  if same:
   assert it+1==len(trace['rounds']) and len(history)==len(trace['rounds']);break
  updated=np.empty((20,24));used=set();events=[]
  for k in range(20):
   if counts[k]:updated[k]=x[z==k].mean(0)
   else:
    ordering=sorted(range(len(x)),key=lambda i:(-float(e[i]),i));p=next(i for i in ordering if i not in used);used.add(p);updated[k]=x[p];events.append(dict(mode=k,index=p))
  assert events==row['empty_reinitializations'] and np.array_equal(updated,history[it+1]);last=z.copy()
 order=np.array(sorted(range(20),key=lambda k:tuple(history[-1,k].tolist())+(k,)))
 assert np.array_equal(order,d['final_sort_order'])
 centers=history[-1,order];assert np.array_equal(centers,d['centers'].reshape(20,24))
 z,e=nearest(x,centers);assert np.array_equal(z,d['labels']) and np.array_equal(e,d['final_squared_errors'])
 rows=[]
 for k in range(20):
  mask=z==k;families,counts=np.unique(data['scene'][tr][mask],return_counts=True)
  rows.append(dict(mode=k,windows=int(mask.sum()),tracks=len(np.unique(data['track'][tr][mask])),scenes=dict(zip(families.tolist(),counts.tolist())),center_rms_m=float(np.sqrt(np.square(centers[k]).mean())),squared_reconstruction_error=float(e[mask].sum())))
 c.eq(rows,report['modes'],0.)
 passed=all(r['windows']>=50 and r['tracks']>=5 for r in rows)
 assert passed==(report['status']=='PASS_DICTIONARY_SUPPORT')
 split=c.npz(c.H/'split_ids.npz')
 for name,ix in data['split'].items():assert np.array_equal(split[name],data['ids'][ix])
 c.verify(manifest['files']);c.verify(report['output_sha']);c.verify(before);c.guard()
 c.write('dictionary_review/REVIEW.json',dict(status='PASS',at=c.now(),support_gate_passed=passed,independent_full_init_Lloyd_reconstruction_and_final_assignments='EXACT',source_preservation='PASS_INITIAL_FINAL',source_count=len(manifest['files']),source_sha=manifest['source_sha'],output_sha=report['output_sha'],audit_input_sha=before,mode_table=rows,elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE,no_Torch_no_GPU_no_model_no_optimizer_no_refit_search=True,not_external_review=True))
 print(dict(status='PASS',support_gate_passed=passed,iterations=len(trace['rounds']),elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE),flush=True)

if __name__=='__main__':
 try:main()
 except Exception:
  c.write('dictionary_review/FAILURE.json',dict(status='FAILURE',at=c.now(),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
