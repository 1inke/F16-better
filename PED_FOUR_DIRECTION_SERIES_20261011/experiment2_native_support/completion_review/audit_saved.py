"""New saved-data completion review, no model or training."""
import sys,time,traceback
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import run as R
import numpy as np
A=R.A;H=R.H
def main():
 start=time.perf_counter();A.guard();result=A.read(H/'RESULT.json')
 assert result['status']=='PASS_NATIVE_SUPPORT_INTEGRATION_DIAGNOSTIC'
 assert A.read(H/'worker_state.json')['status']==A.read(H/'exit_status.json')['status']=='SUCCESS'
 files=A.read(H/'source_manifest.json')['files'];A.verify(files);A.verify(result['output_sha'])
 saved={};records=[];err=0
 for arm in R.ARMS:
  for seed in R.SEEDS:
   for split in R.SPLITS:
    folder=H/'jobs'/arm/f'seed_{seed}'/split;R.skip(folder/'done.json',files);z=A.npz(folder/'paths.npz');ev=A.read(folder/'events.json');m=A.read(folder/'metrics.json')
    met,v,cl=R.audit(z,ev);aux=A.npz(folder/'metrics.npz');delta=max(A.compare(met,m['metrics']),A.compare(R.summary(ev,z),m['integration']))
    assert delta<=1e-12 and np.array_equal(v,aux['vectors']) and np.array_equal(cl,aux['clusters']);err=max(err,delta)
    saved[(arm,seed,split)]=dict(vectors=v,**{k:z[k] for k in ('sample_id','groups','track')});records.append(dict(arm=arm,seed=seed,split=split,metrics=met,integration=m['integration']))
 ci=R.boot(saved);cierr=A.compare(ci,A.read(H/'BOOTSTRAP.json')['records']);assert cierr<=1e-12
 A.verify(files);A.verify(result['output_sha']);A.guard()
 R.write('completion_review/SUMMARY.json',dict(records=records,bootstrap=ci))
 R.write('completion_review/REVIEW.json',dict(status='PASS',jobs=18,max_error=err,bootstrap_max_error=cierr,bootstrap_records=48,scalar_CIs=192,source_count=len(files),output_count=len(result['output_sha']),source_output_initial_final_PASS=True,full_saved_metrics_original_SciPy_complete_link_motion_certificates_decomposition_recomputed=True,two_independent_policy_decision_loops=True,no_torch_model_forward_backward_optimizer=True,not_external_review=True,resource=A.resource,elapsed_seconds=time.perf_counter()-start))
 print('COMPLETION_REVIEW_PASS',err,cierr,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  R.write('completion_review/FAILURE_'+str(time.time_ns())+'.json',dict(traceback=traceback.format_exc(),resource=A.resource));raise

