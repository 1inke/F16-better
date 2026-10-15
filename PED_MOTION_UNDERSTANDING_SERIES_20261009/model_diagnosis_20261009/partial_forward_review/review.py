"""Salvage only the six completed frozen-model outputs. No torch/new forward."""
import os,sys,time,importlib.util,traceback
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
H=Path(__file__).resolve().parent;P=H.parent.parent;F=H.parent/'forward_diagnostics';T=time.perf_counter()
sp=importlib.util.spec_from_file_location('partial_saved_utils',P/'claim_evidence_followup_20261009/saved_reader_recovery/analyze_saved.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a);a.H=H;a.Q={**a.read(F/'protocol.json'),'max_compute_seconds':180};a.START=T
checks=0;maxerror=0.
def matching(p,q):
    global checks,maxerror
    same=np.linalg.norm(p-q,axis=-1).mean((1,2));value=np.empty(len(p))
    for b in range(0,len(p),128):
        a.guard();C=np.linalg.norm(p[b:b+128,:,None]-q[b:b+128,None,:],axis=-1).mean(-1)
        for k,c in enumerate(C):rr,cc=linear_sum_assignment(c);value[b+k]=c[rr,cc].mean()
    return same,value
def metric(pred,gt,cv,t,m):
    aa,ff=a.err(pred,gt);den=((gt-cv)**2).sum((1,2));num=((gt-pred)**2).sum((1,2));return dict(n=int(m.sum()),tracks=len(np.unique(t[m])),skill=a.ci(num[m],den[m],t[m],True),ADE=a.mean_ci(aa[m],t[m]),FDE=a.mean_ci(ff[m],t[m]),coverage=a.mean_ci(((aa<=.2)&(ff<=.4))[m].astype(float),t[m]))
def main():
    global checks,maxerror
    a.guard();a.lock(F/'FAILURE.json');failure=a.read(F/'FAILURE.json');assert 'STOP_MIRROR_CLOSEST' in failure['traceback'] and 'univ' in failure['traceback'];a.lock(F/'run.py');a.lock(F/'DESIGN.md');a.lock(F/'protocol.json');a.lock(Path(__file__));r1=[];rgeo=[];rs=[]
    for fold in ('eth','hotel'):
        raw=a.load(F/f'RAW_{fold}.npz');t=raw['track'];gt=raw['GT'];cv=raw['CV'];groups=dict(all=np.ones(len(t),bool),P1=raw['mask_P1'],fallback=raw['fallback'],no_fallback=~raw['fallback'])
        for seed in (1,2,3):
            for name in a.Q['heading_variants']:
                z=a.load(F/f'{fold}_seed{seed}_{name}.npz');p=z['paths'];ll=z['logits'];assert np.array_equal(z['sample_id'],raw['sample_id'])
                for method,pred in [('top1',p[np.arange(len(t)),ll.argmax(1)]),('meanK',p.mean(1))]:
                    for tag,m in groups.items():r1.append(dict(fold=fold,seed=seed,variant=name,method=method,subset=tag,**metric(pred,gt,cv,t,m)))
                for tr in ('rotate','mirror'):
                    q=a.load(F/f'{fold}_seed{seed}_{name}_{tr}.npz');s,d=matching(p,q['inverse_paths']);error=max(abs(s-q['same_slot']).max(),abs(d-q['matched']).max());maxerror=max(maxerror,float(error));assert error<=1e-12;checks+=2*len(t)
                    for tag,m in groups.items():rgeo.append(dict(fold=fold,seed=seed,variant=name,transform=tr,subset=tag,n=int(m.sum()),same_slot_mean=a.mean_ci(s[m],t[m]),matched_mean=a.mean_ci(d[m],t[m]),same_slot_p95=float(np.quantile(s[m],.95)) if m.any() else None,matched_p95=float(np.quantile(d[m],.95)) if m.any() else None))
            angular=[]
            for angle in a.Q['angles_degrees']:
                z=a.load(F/f'{fold}_seed{seed}_angle{angle}.npz');angular.append(z['paths'][np.arange(len(t)),z['logits'].argmax(1)])
            z=a.load(F/f'{fold}_seed{seed}_TTA.npz');ensemble=np.mean(angular,axis=0);amp=np.linalg.norm(np.stack(angular)-ensemble[None],axis=-1).mean((0,2));assert np.array_equal(ensemble,z['top1_average']) and np.array_equal(amp,z['angular_path_deviation']);checks+=len(t)*25
            for tag,m in groups.items():r1.append(dict(fold=fold,seed=seed,variant='four_fixed_orientation_top1_TTA_not_original_candidate',method='top1_average',subset=tag,angular_variation_mean=a.mean_ci(amp[m],t[m]),angular_variation_p95=float(np.quantile(amp[m],.95)) if m.any() else None,**metric(ensemble,gt,cv,t,m)))
            for method in ('top1','meanK'):
                z=a.load(F/f'{fold}_seed{seed}_SOCIAL_{method}.npz');l=z['l'];lp=z['lprime'];den=abs(l)+abs(lp)
                for tag,m in [('all',raw['axis_defined']),('P1',raw['axis_defined']&raw['mask_P1']),('C3',raw['axis_defined']&raw['C3'])]:rs.append(dict(fold=fold,seed=seed,method=method,subset=tag,n=int(m.sum()),undefined_axis=int((~raw['axis_defined']).sum()),A=a.ci((l+lp)[m],den[m],t[m]),magnitude=a.mean_ci(den[m],t[m]),absolute_antisymmetry_error=a.ci(abs(l+lp)[m],den[m],t[m])))
    a.emit('D1b_PARTIAL',dict(records=r1));a.emit('D1b_GEOMETRY_PARTIAL',dict(records=rgeo));a.emit('D5b_PARTIAL',dict(records=rs))
    for p,d in a.LOCKS.items():a.guard();assert a.sha(p)==d
    a.write('input_manifest.json',dict(files=a.LOCKS));a.write('REVIEW.json',dict(status='PASS_PARTIAL_SIX_MODELS_SAVED_ARITHMETIC_NOT_COMPLETE_T2',completed_models=6,folds=['eth','hotel'],univ_zara_unexecuted=True,no_model_forward=True,checks=checks,max_numeric_error=maxerror,source_SHA_preserved=True,summary_sha=a.OUT,resource=a.RES,elapsed_seconds=time.perf_counter()-T,frozen_assertions_for_six_ran_before_original_FORWARD_console_markers_but_full15_FROZEN_AUDIT_not_saved=True))
if __name__=='__main__':
    try:main()
    except Exception:a.write('FAILURE.json',dict(status='STOP_PARTIAL_REVIEW',traceback=traceback.format_exc()));raise
