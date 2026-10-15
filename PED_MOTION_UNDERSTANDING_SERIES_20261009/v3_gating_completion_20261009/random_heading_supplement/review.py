import os,sys,time,importlib.util
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent
sp=importlib.util.spec_from_file_location('random_saved_review',V/'saved.py');v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v);a=v.a;a.H=H;a.START=time.perf_counter();Q=v.Q
def main():
    a.guard();a.lock(Path(__file__));result=a.read(H/'RESULT.json');a.lock(H/'RESULT.json');rows=[];ncheck=0
    for p,s in result['output_sha'].items():a.lock(p,s)
    for fold in Q['folds']:
        raw=a.load(V/f'RAW_{fold}.npz');base=dict(GT=raw['GT'],CV=raw['CV'],track=raw['track'],P1=raw['mask_P1']);masks={**v.groups(base),**{k:raw[k] for k in ['fallback']},'no_fallback':~raw['fallback']}
        for seed in Q['seeds']:
            z=a.load(H/f'{fold}_seed{seed}.npz');old=a.load(V/f'G4_{fold}_seed{seed}.npz');assert np.array_equal(z['sample_id'],raw['sample_id']);assert np.array_equal(z['paths'][~raw['fallback']],old['post_paths'][~raw['fallback']]);ncheck+=int((~raw['fallback']).sum())*20*12*2
            for method in ('top1','meanK'):
                select=lambda p,l:p[np.arange(len(p)),l.argmax(1)] if method=='top1' else p.mean(1);pred=select(z['paths'],z['logits']);ref=select(old['post_paths'],old['logits'])
                for tag,m in masks.items():rows.append(dict(fold=fold,seed=seed,method=method,subset=tag,**v.point(pred,base,m,ref)))
    a.emit('SUMMARY',dict(records=rows,one_random_angle_per_window_no_best_selection=True))
    for p,s in a.LOCKS.items():assert a.sha(p)==s
    a.write('review_input_manifest.json',dict(files=a.LOCKS));a.write('REVIEW.json',dict(status='PASS_RANDOM15_SAVED_OUTCOMES_AND_NONFALLBACK_IDENTITY',checks=ncheck,summary_sha=a.OUT,resource=a.RES,elapsed_seconds=time.perf_counter()-a.START,no_torch=True));print('RANDOM_REVIEW_PASS')
if __name__=='__main__':main()
