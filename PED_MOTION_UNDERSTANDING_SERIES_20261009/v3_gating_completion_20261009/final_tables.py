import os,sys,time,importlib.util
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('v3_tables',H/'saved.py');v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v);a=v.a;a.START=time.perf_counter();Q=v.Q
def directionCI(r,probe,g,t,m):
    def moments(x):return np.stack(((x*g).sum((1,2)),(x*x).sum((1,2)),(g*g).sum((1,2))),1)[m]
    mm=moments(r);mp=moments(probe);_,ii=np.unique(t[m],return_inverse=True);T=ii.max()+1;A=np.zeros((T,3));B=np.zeros((T,3));np.add.at(A,ii,mm);np.add.at(B,ii,mp)
    def calc(x):return np.divide(x[...,0]**2,x[...,1]*x[...,2],out=np.full_like(x[...,0],np.nan),where=x[...,1]*x[...,2]>0)
    rng=np.random.default_rng(Q['stat_seed']);vv=[]
    for b in range(0,Q['bootstrap_reps'],100):a.guard();ix=rng.integers(T,size=(min(100,Q['bootstrap_reps']-b),T));x=calc(A[ix].sum(1))-calc(B[ix].sum(1));vv.extend(x[np.isfinite(x)].tolist())
    return dict(value=float(calc(A.sum(0))-calc(B.sum(0))),CI=np.quantile(vv,[.025,.5,.975]).tolist(),n=int(m.sum()),tracks=T)
def main():
    a.guard();a.lock(Path(__file__));res=a.read(H/'RESULT_SAVED.json');a.lock(H/'RESULT_SAVED.json');rows=[];directions=[];gates=a.read(H/'G1.json')['records'];a.lock(H/'G1.json');comparison=[]
    for fold in Q['folds']:
        for seed in Q['seeds']:
            path=H/f'POINTS_{fold}_seed{seed}.npz';a.lock(path,res['output_sha'][str(path)]);z=a.load(path);base=dict(GT=z['GT'],CV=z['CV'],track=z['track'],P1=z['P1']);g=z['GT']-z['CV']
            for subset,m in v.groups(base).items():
                for method in ('top1','meanK','probe_self','probe_social'):
                    rows.append(dict(fold=fold,seed=seed,subset=subset,method=method,**v.point(z[method],base,m)))
                    if method in ('top1','meanK'):
                        for probe in ('probe_self','probe_social'):directions.append(dict(fold=fold,seed=seed,subset=subset,method=method,probe=probe,diagnostic_cosine_squared_gain=directionCI(z[method]-z['CV'],z[probe]-z['CV'],g,z['track'],m)))
            for row in gates:
                if row['fold']!=fold or row['seed']!=seed or row['rule']!='gate':continue
                old=next(x for x in gates if x['fold']==fold and x['seed']==seed and x['rule']=='identity' and x['method']==row['method'] and x['subset']==row['subset']);ratio={k:row['motion'][k]/old['motion'][k] if old['motion'][k]>0 else None for k in row['motion']};comparison.append(dict(fold=fold,seed=seed,method=row['method'],subset=row['subset'],physical_ratio_vs_original_point=ratio,all_five_within_1percent=all(value is not None and value<=1.01 for value in ratio.values()),point_not_full_candidate_gate=True))
    a.emit('BASELINE_SCORECARD',dict(records=rows));a.emit('DIRECTION_PAIRED_CI',dict(records=directions));a.emit('GATE_PHYSICAL_PROTECTION',dict(records=comparison))
    for p,s in a.LOCKS.items():assert a.sha(p)==s
    a.write('tables_input_manifest.json',dict(files=a.LOCKS));a.write('TABLES_REVIEW.json',dict(status='PASS_NEW_DERIVED_TABLES_SHA_AND_MOMENT_BOOTSTRAP',input_files=len(a.LOCKS),resource=a.RES,elapsed_seconds=time.perf_counter()-a.START,summary_sha=a.OUT,no_neural_forward=True,not_independent_second_rebuild=True))
if __name__=='__main__':main()
