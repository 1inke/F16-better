"""Saved arithmetic/ID/SHA review and predeclared group contrasts. No models."""
import sys,importlib.util,json,hashlib,time
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('followup_math',H/'analyze_saved.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def mean(x):return float(np.mean(x)) if len(x) else None
result=a.read(H/'RESULT.json');manifest=a.read(H/'input_manifest.json')['files']
for p,d in manifest.items():a.guard();assert a.sha(p)==d
for p,d in result['output_sha'].items():assert a.sha(p)==d
for n,d in result['summary_sha'].items():assert a.sha(H/(n+'.json'))==d
checks=0;tails=[];contrasts=[]
n1=a.read(H/'N1.json')['records'];n2=a.read(H/'N2.json')['records'];n6=a.read(H/'N6.json')['records'];n3=a.read(H/'N3.json')['real_records']
def same(x,y,tol=1e-12):
    global checks
    checks+=1
    if x is None or y is None:assert x is None and y is None;return
    x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape and np.max(abs(x-y))<=tol
def contrast(m,track,event,control):
    u,inv=np.unique(track,return_inverse=True);T=len(u);vals=(m>0).astype(float);groups=[]
    for mask in (event,control):groups.append((np.bincount(inv,weights=vals*mask,minlength=T),np.bincount(inv,weights=mask.astype(float),minlength=T)))
    if not event.any() or not control.any():return dict(value=None,CI=None)
    v=float(vals[event].mean()-vals[control].mean());rng=np.random.default_rng(20261009);out=[]
    for start in range(0,2000,100):
        ix=rng.integers(0,T,(100,T));num0,den0=groups[0];num1,den1=groups[1];d0=den0[ix].sum(1);d1=den1[ix].sum(1);ok=(d0>0)&(d1>0)
        out.extend((num0[ix].sum(1)[ok]/d0[ok]-num1[ix].sum(1)[ok]/d1[ok]).tolist())
    return dict(value=v,CI=np.quantile(out,[.025,.5,.975]).tolist(),tracks=T,exploratory_not_adjusted_confirmatory=True)
for fold in a.Q['folds']:
    a.guard();pairs=[]
    for left,right in ((1,2),(1,3),(2,3)):
        z=load(H/f'N1_{fold}_pair{left}{right}.npz');p=load(a.S/f'jobs/{fold}_seed{left}/mirror/outputs.npz')['base_paths'];q=load(a.S/f'jobs/{fold}_seed{right}/mirror/outputs.npz')['base_paths']
        for i in np.linspace(0,len(p)-1,min(16,len(p)),dtype=int):
            d=np.linalg.norm(p[i,:,None]-q[i,None,:],axis=-1).mean(-1);r,c=linear_sum_assignment(d)
            same(z['values'][i],[d[r,c].mean(),np.linalg.norm(p[i,r,-1]-q[i,c,-1],axis=-1).mean()])
        pairs.append(z['values'])
    pooled=np.concatenate(pairs);median=float(np.median(pooled[:,0]));q95=float(np.quantile(pooled[:,0],.95))
    for seed in a.Q['model_seeds']:
        a.guard();v=load(H/f'N2_N6_{fold}_seed{seed}.npz');source=load(a.P/f'jobs/{fold}_seed{seed}/outputs.npz');p=source['model_paths'];gt=source['GT'];n=len(p);mask=v['mask'];t=v['track'];sel=source['model_logits'].argmax(1)
        assert np.array_equal(v['sample_id'],source['sample_id'])
        sse=((p-gt[:,None])**2).sum((2,3));num=sse[np.arange(n),sel];den=((gt-source['CV'])**2).sum((1,2));oracle=sse.min(1)
        same(v['num'],num);same(v['den'],den);same(v['oracle_num'],oracle)
        aa,ff=a.err(p,gt);good=(aa<=.2)&(ff<=.4);assert np.array_equal(v['good_present'],good.any(1));assert np.array_equal(v['top1_good'],good[np.arange(n),sel])
        rec=next(r for r in n6 if r['fold']==fold and r['seed']==seed)
        same(rec['top1']['value'],1-num[mask].sum()/den[mask].sum());same(rec['oracle_single_complete_SSE']['value'],1-oracle[mask].sum()/den[mask].sum())
        same(rec['top1']['CI'],a.ci(num[mask],den[mask],t[mask],True)['CI']);same(rec['no_good']['value'],(~v['good_present'][mask]).mean())
        for row in n2:
            if row['fold']!=fold or row['seed']!=seed:continue
            sub=np.ones(n,bool) if row['subset']=='all' else mask.copy()
            if row['speed_bin'] is not None:sub&=v['speed_bin']==row['speed_bin']
            for b in row['bins']:
                take=sub&(v['density_bin']==b['bin']);assert int(take.sum())==b['n']
                expected=a.ci(num[take],den[take],t[take],True);same(expected['value'],b['skill']['value']);same(expected['CI'],b['skill']['CI'])
        for name in ('rotate','mirror','translate','scale08','scale12'):
            z=load(H/f'N1_{fold}_seed{seed}_{name}.npz');rec=next(r for r in n1 if r['fold']==fold and r['seed']==seed and r['transform']==name)
            same(rec['seed_path_median'],median);same(rec['path_seed_ratio'],np.median(z['dT'])/median)
            tails.append(dict(fold=fold,seed=seed,transform=name,dT_mean=float(z['dT'].mean()),dT_p95=float(np.quantile(z['dT'],.95)),seed_path_p95=q95,p95_ratio=float(np.quantile(z['dT'],.95)/q95),median_ratio=rec['path_seed_ratio']))
        real=load(a.S/f'jobs/{fold}_seed{seed}/noNbr/outputs.npz');nr=len(real['paths']);idx=np.arange(nr)
        for method in ('top1','meanK'):
            z=load(H/f'N3_{fold}_seed{seed}_{method}.npz')
            before=real['base_paths'][idx,real['base_logits'].argmax(1)] if method=='top1' else real['base_paths'].mean(1)
            after=real['paths'][idx,real['logits'].argmax(1)] if method=='top1' else real['paths'].mean(1)
            same(z['m'],((before[:,-1]-after[:,-1])*z['away']).sum(1))
            rec=next(r for r in n3 if r['fold']==fold and r['seed']==seed and r['method']==method)
            for group,key in (('avoidance_event','event'),('other_C3','other_C3'),('non_C3','non_C3')):
                g=z[key];assert rec['groups'][group]['n']==int(g.sum());same(rec['groups'][group]['response_m']['value'],mean(z['m'][g]));same(rec['groups'][group]['positive']['value'],mean((z['m'][g]>0).astype(float)))
            for label,control in (('other_C3',z['other_C3']),('non_C3',z['non_C3'])):
                contrasts.append(dict(fold=fold,seed=seed,method=method,control=label,positive_probability_difference=contrast(z['m'],z['track'],z['event'],control)))
    print('REVIEW',fold,flush=True)
for p,d in manifest.items():a.guard();assert a.sha(p)==d
for p,d in result['output_sha'].items():assert a.sha(p)==d
a.write('N1_TAIL_SUMMARY.json',tails);a.write('N3_EVENT_CONTROL_CONTRASTS.json',contrasts)
a.write('REVIEW.json',dict(status='PASS_SAVED_ARITHMETIC_ID_AND_SHA',checks=checks,input_files=len(manifest),N1_sampled_full_matching_windows=240,N2_bin_skill_CIs_reconstructed=True,N6_full15_candidate_SSE_and_good_checks=True,N3_all30_real_read_response_checks=True,initial_final_SHA_PASS=True,not_all_synthetic_rows_second_replay=True,not_all_candidate_costs_second_rebuild=True,not_external_review=True,no_torch_forward_optimizer=True,elapsed_seconds=time.perf_counter()-a.START,resource=a.RES,new_contrast_exploratory=True))
print('REVIEW SUCCESS',checks,flush=True)
