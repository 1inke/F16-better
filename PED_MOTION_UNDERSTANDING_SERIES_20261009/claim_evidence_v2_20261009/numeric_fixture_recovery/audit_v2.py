"""Full saved arithmetic/ID verification, no model replay/external review."""
import sys,importlib.util,time
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('v2_math',H/'analyze_v2.py');v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
a=v.a;Q=v.Q;P=v.P;S=v.S
result=a.read(H/'RESULT.json');manifest=a.read(H/'input_manifest.json')['files'];checks=0;maxerr=0.
def same(x,y,tol=1e-9):
    global checks,maxerr
    checks+=1
    if x is None or y is None:assert x is None and y is None;return
    x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape
    if not x.size:return
    e=float(np.max(abs(x.astype(float)-y.astype(float))));maxerr=max(maxerr,e);assert e<=tol,(e,tol)
def checked_stat(x,t,saved):
    rec=a.mean_ci(np.asarray(x,float),t);same(rec['value'],saved['value']);same(rec['CI'],saved['CI']);assert rec['n']==saved['n'] and rec['tracks']==saved['tracks']
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
    for p,d in manifest.items():a.guard();assert a.sha(p)==d
    for p,d in result['output_sha'].items():assert a.sha(p)==d
    for name,d in result['summary_sha'].items():assert a.sha(H/(name+'.json'))==d
    r1=a.read(H/'A1.json')['records'];r2=a.read(H/'A2.json')['records'];cal=a.read(H/'CALIBRATION.json');r5=a.read(H/'A5.json')['records'];r6=a.read(H/'A6.json')['records'];pools={s:[] for s in Q['seeds']}
    for fold in Q['folds']:
        d=load(P/f'data/{fold}_evaluation.npz')
        for seed in Q['seeds']:
            a.guard();z=load(P/f'jobs/{fold}_seed{seed}/outputs.npz');b=load(H/f'A1A2_{fold}_seed{seed}.npz');assert np.array_equal(b['sample_id'],d['ids']) and np.array_equal(z['track'],b['track'])
            p=z['model_paths'];gt=z['GT'];sse=np.sum(np.square(p-gt[:,None]),axis=(2,3));idx=np.arange(len(p));top=z['model_logits'].argmax(1);topSSE=sse[idx,top];mu=p.mean(1);meanSSE=np.square(mu-gt).sum((1,2));randomSSE=sse.mean(1)
            den=np.square(z['CV']-gt).sum((1,2));rank=(np.sum(sse>topSSE[:,None],1)+.5*(np.sum(sse==topSSE[:,None],1)-1))/19
            for key,x in [('top1',topSSE),('meanK',meanSSE),('random',randomSSE),('oracle',sse.min(1)),('den',den),('rank',rank)]:same(x,b[key])
            same(randomSSE-meanSSE,b['dispersion_SSE'],1e-8)
            distance=np.sqrt(np.sum((p-gt[:,None])**2,-1));ade=distance.mean(-1);fde=distance[:,:,-1]
            for rec in [r for r in r1 if r['fold']==fold and r['seed']==seed]:
                m=b['mask_P1'] if rec['subset']=='P1' else np.ones(len(p),bool);t=b['track'][m]
                expected=a.ci((randomSSE-topSSE)[m],den[m],t);same(expected['value'],rec['top1_minus_random']['value']);same(expected['CI'],rec['top1_minus_random']['CI']);checked_stat(rank[m]-.5,t,rec['percentile_minus_half'])
            for j,tau in enumerate(Q['A2_taus']):
                good=(ade<=tau)&(fde<=2*tau);present=good.any(1);sel=good[idx,top];classes=np.stack((~present,present&~sel,sel),1);same(classes,b['gaps'][:,j]);assert (classes.sum(1)==1).all()
                for rec in [r for r in r2 if r['fold']==fold and r['seed']==seed and r['tau']==tau]:
                    m=b['mask_P1'] if rec['subset']=='P1' else np.ones(len(p),bool)
                    for k,key in enumerate(['none','good_not_selected','selected_good']):checked_stat(classes[m,k],b['track'][m],rec[key])
            reg=load(H/f'A6_{fold}_seed{seed}.npz');raw=v.gaussian(d['obs'],d['valid'],cal[fold]['R']);same(raw,reg['density_raw']);density=(raw-cal[fold]['density_mean'])/cal[fold]['density_std'];same(density,reg['density_standardized']);same(den-topSSE,reg['delta_SSE'])
            for rec in [r for r in r6 if r['fold']==fold and r['seed']==seed]:
                m=b['mask_P1'] if rec['subset']=='P1' else np.ones(len(p),bool);X,names=v.design_matrix(density[m],reg['speed_bin'][m],reg['recording'][m]);rr=rec['regression']
                if rr['beta'] is not None:same(np.linalg.lstsq(X,reg['delta_SSE'][m],rcond=1e-12)[0][1],rr['beta'],1e-7)
                rebuilt=v.regression(reg['delta_SSE'][m],X,reg['track'][m]);same(rebuilt['CI'],rr['CI']);same(rebuilt['beta'],rr['beta'])
            pools[seed].append(dict(y=reg['delta_SSE'],density=density,t=np.char.add(fold+':',reg['track']),sb=np.char.add(fold+':',reg['speed_bin'].astype(str)),record=np.char.add(fold+':',reg['recording']),mask=reg['mask_P1']))
        selected=load(S/f'data/{fold}_selected.npz');nb=a.neighborhood(selected['obs'],selected['valid']);g=((selected['gt'][:,-1]-a.cv(selected['obs'])[:,-1])*nb['away']).sum(1)
        for seed in Q['seeds']:
            for control in Q['A5_controls']:
                z=load(S/f'jobs/{fold}_seed{seed}/{control}/outputs.npz');assert np.array_equal(z['sample_id'],selected['ids']);n=len(z['paths'])
                for method in Q['A5_methods']:
                    x=load(H/f'A5_{fold}_seed{seed}_{control}_{method}.npz');before=z['base_paths'][np.arange(n),z['base_logits'].argmax(1)] if method=='top1' else z['base_paths'].mean(1);after=z['paths'][np.arange(n),z['logits'].argmax(1)] if method=='top1' else z['paths'].mean(1)
                    m=np.sum((before[:,-1]-after[:,-1])*nb['away'],1);same(x['m'],m);same(x['g'],g)
                    for rec in [r for r in r5 if r['fold']==fold and r['seed']==seed and r['control']==control and r['method']==method]:
                        ev=nb['C3']&nb['away_valid']&(g>=rec['delta']);assert int(ev.sum())==rec['n'];checked_stat(m[ev],x['track'][ev],rec['response_m']);checked_stat(m[ev]>0,x['track'][ev],rec['positive']);same(a.wilson(int((m[ev]>0).sum()),len(m[ev])),rec['Wilson_window_descriptive'])
        print('AUDIT',fold,flush=True)
    for seed,parts in pools.items():
        c={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        for rec in [r for r in r6 if r['fold']=='pooled' and r['seed']==seed]:
            m=c['mask'] if rec['subset']=='P1' else np.ones(len(c['t']),bool);X,_=v.design_matrix(c['density'][m],c['sb'][m],c['record'][m]);rr=v.regression(c['y'][m],X,c['t'][m]);same(rr['beta'],rec['regression']['beta']);same(rr['CI'],rec['regression']['CI'])
    for p,d in manifest.items():a.guard();assert a.sha(p)==d
    for p,d in result['output_sha'].items():assert a.sha(p)==d
    a.write('REVIEW.json',dict(status='PASS_ALL15_CANDIDATE_ARITHMETIC_A1_A2_A5_A6_AND_INPUT_OUTPUT_SHA',checks=checks,max_numeric_error=maxerr,all15_candidates_full_SSE_rank_thresholds=True,all180_A5_response_records=True,all36_A6_beta_CI_reconstructed=True,input_files=len(manifest),output_files=len(result['output_sha']),A3_A4_descriptive_profiles_need_companion_review=True,not_new_model_replay=True,not_external_review=True,resource=a.RES,elapsed_seconds=time.perf_counter()-a.START))
    print('AUDIT PASS',checks,maxerr,flush=True)
if __name__=='__main__':main()
