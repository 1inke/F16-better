"""Saved-only, post-result exploratory E4 track-sign tests; old files immutable."""
import os,sys,json,hashlib,time,importlib.util,traceback
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np
from scipy.stats import binom,spearmanr
C=Path(__file__).resolve().parent;H=C.parent;P=H.parent
spec=importlib.util.spec_from_file_location('e4_cpu_guard',P/'completion_review/audit_saved.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return a.digest(p)
def load(p):return a.load(p)
def write(n,o):a.write(C/n,o)
def grouped(y,tracks):
    ids,ix=np.unique(tracks,return_inverse=True)
    return ids,np.bincount(ix,weights=y)/np.bincount(ix)
def sign_p(v):
    n=len(v);positive=int((np.asarray(v)>0).sum())
    return {'defined_tracks':n,'positive_tracks':positive,'null_or_negative_tracks':n-positive,'raw_p':float(binom.sf(positive-1,n,.5)) if n else 1.}
def holm(rows):
    order=np.argsort([r['raw_p'] for r in rows],kind='stable');running=0.
    for rank,i in enumerate(order):
        running=max(running,(len(rows)-rank)*rows[i]['raw_p']);rows[i]['holm_p']=min(1.,running);rows[i]['holm_reject_005']=rows[i]['holm_p']<.05
def median_ci(v):
    if not len(v):return None
    rng=np.random.default_rng(20261009);values=[]
    for _ in range(20):values.extend(np.median(v[rng.integers(0,len(v),(100,len(v)))],axis=1).tolist())
    return np.quantile(values,[.025,.5,.975]).tolist()
def fixtures():
    assert sign_p(np.ones(10))['raw_p']==2**-10
    assert sign_p(np.zeros(10))['raw_p']==sign_p(-np.ones(10))['raw_p']==1.
    rows=[{'raw_p':.01},{'raw_p':.04},{'raw_p':.02}];holm(rows)
    assert np.allclose([r['holm_p'] for r in rows],[.03,.04,.04])
    x=np.array([.3,.6,1.,1.5,2.5]);y=3.-2.*x;assert abs(((x-x.mean())*y).sum()/((x-x.mean())**2).sum()+2)<1e-14
def main():
    t=time.perf_counter();a.guard();fixtures()
    assert read(H/'completion_review/REVIEW.json')['status']=='PASS'
    locks=read(H/'source_manifest.json')['files'].copy()
    for n in ('SUMMARY.json','REVIEW.json'):locks[str(H/'completion_review'/n)]=sha(H/'completion_review'/n)
    for n in ('analyze.py','DESIGN.md'):locks[str(C/n)]=sha(C/n)
    def verify():
        for i,(p,h) in enumerate(locks.items()):
            if i%32==0:a.guard()
            assert sha(p)==h,('SOURCE_OR_OUTPUT',p)
    verify()
    summary=read(H/'completion_review/SUMMARY.json');direction=[];dose=[];attention=[]
    for v in summary['synthetic']:
        track=np.array(v['track'])
        for method in ('top1','meanK'):
            y=np.array(v['signed'][method]);ids,fraction=grouped((y>0).astype(float),track)
            direction.append(dict(fold=v['fold'],seed=v['seed'],side=v['side'],distance=v['distance'],method=method,n=len(y),tracks=len(ids),window_away_fraction=float((y>0).mean()),equal_track_away_fraction=float(fraction.mean()),track_median_away_fraction=float(np.median(fraction)),track_fraction_median_CI=median_ci(fraction),track_ids=ids.tolist(),track_fractions=fraction.tolist(),**sign_p(fraction-.5)))
    assert len(direction)==300;holm(direction)
    for fold in ('eth','hotel','univ','zara01','zara02'):
        for seed in (1,2,3):
            for side in (1,-1):
                group=sorted([r for r in summary['synthetic'] if r['fold']==fold and r['seed']==seed and r['side']==side],key=lambda r:r['distance'])
                assert len(group)==5 and all(g['ids']==group[0]['ids'] for g in group)
                x=np.array([r['distance'] for r in group]);xc=x-x.mean();tracks=np.array(group[0]['track'])
                for method in ('top1','meanK'):
                    y=np.array([r['signed'][method] for r in group]);slopes=(xc[:,None]*y).sum(0)/(xc@xc);ids,track_slope=grouped(slopes,tracks)
                    dose.append(dict(fold=fold,seed=seed,side=side,method=method,n=y.shape[1],tracks=len(ids),distance_means=y.mean(1).tolist(),near_minus_far_mean_m=float((y[0]-y[-1]).mean()),window_slope_mean=float(slopes.mean()),track_slope_median=float(np.median(track_slope)),track_slope_median_CI=median_ci(track_slope),track_ids=ids.tolist(),track_slopes=track_slope.tolist(),**sign_p(-track_slope)))
            data=load(H/f'data/{fold}_selected.npz');events=[]
            for slot in range(1,9):
                path=H/f'jobs/{fold}_seed{seed}/leave{slot}';done=read(path/'done.json')
                # SHA schema is read from the actual saved manifest, never guessed.
                for key in ('output_sha','outputs'):
                    if key in done:
                        for p,h in done[key].items():assert sha(p)==h,('OUTPUT',p)
                        locks.update(done[key])
                locks[str(path/'done.json')]=sha(path/'done.json')
                z=load(path/'outputs.npz');att=z['baseline_attention'];effects=np.linalg.norm(z['paths']-z['base_paths'],axis=-1).mean((1,2))
                valid=(data['valid']>slot)&np.isfinite(att)
                events.extend(zip(z['track'][valid].tolist(),att[valid].tolist(),effects[valid].tolist()))
            ids=sorted({e[0] for e in events});values=[];defined=[];undefined=[]
            for tr in ids:
                ev=np.array([[e[1],e[2]] for e in events if e[0]==tr])
                if len(ev)<3 or np.ptp(ev[:,0])==0 or np.ptp(ev[:,1])==0:undefined.append(tr);continue
                r=float(spearmanr(ev[:,0],ev[:,1]).statistic)
                if not np.isfinite(r):undefined.append(tr);continue
                values.append(r);defined.append(tr)
            val=np.array(values)
            attention.append(dict(fold=fold,seed=seed,active_slot_events=len(events),all_active_tracks=len(ids),undefined_tracks=undefined,defined_track_ids=defined,track_rhos=values,track_rho_mean=float(val.mean()) if len(val) else None,track_rho_median=float(np.median(val)) if len(val) else None,track_rho_median_CI=median_ci(val),**sign_p(val)))
    assert len(dose)==60 and len(attention)==15;holm(dose);holm(attention)
    verify();write('input_manifest.json',{'files':locks})
    out={'analysis_timing':'AFTER_RESULTS_EXPLORATORY_NOT_RETROACTIVE_PRIMARY_PREREGISTRATION','direction_300':direction,'dose_60':dose,'attention_15':attention,'direction_rejections':sum(r['holm_reject_005'] for r in direction),'dose_rejections':sum(r['holm_reject_005'] for r in dose),'attention_rejections':sum(r['holm_reject_005'] for r in attention),'far_distance_equivalence_not_tested':'No predeclared scientific equivalence margin; CI contains0 is not equivalence','independence_limitation':'tracks may share recording and social context; model seeds are not independent scenes'}
    write('RESULT.json',out);verify()
    write('REVIEW.json',{'status':'PASS_SAVED_ONLY_E4_STATISTICAL_COMPLETION','direction_family':300,'dose_family':60,'attention_family':15,'fixtures_PASS':True,'source_initial_final_SHA_PASS':True,'locked_files':len(locks),'no_model_forward_optimizer':True,'not_external_review':True,'elapsed_seconds':time.perf_counter()-t,'resource':a.RES})
    print(json.dumps({k:v for k,v in out.items() if k.endswith('rejections')}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        write('FAILURE.json',{'traceback':traceback.format_exc(),'resource':a.RES});raise
