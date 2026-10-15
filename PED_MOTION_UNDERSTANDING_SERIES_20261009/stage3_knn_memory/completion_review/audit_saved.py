"""Independent E6 saved-data review. No torch, model, optimizer, or old writes."""
import os, sys, json, time, hashlib, importlib.util, traceback
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import numpy as np
C = Path(__file__).resolve().parent
H = C.parent
P = H.parent
spec = importlib.util.spec_from_file_location('e6_cpu_resource_helpers', P/'completion_review/audit_saved.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
MAX = 0.; CHECKS = 0
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            h.update(part)
    return h.hexdigest()
def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}
def write(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def eq(x, y, tol=1e-10):
    global MAX, CHECKS
    CHECKS += 1
    if x is None or y is None:
        assert x is None and y is None
        return
    x, y = np.asarray(x), np.asarray(y)
    assert x.shape == y.shape
    err = float(abs(x.astype(float)-y.astype(float)).max()) if x.size else 0.
    MAX = max(MAX, err)
    assert err <= tol, (err, tol)
def exact(x, y):
    global CHECKS
    CHECKS += 1
    assert np.array_equal(x, y)
def cv(o):
    return (o[:,-1]-o[:,-2]).astype(float)[:,None]*np.arange(1,13)[None,:,None]
def errors(p,g):
    d = np.sqrt(((p-g)**2).sum(-1))
    return d.mean(-1), d[:,-1]
def ci(vals, tracks, den=None, ratio=False):
    _, inv = np.unique(tracks, return_inverse=True)
    s = np.bincount(inv, weights=vals)
    n = np.bincount(inv, weights=den) if den is not None else np.bincount(inv)
    rng = np.random.default_rng(20261009); out=[]
    for start in range(0,2000,100):
        ix=rng.integers(0,len(s),(min(100,2000-start),len(s)))
        top=s[ix].sum(1); bottom=n[ix].sum(1); ok=bottom>0
        out.extend((1-top[ok]/bottom[ok] if ratio else top[ok]/bottom[ok]).tolist())
    return np.quantile(out,[.025,.5,.975]).tolist() if out else None
def features(o, kind):
    v=(o[:,-1]-o[:,-2]).astype(float)
    return np.concatenate((o[:,-1].astype(float),v),1) if kind=='position_velocity' else v
def main():
    t=time.perf_counter(); a.guard()
    assert read(H/'worker_state.json')['status']==read(H/'exit_status.json')['status']=='SUCCESS'
    result=read(H/'RESULT.json')
    assert result['status']=='PASS_STAGE3_TRAIN_ONLY_KNN_MEMORY' and len(result['jobs'])==14
    locks=read(H/'source_manifest.json')['files'].copy()
    for d in result['jobs']:
        locks.update(d['outputs'])
        assert d['source_sha']==sha(H/'source_manifest.json')
        for p in d['outputs']:
            locks[str(Path(p).parent/'done.json')]=sha(Path(p).parent/'done.json')
    for name in ('RESULT.json','ARITHMETIC_AUDIT.json','RESOURCE_USAGE.json','worker_state.json','exit_status.json'):
        locks[str(H/name)]=sha(H/name)
    locks[str(Path(__file__))]=sha(__file__)
    def verify():
        for i,(p,h) in enumerate(locks.items()):
            if i%32==0:a.guard()
            assert sha(p)==h, ('SOURCE_OR_OUTPUT',p)
    verify();write(C/'input_manifest.json',{'files':locks})
    output=[]; removal=[]; nearest_rows=0; all_rows=0
    for fold in ('eth','hotel','univ','zara01','zara02'):
        bank=load(H/f'data/{fold}_bank.npz'); test=load(P/f'data/{fold}_evaluation.npz')
        ref=load(P/f'jobs/{fold}_seed1/outputs.npz')
        masks={k[5:]:v for k,v in ref.items() if k.startswith('mask_')}
        g=test['gt']; last=cv(test['obs'][:,0]); den=((g-last)**2).sum((1,2))
        assert not np.intersect1d(bank['track'],test['track']).size
        comparison={}
        for seed in (1,2,3):
            z=load(P/f'jobs/{fold}_seed{seed}/outputs.npz')
            for name,p in [('top1',z['model_paths'][np.arange(len(g)),z['model_logits'].argmax(1)]),('meanK',z['model_paths'].mean(1))]:
                comparison[f'model_seed{seed}_{name}']=errors(p,g)
        cal=read(P/f'calibration/{fold}.json')
        simple={'CV_last':last,'CV_avg3':a.cv(test['obs'],3),'CA':a.ca(test['obs'],cal['ca_cap']),'Kalman':a.kalman(test['obs'],cal['kalman']['q'],cal['kalman']['r'])}
        name='strongest_simple_'+cal['strongest_simple'];comparison[name]=errors(simple[cal['strongest_simple']],g)
        cached={}
        for d in [r for r in result['jobs'] if r['fold']==fold]:
            print('INDEPENDENT_REVIEW', fold,d['features'],d['subset'],flush=True);a.guard()
            directory=H/'jobs'/f'{fold}_{d["features"]}_{d["subset"]}'
            z=load(directory/'outputs.npz'); old=read(directory/'metrics.json')['records']
            exact(z['sample_id'],test['ids']);exact(z['track'],test['track']);eq(z['GT'],g,0);eq(z['CV'],last,0)
            use=np.ones(len(bank['ids']),bool) if d['subset']=='all_train' else bank['scene']!=read(H/'protocol.json')['sibling_exclusion'][fold]
            exact(z['train_ids'],bank['ids'][use]); ids=bank['ids'][use]
            X=features(bank['obs'][use],d['features']);mean=X.mean(0);std=X.std(0);std=np.where(std<1e-6,1.,std)
            eq(mean,z['stats_mean'],0);eq(std,z['stats_std'],0)
            lookup={int(ID):i for i,ID in enumerate(ids)}
            selected=np.array([[lookup[int(ID)] for ID in row] for row in z['neighbor_ids']])
            assert selected.shape==(len(g),32)
            assert all(len(set(row))==32 for row in z['neighbor_ids'])
            q=(features(test['obs'][:,0],d['features'])-mean)/std;X=(X-mean)/std
            for start in range(0,len(q),128):
                a.guard();end=min(len(q),start+128)
                eq(np.linalg.norm(X[selected[start:end]]-q[start:end,None],axis=-1),z['distances'][start:end],0)
            for i in range(min(32,len(q))):
                dist=np.linalg.norm(X-q[i],axis=-1);ix=np.lexsort((ids,dist))[:32]
                exact(ids[ix],z['neighbor_ids'][i]);nearest_rows+=1
            pred=last+(bank['gt'][use]-cv(bank['obs'][use]))[selected].mean(1)
            eq(pred,z['pred'],0); aa,ff=errors(pred,g);good=(aa<=.2)&(ff<=.4);num=((g-pred)**2).sum((1,2))
            v=(test['obs'][:,0,-1]-test['obs'][:,0,-2]).astype(float);sp=np.linalg.norm(v,axis=1)
            direction=np.divide(v,sp[:,None],out=np.zeros_like(v),where=sp[:,None]>0);normal=np.stack((-direction[:,1],direction[:,0]),1)
            records=[]
            for rec in old:
                m=masks[rec['subset']];n=int(m.sum());tracks=len(np.unique(test['track'][m]))
                rr={'subset':rec['subset'],'n':n,'tracks':tracks,'ADE':float(aa[m].mean()),'FDE':float(ff[m].mean()),'single_path_coverage':int(good[m].sum()),'skill':float(1-num[m].sum()/den[m].sum()) if den[m].sum()>0 else None,'skill_track_CI':ci(num[m],test['track'][m],den[m],True) if den[m].sum()>0 else None,'comparisons':[]}
                for k,val in rr.items():
                    if k not in ('subset','comparisons'):eq(val,rec[k])
                for cmp in rec['comparisons']:
                    ba,bf=comparison[cmp['method']]
                    cp={'method':cmp['method'],'delta_ADE':float((aa-ba)[m].mean()),'delta_FDE':float((ff-bf)[m].mean()),'ADE_track_CI':ci((aa-ba)[m],test['track'][m]),'FDE_track_CI':ci((ff-bf)[m],test['track'][m])}
                    for k in cp:
                        if k!='method':eq(cp[k],cmp[k])
                    rr['comparisons'].append(cp)
                for axis,vec in [('longitudinal',direction),('lateral',normal)]:
                    truth=((g[:,-1]-last[:,-1])*vec).sum(1);model=((pred[:,-1]-last[:,-1])*vec).sum(1);valid=m&(sp>=.02)&(abs(truth)>1e-12)
                    nn=int(valid.sum());acc=float((np.sign(truth[valid])==np.sign(model[valid])).mean()) if nn else None
                    eq(nn,rec[axis+'_n'],0);eq(acc,rec[axis+'_sign_accuracy'])
                    rr[axis+'_n']=nn;rr[axis+'_sign_accuracy']=acc
                records.append(rr)
            retained={str(s):int((bank['scene'][use]==s).sum()) for s in np.unique(bank['scene'][use])}
            output.append({'fold':fold,'features':d['features'],'train_subset':d['subset'],'train_n':len(ids),'retained_recordings':retained,'records':records})
            cached[(d['features'],d['subset'])]=(aa,ff,num,good)
            all_rows+=len(g)
        if fold in ('zara01','zara02'):
            for kind in ('position_velocity','velocity_only'):
                base=cached[(kind,'all_train')];without=cached[(kind,'no_sibling')]
                for sub,m in masks.items():
                    if not m.any():continue
                    removal.append({'fold':fold,'features':kind,'subset':sub,'n':int(m.sum()),'tracks':len(np.unique(test['track'][m])),'ADE_delta_without_minus_all':float((without[0]-base[0])[m].mean()),'ADE_track_CI':ci((without[0]-base[0])[m],test['track'][m]),'FDE_delta_without_minus_all':float((without[1]-base[1])[m].mean()),'FDE_track_CI':ci((without[1]-base[1])[m],test['track'][m]),'coverage_delta_without_minus_all':float((without[3].astype(float)-base[3])[m].mean()),'coverage_track_CI':ci((without[3].astype(float)-base[3])[m],test['track'][m])})
    verify()
    write(C/'SUMMARY.json',{'records':output,'paired_sibling_removal':removal})
    write(C/'REVIEW.json',{'status':'PASS','jobs':14,'saved_prediction_rows':all_rows,'all_selected_neighbor_membership_distances_and_predictions_verified':True,'stable_brute_nearest_queries':nearest_rows,'all_registered_metrics_and_CIs_recomputed':True,'retained_recording_composition_recomputed':True,'checks':CHECKS,'max_numeric_error':MAX,'initial_final_SHA_PASS':True,'locked_files':len(locks),'no_torch_models_optimizer':True,'not_external_review':True,'not_full_nearest_search_second_pass':True,'elapsed_seconds':time.perf_counter()-t,'resource':a.RES})
    print(json.dumps(read(C/'REVIEW.json')),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        write(C/'FAILURE.json',{'status':'FAILURE','traceback':traceback.format_exc(),'resource':a.RES});raise
