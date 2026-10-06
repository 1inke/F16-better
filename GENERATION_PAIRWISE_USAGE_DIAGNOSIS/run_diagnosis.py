"""Completed trial re-audit and fixed FIT/CAL branch usage; no optimizer."""
from pathlib import Path
import importlib.util,sys,json,csv,time,traceback
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PARENT=ROOT/'GENERATION_PAIRWISE_HISTORY_SERIES'
spec=importlib.util.spec_from_file_location('pairwise_usage_source',PARENT/'run_head.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(name,value):(HERE/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
def main():
    assert not(HERE/'RESULT.json').exists(),'Already complete'
    started=time.time();files=read(PARENT/'old_results_manifest.json')['files']
    for path in PARENT.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.name not in ('runtime_state.json','independent_worker.lock'):files[str(path)]=h.p.sha256(path)
    for path,digest in files.items():assert h.p.sha256(Path(path))==digest,path
    write('old_results_manifest.json',dict(files=files));pre=read(PARENT/'preflight.json')
    for path,digest in pre['source_hashes'].items():assert h.p.sha256(Path(path))==digest,path
    lock=read(PARENT/'results/model_lock.json');assert len(lock['models'])==18
    for path,digest in lock['models'].items():assert h.p.sha256(Path(path))==digest,path
    assert read(PARENT/'results/exit_status.json')['status']=='SUCCESS'
    _,_,original=h.n.prepare_backend(torch.device('cpu'));del original
    cache=h.p.load_npz(h.p.DEV_CACHE);dim=cache['motion'].shape[1]+cache['social'].shape[1]
    values=h.r.prepared(cache,False,mean=np.zeros((1,dim),np.float32),std=np.ones((1,dim),np.float32));base,speed,future=values[1:4]
    stats={};history=[];summary=read(PARENT/'results/summary.json')
    for arm in h.c.ARMS:
        for seed in (1,2,3):
            modeldir=PARENT/f'results/models/{arm}/seed_{seed}/balanced_protected'
            with (modeldir/'history.csv').open(encoding='utf-8-sig',newline='') as stream:curve=list(csv.DictReader(stream))
            assert [int(row['epoch']) for row in curve]==list(range(1,81));history.append(dict(arm=arm,seed=seed,rows=curve))
            for tag in ('constrained','legacy','final'):
                saved=torch.load(modeldir/(tag+'.pth'),map_location='cpu',weights_only=False);model=h.PairwiseHead(saved['input_dim'],saved['config']);model.load_state_dict(saved['state_dict']);h.frozen_audit(model)
                data=h.n.RAW_LOAD(PARENT/f'results/seed{seed}_{arm}_{tag}.npz')
                assert np.array_equal(data['sample_id'],cache['sample_id']) and data['trajectory'].shape==(1516,20,12,2) and np.isfinite(data['trajectory']).all()
                actual=h.r.generation_metrics(data['trajectory'],base,speed,future)
                for key,value in actual.items():assert np.allclose(value,data[key],atol=1e-6,rtol=1e-6),(seed,arm,tag,key)
                stats[seed,arm,tag]=actual
    for row in summary['aggregate']:
        mask=next(mask for name,mask in h.m.masks(cache['change_group']) if name==row['group'])
        for key in stats[1,row['arm'],row['policy']]:assert np.isclose(np.mean([stats[s,row['arm'],row['policy']][key][mask].mean() for s in (1,2,3)]),row[key],atol=1e-6,rtol=1e-6),key
    print('INDEPENDENT18SHA/18pred/allmetrics/frozen/history/source PASS',flush=True)
    device=torch.device('cuda:0');cache=h.p.load_npz(h.p.TRAIN_CACHE);fit,cal=h.p.internal_split(len(cache['sample_id']),.1);values=h.r.prepared(cache,False,fit=fit)
    rng=np.random.default_rng(20261008);subsets={'FIT':np.sort(rng.choice(fit,2048,replace=False)),'CAL':np.sort(rng.choice(cal,512,replace=False))}
    np.savez_compressed(HERE/'split_ids.npz',**{key:cache['sample_id'][ids] for key,ids in subsets.items()})
    rows=[]
    for seed in (1,2,3):
        for tag in ('constrained','final'):
            path=PARENT/f'results/models/mean_std/seed_{seed}/balanced_protected/{tag}.pth';saved=torch.load(path,map_location='cpu',weights_only=False)
            model=h.PairwiseHead(saved['input_dim'],saved['config']).to(device).eval();model.load_state_dict(saved['state_dict']);h.frozen_audit(model)
            for split,ids in subsets.items():
                factual=[];zeroed=[];res=[];pair=[]
                with torch.no_grad():
                    for start in range(0,len(ids),128):
                        current=ids[start:start+128];o,b,s=[torch.from_numpy(v[current]).to(device) for v in values[:3]]
                        model.arm='mean_std';out,r=model(o,b,s);pair.append(model.pair_inputs(o).cpu().numpy());factual.append(out.cpu().numpy());res.append(r.cpu().numpy())
                        model.arm='zero';counter,_=model(o,b,s);zeroed.append(counter.cpu().numpy());model.arm='mean_std'
                factual,zeroed,res,pair=[np.concatenate(v) for v in (factual,zeroed,res,pair)]
                assert np.isfinite(factual).all() and np.isfinite(zeroed).all()
                b,s,g=[v[ids] for v in values[1:4]];groups=cache['change_group'][ids];native=h.r.generation_metrics(b,b,s,g);actual=h.r.generation_metrics(factual,b,s,g);counter=h.r.generation_metrics(zeroed,b,s,g)
                limit=1.5*(s[:,0,None,None,None]+.03);group_rows=[]
                for name,mask in h.m.masks(groups):
                    group_rows.append(dict(group=name,n=int(mask.sum()),native={k:float(v[mask].mean()) for k,v in native.items()},factual={k:float(v[mask].mean()) for k,v in actual.items()},zeroed_same_checkpoint={k:float(v[mask].mean()) for k,v in counter.items()},prediction_sensitivity_mean_m=float(np.linalg.norm(factual[mask]-zeroed[mask],axis=-1).mean()),residual_rms=float(np.sqrt(np.mean(res[mask]**2))),postclip_bound_fraction=float((np.abs(res[mask])>=limit[mask]-1e-7).mean()),pooled_feature_std_mean=float(pair[mask].std(0).mean())))
                rows.append(dict(seed=seed,policy=tag,epoch=saved['epoch'],native_fallback=bool(model.native_fallback),split=split,groups=group_rows));print('DIAGNOSED '+str((seed,tag,split)),flush=True)
    for path,digest in files.items():assert h.p.sha256(Path(path))==digest,path
    sources={str(path):h.p.sha256(path) for path in (Path(__file__),HERE/'README.md',HERE/'split_ids.npz',PARENT/'results/summary.json')}
    write('RESULT.json',dict(status='PASS_AUDIT_AND_FIT_CAL_USAGE_ONLY',audit=dict(models=18,predictions=18,explicit_ids=True,finiteK20=True,allmetrics_recomputed=True,alloriginal_frozen=True,old_files_unchanged=len(files)),formal_summary=summary,all80epoch_histories=history,rows=rows,source_hashes=sources,optimizer_steps=0,fit_n=2048,cal_n=512,seed=20261008,elapsed_seconds=time.time()-started,limitations='DEV stored predictions only re-audited, no new DEV model forward/test. FIT/CAL zeroed same-checkpoint intervention is posthoc model reliance, not causal input benefit or an alternative deployed baseline; standardized zero may be off-distribution. GT/groups only metric loss description, not model forward. Only repeated ETH/three training seeds, not independent data confirmation.'))
    print('FINAL OLDHASH/DIAGNOSIS PASS',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (HERE/('failure_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8');raise
