"""Fixed FIT/CAL paired learning diagnosis. No optimizer, DEV or test."""
from pathlib import Path
import importlib.util,sys,json,csv,time,traceback
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PARENT=ROOT/'GENERATION_CONDITIONING_ADAPTATION_SERIES'
spec=importlib.util.spec_from_file_location('conditioning_learning_source',PARENT/'run_head.py')
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
p,m,r=h.p,h.m,h.r
def write(name,value):m.write_json(HERE/name,value)
def manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        files=json.loads((PARENT/'old_results_manifest.json').read_text())['files']
        for source in PARENT.rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.name!='runtime_state.json':files[str(source)]=p.sha256(source)
        write('old_results_manifest.json',dict(files=files))
    files=json.loads(path.read_text())['files']
    for source,digest in files.items():assert p.sha256(Path(source))==digest,source
    return len(files)
def predict(model,obs,base,speed,device):
    outputs=[];raws=[];residuals=[];features=[]
    with torch.no_grad():
        for start in range(0,len(obs),128):
            o,b,s=[torch.from_numpy(v[start:start+128]).to(device) for v in (obs,base,speed)]
            z=model.inputs(o);anchors=model.head(z).reshape(-1,20,3,2)*(s[:,0,None,None,None]+.03)
            delta=torch.einsum('tp,bkpc->bktc',model.decoder.interpolation,anchors)
            raw=torch.diff(delta,dim=2,prepend=torch.zeros_like(b[:,:,:1]))
            out,res=model(o,b,s);limit=model.strength*(s[:,0,None,None,None]+.03)
            # Epoch0 fallback is reported explicitly, not mistaken for a learned head.
            if not bool(model.native_fallback):assert torch.equal(out,b+raw.clamp(-limit,limit).cumsum(2))
            outputs.append(out.cpu().numpy());raws.append(raw.cpu().numpy());residuals.append(res.cpu().numpy());features.append(z.cpu().numpy())
    return [np.concatenate(x) for x in (outputs,raws,residuals,features)]
def main():
    assert not(HERE/'RESULT.json').exists(),'Already complete; preserve diagnostic'
    started=time.time();oldcount=manifest(not(HERE/'old_results_manifest.json').exists())
    print('OLD HASH PASS '+str(oldcount),flush=True)
    parent=json.loads((PARENT/'preflight.json').read_text())
    for path,digest in parent['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    device=torch.device('cuda:0');_,_,original=h.n.prepare_backend(torch.device('cpu'));del original
    cache=p.load_npz(p.TRAIN_CACHE);cfg=h.c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert (len(fit),len(cal))==(25912,2879)
    rng=np.random.default_rng(20261006);subsets=dict(FIT=np.sort(rng.choice(fit,2048,replace=False)),CAL=cal)
    assert not(set(subsets['FIT'])&set(subsets['CAL']))
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device);scale=torch.tensor(parent['fit_protection_scales'],device=device)
    splitpath=HERE/'split_ids.npz'
    if splitpath.exists():
        with np.load(splitpath) as priorids:
            for key,ids in subsets.items():assert np.array_equal(priorids[key],cache['sample_id'][ids])
    else:np.savez_compressed(splitpath,**{key:cache['sample_id'][ids] for key,ids in subsets.items()})
    p.seed_everything(1);native_model=h.ConditioningHead(940,cfg).to(device).eval();h.frozen_audit(native_model)
    native_features={};native_metrics={}
    for split,ids in subsets.items():
        o,b,s=[v[ids] for v in values[:3]];native_features[split]=predict(native_model,o,b,s,device)[3]
        native_metrics[split]=r.generation_metrics(b,b,s,values[3][ids])
    rows=[];history=[];sources={str(Path(__file__)):p.sha256(Path(__file__)),str(HERE/'README.md'):p.sha256(HERE/'README.md'),str(HERE/'split_ids.npz'):p.sha256(HERE/'split_ids.npz')}
    for arm in h.c.ARMS:
        for seed in (1,2,3):
            directory=PARENT/f'results/models/{arm}/seed_{seed}/balanced_protected'
            with (directory/'history.csv').open(newline='',encoding='utf-8-sig') as stream:curve=list(csv.DictReader(stream))
            assert len(curve)==80 and int(curve[-1]['epoch'])==80
            history.append(dict(arm=arm,seed=seed,rows=curve));sources[str(directory/'history.csv')]=p.sha256(directory/'history.csv')
            for tag in ('constrained','final'):
                path=directory/(tag+'.pth');saved=torch.load(path,map_location='cpu',weights_only=False);sources[str(path)]=p.sha256(path)
                model=h.ConditioningHead(saved['input_dim'],saved['config']).to(device).eval();model.load_state_dict(saved['state_dict']);h.frozen_audit(model)
                assert saved['config']['set_context']==arm
                for split,ids in subsets.items():
                    obs,base,speed,future=[v[ids] for v in values[:4]];groups=cache['change_group'][ids]
                    out,raw,res,z=predict(model,obs,base,speed,device);assert out.shape==(len(ids),20,12,2) and np.isfinite(out).all()
                    metrics=r.generation_metrics(out,base,speed,future);error=np.linalg.norm(out-future[:,None],axis=-1);quality=error.mean(-1)+cfg['fde_loss_weight']*error[:,:,-1];winner=quality.argmin(-1)
                    limit=1.5*(speed[:,0,None,None,None]+.03)
                    weightedloss=0.
                    with torch.no_grad():
                        for start in range(0,len(ids),128):
                            sl=slice(start,start+128);o,b,g,resid,s=[torch.from_numpy(v[sl]).to(device) for v in (out,base,future,res,speed)]
                            group=torch.tensor(groups[sl],device=device)
                            loss=m.objective(o,b,g,resid,s,group,saved['config'],'balanced_protected',scale)
                            weightedloss+=float(loss)*len(out[sl])
                    drift=z-native_features[split]
                    group_rows=[]
                    for name,mask in m.masks(groups):
                        assert mask.any();reference=native_metrics[split]
                        group_rows.append(dict(group=name,n=int(mask.sum()),metrics={key:float(v[mask].mean()) for key,v in metrics.items()},native={key:float(v[mask].mean()) for key,v in reference.items()},raw_clip_fraction=float((np.abs(raw[mask])>=limit[mask]).mean()),raw_limit_max=float((np.abs(raw[mask])/limit[mask]).max()),residual_rms=float(np.sqrt(np.mean(res[mask]**2))),context_drift_rms=float(np.sqrt(np.mean(drift[mask,:,:64]**2))),mode_drift_rms=float(np.sqrt(np.mean(drift[mask,:,64:]**2))),winner_occupancy=np.bincount(winner[mask],minlength=20).tolist()))
                    # Correction time profile isolates where linear-anchor slope changes act;
                    # no claim that temporal concentration proves the cause of dev failure.
                    acceleration=np.diff(res,axis=2)/(.4**2);jerk=np.diff(acceleration,axis=2)/.4
                    row=dict(arm=arm,seed=seed,policy=tag,epoch=saved['epoch'],native_fallback=bool(model.native_fallback),split=split,objective=weightedloss/len(ids),groups=group_rows,correction_acceleration_rms_by_step=np.sqrt(np.mean(acceleration**2,axis=(0,1,3))).tolist(),correction_jerk_rms_by_step=np.sqrt(np.mean(jerk**2,axis=(0,1,3))).tolist())
                    rows.append(row);print('EVALUATED '+json.dumps(dict(arm=arm,seed=seed,tag=tag,split=split,epoch=saved['epoch'],objective=row['objective'])),flush=True)
    manifest()
    for path,digest in sources.items():assert p.sha256(Path(path))==digest,path
    write('RESULT.json',dict(status='PASS_FIT_CAL_DESCRIPTIVE_ONLY',rows=rows,full_calibration_history=history,source_hashes=sources,old_files_unchanged=oldcount,fit_n=2048,cal_n=2879,seed=20261006,optimizer_steps=0,dev_test_access=False,elapsed_seconds=time.time()-started,limitations='Fixed FIT randomsubset and CAL are descriptive, not independent scene/generalization proof or causal bottleneck evidence. CAL used only to diagnose, not select dimensions/coefficients. All checkpoints immutable; no old results overwritten.'))
    print('DIAGNOSIS PASS',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        path=HERE/'failure.txt'
        if path.exists():path=HERE/('failure_'+str(time.time_ns())+'.txt')
        path.write_text(traceback.format_exc(),encoding='utf-8');raise
