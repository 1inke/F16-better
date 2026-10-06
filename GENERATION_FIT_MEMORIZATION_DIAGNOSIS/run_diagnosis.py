"""Fixed FIT256 memorization; unchanged protected objective, not a formal trial."""
from pathlib import Path
import importlib.util,sys,json,time,hashlib,argparse,random,os
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES'
spec=importlib.util.spec_from_file_location('memorization_online',SOURCE/'run_head.py')
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
p,r,m,c=a.p,a.r,a.m,a.c
def write(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def readonly_raw(split,cache):
    saved=a.n.RAW_LOAD(SOURCE/(split+'_observations.npz'))
    assert np.array_equal(saved['sample_id'],cache['sample_id'])
    return saved['observation']
a.raw_observations=readonly_raw
def digest(path):
    return p.sha256(Path(path))
def manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        files=json.loads((ROOT/'GENERATION_TEMPORAL_COMPONENT_SERIES/old_results_manifest.json').read_text())['files']
        for folder in ('GENERATION_TEMPORAL_COMPONENT_SERIES','GENERATION_REACHABILITY_DIAGNOSIS'):
            for f in (ROOT/folder).rglob('*'):
                if f.is_file() and '__pycache__' not in f.parts:files[str(f)]=digest(f)
        write(path.name,dict(files=files))
    files=json.loads(path.read_text())['files']
    for f,h in files.items():assert digest(f)==h,f
    return len(files)
def cached_forward(model,features,base,speed):
    # Exact frozen online observation features; no future/GT in this function.
    anchors=model.head(features).reshape(-1,20,3,2)*(speed[:,0,None,None,None]+.03)
    delta=torch.einsum('tp,bkpc->bktc',model.decoder.interpolation,anchors)
    raw=torch.diff(delta,dim=2,prepend=torch.zeros_like(base[:,:,:1]))
    limit=model.strength*(speed[:,0,None,None,None]+.03)
    residual=raw.clamp(-limit,limit)
    return base+residual.cumsum(2),residual,raw,limit
def setup():
    device=torch.device('cuda:0');assert torch.cuda.is_available()
    a.prepare_backend(torch.device('cpu'))
    cfg=c.settings(7,'zero')
    cache=p.load_npz(p.TRAIN_CACHE)
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert len(fit)==25912 and len(cal)==2879
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    scale=np.array([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32).clip(1e-6)
    prior=json.loads((SOURCE/'preflight.json').read_text())
    assert np.array_equal(scale,np.array(prior['fit_protection_scales'],np.float32))
    rng=np.random.default_rng(20261002)
    chosen={'fit':rng.choice(fit,256,replace=False),'cal':rng.choice(cal,256,replace=False)}
    assert not set(cache['sample_id'][chosen['fit']])&set(cache['sample_id'][chosen['cal']])
    p.seed_everything(7);model=a.OnlineHead(940,cfg).to(device).eval()
    assert sum(x.numel() for x in model.parameters() if x.requires_grad)==25158
    data={}
    for split,ids in chosen.items():
        observation,base,speed,future=[torch.from_numpy(v[ids]).to(device) for v in values[:4]]
        features=[]
        with torch.no_grad():
            for start in range(0,len(ids),32):
                f,t,native=model.observed_features(observation[start:start+32])
                assert (native-base[start:start+32]).abs().max()<2e-5
                features.append(f)
            features=torch.cat(features)
            out,res,_,_=cached_forward(model,features,base,speed)
            assert torch.equal(out,base) and res.abs().sum()==0
        data[split]=(features,base,speed,future,torch.from_numpy(cache['change_group'][ids]).to(device))
    return model,cfg,data,torch.from_numpy(scale).to(device),{k:cache['sample_id'][v] for k,v in chosen.items()},prior
def loss(model,batch,cfg,scale):
    features,base,speed,future,groups=batch
    out,res,raw,limit=cached_forward(model,features,base,speed)
    obj=m.objective(out,base,future,res,speed,groups,cfg,'balanced_protected',scale)
    return obj,out,res,raw,limit
def optimizer_for(model,cfg):
    return torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
def check_frozen(model):a.frozen_audit(model)
def metrics(model,batch,cfg,scale):
    with torch.no_grad():
        obj,out,res,raw,limit=loss(model,batch,cfg,scale)
        assert torch.isfinite(out).all() and out.shape==(256,20,12,2)
        assert (res.abs()<=limit+1e-6).all()
        error=(out-batch[3][:,None]).norm(dim=-1)
        ade=error.mean(-1);fde=error[:,:,-1];joint=ade+cfg['fde_loss_weight']*fde
        winner=joint.argmin(-1);index=torch.arange(len(out),device=out.device)
        saturated=(raw.abs()>=limit).float()
        rows=[]
        for name,mask in [('overall',torch.ones(256,dtype=torch.bool,device=out.device))]+[(name,batch[4]==i) for i,name in enumerate(('stable','transition','change'))]:
            if mask.any():rows.append(dict(group=name,n=int(mask.sum()),minade=float(ade.min(-1).values[mask].mean()),minfde=float(fde.min(-1).values[mask].mean()),samepathjoint=float(joint.min(-1).values[mask].mean()),winner_clip_fraction=float(saturated[index,winner][mask].mean()),allslot_clip_fraction=float(saturated[mask].mean())))
        return dict(objective=float(obj),rows=rows)
def sources(prior):
    paths=[Path(__file__),HERE/'independent_worker.py',HERE/'start_independent.ps1',SOURCE/'train_observations.npz',SOURCE/'protocol.json']
    return {**prior['source_hashes'],**{str(f):digest(f) for f in paths}}
def main():
    mode=argparse.ArgumentParser();mode.add_argument('--preflight',action='store_true');args=mode.parse_args()
    if args.preflight:
        count=manifest(create=True);model,cfg,data,scale,ids,prior=setup();opt=optimizer_for(model,cfg)
        b=tuple(x[:32] for x in data['fit'])
        for step in range(4):
            opt.zero_grad();obj,*_=loss(model,b,cfg,scale);obj.backward()
            assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.head.parameters())
            if step==3:assert model.head[0].weight.grad.abs().sum()>0
            opt.step();check_frozen(model)
        # Cached forward must remain identical after learning, not only at zero init.
        full=p.load_npz(p.TRAIN_CACHE);indices=np.array([np.flatnonzero(full['sample_id']==i)[0] for i in ids['fit'][:32]])
        values=r.prepared(full,False,fit=p.internal_split(len(full['sample_id']),cfg['calibration_fraction'])[0])
        obs=torch.from_numpy(values[0][indices]).cuda()
        with torch.no_grad():
            online,_=model(obs,b[1],b[2]);cached,*_=cached_forward(model,b[0],b[1],b[2]);assert torch.equal(online,cached)
        manifest();np.savez_compressed(HERE/'split_ids.npz',**ids)
        write('preflight.json',dict(status='PASS',old_files=count,source_hashes=sources(prior),fit_n=256,cal_n=256,all_fit_normalization_n=25912,parameters=25158,online_cached_forward_exact=True,all_original_weights_frozen=True,no_dev_test=True,formal_trial=False,epochs=120,batch_size=32,subset_seed=20261002,initial_seed=7))
        print('PREFLIGHT PASS fixed FIT/CAL256 original objective frozen25158',flush=True);return
    assert not (HERE/'RESULT.json').exists(),'Already completed'
    audit=json.loads((HERE/'preflight.json').read_text());assert audit['status']=='PASS'
    for f,h in audit['source_hashes'].items():assert digest(f)==h,f
    manifest();model,cfg,data,scale,ids,prior=setup()
    with np.load(HERE/'split_ids.npz') as saved:
        for key in ids:assert np.array_equal(saved[key],ids[key])
    opt=optimizer_for(model,cfg);generator=torch.Generator().manual_seed(7)
    history=[];start=time.perf_counter();begin=0
    resume=HERE/'resume.pth'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        assert saved['source_hashes']==audit['source_hashes'];model.load_state_dict(saved['state_dict']);opt.load_state_dict(saved['optimizer'])
        generator.set_state(saved['loader_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng']);history=saved['history'];begin=saved['epoch']+1
    for epoch in range(begin,121):
        gradients={}
        if epoch:
            model.train()
            for index in torch.randperm(256,generator=generator).split(32):
                batch=tuple(x[index.to(x.device)] for x in data['fit']);opt.zero_grad();obj,*_=loss(model,batch,cfg,scale);obj.backward()
                assert torch.isfinite(obj)
                gradients={name:float(param.grad.norm()) for name,param in model.head.named_parameters()}
                assert all(np.isfinite(v) for v in gradients.values());opt.step()
            check_frozen(model)
        model.eval()
        if epoch in (0,1,5,10,20,40,60,80,100,120):
            row=dict(epoch=epoch,fit=metrics(model,data['fit'],cfg,scale),cal=metrics(model,data['cal'],cfg,scale),gradient_norms=gradients)
            history.append(row);write('learning_curve.json',history);print('EPOCH '+str(epoch)+' '+json.dumps(row),flush=True)
        checkpoint=dict(epoch=epoch,state_dict=model.state_dict(),optimizer=opt.state_dict(),source_hashes=audit['source_hashes'],loader_rng=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),history=history)
        temporary=HERE/'resume.tmp';torch.save(checkpoint,temporary);os.replace(temporary,resume)
    check_frozen(model);count=manifest()
    for f,h in audit['source_hashes'].items():assert digest(f)==h,f
    write('RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY',seconds=time.perf_counter()-start,history=history,old_files_unchanged=count,source_hashes_unchanged=True,all_original_weights_frozen=True,future_in_forward=False,dev_test_access=False,formal_trial=False,scope='One fresh head, fixed FIT256 memorization and CAL256 description. Frozen online observed features cached exactly. Unchanged allFIT weights/scales and original protected loss/bounds. No checkpoint selection or major/generalization claim.'))
    print('COMPLETE PASS_DIAGNOSTIC_ONLY',flush=True)
if __name__=='__main__':main()
