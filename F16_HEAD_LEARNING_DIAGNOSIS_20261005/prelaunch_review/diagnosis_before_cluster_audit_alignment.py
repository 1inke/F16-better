"""Actual pulse parameter feedback and tinyFIT learning; original objective only."""
from pathlib import Path
from datetime import datetime
import argparse,ctypes,gc,hashlib,importlib.util,json,os,random,sys,time,traceback
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import numpy as np
import psutil
import torch
torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
PREV=ROOT/'F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004'
PULSE=ROOT/'GENERATION_PULSE_RESPONSE_SERIES_20261004'
GROUPS=ROOT/'F16_CUE_RESPONSE_CONCORDANCE_20261004/recovery_memory_20261004/jobs'
TASK='Trajectory-HeadLearning-Diagnosis-20261005'
SNAPS=(0,1,5,10,20,40,60,80,100,120)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);assert HERE in path.parents
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for z in iter(lambda:f.read(2**20),b''):h.update(z)
    return h.hexdigest()
def npz(path,keys):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in keys}
def stage(name,**kw):
    write(HERE/'progress.json',dict(stage=name,at=datetime.now().astimezone().isoformat(),**kw))
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
class Guard:
    def __init__(self):self.rss=0;self.private=0;self.physical=2**64;self.commit=2**64
    def check(self):
        v=psutil.Process().memory_info();s=Mem();s.length=ctypes.sizeof(s)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        self.rss=max(self.rss,v.peak_wset);self.private=max(self.private,v.peak_pagefile)
        self.physical=min(self.physical,s.avail_phys);self.commit=min(self.commit,s.avail_page)
        if max(v.rss,v.private)>6*2**30 or min(s.avail_phys,s.avail_page)<6*2**30:raise MemoryError('6GiB process or system reserve obstruction; STOP human')
        return dict(peak_RSS_GiB=self.rss/2**30,peak_private_GiB=self.private/2**30,min_physical_GiB=self.physical/2**30,min_commit_GiB=self.commit/2**30)
guard=Guard();guard.check()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);v=importlib.util.module_from_spec(spec);sys.modules[name]=v;spec.loader.exec_module(v);return v
h=module('tiny_pulse_original',PULSE/'run_head.py')
def verify(files,label):
    for i,(path,digest) in enumerate(files.items()):
        assert sha(path)==digest,('SHA mismatch',path)
        if i%4000==0:guard.check();stage(label,files_done=i,files_total=len(files))
    return len(files)
def snapshot():
    path=HERE/'old_results_manifest.json'
    if path.exists():return read(path)['files']
    files=dict(read(PREV/'old_results_manifest.json')['files'])
    for p in PREV.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.pending','.tmp') and p.name!='runtime_state.json':files[str(p)]=sha(p)
    write(path,dict(files=files,excluded_controls=['.memory','AUTONOMOUS_GENERATION_CYCLE.json','runtime_state.json'],all_old_failures_preserved=True));return files
def sources():
    files=dict(read(PREV/'preflight.json')['source_hashes'])
    files.update({str(p):sha(p) for p in list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/'DESIGN.md',HERE/'README.md',HERE/'protocol.json',HERE/'old_results_manifest.json',PULSE/'run_head.py']})
    for p in model_paths():files[str(p)]=sha(p)
    return files
def model_paths():return [PULSE/f'results/models/mean_std/seed_{s}/balanced_protected/constrained.pth' for s in (1,2,3)]
def setup(device):
    guard.check();cfg=read(HERE/'protocol.json')
    saved=torch.load(model_paths()[0],map_location='cpu',weights_only=False);original_cfg=saved['config'];del saved
    assert (original_cfg['learning_rate'],original_cfg['weight_decay'])==(.0003,.0001)
    _,_,original=h.n.prepare_backend(device);assert not any(p.requires_grad for p in original.parameters())
    original_state={k:v.detach().cpu().clone() for k,v in original.state_dict().items()}
    cache=npz(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz',['sample_id','candidate','motion','ade','fde','change_group'])
    enc=npz(ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz',['sample_id','past','tokens'])
    labels=npz(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz',['sample_id','future'])
    assert len(cache['sample_id'])==28791 and len(np.unique(cache['sample_id']))==28791
    assert np.array_equal(cache['sample_id'],enc['sample_id']) and np.array_equal(cache['sample_id'],labels['sample_id'])
    fit,cal=h.p.internal_split(28791,original_cfg['calibration_fraction'])
    assert np.array_equal(fit,np.arange(25912)) and np.array_equal(cal,np.arange(25912,28791))
    obs=np.zeros((28791,940),np.float32);obs[:,:16]=enc['past'].reshape(-1,16);obs[:,16:784]=enc['tokens'].reshape(-1,768)
    base=cache['candidate'][...,:24].reshape(-1,20,12,2).astype(np.float32)
    past=cache['motion'][:,:16].reshape(-1,8,2);vel=(past[:,-1]-past[:,-2]).astype(np.float32)
    speed=np.concatenate((np.linalg.norm(vel,axis=-1)[:,None],vel),1).astype(np.float32);future=labels['future'].astype(np.float32)
    assert np.allclose(np.linalg.norm(base-future[:,None],axis=-1).mean(-1),cache['ade'],atol=2e-5,rtol=0)
    values=(obs,base,speed,future,np.zeros((1,940),np.float32),np.ones((1,940),np.float32))
    _,weights=h.r.setup_weights(cache,fit,values,original_cfg,device)
    assert np.max(np.abs(weights-np.array(read(ROOT/'GENERATION_PROTECTED_BALANCE_SERIES/preflight.json')['weights'])))<1e-12
    scales=np.array([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32).clip(1e-6)
    assert np.array_equal(scales,np.array(read(ROOT/'GENERATION_CANDIDATE_EXCHANGE_SERIES/preflight.json')['fit_protection_scales'],np.float32))
    gp={}
    for name in ('FIT','CAL'):
        z=npz(GROUPS/(name+'_native.npz'),['sample_id','cue_first_compatible','class_0','GT_event','GT_uncertain','scene_family'])
        assert np.array_equal(z['sample_id'],cache['sample_id'][fit if name=='FIT' else cal]);gp[name]=z
    fitclear=gp['FIT']['GT_event'].astype(bool)&~gp['FIT']['GT_uncertain'].astype(bool)
    calclear=gp['CAL']['GT_event'].astype(bool)&~gp['CAL']['GT_uncertain'].astype(bool)
    targeted=np.flatnonzero(fitclear&gp['FIT']['cue_first_compatible'].astype(bool)&(gp['FIT']['class_0']==7));assert len(targeted)==576
    compatible=np.flatnonzero(calclear&gp['CAL']['cue_first_compatible'].astype(bool));assert len(compatible)==137
    coarse=compatible[gp['CAL']['class_0'][compatible]==7];assert len(coarse)==77
    rng=np.random.default_rng(cfg['subset_seed'])
    picked=np.concatenate((rng.choice(targeted,128,replace=False),rng.choice(np.setdiff1d(fit,targeted),128,replace=False)))
    calpick=25912+np.concatenate((compatible,rng.choice(np.setdiff1d(np.arange(2879),compatible),256,replace=False)))
    assert len(picked)==256 and len(calpick)==393 and not set(picked)&set(calpick)
    data={};indices={'FIT':picked,'CAL':calpick}
    for pop,ix in indices.items():
        local=ix if pop=='FIT' else ix-25912
        group=gp[pop]
        data[pop]=dict(batch=tuple(torch.from_numpy(v[ix].copy()).to(device) for v in (obs,base,speed,future,cache['change_group'])),ids=cache['sample_id'][ix],
            masks=dict(all=np.ones(len(ix),bool),targeted=np.arange(len(ix))<(128 if pop=='FIT' else 137),reference=np.arange(len(ix))>=(128 if pop=='FIT' else 137),
                compatible=group['cue_first_compatible'][local].astype(bool)&~group['GT_uncertain'][local].astype(bool)&group['GT_event'][local].astype(bool),coarse=group['cue_first_compatible'][local].astype(bool)&(group['class_0'][local]==7)&~group['GT_uncertain'][local].astype(bool)&group['GT_event'][local].astype(bool),clear=group['GT_event'][local].astype(bool)&~group['GT_uncertain'][local].astype(bool)),
            scene=group['scene_family'][local])
    # No future or group goes into observed native replay.
    audit_obs=obs[picked[:16]].copy();audit_base=base[picked[:16]].copy()
    bytes_estimate=sum(x.nbytes for x in (obs,base,speed,future))+sum(v.numel()*v.element_size() for d in data.values() for v in d['batch'])
    del cache,enc,labels,values,obs,base,speed,future;gc.collect();guard.check()
    return dict(cfg=original_cfg,data=data,scale=torch.tensor(scales,device=device),weights=weights,indices=indices,original=original,original_state=original_state,audit_obs=audit_obs,audit_base=audit_base,input_bytes=bytes_estimate)
def make(d,seed,arm,device):
    h.p.seed_everything(seed);cfg=dict(d['cfg'],seed=seed,set_context=arm)
    model=h.PulseResponseHead(940,cfg).to(device).eval()
    assert sum(v.numel() for v in model.parameters() if v.requires_grad)==39590
    return model
def frozen(model):h.frozen_audit(model)
def batch(d,pop,ix=None):
    z=d['data'][pop]['batch'];return z if ix is None else tuple(v[ix.to(v.device)] for v in z)
def forward_loss(model,b,d,details=False):
    o,base,speed,future,groups=b
    # Only o/base/speed enter predictor; original full K20 order maintained.
    feature,_=model.inputs(o);codes=model.head(feature);res,parts=model.components(codes,speed);out=base+res.cumsum(2)
    total=h.m.objective(out,base,future,res,speed,groups,d['cfg'],'balanced_protected',d['scale'])
    if not details:return total,out,res
    err=(out-future[:,None]).norm(dim=-1);q=err.mean(-1)+d['cfg']['fde_loss_weight']*err[:,:,-1]
    target=h.r.m.labels(speed[:,1:],future[:,None,-1]/12).squeeze(1)
    quality=(h.r.WEIGHTS[target]*q.min(-1).values).mean()
    protection=d['cfg']['protection_weight']*h.m.protection(out,base,future,groups,d['scale'],d['cfg'])
    return total,out,res,dict(quality=quality,protection=protection,regularizers=total-quality-protection),codes,parts
def grads(model,b,d):
    total,out,res,parts,codes,geometry=forward_loss(model,b,d,True)
    params=[(k,v) for k,v in model.named_parameters() if v.requires_grad]
    report={};vectors={}
    for key,loss in {**parts,'total':total}.items():
        gs=torch.autograd.grad(loss,[v for _,v in params],retain_graph=True,allow_unused=False)
        assert all(torch.isfinite(v).all() for v in gs)
        vectors[key]=torch.cat([v.flatten() for v in gs]);norms={k:float(v.norm()) for (k,_),v in zip(params,gs)}
        rows={name:0. for name in ('drift','amplitude','clock')}
        for (name,_),v in zip(params,gs):
            if name in ('head.4.weight','head.4.bias'):
                for tag,sl in [('drift',slice(0,2)),('amplitude',slice(2,4)),('clock',slice(4,6))]:rows[tag]+=float(v[sl].square().sum())
        report[key]=dict(loss=float(loss.detach()),parameter_norm=float(vectors[key].norm()),layers=norms,output_rows_norm={k:v**.5 for k,v in rows.items()})
    for name in ('protection','regularizers','total'):
        a,bv=vectors['quality'],vectors[name];den=float(a.norm()*bv.norm());report['cos_quality_'+name]=float(torch.dot(a,bv)/den) if den>0 else None
    codegrad=torch.autograd.grad(parts['quality'],codes,retain_graph=False)[0]
    report['quality_code_zero_slot_fraction']=float((codegrad.abs().sum(-1)==0).float().mean())
    report['clock_code_norm']=float(codegrad[...,4:].norm())
    report['scope']='Actual batch parameter gradient, not whole population/root cause; key.bias analytically cancels.'
    return report
def array_metrics(paths,future):
    err=np.linalg.norm(paths.astype(np.float64)-future[:,None].astype(np.float64),axis=-1)
    ade=err.mean(-1);fde=err[:,:,-1];good=(ade<=.2)&(fde<=.4)
    v=np.diff(paths.astype(np.float64),axis=2,prepend=np.zeros_like(paths[:,:,:1]))/.4;a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4
    metrics=dict(minade=ade.min(-1),minfde=fde.min(-1),joint=(ade+.5*fde).min(-1),good_count=good.sum(-1),coverage=good.any(-1),loose_coverage=((ade<=.3)&(fde<=.5)).any(-1))
    for key,value in [('speed',v),('accel',a),('jerk',j)]:
        z=np.linalg.norm(value,axis=-1).reshape(len(paths),-1);metrics[key+'_mean']=z.mean(-1);metrics[key+'_window_p95']=np.quantile(z,.95,axis=-1)
    return metrics,good
def clusters(paths,good):
    # Deterministic same complete-link algorithm, ALL K20 GT-free geometry.
    effective=np.zeros(len(paths),np.int16);allcount=effective.copy()
    for row,(path,valid) in enumerate(zip(paths,good)):
        dist=np.linalg.norm(path[:,None].astype(np.float64)-path[None,:].astype(np.float64),axis=-1)
        allowed=(dist.mean(-1)<=.10)&(dist[:,:,-1]<=.20);groups=[]
        for k in range(20):
            for group in groups:
                if all(allowed[k,t] for t in group):group.append(k);break
            else:groups.append([k])
        allcount[row]=len(groups);effective[row]=sum(valid[g].any() for g in groups)
    return effective,allcount
def evaluate(model,d,pop,savepath=None):
    model.eval();b=batch(d,pop);paths=[];resids=[]
    with torch.no_grad():
        for ix in torch.arange(len(b[0])).split(32):
            guard.check();_,out,res=forward_loss(model,batch(d,pop,ix),d)
            assert torch.isfinite(out).all() and out.shape[1:]==(20,12,2)
            limit=1.5*(batch(d,pop,ix)[2][:,0,None,None,None]+.03);assert (res.abs()<=limit+1e-6).all()
            paths.append(out.cpu().numpy());resids.append(res.cpu().numpy())
        paths=np.concatenate(paths);resids=np.concatenate(resids)
        total=h.m.objective(torch.from_numpy(paths).to(b[0].device),b[1],b[3],torch.from_numpy(resids).to(b[0].device),b[2],b[4],d['cfg'],'balanced_protected',d['scale'])
    metric,good=array_metrics(paths,b[3].cpu().numpy());eff,count=clusters(paths,good);metric.update(effective_good=eff,all_distinct=count)
    masks={**d['data'][pop]['masks'],**{name:b[4].cpu().numpy()==i for i,name in enumerate(('stable','transition','change'))}}
    summary={name:dict(n=int(mask.sum()),**{k:float(v[mask].mean()) for k,v in metric.items()}) for name,mask in masks.items() if mask.any()}
    # Original CAL rule descriptive only, never checkpoint/parameter selection.
    native,_=array_metrics(b[1].cpu().numpy(),b[3].cpu().numpy())
    feasible=all(metric[key][mask].mean()<=native[key][mask].mean()*1.01 for mask in (masks['all'],masks['stable'],masks['transition']) if mask.any() for key in ('minade','minfde'))
    if savepath:
        np.savez_compressed(savepath,sample_id=d['data'][pop]['ids'],trajectory=paths,residual=resids,future=b[3].cpu().numpy(),native=b[1].cpu().numpy(),**metric,**{'mask_'+k:v for k,v in masks.items()})
    return dict(objective=float(total),groups=summary,cal_feasible_description=bool(feasible))
def optimizer(model,d):return torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=d['cfg']['learning_rate'],weight_decay=d['cfg']['weight_decay'])
def epoch_step(model,opt,d,generator):
    model.train()
    for ix in torch.randperm(256,generator=generator).split(32):
        guard.check();loss,_,_=forward_loss(model,batch(d,'FIT',ix),d);assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        assert all(v.grad is None for v in model.decoder.parameters())
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad)
        torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],5);opt.step()
    frozen(model)
def rng_state(generator):return dict(loader_rng=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
def checkpoint(model,opt,generator,epoch,history,locks):
    return dict(epoch=epoch,state_dict=model.state_dict(),optimizer=opt.state_dict(),history=history,source_hashes=locks,**rng_state(generator))
def atomic_save(value,path):
    temp=path.with_suffix('.pending');torch.save(value,temp);os.replace(temp,path)
def restore(path,model,opt,generator,locks):
    s=torch.load(path,map_location='cpu',weights_only=False);assert s['source_hashes']==locks
    model.load_state_dict(s['state_dict']);opt.load_state_dict(s['optimizer']);generator.set_state(s['loader_rng']);torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng']);np.random.set_state(s['numpy_rng']);random.setstate(s['python_rng']);return s
def exact(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b
def resume_test(d,locks,device):
    records=[]
    for arm in ('zero','mean_std'):
        outputs=[]
        for mode in ('continuous','interrupted'):
            model=make(d,7,arm,device);opt=optimizer(model,d);gen=torch.Generator().manual_seed(7);history=[]
            epoch_step(model,opt,d,gen);history.append(evaluate(model,d,'CAL'))
            directory=HERE/'resume_smoke'/arm/mode;directory.mkdir(parents=True,exist_ok=True)
            path=directory/'epoch1.pth';atomic_save(checkpoint(model,opt,gen,1,history,locks),path)
            if mode=='interrupted':
                del model,opt;gc.collect();model=make(d,7,arm,device);opt=optimizer(model,d);s=restore(path,model,opt,gen,locks);history=s['history']
            epoch_step(model,opt,d,gen);history.append(evaluate(model,d,'CAL'))
            final=checkpoint(model,opt,gen,2,history,locks);atomic_save(final,directory/'epoch2.pth');outputs.append(torch.load(directory/'epoch2.pth',map_location='cpu',weights_only=False));del model,opt;gc.collect()
        exact(outputs[0],outputs[1]);records.append(dict(arm=arm,params_allbuffers_bool_optimizer_allRNG_history_CAL_exact=True));del outputs;gc.collect()
    write(HERE/'resume_check.json',dict(status='PASS',real_epochs=2,FIT=256,CAL=393,reports=records,no_CAL_checkpoint_selection=True))
def output_skip(done,locks):
    z=read(done);assert z['source_hashes']==locks
    verify(z['outputs'],'VERIFY_COMPLETED_RUN_OUTPUTS');return z
def preflight(device):
    assert torch.cuda.is_available();old=snapshot();verify(old,'PREFLIGHT_INITIAL_OLD_SHA');d=setup(device)
    ids={k:v['ids'] for k,v in d['data'].items()};np.savez_compressed(HERE/'split_ids.npz',**ids)
    np.savez_compressed(HERE/'split_indices.npz',**d['indices'])
    counts={};states=[];reports={};timings={}
    for arm in ('zero','mean_std'):
        model=make(d,7,arm,device);states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        out,res=model(*batch(d,'FIT')[:3]);assert torch.equal(out,batch(d,'FIT')[1]) and res.abs().sum()==0
        opt=optimizer(model,d);b=batch(d,'FIT',torch.arange(32));tim=[]
        for step in range(36):
            torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True);loss,_,_=forward_loss(model,b,d);loss.backward()
            if step==7:
                for name,v in model.named_parameters():
                    if v.requires_grad:
                        assert v.grad is not None and torch.isfinite(v.grad).all(),name
                        if name!='key.bias':assert v.grad.abs().sum()>0,name
                    else:assert v.grad is None,name
                clock=float(model.head[-1].weight.grad[4:].abs().sum()+model.head[-1].bias.grad[4:].abs().sum());assert (clock==0) if arm=='zero' else (clock>0)
            torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],5);opt.step();torch.cuda.synchronize();tim.append(time.perf_counter()-t)
        frozen(model);counts[arm]=sum(v.numel() for v in model.parameters() if v.requires_grad);reports[arm]=grads(model,b,d);timings[arm]=float(np.mean(tim[4:]))
        with torch.no_grad():
            out,res=model(*b[:3]);_,out2,res2=forward_loss(model,b,d);assert torch.equal(out,out2) and torch.equal(res,res2)
        del model,opt;gc.collect()
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0]);del states
    # True observed original F16 alignment, inputs only. Never DEV cache read.
    common=module('tiny_replay_common',ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/common.py');sys.modules['common']=common
    native=module('tiny_replay_native',ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/native.py')
    raw=npz(ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz',['sample_id','observation']);ix=d['indices']['FIT'][:16]
    assert np.array_equal(raw['sample_id'][ix],d['data']['FIT']['ids'][:16]);obs=raw['observation'][ix];original=d['original'];original.device=str(device)
    with torch.no_grad():replay=native.forward(original,torch.tensor(obs[:,160:304].reshape(-1,9,8,2),device=device),torch.tensor(obs[:,304],device=device))
    probe=make(d,7,'mean_std',device)
    with torch.no_grad():ctx=probe.native_inputs(torch.from_numpy(d['audit_obs']).to(device))[:,0,:64]
    alignment=dict(tokens=float((replay['tokens']-torch.from_numpy(d['audit_obs'][:,16:784].reshape(-1,8,96)).to(device)).abs().max()),path=float((replay['paths']-torch.from_numpy(d['audit_base']).to(device)).abs().max()),context=float((replay['E3']-ctx).abs().max()))
    assert max(alignment.values())<2e-5,alignment
    for k,v in original.state_dict().items():assert torch.equal(v.detach().cpu(),d['original_state'][k]),k
    del probe,raw,replay,ctx,original;d.pop('original');d.pop('original_state');gc.collect()
    locks=sources();resume_test(d,locks,device)
    # Exercise exact same source and output rejection/skip function used in run.
    fixture=HERE/'resume_smoke/skip_fixture.json';f=HERE/'split_ids.npz';write(fixture,dict(source_hashes=locks,outputs={str(f):sha(f)}));output_skip(fixture,locks)
    try:output_skip(fixture,{**locks,'bad_source':'0'})
    except AssertionError:pass
    else:raise AssertionError('Source mismatch accepted')
    bad=HERE/'resume_smoke/bad_fixture.json';write(bad,dict(source_hashes=locks,outputs={str(f):'0'}))
    try:output_skip(bad,locks)
    except AssertionError:pass
    else:raise AssertionError('Bad output accepted')
    verify(locks,'PREFLIGHT_FINAL_SOURCE_SHA');verify(old,'PREFLIGHT_FINAL_OLD_SHA')
    estimated_compute=sum(timings.values())*3*120*8/60
    report=dict(status='PASS',source_hashes=locks,old_files=len(old),source_files=len(locks),parameters=counts,paired_weights_allbuffers_exact=True,native_identity_exact=True,actual_GPU8step_gradients=reports,alignment=alignment,online_original_weights_buffers_exact=True,
        split_unique_ids=sum(len(v) for v in ids.values()),FIT=256,CAL=393,fullFIT_weights_scales_n=25912,noDEV_test=True,input_allocation_estimate_bytes=d['input_bytes'],saved_predictions_uncompressed_estimate_bytes=6*len(SNAPS)*649*20*12*2*4,
        memory=guard.check(),seconds_per_batch32=timings,compute_only_minutes=estimated_compute,benchmark_throwaway_not_saved=True,completed_SHA_skip_source_and_corruption_reject=True,resume='PASS',device=torch.cuda.get_device_name(device))
    write(HERE/'preflight.json',report);write(HERE/'MEMORY_PREFLIGHT.json',dict(status='PASS',**guard.check(),input_bytes=d['input_bytes'],process_budget_GiB=6,system_reserve_GiB=6,host_available_GB_approx=17))
    print('PREFLIGHT PASS '+json.dumps({k:report[k] for k in ('old_files','source_files','parameters','alignment','memory','seconds_per_batch32','compute_only_minutes','device')}),flush=True)
def audit_saved(path):
    with np.load(path,allow_pickle=False) as z:
        metrics,good=array_metrics(z['trajectory'],z['future']);eff,count=clusters(z['trajectory'],good);metrics.update(effective_good=eff,all_distinct=count)
        errors={k:float(np.max(np.abs(v.astype(float)-z[k].astype(float)))) for k,v in metrics.items()}
        assert max(errors.values())==0,errors
        assert len(z['sample_id'])==len(np.unique(z['sample_id'])) and np.isfinite(z['trajectory']).all()
        return dict(status='PASS',sample_ids=len(z['sample_id']),shape=list(z['trajectory'].shape),max_saved_metric_error=max(errors.values()))
def run(device):
    pre=read(HERE/'preflight.json');assert pre['status']=='PASS' and read(HERE/'resume_check.json')['status']=='PASS'
    locks=pre['source_hashes'];verify(locks,'INITIAL_SOURCE_SHA');old=snapshot();verify(old,'INITIAL_OLD_SHA');d=setup(device)
    with np.load(HERE/'split_ids.npz') as z:
        for key in d['data']:assert np.array_equal(z[key],d['data'][key]['ids'])
    del d['original'];d.pop('original_state');gc.collect();results=[]
    # Existing checkpoints only diagnostic no-optimizer gradients, never warmstart.
    for seed,path in enumerate(model_paths(),1):
        stage('EXISTING_HEAD_GRADIENT',seed=seed);saved=torch.load(path,map_location='cpu',weights_only=False);model=make(d,seed,'mean_std',device);model.load_state_dict(saved['state_dict']);frozen(model)
        write(HERE/f'gradient_existing_seed{seed}.json',dict(seed=seed,epoch=saved['epoch'],source=sha(path),FIT=grads(model,batch(d,'FIT',torch.arange(32)),d),CAL=grads(model,batch(d,'CAL',torch.arange(32)),d),no_optimizer=True));del model,saved;gc.collect()
    for arm in ('zero','mean_std'):
        for seed in (7,8,9):
            directory=HERE/'runs'/arm/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True);done=directory/'done.json'
            if done.exists():results.append(output_skip(done,locks)['summary']);continue
            model=make(d,seed,arm,device);opt=optimizer(model,d);gen=torch.Generator().manual_seed(seed);history=[];begin=0
            resume=directory/'resume.pth'
            if resume.exists():s=restore(resume,model,opt,gen,locks);history=s['history'];begin=s['epoch']+1
            for epoch in range(begin,121):
                stage('TINY_FIT_TRAINING',arm=arm,seed=seed,epoch=epoch,total_epochs=120,completed_runs=len(results))
                if epoch:epoch_step(model,opt,d,gen)
                if epoch in SNAPS:
                    row=dict(epoch=epoch)
                    for pop in ('FIT','CAL'):row[pop]=evaluate(model,d,pop,directory/f'{pop}_epoch{epoch}.npz')
                    if epoch in (0,1,20,120):row['gradients']=grads(model,batch(d,'FIT',torch.arange(32)),d)
                    history.append(row);write(directory/'history.json',history);print('TRAIN '+arm+' seed'+str(seed)+' epoch'+str(epoch)+'/120',flush=True)
                atomic_save(checkpoint(model,opt,gen,epoch,history,locks),resume)
            frozen(model);atomic_save(checkpoint(model,opt,gen,120,history,locks),directory/'final.pth')
            outputs={str(p):sha(p) for p in directory.glob('*') if p.name!='done.json' and p.suffix!='.pending'}
            for path in directory.glob('*.npz'):audit_saved(path)
            summary=dict(arm=arm,seed=seed,history=history,all_old_decoder_parameters_buffers_exact=True)
            write(done,dict(status='SUCCESS',source_hashes=locks,outputs=outputs,summary=summary));results.append(summary);del model,opt;gc.collect()
    stage('FINAL_PREDICTION_AND_SOURCE_AUDIT');audits=[]
    for done in (HERE/'runs').glob('*/seed*/done.json'):output_skip(done,locks)
    for path in (HERE/'runs').glob('*/seed*/*.npz'):audits.append(audit_saved(path))
    assert len(results)==6 and len(audits)==6*len(SNAPS)*2
    verify(locks,'FINAL_SOURCE_SHA');verify(old,'FINAL_OLD_SHA')
    write(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS',saved_predictions=len(audits),explicit_ids_finite_K20=True,all_saved_metrics_and_complete_link_clusters_recomputed=True,max_error=0,old_files=len(old),sources=len(locks),all_histories_120epochs=True,old_decoder_exact=True))
    write(HERE/'RESOURCE_USAGE.json',dict(status='PASS',**guard.check(),process_budget_GiB=6,system_reserve_GiB=6))
    write(HERE/'RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_HEAD_LEARNING',runs=results,old_files_unchanged=len(old),sources_unchanged=len(locks),formal_trial=False,response_study_completed=2,noDEV_test=True,GT_in_forward=False,scope='Enriched tinyFIT memorization and descriptive CAL only; neither deployment nor major/intent/root proof.'))
    write(HERE/'exit_status.json',dict(status='SUCCESS',at=datetime.now().astimezone().isoformat()));stage('SUCCESS',completed_runs=6)
    print('COMPLETE PASS_DIAGNOSTIC_ONLY_HEAD_LEARNING',flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args();device=torch.device('cuda:0')
    try:
        if args.preflight:preflight(device)
        else:run(device)
    except BaseException:
        folder=HERE/'failures'/datetime.now().strftime('%Y%m%d_%H%M%S');folder.mkdir(parents=True,exist_ok=True)
        write(folder/'failure.json',dict(traceback=traceback.format_exc(),source_sha=sha(__file__),preflight=args.preflight,resource=guard.__dict__,stop_wait_human=True))
        write(HERE/'exit_status.json',dict(status='FAILURE',at=datetime.now().astimezone().isoformat(),source=str(folder)));raise
if __name__=='__main__':main()
