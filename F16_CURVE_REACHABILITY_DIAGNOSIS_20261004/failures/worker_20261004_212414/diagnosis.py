"""GT-assisted curve witnesses; immutable network replay, no model training."""
from pathlib import Path
from datetime import datetime
import argparse, gc, hashlib, importlib.util, json, os, sys, time, traceback, shutil
sys.dont_write_bytecode=True
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
PREV=ROOT/'F16_CUE_RESPONSE_CONCORDANCE_20261004/recovery_memory_20261004'
PULSE=ROOT/'GENERATION_PULSE_RESPONSE_SERIES_20261004'
def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path); v=importlib.util.module_from_spec(spec)
    sys.modules[name]=v;spec.loader.exec_module(v);return v
infra=mod('reachability_memory_infra',PREV/'diagnosis.py')
old=infra.base;old.HERE=HERE
read,write,digest,save=old.read,old.write,old.digest,old.save
monitor=infra.Monitor()
P=read(HERE/'protocol.json');T=np.arange(1,13,dtype=float);MID=(np.arange(12)+.5)/12
def stage(name,**kw):old.stage(name,**kw)
def verify(files,name):return old.verify(files,name)
def snapshot():
    target=HERE/'old_results_manifest.json'
    if target.exists():return read(target)['files']
    files=dict(read(PREV/'old_results_manifest.json')['files'])
    for p in PREV.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.pending','.tmp') and p.name!='runtime_state.json':files[str(p)]=digest(p)
    write(target,dict(files=files,excludes=['.memory','AUTONOMOUS_GENERATION_CYCLE.json','runtime_state.json','locks/temp/bytecode'],prior_completed_and_failure_artifacts_preserved=True))
    return files
def sources():
    locks=dict(read(PREV/'preflight.json')['source_hashes'])
    for p in list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/'README.md',HERE/'DESIGN.md',HERE/'protocol.json',HERE/'old_results_manifest.json']:
        if p.exists():locks[str(p)]=digest(p)
    for name in ('FIT_native','CAL_native'):
        for suffix in ('.npz','.done.json','.audit.json'):locks[str(PREV/'jobs'/(name+suffix))]=digest(PREV/'jobs'/(name+suffix))
    models=read(PULSE/'results/model_lock.json')['models']
    for k,v in models.items():
        if '/mean_std/' in k.replace('\\','/') and k.endswith('constrained.pth'):locks[k]=v
    return locks
def npz(path,keys=None):return old.npz(path,keys)
def load_data():
    # Never call the parent load_populations: it includes DEV. Only locked train arrays.
    cache=npz(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz',['sample_id','candidate','motion'])
    enc=npz(ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz',['sample_id','past','tokens'])
    labels=npz(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz',['sample_id','future'])
    assert np.array_equal(cache['sample_id'],enc['sample_id']) and np.array_equal(cache['sample_id'],labels['sample_id'])
    assert len(cache['sample_id'])==28791 and len(np.unique(cache['sample_id']))==28791
    ids=cache['sample_id'];base=cache['candidate'][...,:24].reshape(-1,20,12,2)
    obs=np.zeros((len(ids),940),np.float32);obs[:,:16]=enc['past'].reshape(-1,16);obs[:,16:784]=enc['tokens'].reshape(-1,768)
    groups={}
    for name,sl in [('FIT',slice(0,25912)),('CAL',slice(25912,None))]:
        z=npz(PREV/'jobs'/(name+'_native.npz'));assert np.array_equal(z['sample_id'],ids[sl])
        groups[name]={k:z[k] for k in ('sample_id','scene_family','GT_event','GT_uncertain','GT_kind','cue_first_compatible','class_0','selected_slot','sequence','track','ambiguous_track')}
    return dict(ids=ids,base=base,past=enc['past'],obs=obs,motion=cache['motion'],future=labels['future'],groups=groups)
def profile(c,s):
    g=np.exp(-.5*((MID-c[...,None])/s[...,None])**2);g-=g.mean(-1,keepdims=True)
    return g/np.maximum(np.abs(g).max(-1,keepdims=True),1e-6)
def decode(x,L):
    pr=profile(x[:,4],x[:,5]);u=.5*L[:,None,None]*(x[:,:2,None].transpose(0,2,1)+pr[:,:,None]*x[:,2:4,None].transpose(0,2,1))
    return np.cumsum(np.clip(u,-L[:,None,None],L[:,None,None]),axis=1)
def metric(delta,y):
    e=np.linalg.norm(delta-y,axis=-1);return e.mean(-1),e[:,-1]
def score(a,f):return np.maximum(a/.2,f/.4)
def bounded(x):
    e=P['parameter_margin'];return np.clip(x,[-1+e]*4+[.05+e,.05+e],[1-e]*4+[.95-e,.5-e])
def codes(x):
    return np.concatenate((np.arctanh(x[:,:4]),np.log((x[:,4:5]-.05)/(.95-x[:,4:5])),np.log((x[:,5:6]-.05)/(.5-x[:,5:6]))),1)
def production_decode(x,L):
    # Independent float32 reconstruction of the actual original code operations.
    import torch
    with torch.inference_mode():
        z=torch.tensor(codes(x).astype(np.float32));lim=torch.tensor(L.astype(np.float32))[:,None,None]
        c=.05+.9*z[:,4].sigmoid();s=.05+.45*z[:,5].sigmoid();tm=(torch.arange(12,dtype=torch.float32)+.5)/12
        g=torch.exp(-.5*((tm-c[:,None])/s[:,None]).square());g-=g.mean(-1,keepdim=True)
        pr=g/g.abs().amax(-1,keepdim=True).clamp_min(1e-6)
        u=.5*lim*(z[:,:2,None].transpose(1,2).tanh()+z[:,2:4,None].transpose(1,2).tanh()*pr[:,:,None])
        u=u.clamp(-lim,lim);return u.cumsum(1).numpy(),u.numpy(),pr.numpy()
def jacobian(x,L):
    n=len(x);pr=profile(x[:,4],x[:,5]);B=np.cumsum(pr,axis=-1)
    j=np.zeros((n,12,2,6));j[:,:,0,0]=.5*L[:,None]*T;j[:,:,1,1]=.5*L[:,None]*T
    j[:,:,0,2]=.5*L[:,None]*B;j[:,:,1,3]=.5*L[:,None]*B
    for k,h in [(4,1e-5),(5,1e-5)]:
        xp=x.copy();xm=x.copy();xp[:,k]+=h;xm[:,k]-=h
        derivative=np.cumsum((profile(xp[:,4],xp[:,5])-profile(xm[:,4],xm[:,5]))/(2*h),-1)
        j[:,:,:,k]=.5*L[:,None,None]*derivative[:,:,None]*x[:,None,2:4]
    return j
def loss(r):return (r*r).sum((1,2))+P['endpoint_squared_error_weight']*(r[:,-1]*r[:,-1]).sum(1)
def pulse_fit(base,gt,L):
    n,k=base.shape[:2]; y=(gt[:,None]-base).reshape(-1,12,2).astype(float);ell=np.repeat(L,20)
    size=len(y);zero=np.tile([0,0,0,0,.5,.275],(size,1)).astype(float)
    best=zero.copy();bestd=decode(best,ell);ba,bf=metric(bestd,y);bs=score(ba,bf)
    candidates=[];profiles=[(c,s) for c in P['grid_centers'] for s in P['grid_spans']]+[tuple(P['include_fixed_profile'])]
    for c,s in profiles:
        pr=profile(np.array([c]),np.array([s]))[0];B=np.cumsum(pr);X=np.stack((T,B),1)
        w=np.ones(12);w[-1]+=P['endpoint_squared_error_weight'];pinv=np.linalg.solve(X.T@(w[:,None]*X),X.T*w)
        coeff=np.einsum('it,ntd->nid',pinv,y)/(ell[:,None,None]*.5)
        x=np.column_stack((coeff[:,0],coeff[:,1],np.full(size,c),np.full(size,s)));x=bounded(x)
        # Endpoint-targeted initialization provides a second shape-fit with exact feasible drift.
        xe=x.copy();xe[:,:2]=y[:,-1]/(ell[:,None]*6);xe=bounded(xe)
        remainder=y-.5*ell[:,None,None]*T[None,:,None]*xe[:,None,:2]
        xe[:,2:4]=(B[None,:,None]*remainder).sum(1)/((B*B).sum()*ell[:,None]*.5);xe=bounded(xe)
        for cur in (x,xe):
            d=decode(cur,ell);a,f=metric(d,y);sco=score(a,f);take=sco<bs
            best[take]=cur[take];bestd[take]=d[take];bs[take]=sco[take]
            candidates.append((sco,cur))
    lbvec=np.maximum(np.abs(y)-ell[:,None,None]*T[None,:,None],0)
    lb=np.linalg.norm(lbvec,axis=-1);excluded=(lb.mean(1)>.2+2e-5)|(lb[:,-1]>.4+2e-5)
    active=(bs>1-1e-4)&~excluded
    order=np.argsort(np.stack([q[0] for q in candidates]),axis=0,kind='stable')[:P['starts']]
    ci=np.stack([q[1] for q in candidates])
    iterations=np.zeros(size,np.int16)
    for start in range(P['starts']):
        idx=np.flatnonzero(active);x=ci[order[start,idx],idx].copy();l=ell[idx];target=y[idx]
        damping=np.full(len(idx),P['LM_initial_damping'])
        if not len(idx):continue
        for it in range(P['iterations']):
            residual=decode(x,l)-target;current=loss(residual);J=jacobian(x,l)
            wr=residual.copy();wr[:,-1]*=(1+P['endpoint_squared_error_weight'])
            wJ=J.copy();wJ[:,-1]*=(1+P['endpoint_squared_error_weight'])
            H=np.einsum('ntci,ntcj->nij',J,wJ);grad=np.einsum('ntci,ntc->ni',J,wr)
            diagonal=np.maximum(np.diagonal(H,axis1=1,axis2=2),1e-8)
            H[:,np.arange(6),np.arange(6)]+=damping[:,None]*diagonal+1e-9
            step=np.linalg.solve(H,grad[...,None])[...,0]
            accepted=np.zeros(len(x),bool)
            for rate in P['line_search']:
                trial=bounded(x-rate*step);pred=decode(trial,l);v=loss(pred-target)
                ok=(v<current)&~accepted
                x[ok]=trial[ok];current[ok]=v[ok];accepted|=ok
            damping=np.clip(damping*np.where(accepted,.5,4),1e-7,1e8)
            d=decode(x,l);a,f=metric(d,target);sc=score(a,f);take=sc<bs[idx]
            z=idx[take];best[z]=x[take];bestd[z]=d[take];bs[z]=sc[take];iterations[idx]+=1
        a,f=metric(bestd,y);active=(score(a,f)>1-1e-4)&~excluded
    # Preserve a valid native/zero witness if float32 conversion is marginal.
    fitted,u,pr=production_decode(best,ell)
    paths=base.astype(np.float32).reshape(-1,12,2)+fitted
    a,f=metric(paths.astype(float),np.repeat(gt,20,axis=0).astype(float))
    ng_a,ng_f=metric(np.zeros_like(y),y);ng=(ng_a<=.2-2e-5)&(ng_f<=.4-2e-5)
    pg=(a<=.2-2e-5)&(f<=.4-2e-5);replace=ng&~pg
    best[replace]=zero[replace];paths[replace]=base.reshape(-1,12,2)[replace]
    return paths.reshape(n,k,12,2),best.reshape(n,k,6),iterations.reshape(n,k)
def superset_fit(base,gt,L):
    desired=(gt[:,None]-base).astype(float);limit=L[:,None,None,None]
    u=np.zeros_like(desired);previous=np.zeros_like(desired[:,:,0])
    for t in range(12):u[:,:,t]=np.clip(desired[:,:,t]-previous,-limit[:,:,0],limit[:,:,0]);previous+=u[:,:,t]
    best=u.copy();a,f=metric(np.cumsum(u,2).reshape(-1,12,2),desired.reshape(-1,12,2));bs=score(a,f).reshape(len(base),20)
    for i in range(P['superset_iterations']):
        error=np.cumsum(u,2)-desired;grad=np.cumsum(error[:,:,::-1],2)[:,:,::-1]+P['endpoint_squared_error_weight']*error[:,:,-1:]
        u=np.clip(u-P['superset_step']*grad,-limit,limit)
        a,f=metric(np.cumsum(u,2).reshape(-1,12,2),desired.reshape(-1,12,2));sc=score(a,f).reshape(len(base),20)
        take=sc<bs;best[take]=u[take];bs[take]=sc[take]
    return base.astype(float)+np.cumsum(best,2),best
def quality(paths,gt):
    d=np.linalg.norm(paths.astype(float)-gt[:,None].astype(float),axis=-1);return d.mean(-1),d[:,:,-1]
def compute(base,gt,L):
    pulse,params,iterations=pulse_fit(base,gt,L);sup,u=superset_fit(base,gt,L)
    # Pulse is itself a superset witness; never report a logically smaller superset coverage.
    pa,pf=quality(pulse,gt);sa,sf=quality(sup,gt);pick=score(pa,pf)<score(sa,sf);sup[pick]=pulse[pick]
    zero=np.zeros_like(pulse[:,:,:1]);u=np.diff(sup-base.astype(float),axis=2,prepend=zero)
    desired=gt[:,None].astype(float)-base.astype(float)
    lower=np.linalg.norm(np.maximum(np.abs(desired)-L[:,None,None,None]*T[None,None,:,None],0),axis=-1)
    pulse_end_lb=np.linalg.norm(np.maximum(np.abs(desired[:,:,-1])-6*L[:,None,None],0),axis=-1)
    sa,sf=quality(sup,gt)
    return dict(pulse_paths=pulse,pulse_params=params,iterations=iterations,superset_paths=sup,superset_increment=u,
        pulse_ade=pa,pulse_fde=pf,superset_ade=sa,superset_fde=sf,
        bound_ade_lower=lower.mean(-1),bound_fde_lower=lower[:,:,-1],pulse_fde_lower=pulse_end_lb)
def audit(base,gt,L,z):
    pa,pf=quality(z['pulse_paths'],gt);sa,sf=quality(z['superset_paths'],gt)
    assert np.max(np.abs(pa-z['pulse_ade']))==0 and np.max(np.abs(pf-z['pulse_fde']))==0
    assert np.max(np.abs(sa-z['superset_ade']))==0 and np.max(np.abs(sf-z['superset_fde']))==0
    x=z['pulse_params'].reshape(-1,6);ell=np.repeat(L,20);d,u,pr=production_decode(x,ell)
    expected=base.reshape(-1,12,2).astype(np.float32)+d
    assert np.max(np.abs(expected-z['pulse_paths'].reshape(-1,12,2)))==0
    assert np.max(np.abs(u)-ell[:,None,None])<2e-6
    assert np.max(np.abs(z['superset_increment'])-L[:,None,None,None])<2e-5
    npd=decode(x,ell);float_error=float(np.max(np.abs(npd-d)))
    assert float_error<2e-5 and np.max(np.abs(pr.sum(-1)))<6e-6
    enderror=float(np.max(np.abs(d[:,-1]-6*ell[:,None]*x[:,:2])))
    assert enderror<2e-5
    desired=gt[:,None].astype(float)-base.astype(float)
    for row in range(len(base)):
        for slot in range(20):
            norms=[]
            for t in range(12):
                v=desired[row,slot,t];r=(t+1)*L[row];norms.append(float(np.linalg.norm(v-np.clip(v,-r,r))))
            assert abs(np.mean(norms)-z['bound_ade_lower'][row,slot])<1e-12
            assert abs(norms[-1]-z['bound_fde_lower'][row,slot])<1e-12
    assert all(np.isfinite(v).all() for v in z.values())
    return dict(status='PASS',metric_error_max=0,float32_vs_float64_path_max=float_error,endpoint_max=enderror,independent_per_slot_lowerbound_recomputed=True)
def setup_replay():
    import torch
    torch.set_num_threads(2)
    h=mod('immutable_pulse_for_replay',PULSE/'run_head.py')
    # Redirect all possible imported output destinations before backend creation.
    h.HERE=HERE;h.c.HERE=HERE;h.m.HERE=HERE;h.m.RESULTS=HERE/'results';h.m.MODELS=HERE/'results/models'
    _,_,native=h.n.prepare_backend(torch.device('cpu'));native.eval()
    assert not any(v.requires_grad for v in native.parameters());del native;gc.collect()
    models=[]
    for seed in (1,2,3):
        path=PULSE/f'results/models/mean_std/seed_{seed}/balanced_protected/constrained.pth'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=h.PulseResponseHead(saved['input_dim'],saved['config']).eval();model.load_state_dict(saved['state_dict'])
        for v in model.parameters():v.requires_grad_(False)
        h.frozen_audit(model)
        models.append((model,{k:v.detach().clone() for k,v in model.state_dict().items()},dict(seed=seed,path=str(path),sha=digest(path),epoch=int(saved['epoch']),fallback=bool(model.native_fallback))))
    return h,models
def replay(h,models,data,indices):
    import torch
    # Exact legacy speed computation is obtained from immutable prepared code.
    if 'speed' not in data:
        cache=h.p.load_npz(h.p.TRAIN_CACHE)
        values=h.r.prepared(cache,False,fit=np.arange(25912))
        assert np.array_equal(cache['sample_id'],data['ids']) and np.array_equal(values[0],data['obs'])
        assert np.array_equal(values[1],data['base'])
        data['speed']=values[2].copy();del cache,values;gc.collect()
    out=[]
    with torch.inference_mode():
        obs=torch.from_numpy(data['obs'][indices]);b=torch.from_numpy(data['base'][indices]);s=torch.from_numpy(data['speed'][indices])
        for model,initial,meta in models:
            path,res=model(obs,b,s);assert torch.isfinite(path).all();h.frozen_audit(model)
            out.append(path.numpy())
            assert all(torch.equal(initial[k],v) for k,v in model.state_dict().items())
    return np.stack(out)
def chunk_done(name,locks):
    p=HERE/'chunks'/(name+'.done.json')
    if not p.exists():return False
    value=read(p);assert value['source_hashes']==locks,'Resume source mismatch, stop'
    for path,sha in value['outputs'].items():assert digest(path)==sha,'Completed output SHA mismatch'
    return True
def summarize(z,learned,gt,group):
    na,nf=quality(z['native_paths'],gt);la=[];lf=[]
    for pred in learned:a,f=quality(pred,gt);la.append(a);lf.append(f)
    from scipy.cluster.hierarchy import linkage,fcluster
    collections={'native':z['native_paths'],'pulse':z['pulse_paths'],'superset':z['superset_paths']}
    collections.update({f'learned{q+1}':v for q,v in enumerate(learned)})
    counts={};clusters={}
    for name,paths in collections.items():
        a,f=quality(paths,gt);good=np.stack([(a<=at)&(f<=ft) for at,ft in P['quality']],1);counts[name]=good.sum(2).astype(np.int8)
        eff=np.zeros((len(paths),2,3),np.int8);i,j=np.triu_indices(20,1)
        for r in range(len(paths)):
            d=np.linalg.norm(paths[r,i].astype(float)-paths[r,j].astype(float),axis=-1)
            for di,(at,ft) in enumerate(P['dedup']):
                c=fcluster(linkage(np.maximum(d.mean(1)/at,d[:,-1]/ft),method='complete'),t=1,criterion='distance')
                for qi in range(2):eff[r,qi,di]=len(np.unique(c[good[r,qi]]))
        clusters[name]=eff
    witnessed=(z['pulse_ade']<=.2-2e-5)&(z['pulse_fde']<=.4-2e-5)
    supw=(z['superset_ade']<=.2-2e-5)&(z['superset_fde']<=.4-2e-5)
    certified=((z['bound_ade_lower']>.2+2e-5)|(z['bound_fde_lower']>.4+2e-5)).all(1)
    kind=np.full(len(gt),3,np.int8);kind[certified]=2;kind[~witnessed.any(1)&supw.any(1)]=1;kind[witnessed.any(1)]=0
    assert not np.any(certified&(supw.any(1)|witnessed.any(1)))
    return dict(native_ade=na,native_fde=nf,classification=kind,pulse_feasible_witness=witnessed,superset_feasible_witness=supw,
      necessary_lowerbound_certified_infeasible=certified,optimizer_unresolved=~witnessed.any(1)&~certified,
      **{name+'_raw_good':v for name,v in counts.items()},**{name+'_effective_good':v for name,v in clusters.items()})
def groupsummary(data,group):
    n=len(data['classification']);clear=group['GT_event'].astype(bool)&~group['GT_uncertain']
    masks=dict(all=np.ones(n,bool),clear=clear,no_event=~group['GT_event'].astype(bool)&~group['GT_uncertain'],
       boundary=group['GT_uncertain'],specific_first_compatible=clear&group['cue_first_compatible'],
       specific_coarse_path_failure=clear&group['cue_first_compatible']&(group['class_0']==4))
    # Class label index is verified in preflight against immutable CLASS_NAMES before use.
    masks['specific_coarse_path_failure']=clear&group['cue_first_compatible']&(group['class_0']==old.rules.CLASS_NAMES.index('coarse_response_covered_path_inadequate'))
    for s in np.unique(group['scene_family']):masks['scene_'+str(s)]=group['scene_family']==s
    for k in (1,2,3,4):masks[f'compatible_kind{k}']=clear&group['cue_first_compatible']&(group['GT_kind']==k)
    summaries=[]
    for name,mask in masks.items():
        count=int(mask.sum());record=dict(group=name,n=count)
        if count:
            record['classification_counts']=np.bincount(data['classification'][mask],minlength=4).tolist()
            record['native_coverage']=int((data['native_raw_good'][mask,0]>0).sum())
            record['pulse_witness_coverage']=int(data['pulse_feasible_witness'][mask].any(1).sum())
            record['superset_witness_coverage']=int(data['superset_feasible_witness'][mask].any(1).sum())
            record['methods']={}
            for method in ('native','pulse','superset','learned1','learned2','learned3'):
                raw=data[method+'_raw_good'][mask];eff=data[method+'_effective_good'][mask]
                good=raw[:,0]>0;native=data['native_raw_good'][mask,0]>0
                record['methods'][method]=dict(coverage=int(good.sum()),rescued=int((good&~native).sum()),lost=int((~good&native).sum()),
                  raw_distribution=np.bincount(raw[:,0],minlength=21).tolist(),effective_distribution=np.bincount(eff[:,0,0],minlength=21).tolist(),
                  sensitivity_coverage=int((raw[:,1]>0).sum()),effective_sensitivity_distributions=[np.bincount(eff[:,0,i],minlength=21).tolist() for i in range(3)])
            record['learned_gap_to_pulse']=[int((data['pulse_feasible_witness'][mask].any(1)&(data[f'learned{s}_raw_good'][mask,0]==0)).sum()) for s in (1,2,3)]
        summaries.append(record)
    return summaries
def preflight():
    monitor.begin();start=time.perf_counter();oldfiles=snapshot();hs=verify(oldfiles,'PREFLIGHT_OLD_SHA')
    data=load_data();h,models=setup_replay();index=np.linspace(0,25911,256,dtype=int)
    pred=replay(h,models,data,index[:64]);assert pred.shape==(3,64,20,12,2)
    L=(1.5*(data['speed'][:,0]+.03)).astype(np.float32).astype(float);assert np.all(L>0)
    assert all(model.strength==1.5 for model,_,_ in models)
    # Check direct old components, not only our independent equivalent equations.
    import torch
    x=np.tile([.2,-.15,.3,-.4,.37,.19],(64*20,1));ell=np.repeat(L[index[:64]],20)
    dd,uu,pp=production_decode(x,ell)
    with torch.inference_mode():
        actual,parts=models[0][0].components(torch.tensor(codes(x).reshape(64,20,6).astype(np.float32)),torch.from_numpy(data['speed'][index[:64]]))
    production_error=float(np.max(np.abs(actual.numpy().reshape(-1,12,2)-uu)));assert production_error<2e-6
    # Analytic Jacobian vs full central differences, including clock geometry.
    ja=jacobian(x[:7],ell[:7]);err=0
    for k in range(6):
        plus=x[:7].copy();minus=plus.copy();plus[:,k]+=1e-5;minus[:,k]-=1e-5
        jd=(decode(plus,ell[:7])-decode(minus,ell[:7]))/2e-5;err=max(err,float(np.max(np.abs(ja[:,:,:,k]-jd))))
    assert err<1e-6
    # Full inputs resident + all chunk machinery. Timing fixed FIT rows only, outputs discarded.
    began=time.perf_counter();reports=[]
    for left in range(0,256,64):
        monitor.check();idx=index[left:left+64];z=compute(data['base'][idx],data['future'][idx],L[idx]);reports.append(audit(data['base'][idx],data['future'][idx],L[idx],z))
        learned=replay(h,models,data,idx);summarize({**z,'native_paths':data['base'][idx]},learned,data['future'][idx],None)
    seconds=time.perf_counter()-began
    fixture=compute(np.zeros((2,20,12,2),np.float32),np.zeros((2,12,2),np.float32),np.array([.1,.1]));audit(np.zeros((2,20,12,2),np.float32),np.zeros((2,12,2),np.float32),np.array([.1,.1]),fixture)
    # All native slots have zero valid pulse; never lose native quality in oracle union.
    assert (fixture['pulse_ade']==0).all()
    resource=monitor.finish();del data,h,models,pred,fixture,z;gc.collect()
    verify(oldfiles,'PREFLIGHT_FINAL_OLD_SHA');locks=sources();verify(locks,'PREFLIGHT_SOURCE_SHA')
    # Source mismatch rejection and done-SHA skip are exercised without a science row output.
    out=HERE/'resume_fixture.npz';save(out,dict(synthetic=np.arange(3)))
    write(HERE/'chunks/resume_fixture.done.json',dict(source_hashes=locks,outputs={str(out):digest(out)}));assert chunk_done('resume_fixture',locks)
    bad=dict(locks);bad[str(HERE/'protocol.json')]='0'*64
    rejected=False
    try:chunk_done('resume_fixture',bad)
    except AssertionError:rejected=True
    assert rejected
    write(HERE/'resume_check.json',dict(status='PASS',done_SHA_skip=True,source_mismatch_rejected=True,model_optimizer_RNG_resume='NOT_APPLICABLE_NO_MODEL_TRAINING'))
    estimate=seconds*28791/256+2*hs+180
    minutes=int(np.ceil(estimate/60));write(HERE/'MEMORY_PREFLIGHT.json',dict(status='PASS_FULL_INPUT_CHUNK_MEMORY_AND_ARITHMETIC',resource=resource,
       input_expected_numpy_bytes=28791*(940+480+24+768+76)*4,learned_output_bytes=3*28791*480*4,
       maximum_pairwise_single_array_bytes=64*190*12*2*8,all_rows=28791,batch_rows=64,model_training=False))
    report=dict(status='PASS',source_hashes=locks,old_files=len(oldfiles),registered_sources=len(locks),production_components_error=production_error,
      jacobian_max_error=err,arithmetic_reports=reports,resource=resource,benchmark_rows=256,benchmark_seconds=seconds,
      measured_single_old_hash_seconds=hs,estimate_minutes=minutes,planning_minutes=minutes,eta_minutes=[int(minutes*.8),int(np.ceil(minutes*1.4))],
      full_input_loaded=True,FIT=25912,CAL=2879,DEV_access=False,model_training=False,model_optimizer=False,posthoc_numeric_fitting=True,
      learned_replay_models=[meta for _,_,meta in []],preflight_seconds=time.perf_counter()-start,benchmark_outputs_discarded=True)
    write(HERE/'preflight.json',report);print('PREFLIGHT PASS '+json.dumps({k:v for k,v in report.items() if k not in ('source_hashes','arithmetic_reports')}),flush=True)
def run():
    global monitor
    monitor=infra.Monitor();monitor.begin();pre=read(HERE/'preflight.json');assert pre['status']=='PASS';locks=pre['source_hashes']
    oldfiles=read(HERE/'old_results_manifest.json')['files'];verify({**oldfiles,**locks},'INITIAL_OLD_AND_SOURCE_SHA')
    data=load_data();h,models=setup_replay();replay(h,models,data,np.arange(1))
    L=(1.5*(data['speed'][:,0]+.03)).astype(np.float32).astype(float)
    count=int(np.ceil(25912/64)+np.ceil(2879/64));done=0;job_seconds=[]
    for pop,lo,hi in [('FIT',0,25912),('CAL',25912,28791)]:
        for left in range(lo,hi,64):
            right=min(hi,left+64);name=f'{pop}_{left-lo:05d}_{right-lo:05d}';done+=1
            stage('CURVE_FITTING',active_job=name,completed_jobs=done-1,total_jobs=count,job_seconds=job_seconds[-8:]);monitor.check()
            if chunk_done(name,locks):continue
            began=time.perf_counter();idx=np.arange(left,right);base=data['base'][idx];gt=data['future'][idx]
            z=compute(base,gt,L[idx]);learned=replay(h,models,data,idx);a=audit(base,gt,L[idx],z)
            results=summarize({**z,'native_paths':base},learned,gt,None)
            output={**z,**results,'sample_id':data['ids'][idx],'learned_paths':learned,'limit':L[idx]}
            dest=HERE/'chunks'/(name+'.npz');save(dest,output);ap=HERE/'chunks'/(name+'.audit.json');write(ap,a)
            write(HERE/'chunks'/(name+'.done.json'),dict(status='SUCCESS',source_hashes=locks,outputs={str(dest):digest(dest),str(ap):digest(ap)},rows=len(idx),model_training=False))
            job_seconds.append(time.perf_counter()-began)
    stage('SAVED_BLOCK_ARITHMETIC_AND_SUMMARY',completed_jobs=count,total_jobs=count)
    all_summaries={};audit_count=0
    for pop,lo,hi in [('FIT',0,25912),('CAL',25912,28791)]:
        parts=[]
        for left in range(lo,hi,64):
            monitor.check();right=min(hi,left+64);name=f'{pop}_{left-lo:05d}_{right-lo:05d}';assert chunk_done(name,locks)
            z=npz(HERE/'chunks'/(name+'.npz'));idx=np.arange(left,right);assert np.array_equal(z['sample_id'],data['ids'][idx])
            audit(data['base'][idx],data['future'][idx],z['limit'],z);audit_count+=1
            fresh=summarize({**z,'native_paths':data['base'][idx]},z['learned_paths'],data['future'][idx],None)
            assert all(np.array_equal(z[k],v) for k,v in fresh.items()),'Saved classification/count arithmetic changed'
            parts.append({k:v for k,v in z.items() if k in fresh or k=='sample_id'})
        arrays={k:np.concatenate([q[k] for q in parts]) for k in parts[0]};save(HERE/'results'/(pop+'_classification.npz'),arrays)
        all_summaries[pop]=groupsummary(arrays,data['groups'][pop]);del parts,arrays;gc.collect()
    import torch
    for model,initial,meta in models:
        h.frozen_audit(model);assert all(torch.equal(initial[k],v) for k,v in model.state_dict().items())
    write(HERE/'FROZEN_REPLAY_AUDIT.json',dict(status='PASS',models=[meta for _,_,meta in models],parameters_buffers_exact=True,all_original_decoder_exact=True,
        original_encoder_scorer_locked_by_SHA_no_model_weight_updates=True,GT_forward=False,read_only_model_forward=True))
    verify({**oldfiles,**locks},'FINAL_OLD_AND_SOURCE_SHA')
    write(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS',blocks=audit_count,all_done_output_SHA=True,all_saved_quality_classification_counts_recomputed=True,
      production_path_reconstruction=True,lowerbound_scalar_recompute=True,source_files_unchanged=len(locks),old_files_unchanged=len(oldfiles),model_training=False))
    write(HERE/'RESOURCE_USAGE.json',monitor.finish())
    write(HERE/'RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_CURVE_REACHABILITY',all_summaries=all_summaries,FIT=25912,CAL=2879,DEV_access=False,
      classification_names=['pulse_feasible_witness','pulse_unresolved_superset_feasible_witness','necessary_box_lowerbound_certified_infeasible','unresolved_no_witness'],
      old_files_SHA_unchanged=len(oldfiles),source_files_SHA_unchanged=len(locks),blocks=audit_count,formal_response_study_completed=2,
      inference='GT-assisted construction is not deployable output. Pulse failure is unresolved, not impossibility. Bound certification is not unpredictability. Three seeds and overlapping windows are not independent people. No causal root-cause/SOTA/major claim.'))
    write(HERE/'exit_status.json',dict(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat()))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    try:preflight() if args.preflight else run()
    except BaseException:
        failure=HERE/'failures'/datetime.now().strftime('preflight_%Y%m%d_%H%M%S');failure.mkdir(parents=True,exist_ok=False)
        (failure/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        for p in HERE.glob('*.py'):shutil.copy2(p,failure/p.name)
        write(failure/'source_hashes.json',{str(p):digest(p) for p in HERE.glob('*.py')});raise
