"""Actual frozen diagnostic preflight, not training resume."""
from pathlib import Path
import gc,time,json,types
import numpy as np
import torch
from torch.nn import functional as F
import analysis_core as c
from resource_guard import guard
HERE=c.HERE;ROOT=c.ROOT
def test_original(data,paths,split,ix):
    vals=c.tensors(data,ix,'cpu',torch.float64)
    base,future,weight,family,group,scale=vals
    n=len(ix);ctx={'n':n,'cross':int((family[:,:,None]!=family[:,None,:]).sum()),'protected':int((group!=2).sum())}
    env=c.extract(ROOT/'GENERATION_MODE_SUPPORT_SERIES/run_mode_support.py',['labels'],dict(torch=torch,np=np,LABEL_CFG=c.read(ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/config.json')))
    env2=dict(torch=torch,F=F,WEIGHTS=torch.tensor(data['class_weights'],dtype=torch.float64),m=types.SimpleNamespace(labels=env['labels']),g=types.SimpleNamespace(pair_distance=c.pair))
    c.extract(ROOT/'GENERATION_DIRECTION_COVERAGE_SERIES/run_direction_coverage.py',['objective'],env2)
    protected=dict(torch=torch,F=F,r=types.SimpleNamespace(objective=env2['objective']))
    c.extract(ROOT/'GENERATION_PROTECTED_BALANCE_SERIES/run_protected.py',['objective','protection'],protected)
    cfg=dict(fde_loss_weight=.5,diversity_weight=.05,diversity_reference_fraction=.5,diversity_distance_cap_m=.3,residual_weight=.01,smoothness_weight=.01,protection_weight=.1,guard_fraction=.01)
    x=torch.tensor(paths,dtype=torch.float64,requires_grad=True)
    ours,_,_,_=c.terms(x,*vals,ctx)
    res=torch.diff(x-base,dim=2,prepend=torch.zeros_like(x[:,:,:1]))
    original=protected['objective'](x,base,future,res,torch.tensor(data['speed'][ix],dtype=torch.float64),group,cfg,'balanced_protected',scale)
    error=abs(float((original-ours['total'].mean()).detach()))
    ga=torch.autograd.grad(original,x,retain_graph=True)[0];gb=torch.autograd.grad(ours['total'].mean(),x)[0]
    ge=float((ga-gb).abs().max());assert error<=1e-10 and ge<=1e-10,(error,ge)
    return dict(original_AST_total_error=error,original_AST_coordinate_gradient_max=ge)
def fixtures():
    # One good candidate need not be hard quality winner.
    a=torch.tensor([[.201,.1]+[2.]*18],dtype=torch.float64);f=torch.tensor([[0.,.3]+[2.]*18],dtype=torch.float64)
    assert (a+.5*f).argmin(-1)==0
    good=(a<=.2)&(f<=.4);assert good.any() and not good[0,0]
    joint=torch.maximum(a/.2,f/.4).min(-1).values;assert joint<=1
    a2=torch.tensor([[.1,.3]]);f2=torch.tensor([[.6,.1]])
    assert a2.min()<=.2 and f2.min()<=.4 and not ((a2<=.2)&(f2<=.4)).any()
    path=np.zeros((1,20,12,2),np.float32);truth=np.zeros((1,12,2),np.float32)
    q,_=c.d.metrics(path,truth,np.zeros((1,20),np.float32));assert q['effective_good'][0]==1 and q['raw_good'][0]==20
    assert c.pt.stats(np.ones(2),-np.ones(2))['cos']==-1
    assert c.pt.stats(np.zeros(2),np.ones(2))['cos'] is None
    # Linear quantile rank derivative agrees with torch, including sparse support.
    x=torch.arange(24,dtype=torch.float64).reshape(1,2,12).requires_grad_()
    q=torch.quantile(x.reshape(-1),.95);g=torch.autograd.grad(q,x)[0]
    r=.95*23;lo=int(r);hi=lo+1;expect=torch.zeros(24,dtype=torch.float64);expect[lo]=hi-r;expect[hi]=r-lo
    assert torch.equal(g.flatten(),expect)
    return 'PASS_HARDWINNER_GOOD_NONWINNER_DISJOINT_MINIMA_COPIES_ONE_CLUSTER_KNOWN_CONFLICT_ZERO_QUANTILE'
def finite_difference(model,params,data,split,paths):
    a,b=c.d.SPLITS[split];ix=(np.flatnonzero(data['group'][a:b]==2)+a)[:32];ctx=data['context'][split]
    dev='cuda';dtype=torch.float64
    obs=torch.tensor(data['obs'][ix],device=dev,dtype=dtype);base=torch.tensor(data['base'][ix],device=dev,dtype=dtype);speed=torch.tensor(data['speed'][ix],device=dev,dtype=dtype)
    vals=c.tensors(data,ix,dev,dtype)
    def evals():
        x,_=model(obs,base,speed);ls,_,_,q=c.terms(x,*vals,ctx);mt=c.motion(x)
        return {'primary':ls['primary'].mean(),'total':ls['total'].mean(),'accel_mean':mt['accel'].mean(),'jerk_mean':mt['jerk'].mean(),'accel_p95':torch.quantile(mt['accel'].reshape(-1),.95)},q.argmin(-1)
    initial=[p.detach().clone() for p in params];res=[]
    for name in ('primary','total','accel_mean','jerk_mean','accel_p95'):
        values,win=evals();gradient=c.pt.vector(values[name],params);norm=float(np.linalg.norm(gradient))
        if norm<=1e-10:res.append(dict(term=name,status='UNDETERMINED_NEARZERO'));continue
        direction=torch.tensor(gradient/norm,device=dev)
        for eps in (1e-4,1e-5):
            def assign(sign):
                off=0
                with torch.no_grad():
                    for p,z in zip(params,initial):
                        p.copy_(z+sign*eps*direction[off:off+p.numel()].reshape_as(p));off+=p.numel()
            assign(1)
            with torch.no_grad():v1,w1=evals()
            assign(-1)
            with torch.no_grad():v0,w0=evals()
            derivative=float((v1[name]-v0[name])/(2*eps));tol=1e-7+1e-3*abs(norm)
            assert abs(derivative-norm)<=tol,('FD_STOP',name,eps,derivative,norm,tol)
            res.append(dict(term=name,eps=eps,analytic=norm,finite_difference=derivative,error=abs(derivative-norm),tolerance=tol,winner_switches=int(((win!=w1)|(win!=w0)).sum())))
            with torch.no_grad():
                for p,z in zip(params,initial):p.copy_(z)
    return res
def run(diagnosis):
    start=time.perf_counter()
    old=diagnosis.snapshot_old();models=c.d.checkpoints();src=diagnosis.sources(models);c.verify(src,'PREFLIGHT_SOURCE_SHA')
    data=c.initialize();reports=[];times=[];cpu_times=[]
    for family in ('raw','centered_raw','centered_own'):
        meta=next(m for m in models if m['family']==family and m['seed']==1 and m['policy']=='constrained')
        model,info=c.build(family,Path(meta['path']));cfg=info['config']
        assert [cfg[k] for k in ('fde_loss_weight','diversity_weight','residual_weight','smoothness_weight','protection_weight','guard_fraction')]==[.5,.05,.01,.01,.1,.01]
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model=model.to('cuda');names,params,layout=c.pt.setup(model)
        paths=c.load_saved(meta,'FIT');ix=np.arange(128)
        with torch.no_grad():
            x,res=model(torch.tensor(data['obs'][ix],device='cuda'),torch.tensor(data['base'][ix],device='cuda'),torch.tensor(data['speed'][ix],device='cuda'))
        align=float((x.cpu()-torch.from_numpy(paths[ix])).abs().max());assert align<=2e-5 and x.shape==(128,20,12,2) and torch.isfinite(x).all()
        assert (res.abs()<=1.5*(torch.tensor(data['speed'][ix,0],device='cuda')[:,None,None,None]+.03)+2e-5).all()
        ast_result=test_original(data,paths[ix],'FIT',ix)
        chunks={}
        for size in (128,64,31):chunks[size]=c.param_accumulate(model,params,data,'FIT',paths,chunk=size,subset=128)
        checks=[]
        for size in (64,31):
            for term in c.TERMS+c.PHYS:
                ref=chunks[128][0][term];diff=float(np.linalg.norm(ref-chunks[size][0][term]));tol=1e-6+2e-4*np.linalg.norm(ref)
                assert diff<=tol,('CHUNK_STOP',family,size,term,diff,tol);checks.append(dict(chunk=size,term=term,error=diff,tolerance=tol))
        assert np.linalg.norm(chunks[128][0]['primary'])>1e-10
        c.pt.unchanged(model,state)
        for k in range(8):
            torch.cuda.synchronize();t=time.perf_counter()
            c.param_accumulate(model,params,data,'FIT',paths,subset=128)
            torch.cuda.synchronize();times.append(time.perf_counter()-t)
        # CPU snapshot throughput includes actual coordinate credit and clustering.
        small=dict(data)
        small['context']={'SMOKE':{'n':128,'cross':int((data['family'][:128,:,None]!=data['family'][:128,None,:]).sum()),'protected':int((data['group'][:128]!=2).sum())}}
        c.d.SPLITS['SMOKE']=(0,128)
        try:
            t=time.perf_counter()
            vals=c.snapshot(small,paths[:128],'SMOKE',(np.full(28791,-1,np.int8),np.zeros((28791,12,2),np.float32)))
            cpu_times.append(time.perf_counter()-t)
            assert np.all(vals['primary_credit20'][np.arange(128)[:,None],np.arange(20)[None,:]][np.arange(20)[None,:]!=vals['quality_winner'][:,None]]==0)
        finally:del c.d.SPLITS['SMOKE']
        c.pt.unchanged(model,state)
        del model,params;torch.cuda.empty_cache();gc.collect()
        numeric,_=c.build(family,Path(meta['path']));numeric=numeric.to('cuda').double();_,pars,_=c.pt.setup(numeric)
        fd=finite_difference(numeric,pars,data,'FIT',paths)
        reports.append(dict(family=family,path_alignment=align,AST=ast_result,chunks=checks,float64_numerical_copy_FD=fd,quantile=chunks[128][2]))
        del numeric,pars,paths;torch.cuda.empty_cache();gc.collect()
    c.verify(src,'PREFLIGHT_FINAL_SOURCE_SHA');c.verify(old,'PREFLIGHT_FINAL_OLD_SHA')
    fixture=fixtures()
    testfile=HERE/'resume_fixture.txt';testfile.write_text('only diagnostic fixture',encoding='utf-8')
    done={'status':'SUCCESS','sources':src,'outputs':{str(testfile):c.sha(testfile)}}
    assert diagnosis.completed(done,src)
    bad=dict(done,sources={'BAD':'digest'})
    try:diagnosis.completed(bad,src)
    except AssertionError:pass
    else:raise AssertionError('BAD_SOURCE_ACCEPTED')
    bad=dict(done,outputs={str(testfile):'0'*64})
    try:diagnosis.completed(bad,src)
    except AssertionError:pass
    else:raise AssertionError('BAD_OUTPUT_ACCEPTED')
    compute=(float(np.mean(cpu_times))*28791/128*19+float(np.mean(times))*7110/128*18)/60
    c.write(HERE/'preflight.json',dict(status='PASS',source_hashes=src,sources=len(src),old_files=len(old),models=18,actual_reports=reports,fixtures=fixture,analysis_resume='PASS_DONE_SHA_SKIP_BADSOURCE_BADOUTPUT_REJECT_NOT_TRAINING_RESUME',GPU_eight_batch_seconds=times,CPU_snapshot128_seconds=cpu_times,nominal_compute_minutes=compute,input_bytes=sum(x.nbytes for x in data.values() if isinstance(x,np.ndarray)),memory=guard.values(),elapsed_seconds=time.perf_counter()-start,scope='EXISTING_FROZEN_DIAGNOSTIC_NO_OPTIMIZER_NO_NEWLOSS'))
    guard.save('PREFLIGHT_PASS')
    print('PREFLIGHT PASS nominalcompute',compute,flush=True)
