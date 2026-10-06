"""Actual scientific gates, no production launch; never relax strict physics."""
import json,time,math,gc,hashlib
import numpy as np
import torch

def independent_physics(h,b,raw,path,up,limit,info):
    b=b.detach().cpu().numpy().astype(np.float64);u=raw.detach().cpu().numpy().astype(np.float64)
    actual=path.detach().cpu().numpy();res=up.detach().cpu().numpy()
    def mean(x):
        v=np.diff(x,axis=2,prepend=np.zeros_like(x[:,:,:1]))/.4
        return np.linalg.norm(np.diff(v,axis=2)/.4,axis=-1).mean((1,2))
    mb=mean(b);mp=mean(actual)
    d=np.linalg.norm(np.diff(u,axis=2)/.16,axis=-1).mean((1,2))
    E=8*h.projection.GAMMA512*(np.max(abs(b),axis=(1,2,3))+12*np.max(abs(u),axis=(1,2,3)))/.16
    alpha=np.ones_like(d);nz=d!=0;alpha[nz]=np.clip(np.maximum(.01*mb[nz]-E[nz],0)/d[nz],0,1)
    av=u.mean(2,keepdims=True);zero=(mb==0)&nz;av=np.where(zero[:,None,None,None],av.astype(np.float32).astype(np.float64),av)
    reference=av+alpha[:,None,None,None]*(u-av);reference=np.where((~nz)[:,None,None,None],u,reference)
    bound=limit.detach().cpu().numpy().astype(np.float64);reference=np.clip(reference,-bound,bound)
    assert (mp<=1.01*mb).all(),('STOP_NUMERIC actual strict physics',float(np.max(mp-1.01*mb)))
    assert np.max(abs(res-reference))<2e-5 and np.max(abs(actual-(b+np.cumsum(reference,axis=2))))<2e-5
    assert np.max(abs(actual[:,:,-1]-(b+np.cumsum(u,axis=2))[:,:,-1]))<2e-5
    assert (abs(res)<=bound+1e-12).all()
    return dict(strict_per_window=True,alpha_reference_max=float(np.max(abs(alpha-info['alpha'].detach().cpu().numpy()))),endpoint_max=float(np.max(abs(actual[:,:,-1]-(b+np.cumsum(u,axis=2))[:,:,-1]))))

def autograd_fixtures(h,device):
    # Double projection arithmetic takes float32 inputs by registered contract.
    b=torch.zeros(3,20,12,2,device=device);u=torch.zeros_like(b)
    b[2,...,0]=torch.arange(1,13,device=device).square()[None]*.003
    u[1,...,0]=torch.arange(1,13,device=device)[None]*.002
    u[2,...,1]=torch.sin(torch.arange(12,device=device))[None]*.03
    u.requires_grad_();lim=torch.ones(3,1,1,1,device=device)
    path,up,info=h.projection.project(b,u,lim)
    independent_physics(h,b,u,path,up,lim,info)
    scalar=(path.square().mean()+up.square().mean())
    grad=torch.autograd.grad(scalar,u)[0]
    assert torch.isfinite(grad).all() and grad[1:].abs().sum()>0
    assert info['delta_mean'][0]==0 and info['zero_budget_mean_quantized'][1]
    # Away from hinges and float32 lattice switches, FD checks a real mixed FIT direction separately.
    return dict(status='PASS',D0=True,zero_budget_cast_gradient_finite=True,not_quantization_differentiability_proof=True)



def verify_parent_transition(h):
    parent=h.HERE.parent
    audit=h.read(parent/'preflight.json')
    allowed={'quality_audit.py','check_once.ps1'}
    changed=[]
    for path,digest in audit['source_hashes'].items():
        source=h.Path(path)
        current=h.sha(source)
        if current!=digest:
            assert source.parent==parent and source.name in allowed,('STOP unapproved parent source change',path)
            locked=parent/'failures/preflight_1791222401792826400'/('locked_'+source.name)
            assert h.sha(locked)==digest,('STOP prior source provenance',path)
            before=locked.read_text(encoding='utf-8').replace('\r\n','\n')
            after=source.read_text(encoding='utf-8').replace('\r\n','\n')
            if source.name=='quality_audit.py':
                expected=before.replace('CANDIDATE_STATE_GATE_FAIL','TRAINED_ACCEL_PROJECTION_GATE_FAIL').replace('Candidate-native-state conditioning','Training-in-loop endpoint-preserving acceleration projection')
                assert expected==after,('STOP metadata-only transition mismatch',source.name)
            else:
                expected=before.replace("$errorFile=Get-Item -LiteralPath (Join-Path $root 'stderr.log') -ErrorAction SilentlyContinue", "$errorFile=Get-Item -LiteralPath (Join-Path $root 'stderr.log') -ErrorAction SilentlyContinue\n$progressPath=Join-Path $root 'progress.json'\n$progress=if(Test-Path -LiteralPath $progressPath){Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json}else{$null}").replace('worker=$state;actual_process=', 'worker=$state;progress=$progress;actual_process=')
                assert expected==after,('STOP display-only transition mismatch',source.name)
            changed.append(dict(path=path,previous_sha=digest,current_sha=current))
    assert {h.Path(x['path']).name for x in changed}==allowed
    h.write(h.HERE/'PARENT_SOURCE_TRANSITION.json',dict(status='PASS',explicit_human_approval='2026-10-06 continue and long autonomous operation',metadata_only=changed,other_previous_sources_unchanged=True))

def active_real_head(h,device,o,b,s):
    """Fixed throwaway actual-head activation, not a fitted model or scientific choice."""
    h.p.seed_everything(20261006)
    model=h.ProjectionHead(940,h.settings(1,'mean_std')).to(device).eval()
    gen=torch.Generator(device='cpu');gen.manual_seed(20261006)
    with torch.no_grad():
        for expert in model.experts:
            last=expert.head[-1]
            last.weight.copy_((torch.randn(last.weight.shape,generator=gen)*.01).to(device))
            last.bias.copy_(torch.tensor([.1,-.1,.35,-.25,.2,-.2],device=device))
    h.frozen_audit(model)
    raw,parts=model.routed_components(o,b,s)
    out,up,info=h.projection.project(b,raw,parts['limit'])
    count=int((info['alpha']<1).sum())
    assert count>0,('STOP fixed actual-head activation absent',count)
    audit=independent_physics(h,b,raw,out,up,parts['limit'],info)
    scalar=out.square().mean()+up.square().mean()
    scalar.backward()
    norms={}
    for name,param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(),name
            norms[name]=float(param.grad.norm())
            if not name.endswith('key.bias'):assert norms[name]>0,('STOP active actual-head disconnected',name)
        else:assert param.grad is None,name
    h.frozen_audit(model)
    result=dict(status='PASS',fixed_seed=20261006,active_windows=count,total_windows=len(b),
        alpha_min=float(info['alpha'].min()),alpha_max=float(info['alpha'].max()),
        gradients=norms,independent_physics=audit,not_optimizer_update=True,
        not_training_result=True,not_quantization_FD_proof=True)
    del model,raw,parts,out,up,info
    return result

def run(h,device):
    start=time.perf_counter();verify_parent_transition(h);h.stage('PREFLIGHT_INITIAL_OLD_SHA');old=h.old_manifest(create=True)
    h.stage('PREFLIGHT_DATA');cache,cfg,fit,cal,values=h.prepared_train(device)
    scale=np.asarray(h.read(h.ROOT/'GENERATION_CANDIDATE_EXCHANGE_SERIES/preflight.json')['fit_protection_scales'],np.float32)
    assert np.array_equal(scale,np.asarray([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32))
    scale_t=torch.from_numpy(scale).to(device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][fit[:128]],device=device)
    assert not set(cache['sample_id'][fit])&set(cache['sample_id'][cal])
    with np.load(h.p.DEV_CACHE,allow_pickle=False) as z:assert not set(cache['sample_id'])&set(z['sample_id'])
    mathcheck=autograd_fixtures(h,device);h.stage('PREFLIGHT_ACTIVE_ACTUAL_HEAD')
    activecheck=active_real_head(h,device,o,b,s);h.stage('PREFLIGHT_GPU')
    gradients={};states=[];physics=[];alignment={};naturalP={}
    for arm in h.c.ARMS:
        h.p.seed_everything(1);model=h.ProjectionHead(940,h.settings(1,arm)).to(device)
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        h.frozen_audit(model);model.eval();out,res=model(o,b,s)
        assert torch.equal(out,b.double()) and torch.equal(res,torch.zeros_like(res))
        if arm=='zero':
            with torch.no_grad():
                pred,logit=model.decoder(o[:,16:784].reshape(-1,1,8,96),o[:,:16].reshape(-1,1,8,2))
                relative=pred[:,0]-o[:,:16].reshape(-1,8,2)[:,-1,None,None]
                error=float((relative-b).abs().max());assert error<2e-5
                logit_error=float((logit[:,0]-torch.from_numpy(cache['logits'][fit[:128]]).to(device)).abs().max());assert logit_error<2e-5
                alignment=dict(native_path_error=error,native_logit_error=logit_error)
        # Existing objective scalar and coordinate gradient unchanged on identical float64 coordinates.
        test=out.detach().requires_grad_();rr=res.detach().requires_grad_()
        original=h.original_objective(test,b,gt,rr,s,groups,cfg,'balanced_protected',scale_t)
        P=h.physical.penalty(test,b,groups,model.physical_accel_scales)
        total=h.m.objective(test,b,gt,rr,s,groups,cfg,'balanced_protected',scale_t)
        assert torch.equal(total,original+.1*P)
        a=torch.autograd.grad(total,(test,rr),retain_graph=True);bb=torch.autograd.grad(original+.1*P,(test,rr))
        assert all(torch.equal(x,y) for x,y in zip(a,bb))
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        Ps=[]
        for step in range(8):
            out,res=model(o,b,s);loss=h.m.objective(out,b,gt,res,s,groups,h.settings(1,arm),'balanced_protected',scale_t)
            opt.zero_grad();loss.backward();assert torch.isfinite(loss)
            for name,v in model.named_parameters():
                if v.requires_grad:
                    assert v.grad is not None and torch.isfinite(v.grad).all(),('STOP gradient',arm,name,step)
                    if step>=3 and not name.endswith('key.bias'):assert v.grad.abs().sum()>0,('STOP disconnected',arm,name,step)
                else:assert v.grad is None,name
            Ps.append(float(h.physical.penalty(out,b,groups,model.physical_accel_scales).detach()))
            opt.step();h.frozen_audit(model);h.guard.check()
        gradients[arm]={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.requires_grad}
        raw,parts=model.routed_components(o,b,s);proj,up,info=h.projection.project(b,raw,parts['limit'])
        physics.append(dict(arm=arm,**independent_physics(h,b,raw,proj,up,parts['limit'],info)))
        assert out.shape==(128,20,12,2) and torch.isfinite(out).all()
        assert (res.abs()<=parts['limit'].double()+1e-12).all()
        assert torch.equal(parts['weights'],torch.ones_like(parts['weights'])*.5)
        # Independent finite difference through production float32 head on nonzero trained throwaway weights.
        params=[v for v in model.parameters() if v.requires_grad];direction=[torch.randn_like(v) for v in params]
        norm=torch.sqrt(sum((v.square().sum() for v in direction)));direction=[v/norm for v in direction]
        output,residual=model(o,b,s);scalar=output.square().mean();gr=torch.autograd.grad(scalar,params)
        derivative=float(sum((x*y).sum() for x,y in zip(gr,direction)))
        errors=[]
        # Float32 production FD is limited by rounding; double-copy established arithmetic fixture is below.
        for eps in (1e-3,3e-4):
            original_params=[v.detach().clone() for v in params]
            with torch.no_grad():
                for v,z,d in zip(params,original_params,direction):v.copy_(z+eps*d)
                plus=float(model(o,b,s)[0].square().mean())
                for v,z,d in zip(params,original_params,direction):v.copy_(z-eps*d)
                minus=float(model(o,b,s)[0].square().mean())
                for v,z in zip(params,original_params):v.copy_(z)
            fd=(plus-minus)/(2*eps);error=abs(fd-derivative);errors.append(error)
            assert error<2e-5+2e-2*abs(derivative),('STOP FD',arm,eps,fd,derivative)
        naturalP[arm]=dict(eight_steps_P=Ps,production_float32_FD_errors=errors,parameter_derivative=derivative)
        model.native_fallback.fill_(True);identity,zero=model(o,b,s);assert torch.equal(identity,b.double()) and not zero.any()
        del model,opt,params,gr
    assert states[0].keys()==states[1].keys() and all(torch.equal(states[0][key],states[1][key]) for key in states[0])
    del states;gc.collect();h.old_manifest();sources=h.source_hashes()
    record=dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=dict(zero=91596,mean_std=91596),source_hashes=sources,
        fit_protection_scales=scale.tolist(),old_files=old,gradients=gradients,physics=physics,autograd_fixtures=mathcheck,
        active_actual_head=activecheck,paired_weights_buffers_exact=True,native_identity_exact=True,all_old_F16_decoder_scorer_frozen=True,
        online_cache_alignment=alignment,both_originaltotal_plus_P_scalar_gradient_EXACT=True,natural_P=naturalP,
        precision='both float64 cumulative trajectories; head math float32; original loss equations',memory=h.guard.check(),elapsed_seconds=time.perf_counter()-start)
    h.write(h.HERE/'preflight.json',record);h.guard.save('PREFLIGHT_PASS')
    print('PREFLIGHT PASS',json.dumps(dict(old=old,seconds=record['elapsed_seconds'],memory=record['memory'],alignment=alignment,natural_P=naturalP)),flush=True)
