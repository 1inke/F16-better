"""Real model preflight. No scientific threshold or architecture search."""
import adapter as h
import numpy as np,torch,time,traceback,gc
from prepare_statistics import prepare
def independent_projection(base,raw,path,res,limit,info):
    b=base.detach().cpu().numpy().astype(np.float64);u=raw.detach().cpu().numpy().astype(np.float64)
    v=np.diff(b,axis=2,prepend=np.zeros_like(b[:,:,:1]))/.4
    mb=np.linalg.norm(np.diff(v,axis=2)/.4,axis=-1).mean((1,2));d=np.linalg.norm(np.diff(u,axis=2)/.16,axis=-1).mean((1,2))
    E=8*h.projection.GAMMA512*(abs(b).max((1,2,3))+12*abs(u).max((1,2,3)))/.16
    alpha=np.ones_like(mb);nz=d!=0;alpha[nz]=np.clip(np.maximum(.01*mb[nz]-E[nz],0)/d[nz],0,1)
    mean=u.mean(2,keepdims=True);zero=(mb==0)&nz;mean[zero]=mean[zero].astype(np.float32).astype(np.float64)
    projected=mean+alpha[:,None,None,None]*(u-mean);projected[~nz]=u[~nz]
    bound=limit.detach().cpu().numpy().astype(np.float64);projected=np.maximum(-bound,np.minimum(bound,projected))
    expected=b+projected.cumsum(2);actual=path.detach().cpu().numpy()
    assert abs(expected-actual).max()<=2e-5 and abs(alpha-info['alpha'].detach().cpu().numpy()).max()<1e-12
    assert (h.physical.numpy_accel(actual)<=1.01*h.physical.numpy_accel(b)).all(),'STRICT_PHYSICAL_GATE'
    endpoint=float(abs(actual[:,:,-1]-(b+u.cumsum(2))[:,:,-1]).max());assert endpoint<=2e-5
    assert (abs(res.detach().cpu().numpy())<=bound+1e-12).all()
    return dict(status='PASS',path_max=float(abs(expected-actual).max()),endpoint_max=endpoint,strict_physical=True)
def main():
    begin=time.perf_counter();h.stage('PREFLIGHT_INITIAL_OLD_SHA');old=h.read(h.HERE/'old_results_manifest.json')['files'];h.verify(old)
    cache,values,splits=h.load_data();stat=prepare(cache,values,splits);Class=h.initialize_classes();device=torch.device('cuda:0')
    h.setup_weights(cache,splits['common_statistics_index'],values,h.settings(1),device)
    scale=torch.from_numpy(np.load(h.HERE/'common_statistics.npz')['protection_scales']).to(device)
    h.write(h.HERE/'preflight.json',dict(status='INCOMPLETE_GPU_PENDING',fit_protection_scales=scale.cpu().tolist()))
    states=[];records=[]
    for arm in ('random','grouped'):
        h.stage('PREFLIGHT_GPU',arm=arm);idx=splits[arm+'_train_index'][:128]
        o,b,s,f=[torch.from_numpy(v[idx]).to(device) for v in values[:4]];groups=torch.from_numpy(cache['change_group'][idx]).to(device)
        h.p.seed_everything(1);model=Class(940,h.settings(1,arm)).to(device);h.frozen_audit(model)
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        model.eval();out,res=model(o,b,s);assert torch.equal(out,b.double()) and torch.count_nonzero(res)==0
        with torch.no_grad():
            past=o[:,:16].reshape(-1,1,8,2);pred,logit=model.decoder(o[:,16:784].reshape(-1,1,8,96),past)
            path_error=float((pred[:,0]-past[:,0,-1,None,None]-b).abs().max());logit_error=float((logit[:,0]-torch.from_numpy(cache['logits'][idx]).to(device)).abs().max())
            assert path_error<=2e-5 and logit_error<=2e-5
        detached=out.detach().requires_grad_();rr=res.detach().requires_grad_()
        original=h.original_objective(detached,b,f,rr,s,groups,h.settings(1,arm),'balanced_protected',scale)
        P=h.physical.penalty(detached,b,groups,model.physical_accel_scales)
        total=h.objective(detached,b,f,rr,s,groups,h.settings(1,arm),'balanced_protected',scale)
        assert torch.equal(total,original+.1*P)
        ga=torch.autograd.grad(total,(detached,rr),retain_graph=True);gb=torch.autograd.grad(original+.1*P,(detached,rr));assert all(torch.equal(a,c) for a,c in zip(ga,gb))
        del out,res,detached,rr,original,total,P,ga,gb
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=h.settings(1)['learning_rate'],weight_decay=h.settings(1)['weight_decay']);natural=[]
        for step in range(8):
            out,res=model(o,b,s);loss=h.objective(out,b,f,res,s,groups,h.settings(1,arm),'balanced_protected',scale)
            opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(loss)
            for name,v in model.named_parameters():
                if v.requires_grad:
                    assert v.grad is not None and torch.isfinite(v.grad).all(),name
                    if step>=3 and not name.endswith('key.bias'):assert v.grad.abs().sum()>0,('DISCONNECTED',name)
                else:assert v.grad is None,name
            natural.append(float(h.physical.penalty(out,b,groups,model.physical_accel_scales).detach()));opt.step();h.frozen_audit(model)
        raw,parts=model.routed_components(o,b,s);path,up,info=h.projection.project(b,raw,parts['limit']);physics=independent_projection(b,raw,path,up,parts['limit'],info)
        assert torch.equal(parts['weights'],torch.ones_like(parts['weights'])*.5)
        norms={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.requires_grad}
        # Fixed actual-head active projection; not a trained/saved model or outcome choice.
        gen=torch.Generator().manual_seed(20261006)
        with torch.no_grad():
            for expert in model.experts:
                last=expert.head[-1];last.weight.copy_((torch.randn(last.weight.shape,generator=gen)*.01).to(device));last.bias.copy_(torch.tensor([.1,-.1,.35,-.25,.2,-.2],device=device))
        model.zero_grad(set_to_none=True);raw,parts=model.routed_components(o,b,s);path,up,info=h.projection.project(b,raw,parts['limit'])
        active=int((info['alpha']<1).sum());assert active>0
        activephysics=independent_projection(b,raw,path,up,parts['limit'],info)
        (path.square().mean()+up.square().mean()).backward()
        for name,v in model.named_parameters():
            if v.requires_grad:
                assert v.grad is not None and torch.isfinite(v.grad).all(),name
                if not name.endswith('key.bias'):assert v.grad.abs().sum()>0,('ACTIVE_DISCONNECTED',name)
        h.frozen_audit(model)
        records.append(dict(arm=arm,gradient_norms=norms,natural_P=natural,path_cache_error=path_error,logit_cache_error=logit_error,physics=physics,active_fixed_head_windows=active,active_physics=activephysics,not_quantization_FD_proof=True))
        del model,opt,out,res,loss,raw,parts,path,up,info;gc.collect();torch.cuda.empty_cache()
    assert states[0].keys()==states[1].keys() and all(torch.equal(states[0][k],states[1][k]) for k in states[0]);del states
    h.stage('PREFLIGHT_FINAL_OLD_SHA');h.verify(old)
    report=dict(status='PASS_GPU_STATISTICS',fit_protection_scales=scale.cpu().tolist(),source_hashes=h.source_hashes(),old_files=len(old),records=records,paired_allweights_buffers_EXACT=True,parameters=91596,statistics=stat,resource=h.guard.check(),elapsed_seconds=time.perf_counter()-begin,resume_status='NOT_YET_RUN',production_started=False)
    h.write(h.HERE/'preflight.json',report);h.guard.save('PREFLIGHT_GPU_PASS');print('GPU_PREFLIGHT_PASS',json_summary(report),flush=True)
def json_summary(report):return str({k:report[k] for k in ('status','parameters','old_files','elapsed_seconds','resource')})
if __name__=='__main__':
    try:main()
    except Exception:
        number=1
        while (h.HERE/f'failures/preflight{number}/traceback.txt').exists():number+=1
        directory=h.HERE/f'failures/preflight{number}';directory.mkdir(parents=True,exist_ok=True)
        (directory/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');h.write(directory/'SOURCE.json',{str(p):h.sha(p) for p in h.HERE.glob('*.py')})
        h.write(h.HERE/'preflight_failure.json',dict(status='STOP_PREFLIGHT',production_started=False));raise
