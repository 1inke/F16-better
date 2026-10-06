"""Actual GPU/route/frozen/source gates, before any production training."""
import json,time,sys
from pathlib import Path
import numpy as np
import torch
def run(h,device):
    start=time.perf_counter();old=h.old_manifest(not(h.HERE/'old_results_manifest.json').exists())
    _,_,original=h.n.prepare_backend(device);original.device=str(device)
    cache,cfg,fit,cal,values=h.prepared_train(device)
    physical_math=h.physical.fixtures();state_math=sys.modules['state_features'].validate_state_math();route_math=h.route.fixtures()
    route_record=json.loads((h.HERE/'route_threshold.json').read_text(encoding='utf-8'))
    assert min(route_record['candidate_counts'])>=.1*518240
    parent=json.loads((h.ROOT/'GENERATION_CANDIDATE_EXCHANGE_SERIES/preflight.json').read_text(encoding='utf-8'))
    scale=torch.tensor(parent['fit_protection_scales'],device=device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][fit[:128]],device=device)
    common=h.module('routed_replay_common',h.ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/common.py');sys.modules['common']=common
    native=h.module('routed_replay_native',h.ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/native.py')
    raw=np.load(h.ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz',allow_pickle=False)
    assert np.array_equal(raw['sample_id'],cache['sample_id'])
    obs=raw['observation'][fit[:128]]
    with torch.no_grad():online=native.forward(original,torch.tensor(obs[:,160:304].reshape(-1,9,8,2),device=device),torch.tensor(obs[:,304],device=device))
    probe=h.PhysicalHead(940,h.c.settings(1,'zero')).to(device)
    alignment=dict(tokens=float((online['tokens']-o[:,16:784].reshape(-1,8,96)).abs().max()),native_path=float((online['paths']-b).abs().max()),
        context=float((online['E3']-probe.native_inputs(o)[:,0,:64]).abs().max()))
    assert max(alignment.values())<2e-5
    original_state={k:v.detach().cpu().clone() for k,v in original.state_dict().items()};del probe,raw
    states=[];counts={};gradient={};responses={}
    for arm in h.c.ARMS:
        h.p.seed_everything(1);model=h.PhysicalHead(940,h.c.settings(1,arm)).to(device).eval()
        counts[arm]=sum(p.numel() for p in model.parameters() if p.requires_grad)
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        out,res=model(o,b,s);assert torch.equal(out,b) and torch.equal(res,torch.zeros_like(res))
        assert model.route_median.dtype==torch.float64
        with torch.no_grad():
            for expert in model.experts:
                feature,w=expert.inputs(o,b);assert torch.isfinite(feature).all() and torch.allclose(w.sum(-1),torch.ones_like(w.sum(-1)),atol=1e-6)
        _,part=model.routed_components(o,b,s)
        support=[int((part['branch']==i).sum()) for i in (0,1)];assert min(support)>0,('STOP no GPU FIT branch support',support)
        # Same physical feedback both arms, independent original loss + registered P.
        candidate=out.detach().clone().requires_grad_();residual=res.detach().clone().requires_grad_()
        old=h.original_objective(candidate,b,gt,residual,s,groups,h.c.settings(1,arm),'balanced_protected',scale)
        P=h.physical.penalty(candidate,b,groups,model.physical_accel_scales)
        expected=old+.1*P
        actual=h.m.objective(candidate,b,gt,residual,s,groups,h.c.settings(1,arm),'balanced_protected',scale)
        assert torch.equal(actual,expected)
        ga=torch.autograd.grad(actual,(candidate,residual),retain_graph=True);ge=torch.autograd.grad(expected,(candidate,residual))
        assert all(torch.equal(x,y) for x,y in zip(ga,ge))
        # Under nonzero same weights each expert matches the previous raw head.
        reference=h.RawParent(940,h.c.settings(1,'mean_std')).to(device).eval()
        expert=model.experts[0]
        selected={k:v for k,v in expert.state_dict().items() if k!='physical_accel_scales'}
        reference.load_state_dict(selected,strict=True)
        with torch.no_grad():
            synthetic=torch.linspace(-.002,.002,expert.head[-1].weight.numel(),device=device).reshape_as(expert.head[-1].weight)
            expert.head[-1].weight.copy_(synthetic);reference.head[-1].weight.copy_(synthetic)
            f0=expert.inputs(o,b)[0];f1=reference.inputs(o,b)[0]
            u0=expert.components(expert.head(f0),s)[0];u1=reference.components(reference.head(f1),s)[0]
            assert torch.equal(f0,f1) and torch.equal(u0,u1)
            expert.head[-1].weight.zero_()
        del reference
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        active=[]
        for step in range(8):
            opt.zero_grad();out,res=model(o,b,s)
            loss=h.m.objective(out,b,gt,res,s,groups,h.c.settings(1,arm),'balanced_protected',scale)
            assert torch.isfinite(loss);loss.backward()
            for name,v in model.named_parameters():
                if v.requires_grad:
                    assert v.grad is not None and torch.isfinite(v.grad).all(),name
                    if step>=3 and not name.endswith('key.bias'):assert v.grad.abs().sum()>0,('STOP expert disconnected',arm,name,step)
                else:assert v.grad is None,name
            for expert in model.experts:
                if step>=3:assert expert.head[-1].weight.grad[4:].abs().sum()+expert.head[-1].bias.grad[4:].abs().sum()>0
            P=h.physical.penalty(out,b,groups,model.physical_accel_scales);active.append(float(P.detach()))
            opt.step();h.frozen_audit(model);h.guard.check()
        gradient[arm]={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.requires_grad}
        residual,parts=model.routed_components(o,b,s)
        assert torch.isfinite(out).all() and out.shape==(128,20,12,2)
        recombined=(parts['expert_increments']*parts['weights'][...,None]).sum(2).clamp(-parts['limit'],parts['limit'])
        assert torch.equal(residual,recombined) and (residual.abs()<=parts['limit']+1e-6).all()
        for expert in model.experts:
            for name,value in expert.decoder.state_dict().items():assert torch.equal(value.cpu(),h.n.DECODER_STATE[name]),name
        responses[arm]=dict(branch_support=support,natural_P=active,both_experts_connected=True,raw_expert_parent_nonzero_EXACT=True)
        model.native_fallback.fill_(True);out,z=model(o,b,s);assert torch.equal(out,b) and torch.equal(z,torch.zeros_like(z))
        del model,opt
    assert counts==dict(zero=91596,mean_std=91596)
    assert states[0].keys()==states[1].keys() and all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    for k,v in original.state_dict().items():assert torch.equal(v.cpu(),original_state[k])
    del original,original_state,states
    # Fixed synthetic active-output physical parameter path, no changed protocol.
    h.p.seed_everything(3);model=h.PhysicalHead(940,h.c.settings(3,'mean_std')).to(device)
    out,res=model(o,b,s);ramp=torch.arange(1,13,device=device,dtype=out.dtype).square()[None,None,:,None]*.01
    P=h.physical.penalty(out+ramp,b,groups,model.physical_accel_scales)
    gr=torch.autograd.grad(P,[v for v in model.parameters() if v.requires_grad],allow_unused=True)
    physical_norm=float(torch.sqrt(sum((v.square().sum() for v in gr if v is not None),start=torch.tensor(0.,device=device))))
    assert P>0 and physical_norm>0 and np.isfinite(physical_norm)
    del model
    fixture=h.HERE/'preflight_fixture.txt';fixture.write_text('allowed-source',encoding='utf-8');digest=h.p.sha256(fixture)
    h.verify({str(fixture):digest})
    try:h.verify({str(fixture):'0'*64})
    except AssertionError:pass
    else:raise AssertionError('bad source accepted')
    modelpath=h.HERE/'route_threshold.npz';assert h.p.sha256(modelpath)==route_record['npz_SHA']
    h.old_manifest()
    record=dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=counts,gradients=gradient,response=responses,
        source_hashes=h.source_hashes(),fit_protection_scales=parent['fit_protection_scales'],physical_math=physical_math,
        state_math=state_math,route_math=route_math,route_FIT=route_record,synthetic_active_parameter_Pnorm=physical_norm,
        paired_weights_buffers_exact=True,native_identity_exact=True,all_original_F16_decoder_scorer_frozen=True,
        online_cache_alignment=alignment,old_files=old,both_originaltotal_plus_P_scalar_gradient_EXACT=True,
        badsource_reject=True,raw_expert_original_nonzero_alignment_EXACT=True,memory=h.guard.check(),elapsed_seconds=time.perf_counter()-start)
    h.m.write_json(h.HERE/'preflight.json',record);h.guard.save('PREFLIGHT_PASS')
    print('PREFLIGHT PASS '+json.dumps(dict(counts=counts,route=route_record,old=old,seconds=record['elapsed_seconds'],memory=record['memory'])),flush=True)
