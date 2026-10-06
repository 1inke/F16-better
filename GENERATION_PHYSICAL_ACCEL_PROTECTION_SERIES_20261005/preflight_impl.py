"""Actual frozen GPU, sole loss factor, math and cache verification."""
import json,time,sys
from pathlib import Path
import numpy as np
import torch
def run(h,device):
    h.guard.check();oldcount=h.old_manifest(not(h.HERE/'old_results_manifest.json').exists())
    _,_,original=h.n.prepare_backend(device);original.device=str(device)
    cache,cfg,fit,cal,values=h.prepared_train(device)
    assert len(fit)==25912 and len(cal)==2879 and not set(fit)&set(cal)
    assert set(np.unique(cache['change_group'][fit]))=={0,1,2}
    prior=json.loads((h.ROOT/'GENERATION_CANDIDATE_STATE_SERIES_20261005/preflight.json').read_text(encoding='utf-8'))
    assert np.array_equal(np.load(h.HERE/'state_scalers.npz')['mean'],np.load(h.ROOT/'GENERATION_CANDIDATE_STATE_SERIES_20261005/state_scalers.npz')['mean'])
    assert np.array_equal(np.load(h.HERE/'state_scalers.npz')['std'],np.load(h.ROOT/'GENERATION_CANDIDATE_STATE_SERIES_20261005/state_scalers.npz')['std'])
    oldcfg=torch.load(h.ROOT/'GENERATION_CANDIDATE_STATE_SERIES_20261005/results/models/mean_std/seed_1/balanced_protected/final.pth',map_location='cpu',weights_only=False)['config']
    current=h.c.settings(1,'mean_std')
    for key in ('learning_rate','weight_decay','batch_size','epochs','velocity_residual_scale','protection_weight','guard_fraction','dropout','calibration_fraction','bootstrap_repetitions','bootstrap_seed'):
        assert current[key]==oldcfg[key],(key,current[key],oldcfg[key])
    physical_math=h.physical.fixtures()
    from state_features import validate_state_math
    state_math=validate_state_math()
    protection=torch.tensor(prior['fit_protection_scales'],device=device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][fit[:128]],device=device)
    common=h.module('acc_replay_common',h.ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/common.py');sys.modules['common']=common
    native=h.module('acc_replay_native',h.ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/native.py')
    with np.load(h.ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz',allow_pickle=False) as raw:
        assert np.array_equal(raw['sample_id'],cache['sample_id'])
        obs=raw['observation'][fit[:128]]
    with torch.no_grad():result=native.forward(original,torch.tensor(obs[:,160:304].reshape(-1,9,8,2),device=device),torch.tensor(obs[:,304],device=device))
    h.p.seed_everything(1);model=h.TimeResponseHead(940,current).to(device).eval()
    alignment=dict(tokens=float((result['tokens']-o[:,16:784].reshape(-1,8,96)).abs().max()),
        native_path=float((result['paths']-b).abs().max()),context=float((result['E3']-model.native_inputs(o)[:,0,:64]).abs().max()))
    assert max(alignment.values())<2e-5,alignment
    del result
    h.p.seed_everything(1);parent=h.RawParent(940,current).to(device).eval()
    paired={k:v for k,v in model.state_dict().items() if k!='physical_accel_scales'}
    assert all(torch.equal(v,parent.state_dict()[k]) for k,v in paired.items())
    with torch.no_grad():
        w=torch.linspace(-.002,.002,model.head[-1].weight.numel(),device=device).reshape_as(model.head[-1].weight)
        model.head[-1].weight.copy_(w);parent.head[-1].weight.copy_(w)
        fm,wm=model.inputs(o,b);fp,wp=parent.inputs(o,b);om,rm=model(o,b,s);op,rp=parent(o,b,s)
    assert torch.equal(fm,fp) and torch.equal(wm,wp) and torch.equal(om,op) and torch.equal(rm,rp),'rawcontrol alignment'
    detached=om.detach().requires_grad_()
    old=h.original_objective(detached,b,gt,rm.detach(),s,groups,current,'balanced_protected',protection)
    controlcfg=h.c.settings(1,'zero')
    control=h.m.objective(detached,b,gt,rm.detach(),s,groups,controlcfg,'balanced_protected',protection)
    assert torch.equal(control,old)
    a=torch.autograd.grad(old,detached,retain_graph=True)[0];cc=torch.autograd.grad(control,detached)[0]
    assert torch.equal(a,cc),'control originaltotalgradient'
    del parent,model,detached,fm,fp,om,op,rm,rp,a,cc
    states=[];gradients={};responses={};timings={}
    for arm in h.c.ARMS:
        h.p.seed_everything(1);current=h.c.settings(1,arm);model=h.TimeResponseHead(940,current).to(device).eval()
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        assert sum(v.numel() for v in model.parameters() if v.requires_grad)==45798
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
        assert float(h.physical.penalty(out,b,groups,model.physical_accel_scales))==0
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        active_steps=0;physical_norms=[];clock=0.
        torch.cuda.synchronize();start=time.perf_counter()
        for step in range(8):
            opt.zero_grad();out,res=model(o,b,s)
            extra,detail=h.physical.penalty(out,b,groups,model.physical_accel_scales,True)
            if any(detail['active']):
                vector=torch.autograd.grad(extra,[v for v in model.parameters() if v.requires_grad],retain_graph=True,allow_unused=True)
                norm=sum(float(v.square().sum()) for v in vector if v is not None)**.5
                assert np.isfinite(norm) and norm>0
                physical_norms.append(norm);active_steps+=1
            orig=h.original_objective(out,b,gt,res,s,groups,current,'balanced_protected',protection)
            loss=h.m.objective(out,b,gt,res,s,groups,current,'balanced_protected',protection)
            expected=orig+.1*extra if arm=='mean_std' else orig
            assert torch.equal(loss,expected);loss.backward()
            for name,v in model.named_parameters():
                if v.requires_grad:
                    assert v.grad is not None and torch.isfinite(v.grad).all(),name
                    if step>=3 and name!='key.bias':assert v.grad.abs().sum()>0,name
                else:assert v.grad is None,name
            if step>=3:
                clock=float(model.head[-1].weight.grad[4:].abs().sum()+model.head[-1].bias.grad[4:].abs().sum());assert clock>0
            opt.step();h.frozen_audit(model)
        torch.cuda.synchronize();timings[arm]=(time.perf_counter()-start)/8
        assert torch.isfinite(out).all() and out.shape==(128,20,12,2)
        codes=model.head(model.inputs(o,b)[0]);res,parts=model.components(codes,s)
        assert (res.abs()<=parts['limit']+1e-6).all()
        endpoint=float((res.sum(2)-(12*parts['drift'].squeeze(2)+parts['pulse'].sum(2))).abs().max());assert endpoint<2e-5
        perturb=b.clone();perturb[:,:,5:,0]+=.02
        sensitivity=float((model.inputs(o,perturb)[0]-model.inputs(o,b)[0]).abs().max());assert sensitivity>1e-7
        gradients[arm]={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.grad is not None}
        responses[arm]=dict(natural_active_steps=active_steps,physical_parameter_gradient_norms=physical_norms,
            clock_gradient=clock,state_sensitivity=sensitivity,endpoint_decomposition_error=endpoint)
        model.native_fallback.fill_(True);identity,zero=model(o,b,s);assert torch.equal(identity,b) and zero.abs().sum()==0
        del opt,model
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0]),'fullpaired weightbuffers'
    assert not any(v.requires_grad for v in original.parameters())
    del original
    # Reject corrupted metadata with exact production checks without modifying real scales.
    metadata=json.loads((h.HERE/'physical_accel_scales.json').read_text(encoding='utf-8'))
    assert metadata['provenance']['fit_n']==25912 and metadata['npz_SHA']==h.p.sha256(h.HERE/'physical_accel_scales.npz')
    bad=metadata['provenance'].copy();bad['ids_SHA']='0'*64
    assert bad!=metadata['provenance']
    rejected=0
    for path in (h.HERE/'physical_accel_scales.npz',h.HERE/'run_head.py'):
        try:h.verify({str(path):'0'*64})
        except AssertionError:rejected+=1
        else:raise AssertionError('Bad source/output accepted')
    assert rejected==2
    h.old_manifest();source=h.source_hashes()
    h.m.write_json(h.HERE/'preflight.json',dict(status='PASS',source_hashes=source,fit_n=len(fit),cal_n=len(cal),
        fit_protection_scales=prior['fit_protection_scales'],physical_math=physical_math,state_math=state_math,
        physical_scales=metadata,rawcontrol_originaltotal_and_gradient_exact=True,
        rawcontrol_actual_parent_nonzero_features_paths_residual_exact=True,original_config_preserved=True,
        online_cache_alignment=alignment,gradients=gradients,response=responses,seconds_per_128step=timings,
        all_weights_buffers_paired_exact=True,all_original_F16_decoder_scorer_frozen=True,
        parameters_trainable={'zero':45798,'mean_std':45798},old_files=oldcount,memory=h.guard.check(),
        synthetic_active_output_verified=True,future_forward=False,test_access=False))
    h.guard.save('PREFLIGHT_PASS')
    print('PREFLIGHT PASS '+json.dumps(dict(old_files=oldcount,sources=len(source),response=responses,timing=timings)),flush=True)
