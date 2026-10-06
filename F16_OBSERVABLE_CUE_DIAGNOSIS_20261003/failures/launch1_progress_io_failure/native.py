"""Frozen actual F16 replay, hooked representations; future has no forward route."""
from common import *
import torch

def build():
    torch.set_num_threads(4)
    loader = module('observable_cues_native_loader', ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/run_formal_regime_selection.py')
    device = torch.device('cuda:0')
    cfg, _, front, _, adapter = loader.setup_backend(device)
    model = loader.build_baseline('eth', cfg, front, adapter, device)
    model.device = str(device)
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training
    return model

def forward(model, local, num_valid):
    """local contains exclusively normalized observed positions, not futures."""
    center = local.mean(-2, keepdim=True)
    captured = []
    def hook(owner, args): captured.append(args)
    handle = model.decoder.register_forward_pre_hook(hook)
    try:
        pred = model(local-center, None, None, mode='testing', num_valid=num_valid.int(), graph_loc=local)[2]
    finally: handle.remove()
    assert len(captured) == 1
    tokens = captured[0][0][:,0]
    decoder_past = captured[0][1][:,:1]
    decoder = model.decoder
    motion = decoder.motion_features(decoder_past)[:,0]
    ctx = decoder.context(torch.cat((tokens[:,-1],tokens.mean(1),motion),-1))
    mode = decoder.mode_mlp(ctx[:,None]+decoder.mode_embedding[None])
    encoder = model.agent_aware_trans_encoder
    relative = pred[:,0]+center[:,0,None]-local[:,0,-1,None,None]
    # Actual score output is obtained by replaying the same frozen decoder.
    _, logits = decoder(captured[0][0], captured[0][1])
    values = dict(E1=encoder.last_temporal_features[:,0].reshape(len(local),-1),
                  E2=tokens.reshape(len(local),-1),
                  E2_motion=torch.cat((tokens.reshape(len(local),-1), motion),-1),
                  E3=ctx, E4=torch.cat((mode.mean(1),mode.std(1,unbiased=False)),-1),
                  social=encoder.last_agent_features[:,0], tokens=tokens,
                  paths=relative, logits=logits[:,0], allowed=encoder.last_allowed[:,0])
    return values

def extract_population(split, model):
    dest = HERE/f'features/{split}.npz'
    if dest.exists():
        record = read(f'features/{split}_audit.json')
        assert sha(dest) == record['sha256'] and record['status'] == 'PASS'
        return npz(dest)
    cache, enc, obs = load_population(split)
    parts = {}; max_path = max_token = max_score = max_social = 0.
    t = time.perf_counter()
    with torch.no_grad():
        for begin in range(0,len(obs),256):
            stage('FEATURE_EXTRACTION', split=split, completed_rows=begin, total_rows=len(obs))
            rows = slice(begin,begin+256)
            local = torch.tensor(obs[rows,160:304].reshape(-1,9,8,2), device='cuda:0')
            valid = torch.tensor(obs[rows,304], device='cuda:0')
            result = forward(model,local,valid)
            values = {k:v.detach().cpu().numpy() for k,v in result.items()}
            max_path=max(max_path,float(np.abs(values['paths']-cache['candidate'][rows,:,:24].reshape(-1,20,12,2)).max()))
            max_token=max(max_token,float(np.abs(values['tokens']-enc['tokens'][rows]).max()))
            max_score=max(max_score,float(np.abs(values['logits']-cache['logits'][rows]).max()))
            max_social=max(max_social,float(np.abs(values['social']-cache['social'][rows,:96]).max()))
            assert max(max_path,max_token,max_score,max_social) < 2e-5
            for k,v in values.items():
                assert np.isfinite(v).all(), k
                parts.setdefault(k,[]).append(v)
    data={k:np.concatenate(v) for k,v in parts.items()}
    data.update(raw_features(obs)); data['sample_id']=cache['sample_id']
    save_npz(f'features/{split}.npz', **data)
    write(f'features/{split}_audit.json',dict(status='PASS',sha256=sha(dest),n=len(obs),seconds=time.perf_counter()-t,
            max_path_error=max_path,max_token_error=max_token,max_logit_error=max_score,max_social_error=max_social,
            all_original_parameters_frozen=True,forward_fields=['observed_local_9x8x2','observed_valid_count'],future_forward=False))
    return data

def functional(model, observation, ids):
    local=torch.tensor(observation[:,160:304].reshape(-1,9,8,2),device='cuda:0')
    count=torch.tensor(observation[:,304],device='cuda:0')
    with torch.no_grad(): base=forward(model,local,count)
    records=[]
    for sign in (-1,1):
        modified=local.clone()
        # Rigid 5cm translation of the first valid neighbor, whole observed history.
        modified[:,1,:,0] += sign*.05*(count>1)[:,None]
        with torch.no_grad(): val=forward(model,modified,count)
        records.append(dict(kind='neighbor_rigid_translation',shift_m=sign*.05,
            path_max_delta=np.abs((val['paths']-base['paths']).cpu().numpy()).max((1,2,3)).tolist(),
            logit_max_delta=np.abs((val['logits']-base['logits']).cpu().numpy()).max(1).tolist(),
            argmax_switched=(val['logits'].argmax(1)!=base['logits'].argmax(1)).cpu().tolist()))
    modified=local.clone()
    invalid=torch.arange(9,device='cuda:0')[None]>=count[:,None]
    modified += invalid[:,:,None,None]*.01
    with torch.no_grad(): sanity=forward(model,modified,count)
    sanity_err=float((sanity['paths']-base['paths']).abs().max())
    # Piecewise local Jacobian. It measures dependence, not beneficial cue use.
    x=local[:16].clone().requires_grad_(True)
    result=forward(model,x,count[:16])
    target=result['paths'][:,:,-1].square().mean()+result['logits'].square().mean()
    grad=torch.autograd.grad(target,x)[0]
    assert torch.isfinite(grad).all()
    assert not any(p.grad is not None for p in model.parameters())
    return dict(status='PASS_FUNCTIONAL_DEPENDENCE_ONLY',sample_id=ids.tolist(),records=records,
                invalid_neighbor_sanity_path_max_delta=sanity_err,
                input_jacobian_finite=True,input_jacobian_abs_mean=float(grad.abs().mean()),
                no_parameter_optimizer=True,no_GT_counterfactual_scoring=True,
                support_limit='Fixed small perturbation; per-case observed-support membership is not certified. Do not infer quality improvement or causal benefit.')
