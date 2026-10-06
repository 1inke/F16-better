"""Read-only checkpoint diagnostics on fixed stratified fit/cal batches, never dev.
No optimizer step; outputs do not change checkpoint selection or guards.
"""
import json,sys
from pathlib import Path
import numpy as np
import torch
import run_modes as x

def gradient(loss,parameters):
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(parameters,values)])

def cosine(a,b):
    denominator=(a.norm()*b.norm()).item()
    return float((a@b).item()/denominator) if denominator>1e-15 else None

def main():
    device=torch.device('cuda:0')
    _,_,original=x.n.prepare_backend(device);del original
    cfg=x.c.settings(1,'mean_std')
    cache=x.p.load_npz(x.p.TRAIN_CACHE)
    fit,cal=x.p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    values=x.r.prepared(cache,False,fit=fit)
    x.r.setup_weights(cache,fit,values,cfg,device)
    scale=torch.tensor(json.loads((x.HERE/'preflight.json').read_text())['fit_protection_scales'],device=device)
    rows=[];lock=json.loads((x.HERE/'results/model_lock.json').read_text())['models'];sampling={}
    for split,indices in [('fit',fit),('cal',cal)]:
        for group in (0,1,2):
            available=indices[cache['change_group'][indices]==group]
            ids=np.random.default_rng(20261001+group).permutation(available)[:128]
            sampling[f'{split}_{group}']=cache['sample_id'][ids].tolist()
            o,b,s,gt=[torch.from_numpy(v[ids]).to(device) for v in values[:4]]
            groups=torch.tensor(cache['change_group'][ids],device=device)
            for seed in (1,2,3):
                for policy in ('constrained','final'):
                    path=x.HERE/f'results/models/mean_std/seed_{seed}/balanced_protected/{policy}.pth'
                    assert x.p.sha256(path)==lock[str(path)]
                    saved=torch.load(path,map_location='cpu',weights_only=False)
                    model=x.ConditionalGenerator(o.shape[1],x.c.settings(seed,'mean_std')).to(device)
                    model.load_state_dict(saved['state_dict']);model.eval()
                    captured={}
                    def hook(module,inputs,output):captured['anchors']=inputs[1]+output
                    handle=model.decoder.refiner.register_forward_hook(hook)
                    output,residual=model(o,b,s);handle.remove()
                    raw=torch.einsum('tp,bakpc->baktc',model.decoder.interpolation,captured['anchors'])[:,0]
                    raw_v=torch.diff(raw-b,dim=2,prepend=torch.zeros_like(b[:,:,:1]))
                    limit=model.strength*(s[:,0,None,None,None]+.03)
                    assert torch.allclose(output,b+raw_v.clamp(-limit,limit).cumsum(2),atol=2e-5,rtol=0)
                    winner_cfg=dict(cfg,diversity_weight=0.,residual_weight=0.,smoothness_weight=0.)
                    winner=x.r.objective(output,b,gt,residual,s,winner_cfg,'native_balanced')
                    regular=x.r.objective(output,b,gt,residual,s,cfg,'native_balanced')-winner
                    protected=cfg['protection_weight']*x.m.protection(output,b,gt,groups,scale,cfg)
                    total=x.m.objective(output,b,gt,residual,s,groups,cfg,'balanced_protected',scale)
                    assert torch.allclose(total,winner+regular+protected,atol=1e-6,rtol=0)
                    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];parameters=[p for n,p in named]
                    gw,gr,gp=[gradient(term,parameters) for term in (winner,regular,protected)]
                    error=(output-gt[:,None]).norm(dim=-1);ade=error.mean(-1);fde=error[:,:,-1];choice=(ade+cfg['fde_loss_weight']*fde).argmin(-1)
                    native_error=(b-gt[:,None]).norm(dim=-1)
                    target=gt[:,None]-b;target_v=torch.diff(target,dim=2,prepend=torch.zeros_like(target[:,:,:1]))
                    anchor_error=(raw[:,:, [3,7,11]]-gt[:,None,[3,7,11]]).norm(dim=-1).mean(-1).min(-1).values
                    layers={};offset=0
                    for name,parameter in named:
                        end=offset+parameter.numel();prefix=name.split('.')[1] if name.startswith('decoder.') else name.split('.')[0]
                        layers[prefix]=layers.get(prefix,0.)+float(gw[offset:end].square().sum());offset=end
                    row=dict(split=split,group=group,n=len(ids),seed=seed,policy=policy,
                        winner_loss=float(winner),regularizer_loss=float(regular),protection_loss=float(protected),
                        winner_gradient_norm=float(gw.norm()),regularizer_gradient_norm=float(gr.norm()),protection_gradient_norm=float(gp.norm()),
                        winner_regularizer_cosine=cosine(gw,gr),winner_protection_cosine=cosine(gw,gp),
                        winner_layer_gradient_norm={k:v**.5 for k,v in layers.items()},
                        raw_velocity_clip_fraction=float((raw_v.abs()>=limit).float().mean()),
                        winner_clip_fraction=float((raw_v.abs()>=limit)[torch.arange(len(ids),device=device),choice].float().mean()),
                        gt_any_candidate_velocity_bound_feasible_fraction=float((target_v.abs()<=limit).flatten(2).all(-1).any(-1).float().mean()),
                        joint_selected_ade=float(ade[torch.arange(len(ids),device=device),choice].mean()),
                        minade=float(ade.min(-1).values.mean()),minfde=float(fde.min(-1).values.mean()),
                        native_minade=float(native_error.mean(-1).min(-1).values.mean()),native_minfde=float(native_error[:,:,-1].min(-1).values.mean()),
                        raw_anchor_min_error=float(anchor_error.mean()))
                    assert all(np.isfinite(row[k]) for k in ('winner_gradient_norm','regularizer_gradient_norm','raw_velocity_clip_fraction','minade'))
                    rows.append(row);del model
    sources={str(Path(__file__).resolve()):x.p.sha256(Path(__file__)),str(x.HERE/'run_modes.py'):x.p.sha256(x.HERE/'run_modes.py'),str(x.p.TRAIN_CACHE):x.p.sha256(x.p.TRAIN_CACHE),**lock}
    x.m.write_json(x.HERE/'trainability_diagnosis.json',dict(status='PASS',scope='fixed128 per group fit/cal only, no dev/test, no optimizer updates; descriptive checkpoint diagnostics not population estimates or independent replication',sample_ids=sampling,source_hashes=sources,rows=rows))
    print('PASS 36 checkpoint/batch diagnoses fit/cal only; no optimizer steps',flush=True)
    for split in ('fit','cal'):
        for policy in ('constrained','final'):
            subset=[r for r in rows if r['split']==split and r['policy']==policy and r['group']==2]
            print(split,policy,{key:float(np.mean([r[key] for r in subset])) for key in ('raw_velocity_clip_fraction','winner_clip_fraction','winner_gradient_norm','regularizer_gradient_norm','protection_gradient_norm','gt_any_candidate_velocity_bound_feasible_fraction','minade','native_minade')},flush=True)

if __name__=='__main__':main()
