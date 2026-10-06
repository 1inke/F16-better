"""Frozen completion FIT/CAL physical and original-objective population audit."""
import json
import numpy as np
import torch
from pathlib import Path
def scalar(arr,cfg,scales,arm):
    groups=arr['group'];n=len(groups)
    primary=arr['primary'].astype(np.float64).mean()
    diversity=arr['div_num'].astype(np.float64).sum()/max(1,arr['div_den'].sum())
    residual=arr['residual'].astype(np.float64).mean();smooth=arr['smooth'].astype(np.float64).mean()
    protection=arr['protection'].astype(np.float64).sum()/max(1,arr['protected'].sum())
    original=primary+cfg['diversity_weight']*diversity+cfg['residual_weight']*residual+cfg['smoothness_weight']*smooth+cfg['protection_weight']*protection
    extra=0.;details=[]
    for g in range(3):
        idx=groups==g;count=int(idx.sum())
        current=float(arr['accel'][idx].astype(np.float64).mean()) if count else 0.
        native=float(arr['native_accel'][idx].astype(np.float64).mean()) if count else 0.
        part=(count/n)*max(current-1.01*native,0)/float(scales[g]);extra+=part
        details.append(dict(group=g,n=count,mean=current,native=native,active=current>1.01*native,contribution=part))
    return dict(original_total=original,P=extra,added_total=original+.1*extra,groups=details,
        coverage=int(arr['good'].sum()),n=n,definition='population denominator; NOT original batch objective or training AdamW direction')

def parts(h,out,base,gt,res,speed,groups,cfg,scale):
    error=torch.linalg.vector_norm(out-gt[:,None],dim=-1);ade=error.mean(-1);fde=error[:,:,-1]
    target=h.r.m.labels(speed[:,1:],gt[:,None,-1]/12).squeeze(1)
    quality=(ade+cfg['fde_loss_weight']*fde).min(-1).values*h.r.WEIGHTS[target]
    family=h.r.m.labels(speed[:,1:],base[:,:,-1]/12);cross=family[:,:,None]!=family[:,None,:]
    reference=(h.r.g.pair_distance(base)*cfg['diversity_reference_fraction']).clamp(max=cfg['diversity_distance_cap_m'])
    div=(reference-h.r.g.pair_distance(out)).relu()*cross
    native=torch.linalg.vector_norm(base-gt[:,None],dim=-1)
    harm=(ade.min(-1).values-1.01*native.mean(-1).min(-1).values).relu()/scale[0]+(fde.min(-1).values-1.01*native[:,:,-1].min(-1).values).relu()/scale[1]
    return dict(primary=quality,div_num=div.sum((1,2)),div_den=cross.sum((1,2)),
        residual=res.square().mean((1,2,3)),smooth=torch.diff(res,dim=2).square().mean((1,2,3)),
        protection=harm*(groups!=2),protected=groups!=2,group=groups,
        accel=h.physical.window_accel(out),native_accel=h.physical.window_accel(base),
        good=((ade<=.2)&(fde<=.4)).any(-1))

def audit(h,device):
    cache,cfg,fit,cal,values=h.prepared_train(device)
    scale=torch.tensor(json.loads((h.HERE/'preflight.json').read_text())['fit_protection_scales'],device=device)
    output=h.m.RESULTS/'physical_population';output.mkdir(exist_ok=True)
    records=[];hashes={};maxsample=0.
    with torch.no_grad():
        for arm in h.c.ARMS:
            for seed in (1,2,3):
                for policy in h.c.POLICIES:
                    path=h.c.checkpoint(seed,arm,policy);before=h.p.sha256(path)
                    saved=torch.load(path,map_location='cpu',weights_only=False)
                    model=h.TimeResponseHead(saved['input_dim'],saved['config']).to(device).eval();model.load_state_dict(saved['state_dict'])
                    startstate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    for split,ids in (('FIT',fit),('CAL',cal)):
                        store={}
                        for start in range(0,len(ids),128):
                            idx=ids[start:start+128];h.guard.check()
                            o,b,s,gt=[torch.from_numpy(v[idx]).to(device) for v in values[:4]]
                            g=torch.tensor(cache['change_group'][idx],device=device)
                            out,res=model(o,b,s);column=parts(h,out,b,gt,res,s,g,saved['config'],scale)
                            arrays={k:v.cpu().numpy() for k,v in column.items()}
                            if start==0:
                                deriv=scalar(arrays,saved['config'],model.physical_accel_scales.cpu().numpy(),arm)
                                direct=float(h.original_objective(out,b,gt,res,s,g,saved['config'],'balanced_protected',scale))
                                err=abs(deriv['original_total']-direct);assert err<2e-6;maxsample=max(maxsample,err)
                            for key,val in arrays.items():store.setdefault(key,[]).append(val)
                        arrays={k:np.concatenate(v) for k,v in store.items()};arrays['sample_id']=cache['sample_id'][ids]
                        scalers=model.physical_accel_scales.cpu().numpy()
                        first=scalar(arrays,saved['config'],scalers,arm)
                        dest=output/f'seed{seed}_{arm}_{policy}_{split}.npz'
                        np.savez_compressed(dest,**arrays);hashes[str(dest)]=h.p.sha256(dest)
                        with np.load(dest,allow_pickle=False) as z:
                            assert np.array_equal(z['sample_id'],cache['sample_id'][ids])
                            second=scalar(z,saved['config'],scalers,arm)
                        assert first==second
                        records.append(dict(seed=seed,arm=arm,policy=policy,split=split,**first))
                    for key,v in model.state_dict().items():assert torch.equal(v.detach().cpu(),startstate[key]),key
                    h.frozen_audit(model);assert h.p.sha256(path)==before
                    del model,saved,startstate
    # Exact recorded training diagnostics, not population means.
    trace_checks=[]
    for arm in h.c.ARMS:
        for seed in (1,2,3):
            parent=h.c.checkpoint(seed,arm,'final').parent
            rows=json.loads((parent/'objective_history.json').read_text(encoding='utf-8'))['rows']
            resume=torch.load(parent/'resume.pth',map_location='cpu',weights_only=False)
            assert len(rows)==80 and rows==resume['objective_diagnostics']
            for row in rows:
                assert row['n']==25912 and sum(row['groups_n'])==25912
                assert abs(row['total']-row['original']-row['added'])<2e-6
                assert abs(row['added']-.1*row['P'])<2e-6
            trace_checks.append(dict(arm=arm,seed=seed,epochs=80,exact_resume_prefix=True))
            del resume
    result=dict(status='PASS',records=records,output_SHA=hashes,sampled_original_scalar_max=maxsample,
        training_trace_checks=trace_checks,original_total_and_P_population_recompute=True,
        no_optimizer=True,weights_buffers_fixed=True,saved_arithmetic_not_allmodels_second_replay=True)
    h.m.write_json(output/'summary.json',result);return result
