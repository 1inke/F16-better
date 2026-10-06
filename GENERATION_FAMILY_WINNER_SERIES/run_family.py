"""Matched family-winner auxiliary loss: no multi-candidate concentration."""
from pathlib import Path
import importlib.util
import sys
import argparse
import json
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
CONTROL=HERE.parent/'GENERATION_RESIDUAL_CAPACITY_SERIES'
spec=importlib.util.spec_from_file_location('family_capacity_source',CONTROL/'run_capacity.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
m=c.m;r=c.r;p=c.p
m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
POLICIES=c.POLICIES;ARM='balanced_protected'
original_objective=m.objective


def family_minimum(quality,matching,weight):
    exists=matching.any(-1)
    minimum=quality.masked_fill(~matching,float('inf')).min(-1).values
    minimum=torch.where(exists,minimum,torch.zeros_like(minimum))
    return (minimum*weight).sum()/exists.sum().clamp_min(1)


def family_objective(output,base,future,residual,speed,groups,cfg,arm,scale):
    loss=original_objective(output,base,future,residual,speed,groups,cfg,arm,scale)
    with torch.no_grad():
        target=r.m.labels(speed[:,1:],future[:,None,-1]/12).squeeze(1)
        matching=r.m.labels(speed[:,1:],base[:,:,-1]/12)==target[:,None]
    errors=(output-future[:,None]).norm(dim=-1)
    quality=errors.mean(-1)+cfg['fde_loss_weight']*errors[:,:,-1]
    return loss+cfg['family_winner_weight']*family_minimum(quality,matching,r.WEIGHTS[target])


m.objective=family_objective


def checkpoint(seed,arm,tag):
    root=CONTROL/'results/models' if arm=='control' else m.MODELS
    return root/f'seed_{seed}'/ARM/f'{tag}.pth'


def preflight(device):
    cfg=m.settings(1);assert cfg['test_access']=='forbidden' and cfg['seeds']==[1,2,3]
    assert cfg['velocity_residual_scale']==1.5 and cfg['family_winner_weight']==.25
    prior=json.loads((CONTROL/'preflight.json').read_text())
    for path,digest in {**prior['source_hashes'],**prior['controls']}.items():assert p.sha256(Path(path))==digest,path
    lock=json.loads((CONTROL/'results/model_lock.json').read_text())['models']
    controls={str(checkpoint(seed,'control',tag)):lock[str(checkpoint(seed,'control',tag))] for seed in cfg['seeds'] for tag in POLICIES}
    for path,digest in controls.items():assert p.sha256(Path(path))==digest
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    assert not(set(cache['sample_id'][fit]) & set(cache['sample_id'][cal]))
    _,weights=r.setup_weights(cache,fit,values,cfg,device)
    scale=np.array([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32).clip(1e-6)
    assert np.array_equal(scale,np.array(prior['fit_protection_scales'],np.float32))
    for seed in cfg['seeds']:
        saved=torch.load(checkpoint(seed,'control','constrained'),map_location='cpu',weights_only=False)
        for key,value in saved['config'].items():
            if key!='protocol':assert m.settings(seed)[key]==value,(key,value)
        assert np.array_equal(saved['mean'],values[4]) and np.array_equal(saved['std'],values[5])
    # Single matching minimum, no contribution from wrong-family or missing-family samples.
    q=torch.tensor([[.1,2.,3.],[1.,2.,3.]],device=device,requires_grad=True)
    matching=torch.tensor([[False,True,True],[False,False,False]],device=device)
    auxiliary=family_minimum(q,matching,torch.ones(2,device=device));assert auxiliary.item()==2
    auxiliary.backward();assert q.grad[0,1]==1 and q.grad.sum()==1
    q2=q.detach().clone().requires_grad_(True);zero=family_minimum(q2,torch.zeros_like(matching),torch.ones(2,device=device));zero.backward();assert zero==0 and q2.grad.abs().sum()==0
    o,b,s,f=[torch.from_numpy(v[:16]).to(device) for v in values[:4]];groups=torch.tensor(cache['change_group'][:16],device=device)
    p.seed_everything(1);model=r.m.Correction(o.shape[1],cfg).to(device);model.eval();output,res=model(o,b,s);assert torch.equal(output,b)
    actual=family_objective(output,b,f,res,s,groups,cfg,ARM,torch.tensor(scale,device=device));actual.backward()
    assert torch.isfinite(actual) and model.network[-1].weight.grad.abs().sum()>0
    perm=torch.randperm(20,device=device)
    with torch.no_grad():
        model.network[-1].bias.fill_(.1);out,res=model(o,b,s);out_p,res_p=model(o,b[:,perm],s)
        assert torch.allclose(out_p,out[:,perm],atol=1e-6)
        a=family_objective(out,b,f,res,s,groups,cfg,ARM,torch.tensor(scale,device=device))
        ap=family_objective(out_p,b[:,perm],f,res_p,s,groups,cfg,ARM,torch.tensor(scale,device=device));assert torch.allclose(a,ap,atol=1e-6)
    sources={**prior['source_hashes'],str(CONTROL/'run_capacity.py'):p.sha256(CONTROL/'run_capacity.py'),str(Path(__file__)):p.sha256(Path(__file__)),str(HERE/'protocol.json'):p.sha256(HERE/'protocol.json')}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),candidate_count=20,new_trainings=3,fit_protection_scales=scale.tolist(),weights=weights.tolist(),source_hashes=sources,controls=controls,test_access=False))
    print('PREFLIGHT PASS: hash/config/scales/split/GPU/backward/single-minimum/missing-family/permutation',flush=True)


def evaluate(device):
    cfg=m.settings(1);seeds=cfg['seeds'];paths=[checkpoint(seed,arm,tag) for seed in seeds for arm in ('control','family') for tag in POLICIES]
    assert all(path.exists() for path in paths)
    m.write_json(m.RESULTS/'model_lock.json',dict(models={str(path):p.sha256(path) for path in paths},dev_used_for_selection=False,test_access=False))
    cache=p.load_npz(p.DEV_CACHE);dim=cache['motion'].shape[1]+cache['social'].shape[1]
    values=r.prepared(cache,False,mean=np.zeros((1,dim),np.float32),std=np.ones((1,dim),np.float32));base,speed,future=values[1:4]
    group=cache['change_group'];native=r.generation_metrics(base,base,speed,future);stats={};rows=[]
    for seed in seeds:
        for arm in ('control','family'):
            for tag in POLICIES:
                saved=torch.load(checkpoint(seed,arm,tag),map_location='cpu',weights_only=False);assert saved['config']['velocity_residual_scale']==1.5
                current=r.prepared(cache,False,mean=saved['mean'],std=saved['std']);model=r.m.Correction(dim,saved['config']).to(device);model.load_state_dict(saved['state_dict'])
                pred=r.g.predict(model,*current[:3],device);assert pred.shape==(len(group),20,12,2) and np.isfinite(pred).all()
                if arm=='control':
                    previous=p.load_npz(CONTROL/f'results/seed{seed}_large_{tag}.npz');assert np.array_equal(previous['sample_id'],cache['sample_id']) and np.allclose(previous['trajectory'],pred,atol=1e-7,rtol=0)
                value=r.generation_metrics(pred,base,speed,future);stats[seed,arm,tag]=value
                np.savez_compressed(m.RESULTS/f'seed{seed}_{arm}_{tag}.npz',sample_id=cache['sample_id'],trajectory=pred,**value)
                for name,metrics in m.grouped(value,group).items():rows.append(dict(seed=seed,arm=arm,policy=tag,group=name,epoch=saved['epoch'],trained_feasible=saved['trained_feasible'],**metrics))
    aggregate=[];boot=[]
    for arm in ('control','family'):
        for tag in POLICIES:
            for name,mask in m.masks(group):aggregate.append(dict(arm=arm,policy=tag,group=name,**{k:float(np.stack([stats[seed,arm,tag][k][mask] for seed in seeds]).mean()) for k in native}))
    for tag in POLICIES:
        for reference in ('native','control'):
            for name,mask in m.masks(group):
                for key in ('minade20','minfde20','top3_ade'):
                    delta=np.stack([stats[seed,'family',tag][key][mask]-(native[key][mask] if reference=='native' else stats[seed,'control',tag][key][mask]) for seed in seeds])
                    mean,lo,hi=m.h.hierarchical_delta(delta,cfg['bootstrap_repetitions'],cfg['bootstrap_seed']);boot.append(dict(policy=tag,reference=reference,group=name,metric=key,delta=mean,ci_low=lo,ci_high=hi))
    lookup={(x['arm'],x['policy'],x['group']):x for x in aggregate};ref=m.grouped(native,group);primary=lookup['family','constrained','change'];overall=lookup['family','constrained','overall']
    gates={
        'change_ade_fde_gain_5pct':all(primary[k]<=ref['change'][k]*.95 for k in ('minade20','minfde20')),
        'protected_groups_guard':all(lookup['family','constrained',name][k]<=ref[name][k]*1.01 for name in ('overall','stable','transition') for k in ('minade20','minfde20')),
        'each_seed_change_improves':all(stats[seed,'family','constrained'][k][group==2].mean()<native[k][group==2].mean() for seed in seeds for k in ('minade20','minfde20')),
        'native_and_control_ci':all(x['ci_high']<0 for x in boot if x['policy']=='constrained' and x['group']=='change' and x['metric'] in ('minade20','minfde20')),
        'all_seed_feasible':all(x['trained_feasible'] for x in rows if x['arm']=='family' and x['policy']=='constrained'),
        'coverage_guard':overall['maneuver_class_count']>=ref['overall']['maneuver_class_count']*.95 and overall['gt_family_covered']>=ref['overall']['gt_family_covered']-.01,
        'duplicate_guard':overall['duplicate_fraction_10cm']<=ref['overall']['duplicate_fraction_10cm']+.02,
        'kinematic_guard':all(overall[k]<=ref['overall'][k]+.01 for k in ('speed_over3mps_fraction','acceleration_over4mps2_fraction'))}
    p.write_csv(m.RESULTS/'per_seed_metrics.csv',rows);p.write_csv(m.RESULTS/'aggregate_metrics.csv',aggregate);p.write_csv(m.RESULTS/'bootstrap.csv',boot)
    summary=dict(status='COMPLETE',verdict='MAJOR_DEV_SIGNAL_REQUIRES_INDEPENDENT_VALIDATION' if all(gates.values()) else 'FAMILY_WINNER_GATE_FAIL',gates=gates,native=ref,aggregate=aggregate,bootstrap=boot,scope='ETH reused dev; K20; three paired correction seeds; verified reused controls; family GT labels loss-only; no test/scorer/backbone training')
    m.write_json(m.RESULTS/'summary.json',summary)
    lines=['# Single family winner experiment','',summary['verdict'],'',summary['scope'],'','| Arm | Policy | Group | minADE | minFDE |','|---|---|---|---:|---:|']
    for x in aggregate:
        if x['group'] in ('overall','change'):lines.append(f'| {x["arm"]} | {x["policy"]} | {x["group"]} | {x["minade20"]:.6f} | {x["minfde20"]:.6f} |')
    lines+=['',json.dumps({k:bool(v) for k,v in gates.items()},indent=2)];(m.RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');parser.add_argument('--device',default='cuda:0');args=parser.parse_args();device=torch.device(args.device)
    if args.action=='preflight':preflight(device);return
    audit=json.loads((HERE/'preflight.json').read_text())
    for path,digest in {**audit['source_hashes'],**audit['controls']}.items():assert p.sha256(Path(path))==digest,path
    if (m.RESULTS/'summary.json').exists():raise RuntimeError('Already complete')
    m.MODELS.mkdir(parents=True,exist_ok=True);cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),m.settings(1)['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    for seed in m.settings(1)['seeds']:
        manifest=m.MODELS/f'seed_{seed}'/ARM/'complete.json'
        if manifest.exists():
            for path,digest in json.loads(manifest.read_text())['model_hashes'].items():assert p.sha256(Path(path))==digest
            print(f'[SKIP VERIFIED] seed{seed}',flush=True);continue
        print(f'[TRAIN] family winner seed{seed}',flush=True);m.train(seed,ARM,cache,values,fit,cal,device)
    evaluate(device);m.write_json(m.RESULTS/'exit_status.json',dict(status='SUCCESS',time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
