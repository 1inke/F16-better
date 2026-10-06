"""One-factor correction-bound pilot; verified existing small-bound controls."""
from pathlib import Path
import importlib.util
import sys
import json
import argparse
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
CONTROL=HERE.parent/'GENERATION_PROTECTED_BALANCE_SERIES'
spec=importlib.util.spec_from_file_location('capacity_source',CONTROL/'run_protected.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
r=m.r;p=m.p
POLICIES=('constrained','legacy','final')
ARM='balanced_protected'


def checkpoint(seed,bound,policy):
    root=CONTROL/'results/models' if bound=='small' else m.MODELS
    return root/f'seed_{seed}'/ARM/f'{policy}.pth'


def preflight(device):
    cfg=m.settings(1)
    assert cfg['velocity_residual_scale']==1.5 and cfg['control_velocity_residual_scale']==.5
    assert cfg['test_access']=='forbidden' and cfg['seeds']==[1,2,3]
    prior=json.loads((CONTROL/'preflight.json').read_text())
    for path,digest in {**prior['source_hashes'],**prior['prior_models']}.items():assert p.sha256(Path(path))==digest,path
    locked=json.loads((CONTROL/'results/model_lock.json').read_text())['models']
    controls={str(checkpoint(seed,'small',tag)):locked[str(checkpoint(seed,'small',tag))] for seed in cfg['seeds'] for tag in POLICIES}
    for path,digest in controls.items():assert p.sha256(Path(path))==digest
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert not(set(cache['sample_id'][fit]) & set(cache['sample_id'][cal]))
    values=r.prepared(cache,False,fit=fit)
    r.setup_weights(cache,fit,values,cfg,device)
    scale=np.array([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32).clip(1e-6)
    assert np.array_equal(scale,np.array(prior['fit_protection_scales'],np.float32))
    for seed in cfg['seeds']:
        saved=torch.load(checkpoint(seed,'small','constrained'),map_location='cpu',weights_only=False)
        expected=m.settings(seed)
        for key,value in saved['config'].items():
            if key not in ('protocol','velocity_residual_scale'):assert expected[key]==value,(key,value,expected[key])
        assert saved['config']['velocity_residual_scale']==.5
        assert np.array_equal(saved['mean'],values[4]) and np.array_equal(saved['std'],values[5])
    o,b,s,f=[torch.from_numpy(v[:16]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][:16],device=device)
    p.seed_everything(1);large=r.m.Correction(o.shape[1],cfg).to(device)
    small_cfg=dict(cfg,velocity_residual_scale=.5);small=r.m.Correction(o.shape[1],small_cfg).to(device)
    small.load_state_dict(large.state_dict());large.eval();small.eval()
    output,residual=large(o,b,s);assert torch.equal(output,b)
    loss=m.objective(output,b,f,residual,s,groups,cfg,ARM,torch.tensor(scale,device=device));loss.backward()
    assert torch.isfinite(loss) and large.network[-1].weight.grad.abs().sum()>0
    # Same weights, only the physical correction bound differs; equivariance preserved.
    with torch.no_grad():
        large.network[-1].bias.fill_(.2);small.load_state_dict(large.state_dict())
        out_l,res_l=large(o,b,s);out_s,res_s=small(o,b,s)
        assert torch.allclose(res_l,res_s*3,atol=1e-7)
        assert (res_l.abs()<=1.5*(s[:,0,None,None,None]+.03)+1e-6).all()
        permutation=torch.randperm(20,device=device)
        perm,_=large(o,b[:,permutation],s);assert torch.allclose(perm,out_l[:,permutation],atol=1e-6)
    paths={**prior['source_hashes'],str(Path(__file__)):p.sha256(Path(__file__)),str(HERE/'protocol.json'):p.sha256(HERE/'protocol.json')}
    m.write_json(HERE/'preflight.json',{'status':'PASS','fit_n':len(fit),'cal_n':len(cal),'new_trainings':3,'candidate_count':20,
        'fit_protection_scales':scale.tolist(),'source_hashes':paths,'controls':controls,'control_reused_not_independent':True,'test_access':False})
    print('PREFLIGHT PASS: config/controls/fit scales/GPU identity-gradient/3x bound/permutation',flush=True)


def evaluate(device):
    cfg=m.settings(1);paths=[checkpoint(seed,bound,tag) for seed in cfg['seeds'] for bound in ('small','large') for tag in POLICIES]
    assert all(path.exists() for path in paths)
    m.write_json(m.RESULTS/'model_lock.json',{'models':{str(path):p.sha256(path) for path in paths},'dev_used_for_selection':False,'test_access':False})
    cache=p.load_npz(p.DEV_CACHE);dim=cache['motion'].shape[1]+cache['social'].shape[1]
    values=r.prepared(cache,False,mean=np.zeros((1,dim),np.float32),std=np.ones((1,dim),np.float32))
    base,speed,future=values[1:4];group=cache['change_group'];native=r.generation_metrics(base,base,speed,future)
    stats={};rows=[]
    for seed in cfg['seeds']:
        for bound in ('small','large'):
            for tag in POLICIES:
                saved=torch.load(checkpoint(seed,bound,tag),map_location='cpu',weights_only=False)
                assert saved['config']['velocity_residual_scale']==(.5 if bound=='small' else 1.5)
                current=r.prepared(cache,False,mean=saved['mean'],std=saved['std'])
                model=r.m.Correction(dim,saved['config']).to(device);model.load_state_dict(saved['state_dict'])
                prediction=r.g.predict(model,*current[:3],device)
                assert prediction.shape==(len(group),20,12,2) and np.isfinite(prediction).all()
                value=r.generation_metrics(prediction,base,speed,future);stats[seed,bound,tag]=value
                if bound=='small':
                    previous=p.load_npz(CONTROL/f'results/seed{seed}_{ARM}_{tag}.npz')
                    assert np.array_equal(previous['sample_id'],cache['sample_id'])
                    assert np.allclose(previous['trajectory'],prediction,atol=1e-7,rtol=0)
                np.savez_compressed(m.RESULTS/f'seed{seed}_{bound}_{tag}.npz',sample_id=cache['sample_id'],trajectory=prediction,**value)
                for name,metrics in m.grouped(value,group).items():rows.append(dict(seed=seed,bound=bound,policy=tag,group=name,epoch=saved['epoch'],trained_feasible=saved['trained_feasible'],**metrics))
    aggregate=[];boot=[]
    for bound in ('small','large'):
        for tag in POLICIES:
            for name,mask in m.masks(group):aggregate.append(dict(bound=bound,policy=tag,group=name,**{k:float(np.stack([stats[seed,bound,tag][k][mask] for seed in cfg['seeds']]).mean()) for k in native}))
    for tag in POLICIES:
        for reference in ('native','small'):
            for name,mask in m.masks(group):
                for metric in ('minade20','minfde20','top3_ade'):
                    delta=np.stack([stats[seed,'large',tag][metric][mask]-(native[metric][mask] if reference=='native' else stats[seed,'small',tag][metric][mask]) for seed in cfg['seeds']])
                    mean,lo,hi=m.h.hierarchical_delta(delta,cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])
                    boot.append(dict(policy=tag,reference=reference,group=name,metric=metric,delta=mean,ci_low=lo,ci_high=hi))
    lookup={(x['bound'],x['policy'],x['group']):x for x in aggregate};ref=m.grouped(native,group)
    primary=lookup['large','constrained','change'];overall=lookup['large','constrained','overall']
    gates={
        'change_ade_fde_gain_5pct':all(primary[k]<=ref['change'][k]*.95 for k in ('minade20','minfde20')),
        'protected_groups_guard':all(lookup['large','constrained',name][k]<=ref[name][k]*1.01 for name in ('overall','stable','transition') for k in ('minade20','minfde20')),
        'each_seed_change_improves':all(stats[seed,'large','constrained'][k][group==2].mean()<native[k][group==2].mean() for seed in cfg['seeds'] for k in ('minade20','minfde20')),
        'native_and_small_ci':all(x['ci_high']<0 for x in boot if x['policy']=='constrained' and x['group']=='change' and x['metric'] in ('minade20','minfde20')),
        'all_seed_feasible':all(x['trained_feasible'] for x in rows if x['bound']=='large' and x['policy']=='constrained'),
        'coverage_guard':overall['maneuver_class_count']>=ref['overall']['maneuver_class_count']*.95 and overall['gt_family_covered']>=ref['overall']['gt_family_covered']-.01,
        'duplicate_guard':overall['duplicate_fraction_10cm']<=ref['overall']['duplicate_fraction_10cm']+.02,
        'kinematic_guard':all(overall[k]<=ref['overall'][k]+.01 for k in ('speed_over3mps_fraction','acceleration_over4mps2_fraction'))}
    p.write_csv(m.RESULTS/'per_seed_metrics.csv',rows);p.write_csv(m.RESULTS/'aggregate_metrics.csv',aggregate);p.write_csv(m.RESULTS/'bootstrap.csv',boot)
    summary={'status':'COMPLETE','verdict':'MAJOR_DEV_SIGNAL_REQUIRES_INDEPENDENT_VALIDATION' if all(gates.values()) else 'CAPACITY_GATE_FAIL','gates':gates,'native':ref,'aggregate':aggregate,'bootstrap':boot,
        'scope':'ETH repeatedly used dev; K20; frozen backbone/scorer; reused verified small-bound controls; three paired correction seeds, no independent data or test'}
    m.write_json(m.RESULTS/'summary.json',summary)
    lines=['# Residual capacity experiment','',summary['verdict'],'',summary['scope'],'','| Bound | Policy | Group | minADE | minFDE |','|---|---|---|---:|---:|']
    for row in aggregate:
        if row['group'] in ('overall','change'):lines.append(f'| {row["bound"]} | {row["policy"]} | {row["group"]} | {row["minade20"]:.6f} | {row["minfde20"]:.6f} |')
    lines+=['',json.dumps({k:bool(v) for k,v in gates.items()},indent=2)]
    (m.RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');parser.add_argument('--device',default='cuda:0');args=parser.parse_args();device=torch.device(args.device)
    if args.action=='preflight':preflight(device);return
    audit=json.loads((HERE/'preflight.json').read_text())
    for path,digest in {**audit['source_hashes'],**audit['controls']}.items():assert p.sha256(Path(path))==digest,path
    if (m.RESULTS/'summary.json').exists():raise RuntimeError('Already complete')
    m.MODELS.mkdir(parents=True,exist_ok=True)
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),m.settings(1)['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    for seed in m.settings(1)['seeds']:
        completed=m.MODELS/f'seed_{seed}'/ARM/'complete.json'
        if completed.exists():
            for path,digest in json.loads(completed.read_text())['model_hashes'].items():assert p.sha256(Path(path))==digest
            print(f'[SKIP VERIFIED] seed{seed}',flush=True);continue
        print(f'[TRAIN] enlarged bound seed{seed}',flush=True);m.train(seed,ARM,cache,values,fit,cal,device)
    evaluate(device);m.write_json(m.RESULTS/'exit_status.json',{'status':'SUCCESS','time':time.time()})


if __name__=='__main__':
    try:main()
    except Exception:
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',{'status':'FAILURE','time':time.time()});raise
