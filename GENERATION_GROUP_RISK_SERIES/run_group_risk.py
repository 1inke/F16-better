"""Smooth worst-group relative oracle risk; frozen baseline fit scales only."""
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
source=HERE.parent/'GENERATION_FAMILY_WINNER_SERIES/run_family.py'
spec=importlib.util.spec_from_file_location('group_reporting_source',source)
f=importlib.util.module_from_spec(spec);sys.modules[spec.name]=f;spec.loader.exec_module(f)
m=f.m;r=f.r;p=f.p
m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
CONTROL=HERE.parent/'GENERATION_RESIDUAL_CAPACITY_SERIES';POLICIES=f.POLICIES;ARM='balanced_protected'
GROUP_SCALES=None;MAIN_SCALE=None


def smooth_group_risk(per_sample,groups,scales,temperature):
    means=[per_sample[groups==index].mean()/scales[index] for index in range(3) if (groups==index).any()]
    risks=torch.stack(means)
    return temperature*(torch.logsumexp(risks/temperature,dim=0)-np.log(len(means)))


def objective(output,base,future,residual,speed,groups,cfg,arm,scale):
    original=f.original_objective(output,base,future,residual,speed,groups,cfg,arm,scale)
    error=(output-future[:,None]).norm(dim=-1)
    quality=error.mean(-1)+cfg['fde_loss_weight']*error[:,:,-1]
    target=r.m.labels(speed[:,1:],future[:,None,-1]/12).squeeze(1)
    per_sample=quality.min(-1).values*r.WEIGHTS[target]
    robust=MAIN_SCALE*smooth_group_risk(per_sample,groups,GROUP_SCALES,cfg['group_risk_temperature'])
    return original-per_sample.mean()+robust


m.objective=objective


def checkpoint(seed,arm,tag):
    root=CONTROL/'results/models' if arm=='control' else m.MODELS
    return root/f'seed_{seed}'/ARM/f'{tag}.pth'


def scales_from_fit(cache,fit,values,cfg,device):
    global GROUP_SCALES,MAIN_SCALE
    _,weights=r.setup_weights(cache,fit,values,cfg,device)
    target=p.classify_velocity(values[2][:,1:],values[3][:,-1]/12,r.m.LABEL_CFG)
    quality=(cache['ade'][fit]+cfg['fde_loss_weight']*cache['fde'][fit]).min(-1)*weights[target[fit]]
    group=cache['change_group'][fit]
    scale=np.array([quality[group==index].mean() for index in range(3)],np.float32)
    assert np.isfinite(scale).all() and (scale>0).all()
    GROUP_SCALES=torch.tensor(scale,device=device);MAIN_SCALE=float(quality.mean())
    return scale,MAIN_SCALE


def preflight(device):
    cfg=m.settings(1);assert cfg['test_access']=='forbidden' and cfg['seeds']==[1,2,3]
    prior=json.loads((CONTROL/'preflight.json').read_text())
    for path,digest in {**prior['source_hashes'],**prior['controls']}.items():assert p.sha256(Path(path))==digest,path
    lock=json.loads((CONTROL/'results/model_lock.json').read_text())['models']
    controls={str(checkpoint(seed,'control',tag)):lock[str(checkpoint(seed,'control',tag))] for seed in cfg['seeds'] for tag in POLICIES}
    for path,digest in controls.items():assert p.sha256(Path(path))==digest
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    assert not(set(cache['sample_id'][fit]) & set(cache['sample_id'][cal]))
    group_scales,main_scale=scales_from_fit(cache,fit,values,cfg,device)
    protection_scale=np.array([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32).clip(1e-6)
    assert np.array_equal(protection_scale,np.array(prior['fit_protection_scales'],np.float32))
    for seed in cfg['seeds']:
        saved=torch.load(checkpoint(seed,'control','constrained'),map_location='cpu',weights_only=False)
        for key,value in saved['config'].items():
            if key!='protocol':assert m.settings(seed)[key]==value,key
        assert np.array_equal(saved['mean'],values[4]) and np.array_equal(saved['std'],values[5])
    toy=torch.tensor([1.,1.,2.],device=device,requires_grad=True);groups=torch.arange(3,device=device)
    toy_loss=smooth_group_risk(toy,groups,torch.ones(3,device=device),.1);toy_loss.backward();assert toy.grad[2]>.99 and (toy.grad>=0).all()
    assert torch.allclose(smooth_group_risk(torch.ones(3,device=device),groups,torch.ones(3,device=device),.1),torch.tensor(1.,device=device))
    single=smooth_group_risk(torch.tensor([2.,4.],device=device),torch.zeros(2,device=device,dtype=torch.long),torch.ones(3,device=device),.1);assert torch.allclose(single,torch.tensor(3.,device=device))
    o,b,s,gt=[torch.from_numpy(v[:128]).to(device) for v in values[:4]];g=torch.tensor(cache['change_group'][:128],device=device)
    p.seed_everything(1);model=r.m.Correction(o.shape[1],cfg).to(device);model.eval();out,res=model(o,b,s);assert torch.equal(out,b)
    loss=objective(out,b,gt,res,s,g,cfg,ARM,torch.tensor(protection_scale,device=device));loss.backward()
    assert torch.isfinite(loss) and model.network[-1].weight.grad.abs().sum()>0
    perm=torch.randperm(20,device=device)
    with torch.no_grad():
        out_p,res_p=model(o,b[:,perm],s);lp=objective(out_p,b[:,perm],gt,res_p,s,g,cfg,ARM,torch.tensor(protection_scale,device=device));assert torch.allclose(lp,loss,atol=1e-6)
    sources={**prior['source_hashes'],str(CONTROL/'run_capacity.py'):p.sha256(CONTROL/'run_capacity.py'),str(source):p.sha256(source),str(Path(__file__)):p.sha256(Path(__file__)),str(HERE/'protocol.json'):p.sha256(HERE/'protocol.json')}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),new_trainings=3,fit_protection_scales=protection_scale.tolist(),group_native_risk_scales=group_scales.tolist(),main_risk_scale=main_scale,source_hashes=sources,controls=controls,test_access=False))
    print('PREFLIGHT PASS: fit-only relative-risk scales, worst-group gradient, missing-group handling, GPU/hash/config/permutation',flush=True)


def evaluate(device):
    # Verified identical metric machinery; historical treatment label means group-risk.
    f.checkpoint=checkpoint;f.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text());summary['variant']='group_risk'
    summary['scope']='ETH reused dev; K20; shared128; fit-native-normalized smooth worst-group oracle loss, NO family auxiliary; GT groups loss-only; matched large-bound1.5 shared128 control; no test/backbone/scorer training'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_INDEPENDENT_VALIDATION' if all(summary['gates'].values()) else 'GROUP_RISK_GATE_FAIL'
    for row in summary['aggregate']:
        if row['arm']=='family':row['arm']='group_risk'
    m.write_json(path,summary)
    lines=['# Smooth worst-group relative risk','',summary['verdict'],'',summary['scope'],'','| Arm | Policy | Group | minADE | minFDE |','|---|---|---|---:|---:|']
    for row in summary['aggregate']:
        if row['group'] in ('overall','change'):lines.append(f'| {row["arm"]} | {row["policy"]} | {row["group"]} | {row["minade20"]:.6f} | {row["minfde20"]:.6f} |')
    (m.RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('COMPLETE '+summary['verdict'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');parser.add_argument('--device',default='cuda:0');args=parser.parse_args();device=torch.device(args.device)
    if args.action=='preflight':preflight(device);return
    audit=json.loads((HERE/'preflight.json').read_text())
    for path,digest in {**audit['source_hashes'],**audit['controls']}.items():assert p.sha256(Path(path))==digest,path
    if (m.RESULTS/'summary.json').exists():raise RuntimeError('Already complete')
    m.MODELS.mkdir(parents=True,exist_ok=True);cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),m.settings(1)['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    scale,main_scale=scales_from_fit(cache,fit,values,m.settings(1),device)
    assert np.array_equal(scale,np.array(audit['group_native_risk_scales'],np.float32)) and main_scale==audit['main_risk_scale']
    for seed in m.settings(1)['seeds']:
        manifest=m.MODELS/f'seed_{seed}'/ARM/'complete.json'
        if manifest.exists():
            for path,digest in json.loads(manifest.read_text())['model_hashes'].items():assert p.sha256(Path(path))==digest
            print(f'[SKIP VERIFIED] seed{seed}',flush=True);continue
        print(f'[TRAIN] normalized group risk seed{seed}',flush=True);m.train(seed,ARM,cache,values,fit,cal,device)
    evaluate(device);m.write_json(m.RESULTS/'exit_status.json',dict(status='SUCCESS',time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
