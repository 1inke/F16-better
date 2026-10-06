"""Capacity-matched shared vs disjoint candidate-family correction networks."""
from pathlib import Path
import importlib.util
import sys
import json
import argparse
import traceback
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
CONTROL=HERE.parent/'GENERATION_RESIDUAL_CAPACITY_SERIES'
source=HERE.parent/'GENERATION_FAMILY_WINNER_SERIES/run_family.py'
spec=importlib.util.spec_from_file_location('expert_reporting_source',source)
f=importlib.util.module_from_spec(spec);sys.modules[spec.name]=f;spec.loader.exec_module(f)
m=f.m;r=f.r;p=f.p
m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
m.objective=f.original_objective  # Explicitly remove the failed family auxiliary loss.
ORIGINAL=r.m.Correction
ROOT_RESULTS=m.RESULTS;ROOT_MODELS=m.MODELS
VARIANTS=('wide','experts');POLICIES=f.POLICIES;ARM='balanced_protected'
current_variant='experts'
base_settings=m.settings


class Experts(nn.Module):
    def __init__(self,dimension,cfg):
        super().__init__();self.strength=cfg['velocity_residual_scale']
        self.experts=nn.ModuleList([ORIGINAL(dimension,cfg).network for _ in range(cfg['expert_count'])])

    def forward(self,observation,base,speed):
        labels=r.m.labels(speed[:,1:],base[:,:,-1]/12)
        features=torch.cat((observation[:,None].expand(-1,20,-1),base.flatten(2),F.one_hot(labels,7).to(base.dtype)),-1)
        flattened=features.flatten(0,1);family=labels.flatten();raw=torch.zeros((len(flattened),24),device=base.device,dtype=base.dtype)
        for index,network in enumerate(self.experts):
            indices=torch.nonzero(family==index,as_tuple=False).flatten()
            if len(indices):raw=raw.index_copy(0,indices,network(flattened.index_select(0,indices)))
        residual=raw.reshape_as(base).tanh()*self.strength*(speed[:,0,None,None,None]+.03)
        return base+residual.cumsum(2),residual


def factory(dimension,cfg):
    return Experts(dimension,cfg) if cfg.get('architecture_variant')=='experts' else ORIGINAL(dimension,cfg)


def settings(seed):
    cfg=base_settings(seed);cfg['architecture_variant']=current_variant
    if current_variant=='wide':cfg['hidden']=cfg['wide_hidden']
    return cfg


r.m.Correction=factory;m.settings=settings


def checkpoint(seed,variant,tag):
    root=CONTROL/'results/models' if variant=='control' else ROOT_MODELS/variant
    return root/f'seed_{seed}'/ARM/f'{tag}.pth'


def preflight(device):
    global current_variant
    cfg=settings(1);assert cfg['test_access']=='forbidden' and cfg['seeds']==[1,2,3]
    prior=json.loads((CONTROL/'preflight.json').read_text())
    for path,digest in {**prior['source_hashes'],**prior['controls']}.items():assert p.sha256(Path(path))==digest,path
    locked=json.loads((CONTROL/'results/model_lock.json').read_text())['models']
    controls={str(checkpoint(seed,'control',tag)):locked[str(checkpoint(seed,'control',tag))] for seed in cfg['seeds'] for tag in POLICIES}
    for path,digest in controls.items():assert p.sha256(Path(path))==digest
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    assert not(set(cache['sample_id'][fit]) & set(cache['sample_id'][cal]))
    scale=np.array([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32).clip(1e-6)
    assert np.array_equal(scale,np.array(prior['fit_protection_scales'],np.float32))
    for seed in cfg['seeds']:
        saved=torch.load(checkpoint(seed,'control','constrained'),map_location='cpu',weights_only=False)
        current_variant='experts'
        for key,value in saved['config'].items():
            if key!='protocol':assert settings(seed)[key]==value,key
        assert np.array_equal(saved['mean'],values[4]) and np.array_equal(saved['std'],values[5])
    r.setup_weights(cache,fit,values,cfg,device)
    # Real full-batch memory/backward check; zero initialization and equivariance.
    o,b,s,gt=[torch.from_numpy(v[:128]).to(device) for v in values[:4]];groups=torch.tensor(cache['change_group'][:128],device=device)
    counts={}
    for variant in VARIANTS:
        current_variant=variant;local=settings(1);p.seed_everything(1);model=factory(o.shape[1],local).to(device);model.eval()
        counts[variant]=sum(x.numel() for x in model.parameters());output,res=model(o,b,s);assert torch.equal(output,b)
        loss=m.objective(output,b,gt,res,s,groups,local,ARM,torch.tensor(scale,device=device));loss.backward()
        assert torch.isfinite(loss) and sum(x.grad.abs().sum().item() for x in model.parameters() if x.grad is not None)>0
        permutation=torch.randperm(20,device=device)
        with torch.no_grad():
            networks=model.experts if variant=='experts' else [model.network]
            for net in networks:net[-1].bias.fill_(.1)
            output,_=model(o,b,s);permuted,_=model(o,b[:,permutation],s);assert torch.allclose(permuted,output[:,permutation],atol=1e-6)
        del model
    assert counts=={'wide':253293,'experts':253288},counts
    # Routing parameter isolation: a synthetic one-family batch only updates that expert.
    current_variant='experts';model=factory(o.shape[1],settings(1)).to(device)
    test_speed=torch.tensor([[.1,.1,0]],device=device);test_base=torch.arange(1,13,device=device)[None,None,:,None]*torch.tensor([[[[.1,0.]]]],device=device)
    test_base=test_base.expand(1,20,12,2);label=int(r.m.labels(test_speed[:,1:],test_base[:,:,-1]/12)[0,0]);assert (r.m.labels(test_speed[:,1:],test_base[:,:,-1]/12)==label).all()
    prediction,_=model(o[:1],test_base,test_speed);prediction.sum().backward()
    assert model.experts[label][-1].bias.grad.abs().sum()>0
    assert all(all(x.grad is None or x.grad.abs().sum()==0 for x in net.parameters()) for i,net in enumerate(model.experts) if i!=label)
    sources={**prior['source_hashes'],str(CONTROL/'run_capacity.py'):p.sha256(CONTROL/'run_capacity.py'),str(source):p.sha256(source),str(Path(__file__)):p.sha256(Path(__file__)),str(HERE/'protocol.json'):p.sha256(HERE/'protocol.json')}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),candidate_count=20,new_trainings=6,parameter_counts=counts,fit_protection_scales=scale.tolist(),source_hashes=sources,controls=controls,test_access=False))
    print('PREFLIGHT PASS: full-batch GPU/backward/identity/permutation/routing-isolation/config/hash/parameter matching',flush=True)


def evaluate(device):
    global current_variant
    cfg=settings(1);all_paths=[checkpoint(seed,variant,tag) for seed in cfg['seeds'] for variant in ('control',)+VARIANTS for tag in POLICIES]
    assert all(path.exists() for path in all_paths)
    m.write_json(ROOT_RESULTS/'model_lock.json',dict(models={str(path):p.sha256(path) for path in all_paths},dev_used_for_selection=False,test_access=False))
    # Reuse the verified pairwise metric evaluator, explicitly translating its generic
    # historical "family" treatment label into the architecture being tested.
    summaries={}
    for variant in VARIANTS:
        current_variant=variant;m.RESULTS=ROOT_RESULTS/variant;m.RESULTS.mkdir(exist_ok=True)
        f.checkpoint=lambda seed,arm,tag:checkpoint(seed,'control' if arm=='control' else variant,tag)
        f.evaluate(device)
        path=m.RESULTS/'summary.json';summary=json.loads(path.read_text());summary['architecture_variant']=variant
        summary['scope']='ETH reused dev; K20; same balanced/protected global winner loss, NO family auxiliary; original shared128 controls reused; shared526 vs seven128 experts capacity matched; no test/backbone/scorer training'
        summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_INDEPENDENT_VALIDATION' if all(summary['gates'].values()) else 'ARCHITECTURE_GATE_FAIL'
        for row in summary['aggregate']:
            if row['arm']=='family':row['arm']=variant
        m.write_json(path,summary);summaries[variant]=summary
        lines=['# '+variant+' architecture experiment','',summary['verdict'],'',summary['scope'],'','| Arm | Policy | Group | minADE | minFDE |','|---|---|---|---:|---:|']
        for row in summary['aggregate']:
            if row['group'] in ('overall','change'):lines.append(f'| {row["arm"]} | {row["policy"]} | {row["group"]} | {row["minade20"]:.6f} | {row["minfde20"]:.6f} |')
        (m.RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    # Capacity-matched contrast, same seeds/cases. No dev-based choice of epochs.
    cache=p.load_npz(p.DEV_CACHE);boot=[]
    for tag in POLICIES:
        for name,mask in m.masks(cache['change_group']):
            for metric in ('minade20','minfde20','top3_ade'):
                delta=np.stack([p.load_npz(ROOT_RESULTS/'experts'/f'seed{seed}_family_{tag}.npz')[metric][mask]-p.load_npz(ROOT_RESULTS/'wide'/f'seed{seed}_family_{tag}.npz')[metric][mask] for seed in cfg['seeds']])
                mean,lo,hi=m.h.hierarchical_delta(delta,cfg['bootstrap_repetitions'],cfg['bootstrap_seed']);boot.append(dict(policy=tag,group=name,metric=metric,delta=mean,ci_low=lo,ci_high=hi))
    p.write_csv(ROOT_RESULTS/'experts_vs_wide_bootstrap.csv',boot)
    isolation=all(x['ci_high']<0 for x in boot if x['policy']=='constrained' and x['group']=='change' and x['metric'] in ('minade20','minfde20'))
    m.write_json(ROOT_RESULTS/'summary.json',dict(status='COMPLETE',primary_variant='experts',variants=summaries,experts_vs_wide_bootstrap=boot,isolation_increment_supported=isolation,test_access=False))
    (ROOT_RESULTS/'REPORT.md').write_text('# Expert isolation experiment\n\n'+ '\n\n'.join(f'{variant}: {summaries[variant]["verdict"]}' for variant in VARIANTS)+f'\n\nMatched change ADE/FDE isolation CI support: {isolation}\n\nETH repeatedly used dev; exploratory evidence only.',encoding='utf-8')
    m.RESULTS=ROOT_RESULTS;print('COMPLETE EXPERT_ISOLATION_REVIEW',flush=True)


def main():
    global current_variant
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');parser.add_argument('--device',default='cuda:0');args=parser.parse_args();device=torch.device(args.device)
    if args.action=='preflight':preflight(device);return
    audit=json.loads((HERE/'preflight.json').read_text())
    for path,digest in {**audit['source_hashes'],**audit['controls']}.items():assert p.sha256(Path(path))==digest,path
    if (ROOT_RESULTS/'summary.json').exists():raise RuntimeError('Already complete')
    ROOT_MODELS.mkdir(parents=True,exist_ok=True);cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),settings(1)['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    for seed in settings(1)['seeds']:
        for variant in VARIANTS:
            current_variant=variant;m.MODELS=ROOT_MODELS/variant;manifest=checkpoint(seed,variant,'constrained').parent/'complete.json'
            if manifest.exists():
                for path,digest in json.loads(manifest.read_text())['model_hashes'].items():assert p.sha256(Path(path))==digest
                print(f'[SKIP VERIFIED] {variant} seed{seed}',flush=True);continue
            print(f'[TRAIN] {variant} seed{seed}',flush=True);m.train(seed,ARM,cache,values,fit,cal,device)
    evaluate(device);m.write_json(ROOT_RESULTS/'exit_status.json',dict(status='SUCCESS',time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        ROOT_RESULTS.mkdir(parents=True,exist_ok=True);(ROOT_RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(ROOT_RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
