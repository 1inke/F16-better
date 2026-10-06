"""Matched fresh zero-context versus pooled-context generation correction."""
from pathlib import Path
import importlib.util
import sys
import argparse
import json
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'GENERATION_DUAL_ORACLE_SERIES/run_dual.py'
spec=importlib.util.spec_from_file_location('context_source',SOURCE)
d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)
m=d.m;r=d.r;p=d.p
m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models';m.objective=d.f.original_objective
ARMS=('zero','mean_std');POLICIES=d.POLICIES
BASE_SETTINGS=m.settings


class Correction(nn.Module):
    def __init__(self,dimension,cfg):
        super().__init__();self.strength=cfg['velocity_residual_scale'];self.mode=cfg['set_context']
        self.network=nn.Sequential(nn.Linear(dimension+24+7+48,cfg['hidden']),nn.LayerNorm(cfg['hidden']),nn.GELU(),nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['hidden'],cfg['hidden']//2),nn.GELU(),nn.Linear(cfg['hidden']//2,24))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,observation,base,speed):
        flat=base.flatten(2)
        if self.mode=='mean_std':
            center=flat.mean(1);spread=flat.std(1,unbiased=False)
            context=torch.cat((center,spread),-1)[:,None].expand(-1,base.shape[1],-1)
        elif self.mode=='zero':context=flat.new_zeros((*flat.shape[:2],48))
        else:raise ValueError(self.mode)
        family=F.one_hot(r.m.labels(speed[:,1:],base[:,:,-1]/12),7).to(base.dtype)
        features=torch.cat((observation[:,None].expand(-1,base.shape[1],-1),flat,family,context),-1)
        residual=self.network(features).reshape_as(base).tanh()*self.strength*(speed[:,0,None,None,None]+.03)
        return base+residual.cumsum(2),residual


r.m.Correction=Correction


def settings(seed,arm='mean_std'):
    cfg=BASE_SETTINGS(seed);cfg['set_context']=arm;return cfg


def checkpoint(seed,arm,tag):
    return HERE/'results/models'/arm/f'seed_{seed}'/d.ARM/f'{tag}.pth'


def preflight(device):
    cfg=settings(1);assert cfg['test_access']=='forbidden' and cfg['seeds']==[1,2,3]
    assert 'separate_oracle_fraction' not in cfg and 'family_winner_weight' not in cfg
    prior=json.loads((d.CONTROL/'preflight.json').read_text())
    for path,digest in {**prior['source_hashes'],**prior['controls']}.items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    assert not(set(cache['sample_id'][fit])&set(cache['sample_id'][cal]))
    r.setup_weights(cache,fit,values,cfg,device)
    scale=np.array([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32).clip(1e-6)
    assert np.array_equal(scale,np.array(prior['fit_protection_scales'],np.float32))
    old=torch.load(d.checkpoint(1,'control','constrained'),map_location='cpu',weights_only=False)
    for key,value in old['config'].items():
        if key!='protocol':assert cfg[key]==value,key
    assert np.array_equal(old['mean'],values[4]) and np.array_equal(old['std'],values[5])
    o,b,s,gt=[torch.from_numpy(v[:128]).to(device) for v in values[:4]];groups=torch.tensor(cache['change_group'][:128],device=device)
    states=[];counts=[]
    for arm in ARMS:
        p.seed_everything(1);model=Correction(o.shape[1],settings(1,arm)).to(device);model.eval();out,res=model(o,b,s)
        assert torch.equal(out,b);counts.append(sum(x.numel() for x in model.parameters()));states.append(model.state_dict())
        loss=m.objective(out,b,gt,res,s,groups,cfg,d.ARM,torch.tensor(scale,device=device));loss.backward()
        assert torch.isfinite(loss) and model.network[-1].weight.grad.abs().sum()>0
        # Mean/std context influences an own candidate when another candidate changes; zero control does not.
        with torch.no_grad():
            model.network[-1].weight.normal_(std=.01)
            original,_=model(o,b,s);changed=b.clone();changed[:,19]+=1;other,_=model(o,changed,s)
            difference=(other[:,:19]-original[:,:19]).abs().max().item()
            assert (difference>1e-6) if arm=='mean_std' else (difference==0)
            perm=torch.randperm(20,device=device);out_p,_=model(o,b[:,perm],s)
            assert torch.allclose(out_p,original[:,perm],atol=2e-6)
    assert counts[0]==counts[1]==42328
    assert all(torch.equal(states[0][key],states[1][key]) for key in states[0])
    sources={**prior['source_hashes'],str(d.CONTROL/'run_capacity.py'):p.sha256(d.CONTROL/'run_capacity.py'),str(d.SOURCE):p.sha256(d.SOURCE),str(SOURCE):p.sha256(SOURCE),str(Path(__file__)):p.sha256(Path(__file__)),str(HERE/'protocol.json'):p.sha256(HERE/'protocol.json')}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),new_trainings=6,parameters_each=counts[0],
        fit_protection_scales=scale.tolist(),source_hashes=sources,controls={},test_access=False))
    print('PREFLIGHT PASS: matched initialization/parameters, context-only sensitivity, native identity, GPU gradient/permutation, hashes/split',flush=True)


def train(seed,arm,cache,values,fit,cal,device):
    m.MODELS=HERE/'results/models'/arm;m.settings=lambda seed:settings(seed,arm)
    try:m.train(seed,d.ARM,cache,values,fit,cal,device)
    finally:m.MODELS=HERE/'results/models';m.settings=BASE_SETTINGS


def evaluate(device):
    cfg=settings(1);seeds=cfg['seeds'];paths=[checkpoint(seed,arm,tag) for seed in seeds for arm in ARMS for tag in POLICIES]
    assert len(paths)==18 and all(path.exists() for path in paths)
    m.write_json(m.RESULTS/'model_lock.json',dict(models={str(path):p.sha256(path) for path in paths},dev_used_for_selection=False,test_access=False))
    cache=p.load_npz(p.DEV_CACHE);dim=cache['motion'].shape[1]+cache['social'].shape[1]
    values=r.prepared(cache,False,mean=np.zeros((1,dim),np.float32),std=np.ones((1,dim),np.float32));base,speed,future=values[1:4];group=cache['change_group']
    native=r.generation_metrics(base,base,speed,future);stats={};rows=[]
    for seed in seeds:
        for arm in ARMS:
            for tag in POLICIES:
                saved=torch.load(checkpoint(seed,arm,tag),map_location='cpu',weights_only=False)
                assert saved['config']['set_context']==arm
                current=r.prepared(cache,False,mean=saved['mean'],std=saved['std']);model=Correction(dim,saved['config']).to(device);model.load_state_dict(saved['state_dict'])
                pred=r.g.predict(model,*current[:3],device);assert pred.shape==(len(group),20,12,2) and np.isfinite(pred).all()
                value=r.generation_metrics(pred,base,speed,future);stats[seed,arm,tag]=value
                np.savez_compressed(m.RESULTS/f'seed{seed}_{arm}_{tag}.npz',sample_id=cache['sample_id'],trajectory=pred,**value)
                for name,metrics in m.grouped(value,group).items():rows.append(dict(seed=seed,arm=arm,policy=tag,group=name,epoch=saved['epoch'],trained_feasible=saved['trained_feasible'],**metrics))
    aggregate=[];boot=[]
    for arm in ARMS:
        for tag in POLICIES:
            for name,mask in m.masks(group):aggregate.append(dict(arm=arm,policy=tag,group=name,**{k:float(np.stack([stats[seed,arm,tag][k][mask] for seed in seeds]).mean()) for k in native}))
    for tag in POLICIES:
        for ref in ('native','zero'):
            for name,mask in m.masks(group):
                for metric in ('minade20','minfde20','top3_ade','joint_oracle'):
                    delta=np.stack([stats[seed,'mean_std',tag][metric][mask]-(native[metric][mask] if ref=='native' else stats[seed,'zero',tag][metric][mask]) for seed in seeds])
                    avg,lo,hi=m.h.hierarchical_delta(delta,cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])
                    boot.append(dict(policy=tag,reference=ref,group=name,metric=metric,delta=avg,ci_low=lo,ci_high=hi))
    lookup={(x['arm'],x['policy'],x['group']):x for x in aggregate};reference=m.grouped(native,group);overall=lookup['mean_std','constrained','overall'];change=lookup['mean_std','constrained','change']
    gates=dict(change_ade_fde_gain_5pct=all(change[k]<=reference['change'][k]*.95 for k in ('minade20','minfde20')),
        protected_groups_guard=all(lookup['mean_std','constrained',name][k]<=reference[name][k]*1.01 for name in ('overall','stable','transition') for k in ('minade20','minfde20')),
        each_seed_change_improves=all(stats[seed,'mean_std','constrained'][k][group==2].mean()<native[k][group==2].mean() for seed in seeds for k in ('minade20','minfde20')),
        native_and_zero_ci=all(x['ci_high']<0 for x in boot if x['policy']=='constrained' and x['group']=='change' and x['metric'] in ('minade20','minfde20')),
        all_seed_feasible=all(x['trained_feasible'] for x in rows if x['arm']=='mean_std' and x['policy']=='constrained'),
        coverage_guard=overall['maneuver_class_count']>=reference['overall']['maneuver_class_count']*.95 and overall['gt_family_covered']>=reference['overall']['gt_family_covered']-.01,
        duplicate_guard=overall['duplicate_fraction_10cm']<=reference['overall']['duplicate_fraction_10cm']+.02,
        kinematic_guard=all(overall[k]<=reference['overall'][k]+.01 for k in ('speed_over3mps_fraction','acceleration_over4mps2_fraction')),
        same_path_joint_guard=all(lookup['mean_std','constrained',name]['joint_oracle']<=lookup['zero','constrained',name]['joint_oracle']*1.01 for name in ('overall','stable','transition','change')))
    summary=dict(status='COMPLETE',verdict='MAJOR_DEV_SIGNAL_REQUIRES_INDEPENDENT_VALIDATION' if all(gates.values()) else 'SET_CONTEXT_GATE_FAIL',gates={k:bool(v) for k,v in gates.items()},native=reference,aggregate=aggregate,bootstrap=boot,
        scope='ETH repeatedly inspected dev; fixed nativeK20; six fresh matched paired-seed trainings; mean/std context vs zero inputs, same42328parameter architecture/initialization; no new loss/GT inference/test/backbone/scorer; zero-context48 weights receive no data gradient, count matching not equal effective capacity')
    p.write_csv(m.RESULTS/'per_seed_metrics.csv',rows);p.write_csv(m.RESULTS/'aggregate_metrics.csv',aggregate);p.write_csv(m.RESULTS/'bootstrap.csv',boot);m.write_json(m.RESULTS/'summary.json',summary)
    lines=['# Candidate set mean/std context','',summary['verdict'],'',summary['scope'],'','| Arm | Policy | Group | minADE | minFDE | Joint |','|---|---|---|---:|---:|---:|']
    for row in aggregate:
        if row['group'] in ('overall','change'):lines.append(f'| {row["arm"]} | {row["policy"]} | {row["group"]} | {row["minade20"]:.6f} | {row["minfde20"]:.6f} | {row["joint_oracle"]:.6f} |')
    lines+=['',json.dumps(summary['gates'],indent=2)];(m.RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    audit=json.loads((m.RESULTS/'model_lock.json').read_text())['models']
    for path,digest in audit.items():assert p.sha256(Path(path))==digest,path
    files=list(m.RESULTS.glob('seed*.npz'));assert len(files)==18
    for path in files:
        data=p.load_npz(path);assert np.array_equal(data['sample_id'],cache['sample_id']) and np.isfinite(data['trajectory']).all()
        actual=r.generation_metrics(data['trajectory'],base,speed,future)
        for key,value in actual.items():assert np.allclose(data[key],value,atol=1e-6,rtol=0),(path,key)
    m.write_json(m.RESULTS/'audit.json',dict(status='PASS',models=18,predictions=18,K=20,IDs_exact=True,metrics_recomputed=True))
    print('COMPLETE '+summary['verdict'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');parser.add_argument('--device',default='cuda:0');args=parser.parse_args();device=torch.device(args.device)
    if args.action=='preflight':preflight(device);return
    audit=json.loads((HERE/'preflight.json').read_text())
    for path,digest in audit['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    if (m.RESULTS/'exit_status.json').exists() and json.loads((m.RESULTS/'exit_status.json').read_text())['status']=='SUCCESS':raise RuntimeError('Already complete')
    m.RESULTS.mkdir(exist_ok=True);cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),settings(1)['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    for arm in ARMS:
        for seed in settings(1)['seeds']:
            manifest=checkpoint(seed,arm,'constrained').parent/'complete.json'
            if manifest.exists():
                for path,digest in json.loads(manifest.read_text())['model_hashes'].items():assert p.sha256(Path(path))==digest,path
                print(f'[SKIP VERIFIED] {arm} seed{seed}',flush=True);continue
            print(f'[TRAIN] {arm} seed{seed}',flush=True);train(seed,arm,cache,values,fit,cal,device)
    evaluate(device);m.write_json(m.RESULTS/'exit_status.json',dict(status='SUCCESS',time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
