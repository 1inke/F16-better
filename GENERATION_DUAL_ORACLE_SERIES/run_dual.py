"""One-factor blend of joint and independently assigned ADE/FDE supervision."""
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
SOURCE=HERE.parent/'GENERATION_FAMILY_WINNER_SERIES/run_family.py'
spec=importlib.util.spec_from_file_location('dual_reporting_source',SOURCE)
f=importlib.util.module_from_spec(spec);sys.modules[spec.name]=f;spec.loader.exec_module(f)
m=f.m;r=f.r;p=f.p
m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
CONTROL=HERE.parent/'GENERATION_RESIDUAL_CAPACITY_SERIES'
POLICIES=f.POLICIES;ARM='balanced_protected'
ORIGINAL_METRICS=r.generation_metrics


def oracle_terms(ade,fde,weight):
    joint=(ade+weight*fde).min(-1).values
    separate=ade.min(-1).values+weight*fde.min(-1).values
    return joint,separate


def objective(output,base,future,residual,speed,groups,cfg,arm,scale):
    original=f.original_objective(output,base,future,residual,speed,groups,cfg,arm,scale)
    error=(output-future[:,None]).norm(dim=-1)
    joint,separate=oracle_terms(error.mean(-1),error[:,:,-1],cfg['fde_loss_weight'])
    target=r.m.labels(speed[:,1:],future[:,None,-1]/12).squeeze(1)
    # Regularizers/protection unchanged. Half joint remains to preserve full-path pressure.
    return original+cfg['separate_oracle_fraction']*((separate-joint)*r.WEIGHTS[target]).mean()


def generation_metrics(paths,base,speed,future):
    stats=ORIGINAL_METRICS(paths,base,speed,future)
    error=np.linalg.norm(paths-future[:,None],axis=-1);ade=error.mean(-1);fde=error[:,:,-1]
    rows=np.arange(len(paths));ade_slot=ade.argmin(-1);fde_slot=fde.argmin(-1)
    weight=m.settings(1)['fde_loss_weight']
    stats.update(joint_oracle=(ade+weight*fde).min(-1),
                 fde_at_ade_oracle=fde[rows,ade_slot],ade_at_fde_oracle=ade[rows,fde_slot],
                 oracle_slot_disagreement=(ade_slot!=fde_slot).astype(np.float32))
    return stats


m.objective=objective;r.generation_metrics=generation_metrics


def checkpoint(seed,arm,tag):
    root=CONTROL/'results/models' if arm=='control' else m.MODELS
    return root/f'seed_{seed}'/ARM/f'{tag}.pth'


def preflight(device):
    cfg=m.settings(1)
    assert cfg['test_access']=='forbidden' and cfg['seeds']==[1,2,3]
    assert cfg['separate_oracle_fraction']==.5 and cfg['velocity_residual_scale']==1.5
    assert not cfg.get('geometry_augment',False) and 'family_winner_weight' not in cfg
    prior=json.loads((CONTROL/'preflight.json').read_text())
    for path,digest in {**prior['source_hashes'],**prior['controls']}.items():assert p.sha256(Path(path))==digest,path
    lock=json.loads((CONTROL/'results/model_lock.json').read_text())['models']
    controls={str(checkpoint(seed,'control',tag)):lock[str(checkpoint(seed,'control',tag))] for seed in cfg['seeds'] for tag in POLICIES}
    for path,digest in controls.items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    values=r.prepared(cache,False,fit=fit)
    assert not(set(cache['sample_id'][fit])&set(cache['sample_id'][cal]))
    r.setup_weights(cache,fit,values,cfg,device)
    scale=np.array([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32).clip(1e-6)
    assert np.array_equal(scale,np.array(prior['fit_protection_scales'],np.float32))
    for seed in cfg['seeds']:
        saved=torch.load(checkpoint(seed,'control','constrained'),map_location='cpu',weights_only=False)
        for key,value in saved['config'].items():
            if key!='protocol':assert m.settings(seed)[key]==value,key
        assert np.array_equal(saved['mean'],values[4]) and np.array_equal(saved['std'],values[5])
    # Separate winners send gradients to distinct paths, not top3/GT-family attraction.
    ade=torch.tensor([[1.,3.]],device=device,requires_grad=True)
    fde=torch.tensor([[4.,1.]],device=device,requires_grad=True)
    joint,separate=oracle_terms(ade,fde,.5)
    assert joint.item()==3 and separate.item()==1.5
    (.5*joint+.5*separate).sum().backward()
    assert torch.equal(ade.grad,torch.tensor([[1.,0.]],device=device))
    assert torch.equal(fde.grad,torch.tensor([[.25,.25]],device=device))
    o,b,s,gt=[torch.from_numpy(v[:128]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][:128],device=device)
    p.seed_everything(1);model=r.m.Correction(o.shape[1],cfg).to(device);model.eval();out,res=model(o,b,s)
    assert torch.equal(out,b) and sum(x.numel() for x in model.parameters())==36184
    loss=objective(out,b,gt,res,s,groups,cfg,ARM,torch.tensor(scale,device=device));loss.backward()
    assert torch.isfinite(loss) and model.network[-1].weight.grad.abs().sum()>0
    zero_cfg=dict(cfg,separate_oracle_fraction=0)
    with torch.no_grad():
        original=f.original_objective(out,b,gt,res,s,groups,cfg,ARM,torch.tensor(scale,device=device))
        assert torch.equal(objective(out,b,gt,res,s,groups,zero_cfg,ARM,torch.tensor(scale,device=device)),original)
        model.network[-1].bias.fill_(.1);out,res=model(o,b,s)
        perm=torch.randperm(20,device=device);out_p,res_p=model(o,b[:,perm],s)
        assert torch.allclose(out_p,out[:,perm],atol=1e-6)
        assert torch.allclose(objective(out,b,gt,res,s,groups,cfg,ARM,torch.tensor(scale,device=device)),
                              objective(out_p,b[:,perm],gt,res_p,s,groups,cfg,ARM,torch.tensor(scale,device=device)),atol=1e-6)
    diagnostic={}
    for split,ids in [('fit',fit),('calibration',cal)]:
        ade,fde=cache['ade'][ids],cache['fde'][ids]
        joint=(ade+cfg['fde_loss_weight']*fde).argmin(-1)
        diagnostic[split]={name:dict(n=int(mask.sum()),
            ade_fde_slot_disagreement=float((ade.argmin(-1)[mask]!=fde.argmin(-1)[mask]).mean()),
            fde_winner_not_joint=float((fde.argmin(-1)[mask]!=joint[mask]).mean()),
            joint_assignment_excess_fde=float((fde[np.arange(len(ids)),joint]-fde.min(-1))[mask].mean()))
            for name,mask in m.masks(cache['change_group'][ids])}
    m.write_json(HERE/'internal_assignment_diagnostic.json',diagnostic)
    sources={**prior['source_hashes'],str(CONTROL/'run_capacity.py'):p.sha256(CONTROL/'run_capacity.py'),str(SOURCE):p.sha256(SOURCE),
             str(Path(__file__)):p.sha256(Path(__file__)),str(HERE/'protocol.json'):p.sha256(HERE/'protocol.json')}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),new_trainings=3,
        fit_protection_scales=scale.tolist(),source_hashes=sources,controls=controls,test_access=False))
    print('PREFLIGHT PASS: separate-winner gradients, alpha0 original, identity, hash/config/GPU/permutation; no replacement/family loss',flush=True)


def audit_predictions():
    lock=json.loads((m.RESULTS/'model_lock.json').read_text())['models']
    assert len(lock)==18
    for path,digest in lock.items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.DEV_CACHE);values=r.prepared(cache,False,fit=None,mean=np.zeros((1,cache['motion'].shape[1]+cache['social'].shape[1]),np.float32),std=np.ones((1,cache['motion'].shape[1]+cache['social'].shape[1]),np.float32))
    files=list(m.RESULTS.glob('seed*.npz'));assert len(files)==18
    for path in files:
        data=p.load_npz(path);assert np.array_equal(data['sample_id'],cache['sample_id'])
        assert data['trajectory'].shape==(len(cache['sample_id']),20,12,2) and np.isfinite(data['trajectory']).all()
        actual=generation_metrics(data['trajectory'],values[1],values[2],values[3])
        for key,value in actual.items():assert np.allclose(data[key],value,atol=1e-6,rtol=0),(path,key)
    m.write_json(m.RESULTS/'audit.json',dict(status='PASS',model_hashes=18,predictions=18,explicit_ids=True,K=20,finite=True,recomputed_metrics=True))


def evaluate(device):
    f.checkpoint=checkpoint;f.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    lookup={(x['arm'],x['policy'],x['group']):x for x in summary['aggregate']}
    summary['gates']['same_path_joint_quality_guard']=all(lookup['family','constrained',group]['joint_oracle']<=lookup['control','constrained',group]['joint_oracle']*1.01 for group in ('overall','stable','transition','change'))
    summary['variant']='dual_oracle';summary['scope']='Repeated ETH dev; K20 native paths; shared128/bound1.5; loss blend .5 joint+.5 independent ADE/FDE minima; original regularizers/protection; frozen backbone/scorer; reused control not independent replication; GT loss-only; no test. Generic NPZ/CSV family label denotes dual_oracle, not family loss.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_INDEPENDENT_VALIDATION' if all(summary['gates'].values()) else 'DUAL_ORACLE_GATE_FAIL'
    for row in summary['aggregate']:
        if row['arm']=='family':row['arm']='dual_oracle'
    m.write_json(path,summary)
    lines=['# Joint/separate oracle supervision','',summary['verdict'],'',summary['scope'],'','| Arm | Policy | Group | minADE | minFDE | same-path Joint |','|---|---|---|---:|---:|---:|']
    for row in summary['aggregate']:
        if row['group'] in ('overall','change'):lines.append(f'| {row["arm"]} | {row["policy"]} | {row["group"]} | {row["minade20"]:.6f} | {row["minfde20"]:.6f} | {row["joint_oracle"]:.6f} |')
    lines+=['',json.dumps(summary['gates'],indent=2)]
    (m.RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    audit_predictions();print('COMPLETE '+summary['verdict'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');parser.add_argument('--device',default='cuda:0');args=parser.parse_args();device=torch.device(args.device)
    if args.action=='preflight':preflight(device);return
    audit=json.loads((HERE/'preflight.json').read_text())
    for path,digest in {**audit['source_hashes'],**audit['controls']}.items():assert p.sha256(Path(path))==digest,path
    if (m.RESULTS/'exit_status.json').exists() and json.loads((m.RESULTS/'exit_status.json').read_text())['status']=='SUCCESS':raise RuntimeError('Already complete')
    m.MODELS.mkdir(parents=True,exist_ok=True);cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),m.settings(1)['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    for seed in m.settings(1)['seeds']:
        manifest=checkpoint(seed,'family','constrained').parent/'complete.json'
        if manifest.exists():
            for path,digest in json.loads(manifest.read_text())['model_hashes'].items():assert p.sha256(Path(path))==digest,path
            print(f'[SKIP VERIFIED] seed{seed}',flush=True);continue
        print(f'[TRAIN] dual oracle seed{seed}',flush=True);m.train(seed,ARM,cache,values,fit,cal,device)
    evaluate(device);m.write_json(m.RESULTS/'exit_status.json',dict(status='SUCCESS',time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
