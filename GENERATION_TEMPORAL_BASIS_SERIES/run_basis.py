"""One-factor convex temporal velocity basis, original joint objective."""
from pathlib import Path
import importlib.util
import sys
import math
import json
import torch
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'GENERATION_DUAL_ORACLE_SERIES/run_dual.py'
spec=importlib.util.spec_from_file_location('basis_source',SOURCE)
d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)
m=d.m;r=d.r;p=d.p
d.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
m.objective=d.f.original_objective
ORIGINAL=r.m.Correction


def bernstein_basis():
    t=torch.linspace(0,1,12,dtype=torch.float64)
    matrix=torch.stack([math.comb(11,j)*t.pow(j)*(1-t).pow(11-j) for j in range(12)],-1)
    return matrix.float()


class TemporalBasis(ORIGINAL):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        self.register_buffer('basis',bernstein_basis())

    def forward(self,observation,base,speed):
        _,coefficients=super().forward(observation,base,speed)
        velocity=torch.einsum('tc,bkcd->bktd',self.basis,coefficients)
        return base+velocity.cumsum(2),velocity


def factory(dimension,cfg):
    return TemporalBasis(dimension,cfg) if cfg.get('temporal_basis')=='bernstein11' else ORIGINAL(dimension,cfg)


r.m.Correction=factory


def preflight(device):
    cfg=m.settings(1)
    assert cfg['test_access']=='forbidden' and cfg['seeds']==[1,2,3]
    assert cfg['temporal_basis']=='bernstein11' and 'separate_oracle_fraction' not in cfg
    assert not cfg.get('geometry_augment',False) and 'family_winner_weight' not in cfg
    prior=json.loads((d.CONTROL/'preflight.json').read_text())
    for path,digest in {**prior['source_hashes'],**prior['controls']}.items():assert p.sha256(Path(path))==digest,path
    lock=json.loads((d.CONTROL/'results/model_lock.json').read_text())['models']
    controls={str(d.checkpoint(seed,'control',tag)):lock[str(d.checkpoint(seed,'control',tag))] for seed in cfg['seeds'] for tag in d.POLICIES}
    for path,digest in controls.items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    assert not(set(cache['sample_id'][fit])&set(cache['sample_id'][cal]))
    r.setup_weights(cache,fit,values,cfg,device)
    scale=np.array([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32).clip(1e-6)
    assert np.array_equal(scale,np.array(prior['fit_protection_scales'],np.float32))
    for seed in cfg['seeds']:
        saved=torch.load(d.checkpoint(seed,'control','constrained'),map_location='cpu',weights_only=False)
        for key,value in saved['config'].items():
            if key!='protocol':assert m.settings(seed)[key]==value,key
        assert np.array_equal(saved['mean'],values[4]) and np.array_equal(saved['std'],values[5])
    basis=bernstein_basis().to(device)
    assert (basis>=0).all() and torch.allclose(basis.sum(-1),torch.ones(12,device=device),atol=1e-6)
    assert torch.linalg.matrix_rank(basis.double())==12
    linear=torch.linspace(0,1,12,device=device);assert torch.allclose(basis@linear,linear,atol=1e-6)
    alternating=torch.tensor([1.,-1.]*6,device=device);smoothed=basis@alternating
    assert smoothed[1:-1].abs().max()<.2 and torch.allclose(smoothed[[0,-1]],alternating[[0,-1]])
    o,b,s,gt=[torch.from_numpy(v[:128]).to(device) for v in values[:4]];groups=torch.tensor(cache['change_group'][:128],device=device)
    p.seed_everything(1);model=factory(o.shape[1],cfg).to(device);model.eval();out,res=model(o,b,s)
    assert torch.equal(out,b) and sum(x.numel() for x in model.parameters())==36184
    loss=m.objective(out,b,gt,res,s,groups,cfg,d.ARM,torch.tensor(scale,device=device));loss.backward()
    assert torch.isfinite(loss) and model.network[-1].weight.grad.abs().sum()>0
    with torch.no_grad():
        model.network[-1].bias.normal_();out,res=model(o,b,s)
        assert (res.abs()<=1.5*(s[:,0,None,None,None]+.03)+1e-6).all()
        perm=torch.randperm(20,device=device);out_p,res_p=model(o,b[:,perm],s)
        assert torch.allclose(out_p,out[:,perm],atol=2e-6)
    sources={**prior['source_hashes'],str(d.CONTROL/'run_capacity.py'):p.sha256(d.CONTROL/'run_capacity.py'),str(d.SOURCE):p.sha256(d.SOURCE),str(SOURCE):p.sha256(SOURCE),str(Path(__file__)):p.sha256(Path(__file__)),str(HERE/'protocol.json'):p.sha256(HERE/'protocol.json')}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),new_trainings=3,parameters=36184,
        fit_protection_scales=scale.tolist(),source_hashes=sources,controls=controls,test_access=False))
    print('PREFLIGHT PASS: convex fullrank basis/linear preservation/alternation attenuation/nativeidentity/bounds/GPU/permutation/hash/config',flush=True)


ORIGINAL_EVALUATE=d.evaluate


def evaluate(device):
    ORIGINAL_EVALUATE(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['variant']='temporal_basis'
    summary['scope']='Repeated ETH dev; nativeK20; shared12836184params/bound1.5; degree11 convex Bernstein time basis of velocity coefficients; original jointwinner+regularizers+protection; reused capacity controls not independent replication; no dual/family/top3loss/proposals/test/backbone/scorer. Fullrank basis is not equal bounded function class: convex mixing contracts rapid alternating residuals; no physical or GT-performance guarantee. Generic family NPZ/CSV means temporal basis.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_INDEPENDENT_VALIDATION' if all(summary['gates'].values()) else 'TEMPORAL_BASIS_GATE_FAIL'
    for row in summary['aggregate']:
        if row['arm']=='dual_oracle':row['arm']='temporal_basis'
    m.write_json(path,summary)
    lines=['# Convex temporal residual basis','',summary['verdict'],'',summary['scope'],'','| Arm | Policy | Group | minADE | minFDE | Joint |','|---|---|---|---:|---:|---:|']
    for row in summary['aggregate']:
        if row['group'] in ('overall','change'):lines.append(f'| {row["arm"]} | {row["policy"]} | {row["group"]} | {row["minade20"]:.6f} | {row["minfde20"]:.6f} | {row["joint_oracle"]:.6f} |')
    lines+=['',json.dumps(summary['gates'],indent=2)];(m.RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)


d.preflight=preflight;d.evaluate=evaluate
if __name__=='__main__':
    try:d.main()
    except Exception:
        import traceback,time
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
