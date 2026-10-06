"""K20 generation-only matched pilot; future is loss-only, never model input."""
from pathlib import Path
import argparse
import importlib.util
import sys
import json
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'GENERATION_LOCAL_SUPPORT_SERIES/run_generation_support.py'
spec = importlib.util.spec_from_file_location('mode_support_source', SOURCE)
g = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = g
spec.loader.exec_module(g)
p = g.p
RESULTS = HERE / 'results'
MODELS = RESULTS / 'models'
ARMS = ('mode_oracle', 'mode_global', 'mode_family')
LABEL_CFG = p.load_config()


def labels(velocity, endpoint):
    """Same seven-state heuristic as the earlier pilot, implemented on GPU."""
    observed = velocity[:, None].expand_as(endpoint)
    a, b = observed.norm(dim=-1), endpoint.norm(dim=-1)
    ratio = (b + .01) / (a + .01)
    product = a * b
    moving = product > 1e-6
    cosine = torch.where(moving, (observed * endpoint).sum(-1) / product.clamp_min(1e-6), 1.).clamp(-1, 1)
    sine = torch.where(moving, (observed[..., 0]*endpoint[..., 1]-observed[..., 1]*endpoint[..., 0]) / product.clamp_min(1e-6), 0.)
    angle = torch.atan2(sine, cosine) * (180 / np.pi)
    out = torch.zeros_like(a, dtype=torch.long)
    for condition, value in ((ratio <= LABEL_CFG['decelerate_speed_ratio'], 1),
                             (ratio >= LABEL_CFG['accelerate_speed_ratio'], 3),
                             (angle >= LABEL_CFG['turn_angle_degrees'], 4),
                             (angle <= -LABEL_CFG['turn_angle_degrees'], 5),
                             (cosine <= LABEL_CFG['reverse_cosine_threshold'], 6),
                             ((b <= LABEL_CFG['stop_absolute_speed']) | (ratio <= LABEL_CFG['stop_speed_ratio']), 2)):
        out = torch.where(condition, value, out)
    return out


class Correction(nn.Module):
    def __init__(self, dimension, cfg):
        super().__init__()
        self.strength = cfg['velocity_residual_scale']
        self.network = nn.Sequential(nn.Linear(dimension+24+7, cfg['hidden']),
            nn.LayerNorm(cfg['hidden']), nn.GELU(), nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['hidden'], cfg['hidden']//2), nn.GELU(), nn.Linear(cfg['hidden']//2,24))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observation, base, speed):
        family = F.one_hot(labels(speed[:,1:], base[:,:,-1]/12),7).to(base.dtype)
        repeated = observation[:,None].expand(-1,base.shape[1],-1)
        raw = self.network(torch.cat((repeated,base.flatten(2),family),-1))
        residual = raw.reshape_as(base).tanh()*self.strength*(speed[:,0,None,None,None]+.03)
        return base+residual.cumsum(2), residual


def data(cache, **kwargs):
    values = list(g.data(cache, **kwargs))
    past = cache['motion'][:,:16].reshape(-1,8,2)
    velocity = past[:,-1]-past[:,-2]
    values[2] = np.concatenate((values[2][:,None],velocity),1).astype(np.float32)
    return values


def family_support(quality, matching, k):
    """Average up to k existing GT-family slots; absent family has no support term."""
    ordered = quality.masked_fill(~matching, float('inf')).topk(k,largest=False).values
    valid = torch.isfinite(ordered)
    count = valid.sum(-1)
    per_sample = ordered.masked_fill(~valid,0).sum(-1)/count.clamp_min(1)
    return (per_sample * (count>0)).sum()/(count>0).sum().clamp_min(1)


def loss(output, base, future, residual, speed, cfg, arm):
    errors = (output-future[:,None]).norm(dim=-1)
    quality = errors.mean(-1)+cfg['fde_loss_weight']*errors[:,:,-1]
    family = labels(speed[:,1:],base[:,:,-1]/12)
    target = labels(speed[:,1:],future[:,None,-1]/12)
    support = quality.topk(cfg['support_k'],largest=False).values.mean() if arm=='mode_global' else family_support(quality,family==target,cfg['support_k'])
    with torch.no_grad():
        reference = (g.pair_distance(base)*cfg['diversity_reference_fraction']).clamp(max=cfg['diversity_distance_cap_m'])
        cross = family[:,:,None] != family[:,None,:]
    penalty = F.relu(reference-g.pair_distance(output))
    diversity = (penalty*cross).sum()/cross.sum().clamp_min(1)
    return quality.min(-1).values.mean() + (cfg['support_weight']*support if arm!='mode_oracle' else 0) + cfg['diversity_weight']*diversity + cfg['residual_weight']*residual.square().mean()+cfg['smoothness_weight']*torch.diff(residual,dim=2).square().mean()


def train(arm,cfg,cache,values,fit,cal,device):
    if any((MODELS/f'{arm}_{tag}.pth').exists() for tag in ('best','final')):
        raise RuntimeError('Existing checkpoints: do not overwrite or silently reuse')
    obs,base,speed,future,mean,std=values
    p.seed_everything(cfg['seed'])
    model=Correction(obs.shape[1],cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    loader=DataLoader(TensorDataset(*[torch.from_numpy(v[fit]) for v in (obs,base,speed,future)]),
        batch_size=cfg['batch_size'],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(cfg['seed']))
    best=float('inf');history=[];best_epoch=0;best_state=None
    for epoch in range(1,cfg['epochs']+1):
        model.train();totals=[]
        for o,b,s,f in loader:
            o,b,s,f=[v.to(device) for v in (o,b,s,f)]
            output,residual=model(o,b,s)
            objective=loss(output,b,f,residual,s,cfg,arm)
            if not torch.isfinite(objective): raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();totals.append(objective.item())
        prediction=g.predict(model,obs[cal],base[cal],speed[cal],device)
        stats=g.metrics(prediction,future[cal],cache['logits'][cal]);key=g.cal_key(stats,cfg)
        history.append({'epoch':epoch,'train_loss':float(np.mean(totals)),'cal_key':key,**{k:float(v.mean()) for k,v in stats.items()}})
        if key<best:
            best=key;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch==1 or epoch%5==0: print(f'[{arm}] epoch {epoch}/{cfg["epochs"]} minADE={stats["minade20"].mean():.6f} top3={stats["top3_ade"].mean():.6f}',flush=True)
    payload={'mean':mean,'std':std,'input_dim':obs.shape[1],'best_epoch':best_epoch,'best_cal_key':best,'config':cfg,'arm':arm,'dev_used_for_selection':False}
    torch.save({**payload,'state_dict':best_state},MODELS/f'{arm}_best.pth')
    torch.save({**payload,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}},MODELS/f'{arm}_final.pth')
    p.write_csv(MODELS/f'{arm}_history.csv',history)


def extra_metrics(trajectory,base,speed,future):
    obs=speed[:,1:]
    classify=lambda paths:p.classify_velocity(np.repeat(obs,20,axis=0),paths[:,:,-1].reshape(-1,2)/12,LABEL_CFG).reshape(-1,20)
    before,after=classify(base),classify(trajectory)
    target=p.classify_velocity(obs,future[:,-1]/12,LABEL_CFG)
    matching=after==target[:,None]
    ade=np.linalg.norm(trajectory-future[:,None],axis=-1).mean(-1)
    return {'maneuver_class_count':np.array([len(np.unique(x)) for x in after],np.float32),
        'base_family_retention':(before==after).mean(-1),'gt_family_covered':matching.any(-1).astype(float),
        'gt_family_near30_count':(matching & (ade<=.3)).sum(-1).astype(float)}


def evaluate(cfg,device):
    hashes={str(f):p.sha256(f) for f in MODELS.glob('*.pth')}
    assert len(hashes)==6
    (RESULTS/'model_lock.json').write_text(json.dumps({'models':hashes,'dev_used_for_selection':False,'test_access':False},indent=2))
    cache=p.load_npz(p.DEV_CACHE)
    dim=cache['motion'].shape[1]+cache['social'].shape[1]
    native=data(cache,mean=np.zeros((1,dim),np.float32),std=np.ones((1,dim),np.float32))
    base,speed,future=native[1:4]
    stats={'native':{**g.metrics(base,future,cache['logits']),**extra_metrics(base,base,speed,future)}}
    for arm in ARMS:
        for tag in ('best','final'):
            saved=torch.load(MODELS/f'{arm}_{tag}.pth',map_location='cpu',weights_only=False)
            values=data(cache,mean=saved['mean'],std=saved['std'])
            model=Correction(dim,cfg).to(device);model.load_state_dict(saved['state_dict'])
            trajectory=g.predict(model,*values[:3],device)
            name=f'{arm}_{tag}'
            stats[name]={**g.metrics(trajectory,future,cache['logits']),**extra_metrics(trajectory,base,speed,future)}
            np.savez_compressed(RESULTS/f'{name}_per_sample.npz',sample_id=cache['sample_id'],trajectory=trajectory,**stats[name])
    target=p.classify_velocity(speed[:,1:],future[:,-1]/12,LABEL_CFG)
    rows=[]
    groups=[('overall',np.ones(len(base),bool))]+[(name,cache['change_group']==i) for i,name in enumerate(('stable','transition','change'))]+[(f'intent_{i}',target==i) for i in range(7)]
    for name,values in stats.items():
        for group,mask in groups:
            if mask.any(): rows.append({'arm':name,'group':group,'n':int(mask.sum()),**{k:float(v[mask].mean()) for k,v in values.items()}})
    bootstrap=[]
    for treatment,reference in [('mode_family_best','native'),('mode_family_best','mode_global_best'),('mode_family_best','mode_oracle_best')]:
        for group,mask in groups[:4]:
            for metric in ('minade20','minfde20','top3_ade'):
                delta,lo,hi=p.paired_bootstrap_delta(stats[treatment][metric][mask],stats[reference][metric][mask],cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])
                bootstrap.append(dict(treatment=treatment,reference=reference,group=group,metric=metric,delta=delta,ci_low=lo,ci_high=hi))
    p.write_csv(RESULTS/'metrics.csv',rows);p.write_csv(RESULTS/'paired_bootstrap.csv',bootstrap)
    lookup={(r['arm'],r['group']):r for r in rows}
    n,s=lookup['native','overall'],lookup['mode_family_best','overall']
    gates={'oracle_gain_1pct':s['minade20']<=n['minade20']*.99,'support_gain_2pct':s['top3_ade']<=n['top3_ade']*.98,
        'change_no_harm':all(lookup['mode_family_best','change'][k]<=lookup['native','change'][k] for k in ('minade20','minfde20','top3_ade')),
        'transition_no_harm':lookup['mode_family_best','transition']['minade20']<=lookup['native','transition']['minade20'],
        'family_coverage_guard':s['gt_family_covered']>=n['gt_family_covered']-.01,
        'class_count_guard':s['maneuver_class_count']>=n['maneuver_class_count']*.95,
        'duplicate_guard':s['duplicate_fraction_10cm']<=n['duplicate_fraction_10cm']+.02,
        'acceleration_guard':s['acceleration_mean_mps2']<=n['acceleration_mean_mps2']*1.1}
    verdict='PILOT_SIGNAL_REVIEW_REQUIRED' if all(gates.values()) else 'PILOT_GATE_FAIL_REVIEW_REQUIRED'
    summary={'status':'COMPLETE','verdict':verdict,'gates':gates,'metrics':rows,'bootstrap':bootstrap,'scope':'ETH seed1 reused dev diagnostic; fixed K20; frozen backbone/scorer; no held-out test'}
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Mode-conditioned generation support pilot','',verdict,'',summary['scope'],'','| Arm | Group | minADE | minFDE | Top3 ADE | Classes | GT family coverage |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['group'] in ('overall','change','transition'): lines.append(f'| {r["arm"]} | {r["group"]} | {r["minade20"]:.6f} | {r["minfde20"]:.6f} | {r["top3_ade"]:.6f} | {r["maneuver_class_count"]:.3f} | {r["gt_family_covered"]:.3f} |')
    lines+=['','Heuristic motion families are not learned intentions. Missing base families receive winner loss only. Final checkpoints are diagnostic, not dev-selected replacements. Native logits are unchanged and can be stale after generation changes.','',json.dumps(gates,indent=2)]
    (RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'COMPLETE {verdict}',flush=True)


def preflight(cfg,device):
    p.seed_everything(1)
    velocity=np.random.randn(100,2).astype(np.float32)*.15
    endpoint=np.random.randn(100,2).astype(np.float32)*.2
    assert np.array_equal(labels(torch.tensor(velocity),torch.tensor(endpoint[:,None])).numpy()[:,0],p.classify_velocity(velocity,endpoint,LABEL_CFG))
    q=torch.tensor([[1.,2.,3.,4.]],requires_grad=True)
    support=family_support(q,torch.tensor([[True,True,False,False]]),3)
    assert support.item()==1.5
    support.backward();assert torch.equal(q.grad,torch.tensor([[.5,.5,0.,0.]]))
    assert family_support(q.detach(),torch.zeros_like(q,dtype=torch.bool),3)==0
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    values=data(cache,fit=fit)
    assert not (set(cache['sample_id'][fit]) & set(cache['sample_id'][cal]))
    tensors=[torch.from_numpy(v[:8]).to(device) for v in values[:4]]
    o,b,s,f=tensors;model=Correction(o.shape[1],cfg).to(device);model.eval()
    output,residual=model(o,b,s);assert torch.equal(output,b)
    for arm in ARMS:
        model.zero_grad();output,residual=model(o,b,s);objective=loss(output,b,f,residual,s,cfg,arm);objective.backward()
        assert torch.isfinite(objective) and model.network[-1].weight.grad.abs().sum()>0
    with torch.no_grad(): model.network[-1].weight.normal_(0,.01)
    order=torch.randperm(20,device=device)
    assert torch.allclose(model(o,b,s)[0][:,order],model(o,b[:,order],s)[0],atol=1e-6)
    toy=np.zeros((2,20,12,2),np.float32);m=g.metrics(toy,np.zeros((2,12,2),np.float32),np.zeros((2,20),np.float32))
    assert (m['minade20']==0).all() and (m['near_30cm_count']==20).all() and (m['duplicate_fraction_10cm']==1).all()
    sources=[Path(__file__),HERE/'config.json',SOURCE,g.SOURCE,g.SOURCE.parent/'config.json',p.TRAIN_CACHE,p.DEV_CACHE,p.RAW_TRAIN]
    families=labels(torch.from_numpy(values[2][:,1:]),torch.from_numpy(values[1][:,:,-1]/12)).numpy()
    target=p.classify_velocity(values[2][:,1:],values[3][:,-1]/12,LABEL_CFG)
    report={'status':'PASS','fit_n':len(fit),'calibration_n':len(cal),'K':20,'test_access':False,
        'trainable_parameters':sum(v.numel() for v in model.parameters()),'fit_gt_family_covered':float((families[fit]==target[fit,None]).any(-1).mean()),
        'hashes':{str(path):p.sha256(path) for path in sources}}
    (HERE/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='hashes'},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());device=torch.device(args.device)
    if cfg['test_access']!='forbidden': raise RuntimeError('No test authorization')
    if args.action=='preflight':preflight(cfg,device);return
    audit=json.loads((HERE/'preflight.json').read_text())
    for source,expected in audit['hashes'].items():
        if p.sha256(Path(source))!=expected:raise RuntimeError(f'Source changed: {source}')
    RESULTS.mkdir(exist_ok=True);MODELS.mkdir(exist_ok=True)
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);values=data(cache,fit=fit)
    start=time.time()
    for arm in ARMS:
        print(f'[TRAIN] {arm} K20 epochs={cfg["epochs"]}',flush=True);train(arm,cfg,cache,values,fit,cal,device)
    print('[DEV] locking all six checkpoints before evaluation',flush=True);evaluate(cfg,device)
    print(f'Elapsed seconds: {time.time()-start:.1f}',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        RESULTS.mkdir(exist_ok=True);(RESULTS/'failure.txt').write_text(traceback.format_exc());raise
