"""Fixed-budget generation pilot: best-of-K versus GT-local set support."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/directional_regime_pilot.py'
spec = importlib.util.spec_from_file_location('support_source', SOURCE)
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)
RESULTS = HERE / 'results'
MODELS = RESULTS / 'models'
ARMS = ('oracle_only', 'local_support')


class GeneratorCorrection(nn.Module):
    def __init__(self, observation_dim, cfg):
        super().__init__()
        self.strength = cfg['velocity_residual_scale']
        self.network = nn.Sequential(
            nn.Linear(observation_dim + 24, cfg['hidden']), nn.LayerNorm(cfg['hidden']),
            nn.GELU(), nn.Dropout(cfg['dropout']), nn.Linear(cfg['hidden'], cfg['hidden']//2),
            nn.GELU(), nn.Linear(cfg['hidden']//2, 24))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, observation, base_trajectory, observed_speed):
        repeated = observation[:, None].expand(-1, base_trajectory.shape[1], -1)
        raw = self.network(torch.cat((repeated, base_trajectory.flatten(2)), -1))
        velocity_residual = raw.reshape_as(base_trajectory).tanh()
        velocity_residual = velocity_residual * self.strength * (observed_speed[:, None, None, None] + 0.03)
        return base_trajectory + velocity_residual.cumsum(2), velocity_residual


def pair_distance(trajectory):
    # A mean spatial separation over all forecast times; [batch, modes, modes].
    return torch.linalg.vector_norm(trajectory[:, :, None] - trajectory[:, None, :], dim=-1).mean(-1)


def generation_loss(trajectory, base, future, velocity_residual, cfg, support):
    errors = torch.linalg.vector_norm(trajectory - future[:, None], dim=-1)
    quality = errors.mean(-1) + cfg['fde_loss_weight'] * errors[..., -1]
    winner = quality.min(-1).values.mean()
    near = quality.topk(cfg['support_k'], largest=False).values.mean()
    with torch.no_grad():
        reference = pair_distance(base)
        minimum = (reference * cfg['diversity_reference_fraction']).clamp(max=cfg['diversity_distance_cap_m'])
        off_diagonal = ~torch.eye(base.shape[1], dtype=torch.bool, device=base.device)
    separation = pair_distance(trajectory)
    collapse = F.relu(minimum - separation)[:, off_diagonal].mean()
    residual = velocity_residual.square().mean()
    smooth = torch.diff(velocity_residual, dim=2).square().mean()
    loss = winner + (cfg['support_weight'] * near if support else 0.0)
    return loss + cfg['diversity_weight'] * collapse + cfg['residual_weight'] * residual + cfg['smoothness_weight'] * smooth


def data(cache, mean=None, std=None, fit=None):
    raw = np.concatenate((cache['motion'], cache['social']), 1).astype(np.float32)
    if mean is None:
        mean = raw[fit].mean(0, keepdims=True)
        std = raw[fit].std(0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    observation = ((raw - mean) / std).astype(np.float32)
    past, future = p.raw_focal_trajectories(cache['sample_id'], cache['motion'])
    base = cache['candidate'][..., :24].reshape(-1, 20, 12, 2)
    errors = np.linalg.norm(base - future[:, None], axis=-1)
    if not np.allclose(errors.mean(-1), cache['ade'], atol=2e-5, rtol=0):
        raise RuntimeError('Candidate/GT ADE alignment failed')
    if not np.allclose(errors[..., -1], cache['fde'], atol=2e-5, rtol=0):
        raise RuntimeError('Candidate/GT FDE alignment failed')
    speed = np.linalg.norm(past[:, -1] - past[:, -2], axis=-1).astype(np.float32)
    return observation, base, speed, future, mean.astype(np.float32), std


def predict(model, observation, base, speed, device):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(observation), 256):
            output, _ = model(torch.from_numpy(observation[start:start+256]).to(device),
                               torch.from_numpy(base[start:start+256]).to(device),
                               torch.from_numpy(speed[start:start+256]).to(device))
            outputs.append(output.cpu().numpy())
    return np.concatenate(outputs)


def metrics(trajectory, future, native_logits):
    errors = np.linalg.norm(trajectory - future[:, None], axis=-1)
    ade, fde = errors.mean(-1), errors[..., -1]
    ordered = np.sort(ade, axis=-1)
    diff = np.linalg.norm(trajectory[:, :, None] - trajectory[:, None, :], axis=-1).mean(-1)
    mask = ~np.eye(trajectory.shape[1], dtype=bool)
    origin = np.zeros_like(trajectory[:, :, :1])
    velocity = np.diff(np.concatenate((origin, trajectory), axis=2), axis=2)
    speed = np.linalg.norm(velocity, axis=-1) / .4
    acceleration = np.linalg.norm(np.diff(velocity, axis=2), axis=-1) / .16
    selected = native_logits.argmax(-1)
    row = np.arange(len(selected))
    endpoint_center = trajectory[:, :, -1].mean(1, keepdims=True)
    return {
        'minade20': ade.min(-1), 'minfde20': fde.min(-1), 'top3_ade': ordered[:, :3].mean(-1),
        'near_30cm_count': (ade <= .3).sum(-1).astype(np.float32),
        'near_50cm_count': (ade <= .5).sum(-1).astype(np.float32),
        'three_hit_30cm': ((ade <= .3).sum(-1) >= 3).astype(np.float32),
        'duplicate_fraction_10cm': (diff[:, mask] < .1).mean(-1),
        'endpoint_spread': np.linalg.norm(trajectory[:, :, -1] - endpoint_center, axis=-1).mean(-1),
        'speed_mean_mps': speed.mean((1,2)), 'acceleration_mean_mps2': acceleration.mean((1,2)),
        'speed_over3mps_fraction': (speed > 3.0).mean((1,2)),
        'acceleration_over4mps2_fraction': (acceleration > 4.0).mean((1,2)),
        'fixed_score_selected_ade': ade[row, selected], 'fixed_score_selected_fde': fde[row, selected],
    }


def cal_key(values, cfg):
    return float(values['minade20'].mean() + cfg['fde_loss_weight'] * values['minfde20'].mean()
                 + cfg['calibration_top3_weight'] * values['top3_ade'].mean())


def train(arm, cfg, cache, values, fit, calibration, device):
    observation, base, speed, future, mean, std = values
    best_path, final_path = MODELS / f'{arm}_best.pth', MODELS / f'{arm}_final.pth'
    if best_path.exists() and final_path.exists():
        print(f'[REUSE] {arm}', flush=True)
        return
    if best_path.exists() or final_path.exists():
        raise RuntimeError('Partial arm artifacts found; inspect before retrying')
    p.seed_everything(cfg['seed'])
    model = GeneratorCorrection(observation.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    loader = DataLoader(TensorDataset(torch.from_numpy(observation[fit]), torch.from_numpy(base[fit]),
                                      torch.from_numpy(speed[fit]), torch.from_numpy(future[fit])),
                        batch_size=cfg['batch_size'], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(cfg['seed']))
    best_key, best_state, best_epoch = math.inf, None, -1
    history = []
    for epoch in range(1, cfg['epochs'] + 1):
        model.train(); losses = []
        for obs, native, observed_speed, gt in loader:
            obs, native, observed_speed, gt = obs.to(device), native.to(device), observed_speed.to(device), gt.to(device)
            trajectory, residual = model(obs, native, observed_speed)
            loss = generation_loss(trajectory, native, gt, residual, cfg, arm == 'local_support')
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss {arm} epoch{epoch}')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(loss.item())
        prediction = predict(model, observation[calibration], base[calibration], speed[calibration], device)
        stats = metrics(prediction, future[calibration], cache['logits'][calibration])
        key = cal_key(stats, cfg)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'cal_key': key,
                        **{name: float(v.mean()) for name, v in stats.items()}})
        if key < best_key:
            best_key, best_epoch = key, epoch
            best_state = {name: v.detach().cpu().clone() for name,v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(f'[{arm}] epoch {epoch}/{cfg["epochs"]} cal_minADE={stats["minade20"].mean():.5f} top3={stats["top3_ade"].mean():.5f}', flush=True)
    payload = {'mean': mean, 'std': std, 'input_dim': observation.shape[1], 'arm': arm,
               'best_epoch': best_epoch, 'best_cal_key': best_key, 'config': cfg,
               'trainable_parameters': sum(v.numel() for v in model.parameters()), 'dev_used_for_selection': False}
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, 'state_dict': best_state}, best_path)
    torch.save({**payload, 'state_dict': {name:v.detach().cpu().clone() for name,v in model.state_dict().items()},
                'final_epoch': cfg['epochs']}, final_path)
    p.write_csv(MODELS / f'{arm}_history.csv', history)


def preflight(cfg, device):
    p.seed_everything(1)
    model = GeneratorCorrection(10, cfg).to(device)
    obs = torch.randn(4,10,device=device)
    base = torch.randn(4,20,12,2,device=device)
    speed = torch.ones(4,device=device) * .2
    model.eval(); out, residual = model(obs,base,speed)
    assert torch.equal(out, base), 'Initialization is not exact identity'
    gt = torch.randn(4,12,2,device=device)
    loss = generation_loss(out,base,gt,residual,cfg,True); loss.backward()
    assert model.network[-1].weight.grad.abs().sum() > 0, 'Generator gradient unreachable'
    with torch.no_grad(): model.network[-1].weight.normal_(0,.01)
    order = torch.randperm(20,device=device)
    a,_ = model(obs,base,speed); b,_ = model(obs,base[:,order],speed)
    assert torch.allclose(a[:,order],b,atol=1e-6), 'Candidate permutation mismatch'
    # Metric direction: identical-to-GT set has zero error and 20 near hits.
    tiny = np.zeros((2,20,12,2),np.float32)
    stat = metrics(tiny,np.zeros((2,12,2),np.float32),np.zeros((2,20),np.float32))
    assert np.all(stat['minade20']==0) and np.all(stat['near_30cm_count']==20)
    assert np.all(stat['duplicate_fraction_10cm']==1)
    gt_toy = torch.zeros(1,12,2,device=device)
    good = torch.zeros(1,20,12,2,device=device)
    bad = good.clone(); bad[:,1:] = 1.0
    zero = torch.zeros_like(good)
    assert generation_loss(good,good,gt_toy,zero,cfg,True) < generation_loss(bad,good,gt_toy,zero,cfg,True)
    cache = p.load_npz(p.TRAIN_CACHE)
    fit,cal = p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    values = data(cache,fit=fit)
    assert len(set(cache['sample_id'][fit]) & set(cache['sample_id'][cal])) == 0
    # Real forward/backward over a small batch; future is loss-only.
    real = GeneratorCorrection(values[0].shape[1],cfg).to(device)
    out,res = real(torch.from_numpy(values[0][:8]).to(device),torch.from_numpy(values[1][:8]).to(device),
                   torch.from_numpy(values[2][:8]).to(device))
    generation_loss(out,torch.from_numpy(values[1][:8]).to(device),torch.from_numpy(values[3][:8]).to(device),res,cfg,True).backward()
    sources = [Path(__file__),HERE/'config.json',SOURCE,p.TRAIN_CACHE,p.DEV_CACHE,p.RAW_TRAIN]
    report = {'status':'PASS','fit_n':len(fit),'calibration_n':len(cal),'candidate_count':20,
              'test_access':False,'trainable_parameters':sum(v.numel() for v in real.parameters()),
              'hashes':{str(f):p.sha256(f) for f in sources}}
    (HERE/'preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='hashes'},indent=2),flush=True)


def evaluate(cfg,device):
    MODELS.mkdir(parents=True,exist_ok=True)
    lock = {str(f):p.sha256(f) for f in MODELS.glob('*.pth')}
    assert len(lock)==4
    (RESULTS/'model_lock.json').write_text(json.dumps({'models':lock,'test_access':False,
        'dev_used_for_selection':False,'time':time.time()},indent=2),encoding='utf-8')
    cache=p.load_npz(p.DEV_CACHE)
    obs_dim=cache['motion'].shape[1]+cache['social'].shape[1]
    native_values=data(cache,mean=np.zeros((1,obs_dim),np.float32),std=np.ones((1,obs_dim),np.float32))
    base,future=native_values[1],native_values[3]
    stats={'native':metrics(base,future,cache['logits'])}
    rows=[]
    label_cfg=p.load_config()
    past=cache['motion'][:,:16].reshape(-1,8,2)
    for arm in ARMS:
        for checkpoint in ('best','final'):
            saved=torch.load(MODELS/f'{arm}_{checkpoint}.pth',map_location='cpu',weights_only=False)
            values=data(cache,mean=saved['mean'],std=saved['std'])
            model=GeneratorCorrection(saved['input_dim'],cfg).to(device)
            model.load_state_dict(saved['state_dict'])
            trajectory=predict(model,values[0],values[1],values[2],device)
            name=f'{arm}_{checkpoint}'
            stats[name]=metrics(trajectory,future,cache['logits'])
            labels=p.candidate_intent_labels(past,trajectory,label_cfg)
            stats[name]['maneuver_class_count']=np.asarray([len(np.unique(r)) for r in labels],np.float32)
            np.savez_compressed(RESULTS/f'{name}_per_sample.npz',sample_id=cache['sample_id'],
                                trajectory=trajectory,**stats[name])
    labels=p.candidate_intent_labels(past,base,label_cfg)
    stats['native']['maneuver_class_count']=np.asarray([len(np.unique(r)) for r in labels],np.float32)
    for name,values in stats.items():
        for group,mask in [('overall',np.ones(len(base),bool)),('stable',cache['change_group']==0),
                           ('transition',cache['change_group']==1),('change',cache['change_group']==2)]:
            rows.append({'arm':name,'group':group,'n':int(mask.sum()),**{k:float(v[mask].mean()) for k,v in values.items()}})
    boot=[]
    for i,(treatment,reference) in enumerate([('oracle_only_best','native'),('local_support_best','native'),
                                            ('local_support_best','oracle_only_best'),('local_support_final','native')]):
        for metric in ('minade20','minfde20','top3_ade','three_hit_30cm','duplicate_fraction_10cm'):
            delta,lo,hi=p.paired_bootstrap_delta(stats[treatment][metric],stats[reference][metric],
                                                cfg['bootstrap_repetitions'],cfg['bootstrap_seed']+i)
            boot.append({'treatment':treatment,'reference':reference,'metric':metric,'delta':delta,'ci_low':lo,'ci_high':hi})
    p.write_csv(RESULTS/'metrics.csv',rows);p.write_csv(RESULTS/'paired_bootstrap.csv',boot)
    overall={r['arm']:r for r in rows if r['group']=='overall'}
    n=overall['native'];s=overall['local_support_best'];f=overall['local_support_final']
    gates={
        'oracle_gain':s['minade20'] <= n['minade20']*(1-cfg['minimum_oracle_gain']),
        'support_gain':s['top3_ade'] <= n['top3_ade']*(1-cfg['minimum_support_gain']),
        'duplicate_guard':s['duplicate_fraction_10cm']<=n['duplicate_fraction_10cm']+cfg['max_duplicate_increase'],
        'acceleration_guard':s['acceleration_mean_mps2']<=n['acceleration_mean_mps2']*(1+cfg['max_acceleration_increase_fraction']),
        'final_consistency':f['minade20']<n['minade20'] and f['top3_ade']<n['top3_ade'],
    }
    verdict='LOCAL_SUPPORT_PILOT_SIGNAL_REVIEW_REQUIRED' if all(gates.values()) else 'LOCAL_SUPPORT_PILOT_GATE_FAIL_REVIEW_REQUIRED'
    summary={'status':'COMPLETE','verdict':verdict,'gates':gates,'overall':overall,'bootstrap':boot,
             'scope':'ETH seed1 reused development diagnosis; no held-out test; K=20 for every arm'}
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Generation local support pilot','',verdict,'',summary['scope'],'',
           '| Arm | minADE20 | minFDE20 | Top3 ADE | Near .3m count | Duplicate fraction |',
           '|---|---:|---:|---:|---:|---:|']
    for name,r in overall.items():
        lines.append(f'| {name} | {r["minade20"]:.6f} | {r["minfde20"]:.6f} | {r["top3_ade"]:.6f} | {r["near_30cm_count"]:.3f} | {r["duplicate_fraction_10cm"]:.4f} |')
    lines+=['','## Gates','']+[f'- {k}: {v}' for k,v in gates.items()]
    lines+=['','Top3 improvement alone is not proof of intention understanding. Fixed-score selected error is diagnostic because scorer logits were not retrained.']
    (RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'COMPLETE | {verdict}',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight')
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());device=torch.device(args.device)
    if cfg['test_access']!='forbidden':raise RuntimeError('Test scope changed')
    if args.action=='preflight':preflight(cfg,device);return
    audit=json.loads((HERE/'preflight.json').read_text())
    for source,expected in audit['hashes'].items():
        if p.sha256(Path(source))!=expected:raise RuntimeError(f'Source lock mismatch {source}')
    RESULTS.mkdir(parents=True,exist_ok=True)
    cache=p.load_npz(p.TRAIN_CACHE)
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    values=data(cache,fit=fit)
    for arm in ARMS:
        print(f'[TRAIN] {arm} | fixed K20 | epochs={cfg["epochs"]}',flush=True)
        train(arm,cfg,cache,values,fit,cal,device)
    print('[DEV] Models locked for development diagnostic',flush=True)
    evaluate(cfg,device)


if __name__=='__main__':
    try:main()
    except Exception:
        RESULTS.mkdir(parents=True,exist_ok=True)
        (RESULTS/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
