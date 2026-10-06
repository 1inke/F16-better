"""Matched K20 replacement x maneuver-balanced loss generation pilot."""
from pathlib import Path
import importlib.util
import sys
import json
import argparse
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'GENERATION_MODE_SUPPORT_SERIES/run_mode_support.py'
spec=importlib.util.spec_from_file_location('direction_coverage_source',SOURCE)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
g,p=m.g,m.p
RESULTS=HERE/'results';MODELS=RESULTS/'models'
ARMS=('native_uniform','native_balanced','replacement_uniform','replacement_balanced')
WEIGHTS=None
RAW_METRICS=g.metrics


def score_free_metrics(paths,future,logits):
    return {k:v for k,v in RAW_METRICS(paths,future,logits).items() if not k.startswith('fixed_score_selected')}


# Shared trainer must not log stale/misaligned selection metrics for replaced slots.
g.metrics=score_free_metrics


def prepared(cache,replace,**kwargs):
    values=list(m.data(cache,**kwargs))
    if replace:
        # Stable sort: retain twelve highest native logits. Selection has no GT access.
        keep=np.argsort(-cache['logits'],axis=1,kind='stable')[:,:12]
        retained=values[1][np.arange(len(keep))[:,None],keep]
        past=cache['motion'][:,:16].reshape(-1,8,2)
        values[1]=np.concatenate((retained,p.explicit_candidates(past)),axis=1).astype(np.float32)
    return values


def objective(output,base,future,residual,speed,cfg,arm):
    error=(output-future[:,None]).norm(dim=-1)
    quality=error.mean(-1)+cfg['fde_loss_weight']*error[:,:,-1]
    per_sample=quality.min(-1).values
    target=m.labels(speed[:,1:],future[:,None,-1]/12).squeeze(1)
    weight=WEIGHTS[target] if arm.endswith('balanced') else torch.ones_like(per_sample)
    family=m.labels(speed[:,1:],base[:,:,-1]/12)
    with torch.no_grad():
        reference=(g.pair_distance(base)*cfg['diversity_reference_fraction']).clamp(max=cfg['diversity_distance_cap_m'])
        cross=family[:,:,None]!=family[:,None,:]
    separation=(F.relu(reference-g.pair_distance(output))*cross).sum()/cross.sum().clamp_min(1)
    return (weight*per_sample).mean()+cfg['diversity_weight']*separation+cfg['residual_weight']*residual.square().mean()+cfg['smoothness_weight']*torch.diff(residual,dim=2).square().mean()


def generation_metrics(paths,base,speed,future):
    stats=g.metrics(paths,future,np.zeros(paths.shape[:2],np.float32))
    # Replacement scores are undefined: never treat the first proposal as a selected prediction.
    stats={k:v for k,v in stats.items() if not k.startswith('fixed_score_selected')}
    return {**stats,**m.extra_metrics(paths,base,speed,future)}


def evaluate(cfg,device):
    hashes={str(f):p.sha256(f) for f in MODELS.glob('*.pth')}
    assert len(hashes)==2*len(ARMS)
    (RESULTS/'model_lock.json').write_text(json.dumps({'models':hashes,'test_access':False,'dev_used_for_selection':False},indent=2))
    cache=p.load_npz(p.DEV_CACHE);dim=cache['motion'].shape[1]+cache['social'].shape[1]
    kwargs={'mean':np.zeros((1,dim),np.float32),'std':np.ones((1,dim),np.float32)}
    native=prepared(cache,False,**kwargs);replacement=prepared(cache,True,**kwargs)
    base,speed,future=native[1:4]
    stats={}
    for name,values in [('native',native),('replacement_untrained',replacement)]:
        stats[name]=generation_metrics(values[1],values[1],speed,future)
        np.savez_compressed(RESULTS/f'{name}_per_sample.npz',sample_id=cache['sample_id'],trajectory=values[1],**stats[name])
    for arm in ARMS:
        for tag in ('best','final'):
            saved=torch.load(MODELS/f'{arm}_{tag}.pth',map_location='cpu',weights_only=False)
            values=prepared(cache,arm.startswith('replacement'),mean=saved['mean'],std=saved['std'])
            model=m.Correction(dim,cfg).to(device);model.load_state_dict(saved['state_dict'])
            paths=g.predict(model,*values[:3],device);name=f'{arm}_{tag}'
            stats[name]=generation_metrics(paths,values[1],speed,future)
            np.savez_compressed(RESULTS/f'{name}_per_sample.npz',sample_id=cache['sample_id'],trajectory=paths,**stats[name])
    target=p.classify_velocity(speed[:,1:],future[:,-1]/12,m.LABEL_CFG)
    groups=[('overall',np.ones(len(base),bool))]+[(name,cache['change_group']==i) for i,name in enumerate(('stable','transition','change'))]+[(f'intent_{i}',target==i) for i in range(7)]
    rows=[]
    for name,values in stats.items():
        for group,mask in groups:
            if mask.any():rows.append({'arm':name,'group':group,'n':int(mask.sum()),**{k:float(v[mask].mean()) for k,v in values.items()}})
    bootstrap=[]
    contrasts=[('replacement_untrained','native'),('replacement_balanced_best','native'),('replacement_balanced_best','replacement_untrained'),('replacement_balanced_best','replacement_uniform_best'),('replacement_balanced_best','native_balanced_best'),('native_balanced_best','native_uniform_best')]
    for treatment,reference in contrasts:
        for group,mask in groups[:4]:
            for metric in ('minade20','minfde20','top3_ade'):
                delta,lo,hi=p.paired_bootstrap_delta(stats[treatment][metric][mask],stats[reference][metric][mask],cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])
                bootstrap.append(dict(treatment=treatment,reference=reference,group=group,metric=metric,delta=delta,ci_low=lo,ci_high=hi))
    # Matched-final factorial effects avoid different best-epoch confounding.
    factorial=[]
    for group,mask in groups[:4]:
        for metric in ('minade20','minfde20','top3_ade'):
            a=stats['native_uniform_final'][metric][mask];b=stats['native_balanced_final'][metric][mask]
            c=stats['replacement_uniform_final'][metric][mask];d=stats['replacement_balanced_final'][metric][mask]
            for effect,array in [('replacement_main',.5*(c+d-a-b)),('balance_main',.5*(b+d-a-c)),('interaction',d-c-b+a)]:
                delta,lo,hi=p.paired_bootstrap_delta(array,np.zeros_like(array),cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])
                factorial.append(dict(group=group,metric=metric,effect=effect,delta=delta,ci_low=lo,ci_high=hi))
    p.write_csv(RESULTS/'metrics.csv',rows);p.write_csv(RESULTS/'paired_bootstrap.csv',bootstrap);p.write_csv(RESULTS/'factorial_final.csv',factorial)
    lookup={(r['arm'],r['group']):r for r in rows};chosen='replacement_balanced_best'
    n,s=lookup['native','overall'],lookup[chosen,'overall'];nc,sc=lookup['native','change'],lookup[chosen,'change']
    gates={'overall_oracle_no_harm':s['minade20']<=n['minade20'] and s['minfde20']<=n['minfde20'],
        'change_ade_gain_2pct':sc['minade20']<=nc['minade20']*.98,'change_fde_no_harm':sc['minfde20']<=nc['minfde20'],
        'stable_ade_guard_1pct':lookup[chosen,'stable']['minade20']<=lookup['native','stable']['minade20']*1.01,
        'transition_ade_guard_1pct':lookup[chosen,'transition']['minade20']<=lookup['native','transition']['minade20']*1.01,
        'duplicate_guard':s['duplicate_fraction_10cm']<=n['duplicate_fraction_10cm']+.02,
        'kinematic_guard':s['speed_over3mps_fraction']<=n['speed_over3mps_fraction']+.01 and s['acceleration_over4mps2_fraction']<=n['acceleration_over4mps2_fraction']+.01,
        'change_final_consistency':lookup['replacement_balanced_final','change']['minade20']<nc['minade20']}
    summary={'status':'COMPLETE','verdict':'PILOT_SIGNAL_REVIEW_REQUIRED' if all(gates.values()) else 'PILOT_GATE_FAIL_REVIEW_REQUIRED','gates':gates,'metrics':rows,'bootstrap':bootstrap,'factorial_final':factorial,'scope':'ETH seed1 reused dev; K20; frozen backbone and no scorer training; no test'}
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Fixed-budget direction coverage pilot','',summary['verdict'],'',summary['scope'],'','| Arm | Group | minADE | minFDE | Top3 ADE |','|---|---|---:|---:|---:|']
    for r in rows:
        if r['group'] in ('overall','change','stable','transition'):lines.append(f'| {r["arm"]} | {r["group"]} | {r["minade20"]:.6f} | {r["minfde20"]:.6f} | {r["top3_ade"]:.6f} |')
    lines+=['',json.dumps(gates,indent=2),'','Best selected on common internal calibration. Final is diagnostic. Explicit motion anchors are heuristics, not understood intentions. Replacement slots have no trained scores; selection improvement is not evaluated.']
    (RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('COMPLETE '+summary['verdict'],flush=True)


def setup_weights(cache,fit,values,cfg,device):
    global WEIGHTS
    target=p.classify_velocity(values[2][:,1:],values[3][:,-1]/12,m.LABEL_CFG)
    counts=np.bincount(target[fit],minlength=7)
    if (counts==0).any():raise RuntimeError('Missing fit motion class')
    w=np.sqrt(len(fit)/(7*counts)).clip(.5,3.)
    w/=np.mean(w[target[fit]])
    WEIGHTS=torch.tensor(w,dtype=torch.float32,device=device)
    return counts,w


def sources():
    return [Path(__file__),HERE/'config.json',SOURCE,m.SOURCE,g.SOURCE,g.SOURCE.parent/'config.json',p.TRAIN_CACHE,p.DEV_CACHE,p.RAW_TRAIN]


def preflight(cfg,device):
    p.seed_everything(cfg['seed']);cache=p.load_npz(p.TRAIN_CACHE)
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);values=prepared(cache,False,fit=fit);replacement=prepared(cache,True,fit=fit)
    counts,w=setup_weights(cache,fit,values,cfg,device)
    assert not(set(cache['sample_id'][fit]) & set(cache['sample_id'][cal]))
    assert values[1].shape==replacement[1].shape and values[1].shape[1]==20
    # Candidate construction only consumes history and logits; proposal API has no future input.
    past=cache['motion'][:,:16].reshape(-1,8,2)
    assert np.array_equal(replacement[1][:,12:],p.explicit_candidates(past))
    for arm in ARMS:
        real=replacement if arm.startswith('replacement') else values
        o,b,s,f=[torch.from_numpy(v[:8]).to(device) for v in real[:4]]
        model=m.Correction(o.shape[1],cfg).to(device);model.eval();out,res=model(o,b,s)
        assert torch.equal(out,b)
        obj=objective(out,b,f,res,s,cfg,arm);obj.backward()
        assert torch.isfinite(obj) and model.network[-1].weight.grad.abs().sum()>0
        with torch.no_grad():model.network[-1].weight.normal_(0,.01)
        order=torch.randperm(20,device=device)
        assert torch.allclose(model(o,b,s)[0][:,order],model(o,b[:,order],s)[0],atol=1e-6)
        # Exact-GT candidates improve winner loss; balancing weights stay loss-only.
        perfect=f[:,None].expand_as(b);zero=torch.zeros_like(b)
        assert objective(perfect,perfect,f,zero,s,cfg,arm)<objective(perfect+1,perfect,f,zero,s,cfg,arm)
    toy=np.zeros((2,20,12,2),np.float32);stat=g.metrics(toy,np.zeros((2,12,2),np.float32),np.zeros((2,20),np.float32))
    assert (stat['minade20']==0).all() and (stat['near_30cm_count']==20).all() and (stat['duplicate_fraction_10cm']==1).all()
    report={'status':'PASS','fit_n':len(fit),'calibration_n':len(cal),'K':20,'trainable_parameters':sum(v.numel() for v in model.parameters()),'class_counts_fit':counts.tolist(),'class_weights_fit':w.tolist(),'test_access':False,'hashes':{str(f):p.sha256(f) for f in sources()}}
    (HERE/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='hashes'},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());device=torch.device(args.device)
    if cfg['test_access']!='forbidden':raise RuntimeError('Test forbidden')
    if args.action=='preflight':preflight(cfg,device);return
    audit=json.loads((HERE/'preflight.json').read_text())
    for source,h in audit['hashes'].items():
        if p.sha256(Path(source))!=h:raise RuntimeError('Source lock mismatch: '+source)
    RESULTS.mkdir(exist_ok=True);MODELS.mkdir(exist_ok=True)
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    native=prepared(cache,False,fit=fit);replacement=prepared(cache,True,fit=fit)
    counts,w=setup_weights(cache,fit,native,cfg,device)
    assert np.allclose(w,audit['class_weights_fit'])
    # Reuse the audited trainer, but redirect outputs and explicitly replace only its loss.
    m.MODELS=MODELS;m.loss=objective
    start=time.time()
    for arm in ARMS:
        print(f'[TRAIN] {arm} K20 epochs={cfg["epochs"]}',flush=True)
        m.train(arm,cfg,cache,replacement if arm.startswith('replacement') else native,fit,cal,device)
    print('[DEV] lock eight checkpoints before diagnosis',flush=True);evaluate(cfg,device)
    print(f'Elapsed seconds: {time.time()-start:.1f}',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        RESULTS.mkdir(exist_ok=True);(RESULTS/'failure.txt').write_text(traceback.format_exc());raise
