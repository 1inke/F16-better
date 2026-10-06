"""Retain K20; replicate balanced-vs-uniform generator training across seeds."""
from pathlib import Path
import importlib.util
import sys
import json
import argparse
import csv
import traceback
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'GENERATION_DIRECTION_COVERAGE_SERIES/run_direction_coverage.py'
spec=importlib.util.spec_from_file_location('balance_replication_source',SOURCE)
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
RESULTS=HERE/'results';MODELS=RESULTS/'models'
ARMS=('native_uniform','native_balanced')


def cfg_for(seed):
    cfg=json.loads((SOURCE.parent/'config.json').read_text());cfg['seed']=seed
    return cfg


def model_path(seed,arm,tag):
    directory=r.MODELS if seed==1 else MODELS/f'seed_{seed}'
    return directory/f'{arm}_{tag}.pth'


def validate_arm(seed,arm,values):
    cfg=cfg_for(seed)
    files=[model_path(seed,arm,tag) for tag in ('best','final')]
    history=files[0].parent/f'{arm}_history.csv'
    if not any(f.exists() for f in files+[history]):return False
    if not all(f.exists() for f in files+[history]):raise RuntimeError(f'Partial arm {seed}/{arm}; inspect before restarting')
    rows=list(csv.DictReader(history.open(encoding='utf-8-sig')))
    assert len(rows)==cfg['epochs'] and [int(x['epoch']) for x in rows]==list(range(1,cfg['epochs']+1))
    for path in files:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['config']==cfg and saved['arm']==arm and saved['dev_used_for_selection'] is False
        assert np.array_equal(saved['mean'],values[4]) and np.array_equal(saved['std'],values[5])
        assert abs(min(float(x['cal_key']) for x in rows)-saved['best_cal_key'])<1e-8
        model=r.m.Correction(saved['input_dim'],cfg);model.load_state_dict(saved['state_dict'])
        assert all(torch.isfinite(t).all() for t in saved['state_dict'].values())
    return True


def hierarchical_delta(delta,repetitions,seed):
    # Paired crossed bootstrap: same sampled cases shared by sampled training seeds.
    delta=np.asarray(delta,dtype=np.float64);rng=np.random.default_rng(seed);values=[]
    for start in range(0,repetitions,100):
        count=min(100,repetitions-start)
        sampled_seeds=rng.integers(0,len(delta),(count,len(delta)))
        seed_mean=delta[sampled_seeds].mean(1)
        cases=rng.integers(0,delta.shape[1],(count,delta.shape[1]))
        values.extend(np.take_along_axis(seed_mean,cases,axis=1).mean(1).tolist())
    lo,hi=np.percentile(values,[2.5,97.5]);return float(delta.mean()),float(lo),float(hi)


def preflight(device):
    protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['seeds']==[1,2,3] and protocol['test_access']=='forbidden'
    old=json.loads((SOURCE.parent/'preflight.json').read_text())
    for path,digest in old['hashes'].items():
        if r.p.sha256(Path(path))!=digest:raise RuntimeError('Original source lock mismatch '+path)
    cache=r.p.load_npz(r.p.TRAIN_CACHE);fit,cal=r.p.internal_split(len(cache['sample_id']),cfg_for(1)['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    assert not(set(cache['sample_id'][fit]) & set(cache['sample_id'][cal]))
    oldlock=json.loads((SOURCE.parent/'results/model_lock.json').read_text())
    preserved={}
    for arm in ARMS:
        assert validate_arm(1,arm,values)
        for tag in ('best','final'):
            path=model_path(1,arm,tag);digest=r.p.sha256(path);assert oldlock['models'][str(path)]==digest;preserved[str(path)]=digest
    counts,weights=r.setup_weights(cache,fit,values,cfg_for(2),device)
    for arm in ARMS:
        tensors=[torch.from_numpy(x[:8]).to(device) for x in values[:4]];o,b,s,f=tensors
        model=r.m.Correction(o.shape[1],cfg_for(2)).to(device);model.eval();out,res=model(o,b,s);assert torch.equal(out,b)
        loss=r.objective(out,b,f,res,s,cfg_for(2),arm);loss.backward();assert torch.isfinite(loss) and model.network[-1].weight.grad.abs().sum()>0
    toy=np.zeros((3,8));assert hierarchical_delta(toy,100,1)==(0.,0.,0.)
    assert hierarchical_delta(toy-1,100,1)==(-1.,-1.,-1.)
    paths=[Path(__file__),HERE/'protocol.json']+[Path(x) for x in old['hashes']]
    report={'status':'PASS','fit_n':len(fit),'calibration_n':len(cal),'K':20,'new_training_runs':4,'reused_seed':1,'seeds':[1,2,3],'test_access':False,
        'class_counts_fit':counts.tolist(),'weights_fit':weights.tolist(),'source_hashes':{str(x):r.p.sha256(x) for x in paths},'preserved_models':preserved}
    (HERE/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if 'hash' not in k and k!='preserved_models'},indent=2),flush=True)


def evaluate(protocol,device):
    lock={str(model_path(seed,arm,tag)):r.p.sha256(model_path(seed,arm,tag)) for seed in protocol['seeds'] for arm in ARMS for tag in ('best','final')}
    assert len(lock)==12
    (RESULTS/'model_lock.json').write_text(json.dumps({'models':lock,'test_access':False,'dev_used_for_selection':False},indent=2))
    cache=r.p.load_npz(r.p.DEV_CACHE);dim=cache['motion'].shape[1]+cache['social'].shape[1]
    native=r.prepared(cache,False,mean=np.zeros((1,dim),np.float32),std=np.ones((1,dim),np.float32))
    base,speed,future=native[1:4];native_stats=r.generation_metrics(base,base,speed,future)
    np.savez_compressed(RESULTS/'native_per_sample.npz',sample_id=cache['sample_id'],trajectory=base,**native_stats)
    stats={};rows=[]
    target=r.p.classify_velocity(speed[:,1:],future[:,-1]/12,r.m.LABEL_CFG)
    groups=[('overall',np.ones(len(base),bool))]+[(name,cache['change_group']==i) for i,name in enumerate(('stable','transition','change'))]+[(f'intent_{i}',target==i) for i in range(7)]
    for seed in protocol['seeds']:
        for arm in ARMS:
            for tag in ('best','final'):
                saved=torch.load(model_path(seed,arm,tag),map_location='cpu',weights_only=False);values=r.prepared(cache,False,mean=saved['mean'],std=saved['std'])
                model=r.m.Correction(dim,cfg_for(seed)).to(device);model.load_state_dict(saved['state_dict']);paths=r.g.predict(model,*values[:3],device)
                value=r.generation_metrics(paths,base,speed,future);stats[seed,arm,tag]=value
                np.savez_compressed(RESULTS/f'seed_{seed}_{arm}_{tag}.npz',sample_id=cache['sample_id'],trajectory=paths,**value)
                for group,mask in groups:
                    if mask.any():rows.append({'seed':seed,'arm':arm,'checkpoint':tag,'group':group,'n':int(mask.sum()),'best_epoch':saved['best_epoch'],**{k:float(v[mask].mean()) for k,v in value.items()}})
    aggregate=[];boot=[]
    for scope,seeds in [('all_seeds',[1,2,3]),('new_seeds_only',[2,3])]:
        for tag in ('best','final'):
            for arm in ARMS:
                for group,mask in groups[:4]:aggregate.append({'scope':scope,'arm':arm,'checkpoint':tag,'group':group,'seed_n':len(seeds),'sample_n':int(mask.sum()),**{k:float(np.stack([stats[seed,arm,tag][k][mask] for seed in seeds]).mean()) for k in native_stats}})
            for reference in ('native_uniform','native'):
                for group,mask in groups[:4]:
                    for metric in ('minade20','minfde20','top3_ade'):
                        delta=np.stack([stats[seed,'native_balanced',tag][metric][mask]-(native_stats[metric][mask] if reference=='native' else stats[seed,reference,tag][metric][mask]) for seed in seeds])
                        mean,lo,hi=hierarchical_delta(delta,protocol['bootstrap_repetitions'],protocol['bootstrap_seed'])
                        boot.append(dict(scope=scope,checkpoint=tag,reference=reference,group=group,metric=metric,delta=mean,ci_low=lo,ci_high=hi))
    lookup={(x['scope'],x['arm'],x['checkpoint'],x['group']):x for x in aggregate}
    b=lookup['new_seeds_only','native_balanced','best','change']
    replication=all(all(stats[seed,'native_balanced','best'][k][cache['change_group']==2].mean()<native_stats[k][cache['change_group']==2].mean() for k in ('minade20','minfde20')) for seed in (2,3))
    gates={'both_new_seeds_change_ade_fde_improve_native':replication,'new_mean_change_ade_gain_1pct':b['minade20']<=native_stats['minade20'][cache['change_group']==2].mean()*.99,
        'balanced_vs_uniform_change_CI':all(x['ci_high']<0 for x in boot if x['scope']=='new_seeds_only' and x['checkpoint']=='best' and x['reference']=='native_uniform' and x['group']=='change' and x['metric'] in ('minade20','minfde20')),
        'overall_stable_transition_oracle_guard_1pct':all(lookup['new_seeds_only','native_balanced','best',group][k]<=native_stats[k][mask].mean()*1.01 for group,mask in groups[:3] for k in ('minade20','minfde20')),
        'final_change_ade_fde_no_harm':all(lookup['new_seeds_only','native_balanced','final','change'][k]<=native_stats[k][cache['change_group']==2].mean() for k in ('minade20','minfde20'))}
    gates={name:bool(value) for name,value in gates.items()}
    summary={'status':'COMPLETE','verdict':'REPLICATION_SIGNAL_REVIEW_REQUIRED' if all(gates.values()) else 'REPLICATION_GATE_FAIL_REVIEW_REQUIRED','gates':gates,'per_seed':rows,'aggregate':aggregate,'bootstrap':boot,'scope':'ETH frozen generator seed1; correction seeds1/2/3; seed1 discovery reused, seeds2/3 replication; repeatedly inspected dev, no test; not trajectory ensemble'}
    r.p.write_csv(RESULTS/'per_seed_metrics.csv',rows);r.p.write_csv(RESULTS/'aggregate_metrics.csv',aggregate);r.p.write_csv(RESULTS/'hierarchical_bootstrap.csv',boot)
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Balance-only replication','',summary['verdict'],'',summary['scope'],'','| Seed | Arm | Checkpoint | Group | minADE | minFDE |','|---|---|---|---|---:|---:|']
    for x in rows:
        if x['group'] in ('overall','change'):lines.append(f'| {x["seed"]} | {x["arm"]} | {x["checkpoint"]} | {x["group"]} | {x["minade20"]:.6f} | {x["minfde20"]:.6f} |')
    lines+=['',json.dumps(gates,indent=2),'','Only three correction seeds (two new); uncertainty estimates remain exploratory. No selector improvement or intention understanding claimed.']
    (RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('COMPLETE '+summary['verdict'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');parser.add_argument('--device',default='cuda:0');args=parser.parse_args();device=torch.device(args.device)
    if args.action=='preflight':preflight(device);return
    audit=json.loads((HERE/'preflight.json').read_text())
    for path,h in {**audit['source_hashes'],**audit['preserved_models']}.items():
        if r.p.sha256(Path(path))!=h:raise RuntimeError('Lock mismatch '+path)
    if (RESULTS/'summary.json').exists():raise RuntimeError('Already complete')
    RESULTS.mkdir(exist_ok=True);MODELS.mkdir(exist_ok=True)
    cache=r.p.load_npz(r.p.TRAIN_CACHE);fit,cal=r.p.internal_split(len(cache['sample_id']),cfg_for(1)['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    r.m.loss=r.objective;start=time.time()
    for seed in (2,3):
        cfg=cfg_for(seed);r.setup_weights(cache,fit,values,cfg,device);r.m.MODELS=MODELS/f'seed_{seed}';r.m.MODELS.mkdir(exist_ok=True)
        for arm in ARMS:
            if validate_arm(seed,arm,values):print(f'[SKIP VERIFIED] seed{seed} {arm}',flush=True);continue
            print(f'[TRAIN] seed{seed} {arm} epochs={cfg["epochs"]} K20 unchanged',flush=True);r.m.train(arm,cfg,cache,values,fit,cal,device)
    print('[DEV] lock all12 checkpoints before evaluation',flush=True);evaluate(json.loads((HERE/'protocol.json').read_text()),device)
    assert all(r.p.sha256(Path(path))==h for path,h in audit['preserved_models'].items())
    print(f'Elapsed seconds {time.time()-start:.1f}; original seed1 models unchanged',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        RESULTS.mkdir(exist_ok=True);(RESULTS/'failure.txt').write_text(traceback.format_exc());raise
