"""Generation-only protected balancing, with calibrated constraints and epoch resume."""
from pathlib import Path
import importlib.util
import sys
import json
import argparse
import random
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'GENERATION_BALANCE_REPLICATION_SERIES/run_replication.py'
spec=importlib.util.spec_from_file_location('protected_source',SOURCE)
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
r=h.r;p=r.p
RESULTS=HERE/'results';MODELS=RESULTS/'models'
ARMS=('uniform','balanced','uniform_protected','balanced_protected')


def write_json(path,value):
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,indent=2,default=convert),encoding='utf-8')


def save_atomic(value,path):
    temporary=path.with_suffix('.pending')
    torch.save(value,temporary);temporary.replace(path)


def masks(group):
    return [('overall',np.ones(len(group),bool))]+[(name,group==i) for i,name in enumerate(('stable','transition','change'))]


def grouped(stats,group):
    return {name:{k:float(v[mask].mean()) for k,v in stats.items()} for name,mask in masks(group)}


def select_key(stats,native,group,cfg):
    values=grouped(stats,group);reference=grouped(native,group)
    feasible=all(values[name][k]<=reference[name][k]*(1+cfg['guard_fraction']) for name in ('overall','stable','transition') for k in ('minade20','minfde20'))
    key=sum(values['change'][k]/max(reference['change'][k],1e-6) for k in ('minade20','minfde20'))
    tie=sum(values['overall'][k]/max(reference['overall'][k],1e-6) for k in ('minade20','minfde20'))
    return bool(feasible),(key,tie),values


def protection(output,base,future,groups,scale,cfg):
    errors=(output-future[:,None]).norm(dim=-1)
    with torch.no_grad():reference=(base-future[:,None]).norm(dim=-1)
    ade=errors.mean(-1).min(-1).values;fde=errors[:,:,-1].min(-1).values
    native_ade=reference.mean(-1).min(-1).values;native_fde=reference[:,:,-1].min(-1).values
    harm=F.relu(ade-native_ade*(1+cfg['guard_fraction']))/scale[0]+F.relu(fde-native_fde*(1+cfg['guard_fraction']))/scale[1]
    protected=groups!=2
    return (harm*protected).sum()/protected.sum().clamp_min(1)


def objective(output,base,future,residual,speed,groups,cfg,arm,scale):
    source_arm='native_balanced' if arm.startswith('balanced') else 'native_uniform'
    loss=r.objective(output,base,future,residual,speed,cfg,source_arm)
    if arm.endswith('protected'):loss=loss+cfg['protection_weight']*protection(output,base,future,groups,scale,cfg)
    return loss


def settings(seed):
    cfg=json.loads((r.HERE/'config.json').read_text());cfg.update(json.loads((HERE/'protocol.json').read_text()));cfg['seed']=seed
    return cfg


def diagnostic(device):
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),settings(1)['calibration_fraction'])
    values=r.prepared(cache,False,fit=fit);base,speed,future=values[1:4];native=r.generation_metrics(base[cal],base[cal],speed[cal],future[cal]);rows=[]
    for seed in (1,2,3):
        for arm in h.ARMS:
            for tag in ('best','final'):
                saved=torch.load(h.model_path(seed,arm,tag),map_location='cpu',weights_only=False)
                assert np.array_equal(saved['mean'],values[4]) and np.array_equal(saved['std'],values[5])
                model=r.m.Correction(saved['input_dim'],settings(seed)).to(device);model.load_state_dict(saved['state_dict'])
                paths=r.g.predict(model,*[v[cal] for v in values[:3]],device)
                stats=r.generation_metrics(paths,base[cal],speed[cal],future[cal])
                feasible,key,groups=select_key(stats,native,cache['change_group'][cal],settings(seed))
                for name,metrics in groups.items():rows.append({'seed':seed,'arm':arm,'checkpoint':tag,'group':name,'feasible_under_new_rule':feasible,'epoch':saved['best_epoch'] if tag=='best' else settings(seed)['epochs'],**metrics})
    p.write_csv(HERE/'prior_internal_calibration.csv',rows)
    print('[DIAGNOSTIC] Existing best/final internally evaluated; no dev selection',flush=True)


def preflight(device):
    cfg=settings(1);assert cfg['test_access']=='forbidden' and cfg['seeds']==[1,2,3]
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    assert not(set(cache['sample_id'][fit]) & set(cache['sample_id'][cal]))
    _,weights=r.setup_weights(cache,fit,values,cfg,device)
    scale=np.array([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32).clip(1e-6)
    o,b,s,f=[torch.from_numpy(v[:8]).to(device) for v in values[:4]]
    group=torch.tensor(cache['change_group'][:8],device=device);scale_t=torch.tensor(scale,device=device)
    p.seed_everything(1);model=r.m.Correction(o.shape[1],cfg).to(device);model.eval();out,res=model(o,b,s);assert torch.equal(out,b)
    assert protection(b,b,f,group,scale_t,cfg)==0
    for arm in ARMS:
        model.zero_grad();out,res=model(o,b,s);loss=objective(out,b,f,res,s,group,cfg,arm,scale_t);loss.backward()
        assert torch.isfinite(loss) and model.network[-1].weight.grad.abs().sum()>0
    zeros=torch.zeros(2,20,12,2,device=device);gt=torch.zeros(2,12,2,device=device)
    assert protection(zeros+1,zeros,gt,torch.zeros(2,device=device),scale_t,cfg)>0
    assert protection(zeros+1,zeros,gt,torch.full((2,),2,device=device),scale_t,cfg)==0
    native=r.generation_metrics(values[1][cal],values[1][cal],values[2][cal],values[3][cal]);feasible,_,_=select_key(native,native,cache['change_group'][cal],cfg);assert feasible
    bad={k:v*1.02 if k in ('minade20','minfde20') else v for k,v in native.items()};assert not select_key(bad,native,cache['change_group'][cal],cfg)[0]
    toy={k:np.zeros(len(cal),np.float32) for k in native};assert select_key(toy,native,cache['change_group'][cal],cfg)[0]
    # RNG checkpoint roundtrip test, no future/group is passed to model.forward.
    rng=torch.get_rng_state();expected=torch.rand(4);torch.set_rng_state(rng);assert torch.equal(expected,torch.rand(4))
    diagnostic(device)
    origin=json.loads((r.HERE/'preflight.json').read_text())
    paths=[Path(__file__),HERE/'protocol.json',SOURCE,r.HERE/'config.json']+[Path(x) for x in origin['hashes']]
    previous=json.loads((h.RESULTS/'model_lock.json').read_text())['models']
    for path,digest in previous.items():assert p.sha256(Path(path))==digest
    audit={'status':'PASS','fit_n':len(fit),'cal_n':len(cal),'candidate_count':20,'runs':len(ARMS)*len(cfg['seeds']),
        'fit_protection_scales':scale.tolist(),'weights':weights.tolist(),'source_hashes':{str(x):p.sha256(x) for x in paths},'prior_models':previous,'test_access':False}
    write_json(HERE/'preflight.json',audit);print(json.dumps({k:v for k,v in audit.items() if k not in ('source_hashes','prior_models')},indent=2),flush=True)


def train(seed,arm,cache,values,fit,cal,device):
    cfg=settings(seed);directory=MODELS/f'seed_{seed}'/arm;directory.mkdir(parents=True,exist_ok=True)
    obs,base,speed,future,mean,std=values
    r.setup_weights(cache,fit,values,cfg,device)
    audit=json.loads((HERE/'preflight.json').read_text());scale=torch.tensor(audit['fit_protection_scales'],device=device)
    native=r.generation_metrics(base[cal],base[cal],speed[cal],future[cal])
    p.seed_everything(seed);model=r.m.Correction(obs.shape[1],cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(*[torch.from_numpy(v[fit]) for v in (obs,base,speed,future)],torch.from_numpy(cache['change_group'][fit])),batch_size=cfg['batch_size'],shuffle=True,num_workers=0,generator=generator)
    history=[];first=1;best_key=(float('inf'),float('inf'));best_epoch=0;legacy_key=float('inf');legacy_epoch=0
    identity={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    metadata={'config':cfg,'seed':seed,'arm':arm,'mean':mean,'std':std,'input_dim':obs.shape[1],'dev_used_for_selection':False}
    resume_path=directory/'resume.pth'
    if resume_path.exists():
        saved=torch.load(resume_path,map_location='cpu',weights_only=False);assert saved['metadata']['config']==cfg
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);generator.set_state(saved['loader_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        history=saved['history'];first=saved['epoch']+1;best_key=tuple(saved['best_key']);best_epoch=saved['best_epoch'];legacy_key=saved['legacy_key'];legacy_epoch=saved['legacy_epoch']
        print(f'[RESUME] seed{seed} {arm} next_epoch={first}',flush=True)
    for epoch in range(first,cfg['epochs']+1):
        model.train();losses=[]
        for o,b,s,f,groups in loader:
            o,b,s,f,groups=[v.to(device) for v in (o,b,s,f,groups)];output,residual=model(o,b,s)
            loss=objective(output,b,f,residual,s,groups,cfg,arm,scale)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(loss.item())
        prediction=r.g.predict(model,obs[cal],base[cal],speed[cal],device);stats=r.generation_metrics(prediction,base[cal],speed[cal],future[cal])
        feasible,key,groupstats=select_key(stats,native,cache['change_group'][cal],cfg);legacy=r.g.cal_key(stats,cfg)
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'cal_feasible':feasible,'cal_change_key':key[0],'cal_overall_tie':key[1],'legacy_key':legacy}
        row.update({f'{name}_{k}':v for name,metrics in groupstats.items() for k,v in metrics.items()});history.append(row)
        if feasible and key<best_key:best_key=key;best_epoch=epoch
        if legacy<legacy_key:legacy_key=legacy;legacy_epoch=epoch
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        save_atomic({**metadata,'epoch':epoch,'state_dict':state,'calibration':row},directory/f'epoch_{epoch:03d}.pth')
        save_atomic({'metadata':metadata,'epoch':epoch,'state_dict':state,'optimizer':optimizer.state_dict(),'history':history,'best_key':best_key,'best_epoch':best_epoch,'legacy_key':legacy_key,'legacy_epoch':legacy_epoch,
            'loader_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},resume_path)
        p.write_csv(directory/'history.csv',history)
        if epoch==1 or epoch%5==0:print(f'[seed{seed}/{arm}] epoch{epoch}/{cfg["epochs"]} feasible={feasible} change_key={key[0]:.4f}',flush=True)
    for policy,epoch in [('constrained',best_epoch),('legacy',legacy_epoch),('final',cfg['epochs'])]:
        payload=torch.load(directory/f'epoch_{epoch:03d}.pth',map_location='cpu',weights_only=False) if epoch else {**metadata,'epoch':0,'state_dict':identity,'calibration':None}
        save_atomic({**payload,'selection_policy':policy,'trained_feasible':best_epoch>0,'best_feasible_epoch':best_epoch},directory/f'{policy}.pth')
    write_json(directory/'complete.json',{'status':'COMPLETE','seed':seed,'arm':arm,'best_feasible_epoch':best_epoch,'legacy_epoch':legacy_epoch,'epochs':cfg['epochs'],'model_hashes':{str(directory/f'{tag}.pth'):p.sha256(directory/f'{tag}.pth') for tag in ('constrained','legacy','final')}})


def evaluate(device):
    cfg=settings(1);paths=[MODELS/f'seed_{seed}'/arm/f'{policy}.pth' for seed in cfg['seeds'] for arm in ARMS for policy in ('constrained','legacy','final')]
    assert all(path.exists() for path in paths)
    write_json(RESULTS/'model_lock.json',{'models':{str(path):p.sha256(path) for path in paths},'dev_used_for_selection':False,'test_access':False})
    cache=p.load_npz(p.DEV_CACHE);dim=cache['motion'].shape[1]+cache['social'].shape[1]
    values=r.prepared(cache,False,mean=np.zeros((1,dim),np.float32),std=np.ones((1,dim),np.float32));base,speed,future=values[1:4]
    native=r.generation_metrics(base,base,speed,future);stats={};rows=[];group=cache['change_group']
    for seed in cfg['seeds']:
        for arm in ARMS:
            for policy in ('constrained','legacy','final'):
                path=MODELS/f'seed_{seed}'/arm/f'{policy}.pth';saved=torch.load(path,map_location='cpu',weights_only=False)
                current=r.prepared(cache,False,mean=saved['mean'],std=saved['std']);model=r.m.Correction(dim,settings(seed)).to(device);model.load_state_dict(saved['state_dict'])
                prediction=r.g.predict(model,*current[:3],device);value=r.generation_metrics(prediction,base,speed,future);stats[seed,arm,policy]=value
                np.savez_compressed(RESULTS/f'seed{seed}_{arm}_{policy}.npz',sample_id=cache['sample_id'],trajectory=prediction,**value)
                for name,metrics in grouped(value,group).items():rows.append({'seed':seed,'arm':arm,'policy':policy,'group':name,'epoch':saved['epoch'],'trained_feasible':saved['trained_feasible'],**metrics})
    aggregate=[];boot=[]
    for arm in ARMS:
        for policy in ('constrained','legacy','final'):
            for name,mask in masks(group):aggregate.append({'arm':arm,'policy':policy,'group':name,**{k:float(np.stack([stats[seed,arm,policy][k][mask] for seed in cfg['seeds']]).mean()) for k in native}})
    for policy in ('constrained','legacy','final'):
        for reference in ('native','balanced','uniform_protected','uniform'):
            for name,mask in masks(group):
                for metric in ('minade20','minfde20','top3_ade'):
                    delta=np.stack([stats[seed,'balanced_protected',policy][metric][mask]-(native[metric][mask] if reference=='native' else stats[seed,reference,policy][metric][mask]) for seed in cfg['seeds']])
                    mean,lo,hi=h.hierarchical_delta(delta,cfg['bootstrap_repetitions'],cfg['bootstrap_seed']);boot.append(dict(policy=policy,reference=reference,group=name,metric=metric,delta=mean,ci_low=lo,ci_high=hi))
    lookup={(x['arm'],x['policy'],x['group']):x for x in aggregate};primary=lookup['balanced_protected','constrained','change'];ref=grouped(native,group)
    gates={'change_ade_fde_gain_5pct':all(primary[k]<=ref['change'][k]*.95 for k in ('minade20','minfde20')),
        'protected_groups_guard':all(lookup['balanced_protected','constrained',name][k]<=ref[name][k]*1.01 for name in ('overall','stable','transition') for k in ('minade20','minfde20')),
        'each_seed_change_improves':all(stats[seed,'balanced_protected','constrained'][k][group==2].mean()<native[k][group==2].mean() for seed in cfg['seeds'] for k in ('minade20','minfde20')),
        'native_and_balanced_ci':all(x['ci_high']<0 for x in boot if x['policy']=='constrained' and x['reference'] in ('native','balanced') and x['group']=='change' and x['metric'] in ('minade20','minfde20')),
        'all_seed_feasible':all(x['trained_feasible'] for x in rows if x['arm']=='balanced_protected' and x['policy']=='constrained'),
        'coverage_guard':lookup['balanced_protected','constrained','overall']['maneuver_class_count']>=ref['overall']['maneuver_class_count']*.95 and lookup['balanced_protected','constrained','overall']['gt_family_covered']>=ref['overall']['gt_family_covered']-.01,
        'duplicate_guard':lookup['balanced_protected','constrained','overall']['duplicate_fraction_10cm']<=ref['overall']['duplicate_fraction_10cm']+.02,
        'kinematic_guard':all(lookup['balanced_protected','constrained','overall'][k]<=ref['overall'][k]+.01 for k in ('speed_over3mps_fraction','acceleration_over4mps2_fraction'))}
    p.write_csv(RESULTS/'per_seed_metrics.csv',rows);p.write_csv(RESULTS/'aggregate_metrics.csv',aggregate);p.write_csv(RESULTS/'bootstrap.csv',boot)
    summary={'status':'COMPLETE','verdict':'MAJOR_DEV_SIGNAL_REQUIRES_INDEPENDENT_VALIDATION' if all(gates.values()) else 'PROTECTED_BALANCE_GATE_FAIL','gates':gates,'aggregate':aggregate,'bootstrap':boot,'native':ref,'scope':'ETH reused dev; correction seeds1/2/3; K20; no selector training or test; infeasible trained arms fall back to epoch0 native'}
    write_json(RESULTS/'summary.json',summary)
    lines=['# Protected balance generation experiment','',summary['verdict'],'',summary['scope'],'','| Arm | Policy | Group | minADE | minFDE |','|---|---|---|---:|---:|']
    for row in aggregate:
        if row['group'] in ('overall','change'):lines.append(f'| {row["arm"]} | {row["policy"]} | {row["group"]} | {row["minade20"]:.6f} | {row["minfde20"]:.6f} |')
    lines+=['',json.dumps({k:bool(v) for k,v in gates.items()},indent=2)]
    (RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('COMPLETE '+summary['verdict'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');parser.add_argument('--device',default='cuda:0');args=parser.parse_args();device=torch.device(args.device)
    if args.action=='preflight':preflight(device);return
    audit=json.loads((HERE/'preflight.json').read_text())
    for path,digest in {**audit['source_hashes'],**audit['prior_models']}.items():
        if p.sha256(Path(path))!=digest:raise RuntimeError('Lock mismatch '+path)
    if (RESULTS/'summary.json').exists():raise RuntimeError('Already complete')
    RESULTS.mkdir(exist_ok=True);MODELS.mkdir(exist_ok=True)
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),settings(1)['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    for seed in settings(1)['seeds']:
        for arm in ARMS:
            completed=MODELS/f'seed_{seed}'/arm/'complete.json'
            if completed.exists():
                for path,digest in json.loads(completed.read_text())['model_hashes'].items():assert p.sha256(Path(path))==digest
                print(f'[SKIP VERIFIED] seed{seed}/{arm}',flush=True);continue
            print(f'[TRAIN] seed{seed}/{arm}',flush=True);train(seed,arm,cache,values,fit,cal,device)
    evaluate(device)
    write_json(RESULTS/'exit_status.json',{'status':'SUCCESS','time':time.time()})


if __name__=='__main__':
    try:main()
    except Exception:
        RESULTS.mkdir(exist_ok=True);(RESULTS/'failure.txt').write_text(traceback.format_exc());write_json(RESULTS/'exit_status.json',{'status':'FAILURE','time':time.time()});raise
