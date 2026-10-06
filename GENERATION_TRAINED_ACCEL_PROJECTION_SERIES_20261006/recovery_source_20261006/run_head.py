"""Fresh uniform dual heads; AST-only legacy functions, fixed projection in training."""
from pathlib import Path
from datetime import datetime
from typing import Iterable
import ast, types, json, hashlib, random, csv, os, sys, time, math, traceback, argparse
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from resource_guard import guard
from model_factory import classes,nodes,NATIVE,SOURCES
import projection,physical_accel as physical,route_features as route
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
PARENT=ROOT/'GENERATION_CANDIDATE_ROUTED_MAPPING_SERIES_20261005'
RECOVERY=ROOT/'F16_ENDPOINT_PRESERVING_ACCEL_PROJECTION_DIAGNOSIS_20261005/recovery_numeric_20261006'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x).__name__)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=convert),encoding='utf-8')
    for attempt in range(6):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==5:raise
            time.sleep(.02*2**attempt)
def stage(name,**extra):
    write(HERE/'progress.json',dict(stage=name,at=datetime.now().astimezone().isoformat(),**extra))
    print(name,json.dumps(extra),flush=True)
def verify(files):
    for i,(path,digest) in enumerate(files.items()):
        if i%1000==0:guard.check()
        assert sha(path)==digest,('SOURCE/OLD mismatch',path)
    return len(files)
def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if path.exists():return verify(read(path)['files'])
    assert create
    files=dict(read(HERE.parent/'old_results_manifest.json')['files']);verify(files)
    for folder in ROOT.iterdir():
        if folder==HERE or not folder.is_dir() or not folder.name.startswith(('GENERATION_','F16_','HEAD_','PULSE_','RESEARCH_')):continue
        for source in folder.rglob('*'):
            if not source.is_relative_to(HERE) and source.is_file() and '__pycache__' not in source.parts and source.suffix not in ('.lock','.tmp','.pending') and source.name!='runtime_state.json':
                if str(source) not in files:files[str(source)]=sha(source)
    write(path,dict(files=files,all_prior_digests_preserved=True,controls_excluded=True));return len(files)
def module_ast(name,path,names,extra=None):
    mod=types.ModuleType(name);mod.__dict__.update(dict(Path=Path,np=np,torch=torch,nn=nn,F=F,random=random,csv=csv,json=json,hashlib=hashlib,math=math,time=time,Iterable=Iterable,DataLoader=DataLoader,TensorDataset=TensorDataset))
    if extra:mod.__dict__.update(extra)
    nodes(path,names,mod.__dict__);return mod
p=module_ast('isolated_pilot',ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/directional_regime_pilot.py',['seed_everything','write_csv','load_npz','classify_velocity','internal_split'],dict(CONTINUE=0,DECELERATE=1,STOP=2,ACCELERATE=3,LEFT=4,RIGHT=5,REVERSE=6))
p.sha256=sha
p.TRAIN_CACHE=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
p.DEV_CACHE=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_dev.npz'
RAW_LOAD=p.load_npz
LABEL_CFG=read(ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/config.json')
g=module_ast('isolated_generation',ROOT/'GENERATION_LOCAL_SUPPORT_SERIES/run_generation_support.py',['pair_distance','predict','metrics','cal_key'])
mode=module_ast('isolated_mode',ROOT/'GENERATION_MODE_SUPPORT_SERIES/run_mode_support.py',['labels','extra_metrics'],dict(p=p,LABEL_CFG=LABEL_CFG))
r=module_ast('isolated_direction',ROOT/'GENERATION_DIRECTION_COVERAGE_SERIES/run_direction_coverage.py',['objective','setup_weights','generation_metrics'],dict(m=mode,g=g,p=p,WEIGHTS=None))
# Original score-free sample quantities, joint same-slot and native physics definitions.
old_generation=r.generation_metrics
def generation_metrics(paths,base,speed,future):
    value=old_generation(paths,base,speed,future)
    error=np.linalg.norm(paths-future[:,None],axis=-1);ade=error.mean(-1);fde=error[:,:,-1];rows=np.arange(len(paths))
    ai=ade.argmin(-1);fi=fde.argmin(-1)
    value.update(joint_oracle=(ade+.5*fde).min(-1),fde_at_ade_oracle=fde[rows,ai],ade_at_fde_oracle=ade[rows,fi],oracle_slot_disagreement=(ai!=fi).astype(np.float32))
    velocity=np.diff(paths,axis=2,prepend=np.zeros_like(paths[:,:,:1]))/.4;acc=np.diff(velocity,axis=2)/.4;jerk=np.diff(acc,axis=2)/.4
    for name,array in [('speed',velocity),('acceleration',acc),('jerk',jerk)]:
        mag=np.linalg.norm(array,axis=-1).reshape(len(paths),-1)
        value[name+'_audit_mean']=mag.mean(1);value[name+'_audit_p95']=np.quantile(mag,.95,axis=1)
    return value
r.generation_metrics=generation_metrics
BASE_CONFIG=torch.load(PARENT/'results/models/zero/seed_1/balanced_protected/constrained.pth',map_location='cpu',weights_only=False)['config']
def settings(seed,arm='mean_std'):
    cfg=dict(BASE_CONFIG);cfg.update(protocol='uniform_dual_training_projection_v1',seed=seed,set_context=arm)
    assert cfg['epochs']==80 and cfg['seeds']==[1,2,3] and cfg['velocity_residual_scale']==1.5 and cfg['guard_fraction']==.01
    return cfg
def load_cache(path):
    cache=RAW_LOAD(path);split='train' if Path(path)==p.TRAIN_CACHE else 'dev'
    assert Path(path) in (p.TRAIN_CACHE,p.DEV_CACHE)
    encoded=RAW_LOAD(ROOT/f'GENERATION_NATIVE_DECODER_SERIES/{split}_encoder.npz')
    label=RAW_LOAD(ROOT/f'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/{split}.npz')
    assert np.array_equal(cache['sample_id'],encoded['sample_id']) and np.array_equal(cache['sample_id'],label['sample_id'])
    cache['_past']=encoded['past'];cache['_tokens']=encoded['tokens'];cache['_future']=label['future'];return cache
p.load_npz=load_cache
def prepared(cache,replace=False,**kwargs):
    assert not replace
    n=len(cache['sample_id']);obs=np.zeros((n,940),np.float32)
    obs[:,:16]=cache['_past'].reshape(n,16);obs[:,16:784]=cache['_tokens'].reshape(n,768)
    base=cache['candidate'][...,:24].reshape(n,20,12,2);future=cache['_future']
    past=cache['motion'][:,:16].reshape(n,8,2);v=past[:,-1]-past[:,-2]
    speed=np.concatenate((np.linalg.norm(v,axis=-1)[:,None],v),1).astype(np.float32)
    assert all(x.dtype==np.float32 for x in (obs,base,future,speed))
    err=np.linalg.norm(base-future[:,None],axis=-1)
    assert np.allclose(err.mean(-1),cache['ade'],atol=2e-5,rtol=0) and np.allclose(err[:,:,-1],cache['fde'],atol=2e-5,rtol=0)
    return [obs,base,speed,future,np.zeros((1,940),np.float32),np.ones((1,940),np.float32)]
r.prepared=prepared
def prepared_train(device):
    guard.check();cache=load_cache(p.TRAIN_CACHE);cfg=settings(1)
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert (len(fit),len(cal))==(25912,2879) and len(np.unique(cache['sample_id']))==28791
    values=prepared(cache);r.setup_weights(cache,fit,values,cfg,device)
    for path,key in [('state_scalers.npz','fit_id'),('physical_accel_scales.npz','fit_ids')]:
        with np.load(HERE/path,allow_pickle=False) as z:assert np.array_equal(z[key],cache['sample_id'][fit])
    for name in ('state_scalers.npz','physical_accel_scales.npz','route_threshold.npz'):assert sha(HERE/name)==sha(PARENT/name)
    guard.check();return cache,cfg,fit,cal,values
BasePhysical,DECODER_STATE=classes()
class ProjectionHead(BasePhysical):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,dict(cfg,set_context='zero'))
        self.projection_arm=cfg['set_context']
    def forward(self,observation,base,speed):
        guard.check()
        if bool(self.native_fallback):return base.double(),torch.zeros_like(base,dtype=torch.float64)
        raw,part=self.routed_components(observation,base,speed)
        if self.projection_arm=='mean_std':output,residual,_=projection.project(base,raw,part['limit'])
        else:residual=raw.double();output=base.double()+residual.cumsum(2)
        return output,residual
TimeResponseHead=ProjectionHead;PhysicalHead=ProjectionHead
mode.Correction=ProjectionHead
def frozen_audit(model):
    assert sum(v.numel() for v in model.parameters() if v.requires_grad)==91596
    assert model.experts[0].decoder is model.experts[1].decoder
    assert not any(v.requires_grad for v in model.decoder.parameters())
    for name,val in model.decoder.state_dict().items():assert torch.equal(val.detach().cpu(),DECODER_STATE[name]),name
m=module_ast('isolated_protected',ROOT/'GENERATION_PROTECTED_BALANCE_SERIES/run_protected.py',['write_json','save_atomic','masks','grouped','select_key','protection','objective','train'],dict(r=r,p=p,HERE=HERE,RESULTS=HERE/'results',MODELS=HERE/'results/models',settings=lambda seed:settings(seed)))
m.write_json=write
original_objective=m.objective;original_train=m.train;original_save=m.save_atomic
_TRACE=[];_ROWS=[];_LOGGING=False
def objective(output,base,future,residual,speed,groups,cfg,arm,scales):
    old=original_objective(output,base,future,residual,speed,groups,cfg,arm,scales)
    extra,detail=physical.penalty(output,base,groups,torch.as_tensor(physical.load(HERE),device=output.device,dtype=output.dtype),True)
    total=old+.1*extra
    if _LOGGING:
        derivative=None
        if not _TRACE:derivative=float(torch.autograd.grad(extra,output,retain_graph=True)[0].norm().detach())
        _TRACE.append(dict(n=len(groups),original=float(old.detach()),P=float(extra.detach()),added=float((.1*extra).detach()),total=float(total.detach()),coordinate_P_gradient_norm=derivative,**detail))
    return total
m.objective=objective
# Reuse the exact objective-history/resume bookkeeping without importing its module.
nodes(PARENT/'run_head.py',['trace_summary'],globals())
def save_atomic(value,path):
    if _LOGGING and path.name.startswith('epoch_') and _TRACE:value={**value,'objective_diagnostic':trace_summary(value['epoch'])}
    if _LOGGING and path.name=='resume.pth':
        _ROWS.append(trace_summary(value['epoch']));_TRACE.clear();value={**value,'objective_diagnostics':list(_ROWS)}
        write(path.parent/'objective_history.json',dict(status='RECORDED',rows=_ROWS))
    if path.name=='constrained.pth' and value.get('epoch')==0:
        value={**value,'state_dict':dict(value['state_dict'])};value['state_dict']['native_fallback']=torch.tensor(True)
    original_save(value,path)
m.save_atomic=save_atomic
def train_original(seed,arm,cache,values,fit,cal,device):
    global _LOGGING,_ROWS
    _TRACE.clear();_ROWS=[];resume=m.MODELS/f'seed_{seed}'/arm/'resume.pth'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False);_ROWS=list(saved['objective_diagnostics']);assert len(_ROWS)==saved['epoch'];del saved
    _LOGGING=True
    try:return original_train(seed,arm,cache,values,fit,cal,device)
    finally:_LOGGING=False;_TRACE.clear()
m.train=train_original
def train(seed,arm,cache,values,fit,cal,device):
    previous=m.settings;m.MODELS=HERE/'results/models'/arm;m.settings=lambda seed:settings(seed,arm)
    try:return m.train(seed,'balanced_protected',cache,values,fit,cal,device)
    finally:m.settings=previous;m.MODELS=HERE/'results/models'
def checkpoint(seed,arm,tag):return HERE/f'results/models/{arm}/seed_{seed}/balanced_protected/{tag}.pth'
boot=module_ast('isolated_bootstrap',ROOT/'GENERATION_BALANCE_REPLICATION_SERIES/run_replication.py',['hierarchical_delta'])
m.h=boot
c=module_ast('isolated_context',ROOT/'GENERATION_SET_CONTEXT_SERIES/run_context.py',['evaluate'],dict(HERE=HERE,m=m,r=r,p=p,settings=settings,checkpoint=checkpoint,Correction=ProjectionHead,ARMS=('zero','mean_std'),POLICIES=('constrained','legacy','final'),d=types.SimpleNamespace(ARM='balanced_protected')))
c.settings=settings;c.checkpoint=checkpoint;c.ARMS=('zero','mean_std');c.POLICIES=('constrained','legacy','final')
n=types.SimpleNamespace(RAW_LOAD=RAW_LOAD,DECODER_STATE=DECODER_STATE)
t=types.SimpleNamespace(TimeResponseHead=ProjectionHead)
def source_hashes():
    inherited=dict(read(RECOVERY/'real_preflight.json')['source_hashes']);verify(inherited)
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/x for x in ('README.md','DESIGN.md','protocol.json','old_results_manifest.json','state_scalers.npz','physical_accel_scales.npz','route_threshold.npz')]
    return {**inherited,**{str(path):sha(path) for path in paths if path.is_file()}}
def temporal_projection_outputs(device):
    cache=load_cache(p.DEV_CACHE);values=prepared(cache);gt=values[3];ids=cache['sample_id']
    directory=m.RESULTS/'temporal';directory.mkdir(exist_ok=True)
    projdir=m.RESULTS/'projection';projdir.mkdir(exist_ok=True)
    for arm in c.ARMS:
        for seed in (1,2,3):
            for policy in c.POLICIES:
                saved=torch.load(checkpoint(seed,arm,policy),map_location='cpu',weights_only=False)
                model=ProjectionHead(940,saved['config']).to(device).eval();model.load_state_dict(saved['state_dict']);frozen_audit(model)
                prediction=RAW_LOAD(m.RESULTS/f'seed{seed}_{arm}_{policy}.npz')['trajectory']
                err=np.linalg.norm(prediction-gt[:,None],axis=-1);slot=(err.mean(-1)+.5*err[:,:,-1]).argmin(-1);chosen=err[np.arange(len(ids)),slot]
                np.savez_compressed(directory/f'seed{seed}_{arm}_{policy}_temporal.npz',sample_id=ids,selected_samepath_index=slot,early=chosen[:,:4].mean(-1),middle=chosen[:,4:8].mean(-1),late=chosen[:,8:].mean(-1))
                pieces={}
                with torch.no_grad():
                    for start in range(0,len(ids),128):
                        o,b,s=[torch.from_numpy(v[start:start+128]).to(device) for v in values[:3]]
                        raw,parts=model.routed_components(o,b,s)
                        if bool(model.native_fallback):raw=torch.zeros_like(raw)
                        out,up,info=projection.project(b,raw,parts['limit'])
                        actual=b.double()+raw.double().cumsum(2) if arm=='zero' else out
                        assert np.allclose(actual.cpu().numpy(),prediction[start:start+128],atol=2e-5,rtol=0)
                        arrays=dict(base=b,raw=raw,projected=out,up=up,limit=parts['limit'],**info)
                        for key,v in arrays.items():pieces.setdefault(key,[]).append(v.cpu().numpy())
                np.savez_compressed(projdir/f'seed{seed}_{arm}_{policy}.npz',sample_id=ids,**{k:np.concatenate(v) for k,v in pieces.items()})
                del model,saved
def evaluate(device):
    c.evaluate(device);temporal_projection_outputs(device)
    summary=read(m.RESULTS/'summary.json');summary['scope']=read(HERE/'protocol.json')
    write(m.RESULTS/'summary.json',summary)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');parser.add_argument('--device',default='cuda:0');args=parser.parse_args();device=torch.device(args.device)
    m.RESULTS.mkdir(exist_ok=True)
    if args.action=='preflight':
        import preflight_impl;preflight_impl.run(sys.modules[__name__],device);return
    audit=read(HERE/'preflight.json');assert audit['status']=='PASS';verify(audit['source_hashes']);old_manifest()
    if (m.RESULTS/'exit_status.json').exists() and read(m.RESULTS/'exit_status.json')['status']=='SUCCESS':raise RuntimeError('Already complete')
    cache,cfg,fit,cal,values=prepared_train(device)
    for arm in c.ARMS:
        for seed in (1,2,3):
            complete=checkpoint(seed,arm,'final').parent/'complete.json'
            if complete.exists():verify(read(complete)['model_hashes']);print('[SKIP VERIFIED]',arm,seed,flush=True);continue
            stage('TRAIN',arm=arm,seed=seed);train(seed,arm,cache,values,fit,cal,device)
    stage('EVALUATE');evaluate(device)
    write(m.RESULTS/'exit_status.json',dict(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat()))
if __name__=='__main__':main()
