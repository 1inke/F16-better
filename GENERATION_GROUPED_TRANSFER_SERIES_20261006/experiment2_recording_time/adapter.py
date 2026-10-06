"""Frozen AST-isolated old model/loss; approved FIT-only recording-time contrast."""
from pathlib import Path
from datetime import datetime
from typing import Iterable
import ast,types,json,hashlib,random,csv,os,sys,time,math
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from resource_guard import guard
HERE=Path(__file__).resolve().parent;SERIES=HERE.parent;ROOT=SERIES.parent
OLD=ROOT/'GENERATION_TRAINED_ACCEL_PROJECTION_SERIES_20261006/recovery_source_20261006'
ROUTED=ROOT/'GENERATION_CANDIDATE_ROUTED_MAPPING_SERIES_20261005'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x).__name__)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=convert),encoding='utf-8');os.replace(tmp,path)
def stage(name,**kw):
    write(HERE/'progress.json',dict(stage=name,at=datetime.now().astimezone().isoformat(),**kw));print(name,json.dumps(kw),flush=True)
def verify(files):
    for i,(path,digest) in enumerate(files.items()):
        if i%1000==0:guard.check()
        assert sha(path)==digest,('SOURCE/OUTPUT mismatch',path)
    return len(files)
def nodes(path,names,ns):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    selected=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    assert len(selected)==len(names),(path,names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
def module(name,path,names,extra=None):
    mod=types.ModuleType(name)
    mod.__dict__.update(dict(Path=Path,np=np,torch=torch,nn=nn,Tensor=Tensor,F=F,random=random,csv=csv,json=json,hashlib=hashlib,math=math,time=time,Iterable=Iterable,DataLoader=DataLoader,TensorDataset=TensorDataset))
    if extra:mod.__dict__.update(extra)
    nodes(path,names,mod.__dict__);return mod
# Execute only verified constructor assignments/functions, not legacy module imports/main.
factory=types.ModuleType('frozen_constructor')
factory.__dict__.update(dict(Path=Path,ast=ast,types=types,np=np,torch=torch,nn=nn,Tensor=Tensor,F=F,math=math,guard=guard,ROOT=ROOT,__file__=str(HERE/'adapter.py')))
tree=ast.parse((OLD/'model_factory.py').read_text(encoding='utf-8'))
selected=[x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('FAMILIES','SOURCES','NATIVE') for t in x.targets)]
exec(compile(ast.Module(body=selected,type_ignores=[]),str(OLD/'model_factory.py'),'exec'),factory.__dict__)
nodes(OLD/'model_factory.py',['nodes','classes'],factory.__dict__)
physical=module('physical',ROUTED/'physical_accel.py',['window_accel','numpy_accel','penalty','stats','load','fixtures'])
route=module('route',ROUTED/'route_features.py',['load','statistic'])
state=module('state',ROOT/'GENERATION_CANDIDATE_STATE_SERIES_20261005/state_features.py',['state32','load_scalers','validate_state_math'])
projection=module('projection',OLD/'projection.py',['physical_mean','project'],dict(DT=.4,UNIT=np.finfo(np.float64).eps/2,GAMMA512=(512*(np.finfo(np.float64).eps/2))/(1-512*(np.finfo(np.float64).eps/2))))
p=module('pilot',ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/directional_regime_pilot.py',['seed_everything','write_csv','load_npz','classify_velocity'],dict(CONTINUE=0,DECELERATE=1,STOP=2,ACCELERATE=3,LEFT=4,RIGHT=5,REVERSE=6))
p.sha256=sha;RAW_LOAD=p.load_npz
g=module('generation',ROOT/'GENERATION_LOCAL_SUPPORT_SERIES/run_generation_support.py',['pair_distance','metrics','cal_key'])
LABEL_CFG=read(ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/config.json')
mode=module('mode',ROOT/'GENERATION_MODE_SUPPORT_SERIES/run_mode_support.py',['labels','extra_metrics'],dict(p=p,LABEL_CFG=LABEL_CFG))
r=module('direction',ROOT/'GENERATION_DIRECTION_COVERAGE_SERIES/run_direction_coverage.py',['objective','setup_weights','generation_metrics'],dict(m=mode,g=g,p=p,WEIGHTS=None))
def predict(model,obs,base,speed,device):
    model.eval();pieces=[]
    with torch.inference_mode():
        for start in range(0,len(obs),128):
            guard.check();o,b,s=[torch.from_numpy(v[start:start+128]).to(device) for v in (obs,base,speed)]
            output,_=model(o,b,s);pieces.append(output.cpu().numpy())
    return np.concatenate(pieces)
g.predict=predict
BASE_CONFIG=torch.load(OLD/'results/models/mean_std/seed_1/balanced_protected/final.pth',map_location='cpu',weights_only=False)['config']
def settings(seed,adaptation_arm='grouped'):
    cfg=dict(BASE_CONFIG);cfg.update(protocol='matched_recording_time_transfer_v1',seed=seed,set_context='zero',adaptation_arm=adaptation_arm)
    assert cfg['epochs']==80 and cfg['seeds']==[1,2,3] and cfg['velocity_residual_scale']==1.5 and cfg['guard_fraction']==.01
    return cfg
def load_data():
    guard.check();cache=RAW_LOAD(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz')
    enc=RAW_LOAD(ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz')
    lab=RAW_LOAD(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz')
    assert np.array_equal(cache['sample_id'],enc['sample_id']) and np.array_equal(cache['sample_id'],lab['sample_id'])
    n=25912
    # No DEV file access. Original CAL is not used for training/statistics/selection.
    cache={k:v[:n].copy() for k,v in cache.items()};past=enc['past'][:n];tokens=enc['tokens'][:n];future=lab['future'][:n].copy()
    obs=np.zeros((n,940),np.float32);obs[:,:16]=past.reshape(n,16);obs[:,16:784]=tokens.reshape(n,768)
    base=cache['candidate'][...,:24].reshape(n,20,12,2).copy()
    v=np.diff(cache['motion'][:,:16].reshape(n,8,2)[:,-2:],axis=1)[:,0]
    speed=np.c_[np.linalg.norm(v,axis=-1),v].astype(np.float32)
    error=np.linalg.norm(base-future[:,None],axis=-1)
    assert np.allclose(error.mean(-1),cache['ade'],atol=2e-5,rtol=0) and np.allclose(error[:,:,-1],cache['fde'],atol=2e-5,rtol=0)
    with np.load(HERE/'identity_audit/splits.npz',allow_pickle=False) as z:splits={k:z[k].copy() for k in z.files}
    assert np.array_equal(cache['sample_id'],splits['sample_id'])
    return cache,[obs,base,speed,future,np.zeros((1,940),np.float32),np.ones((1,940),np.float32)],splits
def setup_weights(cache,fit,values,cfg,device):
    with np.load(HERE/'common_statistics.npz',allow_pickle=False) as z:
        weights=z['quality_weights'].copy();counts=z['quality_counts'].copy()
    r.WEIGHTS=torch.tensor(weights,device=device,dtype=torch.float32);return counts,weights
r.setup_weights=setup_weights
def build_class():
    return factory.classes()
_BASE=None;DECODER_STATE=None
def initialize_classes():
    global _BASE,DECODER_STATE
    if _BASE is None:_BASE,DECODER_STATE=build_class()
    class ProjectedHead(_BASE):
        def __init__(self,dimension,cfg):super().__init__(dimension,dict(cfg,set_context='zero'))
        def forward(self,observation,base,speed):
            guard.check()
            if bool(self.native_fallback):return base.double(),torch.zeros_like(base,dtype=torch.float64)
            raw,part=self.routed_components(observation,base,speed)
            output,residual,_=projection.project(base,raw,part['limit']);return output,residual
    mode.Correction=ProjectedHead
    return ProjectedHead
def frozen_audit(model):
    assert sum(x.numel() for x in model.parameters() if x.requires_grad)==91596
    assert model.experts[0].decoder is model.experts[1].decoder
    assert not any(x.requires_grad or x.grad is not None for x in model.decoder.parameters())
    for name,val in model.decoder.state_dict().items():assert torch.equal(val.detach().cpu(),DECODER_STATE[name]),name
m=module('protected',ROOT/'GENERATION_PROTECTED_BALANCE_SERIES/run_protected.py',['write_json','save_atomic','masks','grouped','select_key','protection','objective','train'],dict(r=r,p=p,HERE=HERE,RESULTS=HERE/'results',MODELS=HERE/'results/models',settings=settings))
m.write_json=write
original_objective=m.objective;original_train=m.train;original_save=m.save_atomic
_TRACE=[];_ROWS=[];_LOGGING=False
nodes(ROUTED/'run_head.py',['trace_summary'],globals())
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
def save_atomic(value,path):
    if _LOGGING and path.name.startswith('epoch_') and _TRACE:value={**value,'objective_diagnostic':trace_summary(value['epoch'])}
    if _LOGGING and path.name=='resume.pth':
        _ROWS.append(trace_summary(value['epoch']));_TRACE.clear();value={**value,'objective_diagnostics':list(_ROWS)}
        write(path.parent/'objective_history.json',dict(status='RECORDED',rows=_ROWS))
    if path.name=='constrained.pth' and value.get('epoch')==0:
        value={**value,'state_dict':dict(value['state_dict'])};value['state_dict']['native_fallback']=torch.tensor(True)
    original_save(value,path)
m.save_atomic=save_atomic
def train(seed,adaptation_arm,cache,values,fit,cal,device):
    global _ROWS,_LOGGING
    prior=m.settings;directory=m.MODELS/f'seed_{seed}/balanced_protected';resume=directory/'resume.pth'
    _TRACE.clear();_ROWS=[]
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False);_ROWS=list(saved['objective_diagnostics']);assert len(_ROWS)==saved['epoch']
    m.settings=lambda seed:settings(seed,adaptation_arm);_LOGGING=True
    try:return original_train(seed,'balanced_protected',cache,values,fit,cal,device)
    finally:m.settings=prior;_LOGGING=False;_TRACE.clear()
def source_hashes():
    inherited=dict(read(SERIES/'experiment1_target_track/preflight.json')['source_hashes']);verify(inherited)
    inherited.update(read(HERE/'identity_audit/source_lock.json')['files'])
    extra=list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/'identity_audit/splits.npz',HERE/'identity_audit/RESULT.json',HERE/'old_results_manifest.json']
    extra += [HERE/x for x in ('state_scalers.npz','physical_accel_scales.npz','route_threshold.npz','common_statistics.npz','protocol.json','DESIGN.md')]
    return {**inherited,**{str(x):sha(x) for x in extra if x.is_file()}}
