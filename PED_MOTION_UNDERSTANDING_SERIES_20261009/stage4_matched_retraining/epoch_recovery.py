"""Epoch-boundary recovery for the isolated original H3/V2 training lifecycle.

No batch-restart claim: verified committed epoch boundaries restore exact
sampler/augmentation RNG. Conflicting partial selected files stop for review.
"""
from __future__ import annotations
import copy
import ctypes
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import psutil
import torch

class InterruptedForFixture(RuntimeError):
    pass

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def verify_binding(path):
    binding=json.loads(Path(path).read_text(encoding='utf-8'))
    assert binding['files'], 'Empty source/input binding'
    for name,expected in binding['files'].items():
        if digest(name)!=expected:raise RuntimeError('STOP_SOURCE: '+name)
    return digest(path)

def guard():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError('GlobalMemoryStatusEx')
    p=psutil.Process().memory_info();private=getattr(p,'private',p.rss)
    if private>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:
        raise RuntimeError('STOP_RESOURCE: process/system physical+commit reserve')
    return {'private_GiB':private/2**30,'RSS_GiB':p.rss/2**30,'physical_available_GiB':m.avail_phys/2**30,'commit_available_GiB':m.avail_page/2**30}

def atomic_torch(state,path):
    path=Path(path);temp=path.with_suffix(path.suffix+'.writing')
    torch.save(state,temp);os.replace(temp,path)

def atomic_json(state,path):
    path=Path(path);temp=path.with_suffix(path.suffix+'.writing')
    temp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)

def rng_state(generator):
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'loader':generator.get_state(),'numpy':np.random.get_state(),'python':random.getstate()}

def restore_rng(state,generator):
    torch.set_rng_state(state['torch'].cpu());generator.set_state(state['loader'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    np.random.set_state(state['numpy']);random.setstate(state['python'])

class EpochRecovery:
    def __init__(self,root,branch,binding,resume=False,stop_after=None,fixture=False):
        self.root=Path(root).resolve();self.branch=branch;self.binding=Path(binding).resolve()
        self.binding_sha=verify_binding(binding);self.resume=resume;self.stop_after=stop_after;self.fixture=fixture
        self.resources=[];self.gradient_reports=[];self.contract=None

    def observe(self,scope):
        self.resources.append(guard())
        model=scope['model'];loss=scope['loss']
        if not torch.isfinite(loss):raise RuntimeError('STOP_NUMERIC: loss')
        finite=True;nonzero=[]
        for name,p in model.named_parameters():
            if p.grad is None:continue
            if not p.requires_grad:raise RuntimeError('Frozen parameter received gradient: '+name)
            finite=finite and bool(torch.isfinite(p.grad).all())
            if bool((p.grad!=0).any()):nonzero.append(name)
        if not finite or not nonzero:raise RuntimeError('STOP_GRADIENT: nonfinite/empty gradient')
        if len(self.gradient_reports)<8:self.gradient_reports.append({'finite':finite,'nonzero_parameters':nonzero})

    def begin(self,s):
        self.resources.append(guard())
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(s['args']).items() if k not in ('output_root','force')}
        self.contract={'branch':self.branch,'arguments':args,'source_input_binding_sha':self.binding_sha,'backend':s['contract'],
            'train_indices_sha':hashlib.sha256(np.asarray(s['train_indices'],dtype=np.int64).tobytes()).hexdigest(),
            'val_indices_sha':hashlib.sha256(np.asarray(s['val_indices'],dtype=np.int64).tobytes()).hexdigest(),
            'stage_input_H3_sha':digest(os.environ['EXP143_SOURCE_CHECKPOINT']) if s['args'].decoder_experiment=='v2_wide_endpoint_correction' else None,
            'preflight_subset_only':self.fixture}
        self.contract_sha=hashlib.sha256(json.dumps(self.contract,sort_keys=True).encode()).hexdigest()
        dest=s['run_dir'];self.dest=dest;self.checkpoint=dest/'epoch_resume.pt'
        manifest=dest/'run_manifest.json'
        if self.resume:
            state=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
            if state['contract_sha']!=self.contract_sha:raise RuntimeError('STOP_RESUME_SOURCE_OR_PROTOCOL')
            for name,sha in state['selected_checkpoint_SHA'].items():
                if digest(dest/name)!=sha:raise RuntimeError('STOP_RESUME_SELECTED_OUTPUT: '+name)
            s['model'].load_state_dict(state['model'],strict=True)
            if s['ema'] is not None:s['ema'].module.load_state_dict(state['ema'],strict=True)
            s['optimizer'].load_state_dict(state['optimizer'])
            if s['scheduler'] is not None:s['scheduler'].load_state_dict(state['scheduler'])
            restore_rng(state['RNG'],s['generator'])
            self.gradient_reports=state['gradient_reports'];self.resources=state['resources']+self.resources
            return state['next_epoch'],state['history'],state['best_ade'],state['best_checkpoint_score']
        if self.checkpoint.exists() or manifest.exists():raise RuntimeError('Refusing unapproved overwrite; use compatible resume')
        atomic_json(self.contract|{'contract_sha':self.contract_sha},manifest)
        if self.fixture:
            atomic_torch({k:v.detach().cpu().clone() for k,v in s['model'].state_dict().items()},dest/'initial_state.pt')
        return 0,s['history'],s['best_ade'],s['best_checkpoint_score']

    def commit(self,s):
        self.resources.append(guard())
        selected={name:digest(s['run_dir']/name) for name in ('best_original_ade.pth','best_selected.pth') if (s['run_dir']/name).exists()}
        state={'contract_sha':self.contract_sha,'next_epoch':s['epoch']+1,'model':s['model'].state_dict(),
            'ema':s['ema'].module.state_dict() if s['ema'] is not None else None,
            'optimizer':s['optimizer'].state_dict(),'scheduler':s['scheduler'].state_dict() if s['scheduler'] is not None else None,
            'RNG':rng_state(s['generator']),'history':copy.deepcopy(s['history']),'best_ade':s['best_ade'],
            'best_checkpoint_score':s['best_checkpoint_score'],'selected_checkpoint_SHA':selected,
            'gradient_reports':self.gradient_reports,'resources':self.resources}
        atomic_torch(state,self.checkpoint)
        atomic_json({'epoch':s['epoch']+1,'decoder_stage':s['args'].decoder_experiment,'branch':self.branch,'status':'TRAINING_EPOCH_COMMITTED','preflight_subset_only':self.fixture},self.root/'progress.json')
        if self.stop_after is not None and s['epoch']+1==self.stop_after:
            raise InterruptedForFixture('Intentional epoch-boundary interruption, not production failure')

    def finish(self,s):
        verify_binding(self.binding)
        paths=[s['run_dir']/x for x in ('metrics.json','history.csv','epoch_resume.pt','best_selected.pth','final_model.pth','run_manifest.json')]
        for x in paths:
            if not x.is_file():raise RuntimeError('Missing completed output '+str(x))
        if len(s['history'])!=s['args'].epochs:raise RuntimeError('Incomplete training history')
        atomic_json({'status':'SUCCESS','contract_sha':self.contract_sha,'outputs':{str(x):digest(x) for x in paths},
            'epochs':len(s['history']),'preflight_subset_only':self.fixture,'resources':self.resources,
            'gradient_reports':self.gradient_reports},s['run_dir']/'done.json')

    def skip_compatible(self,scope):
        dest=scope['run_dir'];done=json.loads((dest/'done.json').read_text(encoding='utf-8'))
        manifest=json.loads((dest/'run_manifest.json').read_text(encoding='utf-8'))
        if done['status']!='SUCCESS' or manifest['source_input_binding_sha']!=self.binding_sha or manifest['branch']!=self.branch:
            raise RuntimeError('STOP_COMPLETED_SOURCE_BINDING')
        expected={k:str(v) if isinstance(v,Path) else v for k,v in vars(scope['args']).items() if k not in ('output_root','force')}
        if manifest['arguments']!=expected:raise RuntimeError('STOP_COMPLETED_PROTOCOL')
        if manifest['stage_input_H3_sha'] is not None and manifest['stage_input_H3_sha']!=digest(os.environ['EXP143_SOURCE_CHECKPOINT']):
            raise RuntimeError('STOP_COMPLETED_H3_BINDING')
        for path,sha in done['outputs'].items():
            if digest(path)!=sha:raise RuntimeError('STOP_COMPLETED_OUTPUT_SHA')
