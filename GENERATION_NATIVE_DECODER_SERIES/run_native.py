"""Paired native V2-refiner-only versus whole native generation decoder training.

Frozen encoder input cache is verified against the existing ETH native paths.
Generic zero/mean_std reporter labels mean refiner_only/full_generation here.
"""
from pathlib import Path
import importlib.util
import sys
import json
import time
import copy
import traceback
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value
c=module('native_context_source',ROOT/'GENERATION_SET_CONTEXT_SERIES/run_context.py')
f=module('native_formal_source',ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/run_formal_regime_selection.py')
m=c.m;r=c.r;p=c.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
RAW_LOAD=p.load_npz
RAW_PREPARED=r.prepared
DECODER_STATE=None
DECODER_CLASS=None
TOKEN_DIM=None

def prepare_backend(device):
    global DECODER_STATE,DECODER_CLASS,TOKEN_DIM
    cfg,regime,front,trainer,adapter=f.setup_backend(device)
    model=f.build_baseline('eth',cfg,front,adapter,device)
    DECODER_STATE={k:v.detach().cpu().clone() for k,v in model.decoder.state_dict().items()}
    DECODER_CLASS=adapter.JointTrainableV2Decoder;TOKEN_DIM=model.token_dim
    return cfg,trainer,model

def extract(device):
    cfg,trainer,model=prepare_backend(device)
    dataset=trainer.TrajectoryDataset(*f.fold_data('eth','train'),cfg)
    captured=[]
    def hook(owner,args):captured.append(args[0][:,0].detach().cpu().numpy().astype(np.float32))
    handle=model.decoder.register_forward_pre_hook(hook)
    audit={}
    for split,path in [('train',p.TRAIN_CACHE),('dev',p.DEV_CACHE)]:
        cache=RAW_LOAD(path);destination=HERE/f'{split}_encoder.npz'
        if destination.exists():
            saved=RAW_LOAD(destination);assert np.array_equal(saved['sample_id'],cache['sample_id'])
            prior=json.loads((HERE/'encoder_audit.json').read_text())[split]
            assert prior['max_native_alignment_error']<2e-5
            audit[split]={**prior,'reused':True};continue
        parts=[];captured.clear();maximum=0.;offset=0
        loader=DataLoader(Subset(dataset,cache['sample_id'].tolist()),batch_size=128,shuffle=False,num_workers=0)
        with torch.inference_mode():
            for past,future,num_valid,*_ in loader:
                # Encoder sees observation only. Future is not passed to normalization or forward.
                past=past.to(device);local,_=trainer.normalize_to_focal_heading(past,past)
                center=local.mean(-2,keepdim=True)
                pred=model(local-center,None,None,mode='testing',num_valid=num_valid.to(device).int(),graph_loc=local)[2]
                relative=(pred[:,0]+center[:,0,None]-local[:,0,-1,None,None]).cpu().numpy()
                reference=cache['candidate'][offset:offset+len(past),:,:24].reshape(-1,20,12,2)
                maximum=max(maximum,float(np.abs(relative-reference).max()))
                assert maximum<2e-5,('encoder native alignment',split,maximum)
                parts.append(local[:,0].cpu().numpy().astype(np.float32));offset+=len(past)
        assert offset==len(cache['sample_id'])
        np.savez_compressed(destination,sample_id=cache['sample_id'],tokens=np.concatenate(captured),past=np.concatenate(parts))
        audit[split]=dict(n=offset,max_native_alignment_error=maximum,reused=False)
        print('[ENCODER CACHE] '+split+' '+json.dumps(audit[split]),flush=True)
    handle.remove();del model
    m.write_json(HERE/'encoder_audit.json',audit)

def load_cache(path):
    cache=RAW_LOAD(path)
    if Path(path) not in (p.TRAIN_CACHE,p.DEV_CACHE):return cache
    split='train' if Path(path)==p.TRAIN_CACHE else 'dev'
    encoded=RAW_LOAD(HERE/f'{split}_encoder.npz')
    assert np.array_equal(encoded['sample_id'],cache['sample_id'])
    # Keep original motion for raw-future/history alignment; add exact decoder tokens.
    cache['social']=np.concatenate((cache['social'],encoded['tokens'].reshape(len(cache['sample_id']),-1)),1)
    cache['_tokens']=encoded['tokens'];cache['_past']=encoded['past']
    return cache

def prepared(cache,replace,**kwargs):
    assert not replace
    values=list(RAW_PREPARED(cache,False,**kwargs))
    dim=cache['motion'].shape[1]+cache['social'].shape[1]
    # Native decoder consumes raw observations, not learned normalization of encoder tokens.
    obs=np.zeros((len(cache['sample_id']),dim),np.float32)
    obs[:,:16]=cache['_past'].reshape(-1,16)
    flat=cache['_tokens'].reshape(len(obs),-1);obs[:,16:16+flat.shape[1]]=flat
    values[0]=obs;values[4]=np.zeros((1,dim),np.float32);values[5]=np.ones((1,dim),np.float32)
    return values

class NativeGenerator(nn.Module):
    def __init__(self,dimension,cfg):
        super().__init__();assert DECODER_STATE is not None
        self.decoder=DECODER_CLASS(TOKEN_DIM,12);self.decoder.load_state_dict(DECODER_STATE,strict=True)
        self.strength=cfg['velocity_residual_scale'];self.arm=cfg['set_context']
        allowed=('refiner.',) if self.arm=='zero' else ('context.','mode_mlp.','mode_embedding','anchor_templates','proposal_scale.','proposal_delta.','refiner.')
        for name,parameter in self.decoder.named_parameters():parameter.requires_grad_(name.startswith(allowed))
        assert any(x.requires_grad for x in self.parameters())
        assert not any(x.requires_grad for x in self.decoder.score_head.parameters())

    def forward(self,observation,base,speed):
        past=observation[:,:16].reshape(-1,1,8,2)
        tokens=observation[:,16:16+8*TOKEN_DIM].reshape(-1,1,8,TOKEN_DIM)
        prediction,_=self.decoder(tokens,past)
        raw=prediction[:,0]-past[:,0,-1,None,None]
        delta=raw-base
        velocity=torch.diff(delta,dim=2,prepend=torch.zeros_like(delta[:,:,:1]))
        limit=self.strength*(speed[:,0,None,None,None]+.03)
        residual=velocity.clamp(-limit,limit)
        return base+residual.cumsum(2),residual

c.Correction=NativeGenerator;r.m.Correction=NativeGenerator
p.load_npz=load_cache;r.prepared=prepared
RAW_METRICS=r.generation_metrics
def generation_metrics(paths,base,speed,future):
    value=RAW_METRICS(paths,base,speed,future)
    velocity=np.diff(paths,axis=2,prepend=np.zeros_like(paths[:,:,:1]))/.4
    acceleration=np.diff(velocity,axis=2)/.4
    jerk=np.diff(acceleration,axis=2)/.4
    for name,array in [('speed',velocity),('acceleration',acceleration),('jerk',jerk)]:
        magnitude=np.linalg.norm(array,axis=-1).reshape(len(paths),-1)
        value[name+'_audit_mean']=magnitude.mean(1)
        value[name+'_audit_p95']=np.quantile(magnitude,.95,axis=1)
    return value
r.generation_metrics=generation_metrics

def frozen_audit(model):
    allowed={name for name,value in model.decoder.named_parameters() if value.requires_grad}
    for name,value in model.decoder.state_dict().items():
        if name not in allowed:assert torch.equal(value.cpu(),DECODER_STATE[name]),name

def preflight(device):
    extract(device)
    cfg=c.settings(1);cache=load_cache(p.TRAIN_CACHE)
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);values=prepared(cache,False,fit=fit)
    assert not(set(cache['sample_id'][fit])&set(cache['sample_id'][cal]))
    assert not(set(cache['sample_id'])&set(RAW_LOAD(p.DEV_CACHE)['sample_id']))
    r.setup_weights(cache,fit,values,cfg,device)
    scale=np.array([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32)
    o,b,s,gt=[torch.from_numpy(v[:128]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][:128],device=device);counts={};timings={};whitelists={}
    for arm in c.ARMS:
        p.seed_everything(1);model=NativeGenerator(o.shape[1],c.settings(1,arm)).to(device)
        initial=copy.deepcopy(model.state_dict());model.eval();output,residual=model(o,b,s)
        assert torch.allclose(output,b,atol=2e-5,rtol=0)
        counts[arm]=sum(x.numel() for x in model.parameters() if x.requires_grad)
        whitelists[arm]=[key for key,x in model.named_parameters() if x.requires_grad]
        optimizer=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=cfg['learning_rate'])
        start=time.perf_counter()
        for _ in range(10):
            output,residual=model(o,b,s);loss=m.objective(output,b,gt,residual,s,groups,cfg,c.d.ARM,torch.tensor(scale,device=device))
            assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();optimizer.step()
        torch.cuda.synchronize();timings[arm]=(time.perf_counter()-start)/10
        frozen_audit(model)
        assert any(not torch.equal(initial[k],v) for k,v in model.state_dict().items())
        assert (residual.abs()<=cfg['velocity_residual_scale']*(s[:,0,None,None,None]+.03)+1e-6).all()
    source_paths=[Path(__file__),HERE/'protocol.json',c.SOURCE,ROOT/'GENERATION_SET_CONTEXT_SERIES/run_context.py',m.SOURCE,
        ROOT/'GENERATION_PROTECTED_BALANCE_SERIES/run_protected.py',f.FRONT_MODELS,f.FRONT_TRAINER,f.V2_ADAPTER,
        ROOT/'CONTROLLED_ANCHOR_CORRECTION_SERIES/controlled_anchor_correction_models.py',ROOT/'HIERARCHICAL_DECODER_SERIES/hierarchical_decoder_models.py',
        f.fold_run('eth')/'best_selected.pth',f.fold_run('eth','h3')/'best_selected.pth',f.CONFIG,p.TRAIN_CACHE,p.DEV_CACHE,p.RAW_TRAIN,
        f.fold_data('eth','train')[1],HERE/'train_encoder.npz',HERE/'dev_encoder.npz']
    source_paths+=r.sources()
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=counts,trainable_whitelists=whitelists,
        seconds_per_128step=timings,fit_protection_scales=scale.tolist(),source_hashes={str(x):p.sha256(x) for x in source_paths},
        test_access=False,native_source=str(f.fold_run('eth')/'best_selected.pth'),token_dim=TOKEN_DIM))
    print('PREFLIGHT PASS '+json.dumps(dict(parameters=counts,seconds_per_step=timings)),flush=True)

ORIGINAL_EVALUATE=c.evaluate
def evaluate(device):
    # All frozen native state tensors must remain exact in every saved policy.
    for seed in c.settings(1)['seeds']:
        for arm in c.ARMS:
            for tag in c.POLICIES:
                saved=torch.load(c.checkpoint(seed,arm,tag),map_location='cpu',weights_only=False)
                model=NativeGenerator(saved['input_dim'],saved['config']);model.load_state_dict(saved['state_dict']);frozen_audit(model)
    ORIGINAL_EVALUATE(device)
    summary=json.loads((m.RESULTS/'summary.json').read_text())
    lookup={(x['arm'],x['policy'],x['group']):x for x in summary['aggregate']}
    summary['gates']['motion_distribution_no_harm_1pct']=all(
        lookup['mean_std','constrained',group][key]<=summary['native'][group][key]*1.01+1e-8
        for group in ('overall','stable','transition','change')
        for key in ('speed_audit_p95','acceleration_audit_mean','acceleration_audit_p95','jerk_audit_mean','jerk_audit_p95'))
    summary['scope']='ETH repeated dev; 6fresh80epoch paired-order-seed native-checkpoint trainings. zero=original V2 refiner only; mean_std=original generation decoder/query/templates/anchor heads/refiner. Frozen F16 exact encoder inputs and score_head; decoder score inputs change so no ranking claims. Same loss/fitcal/budget/trust-region velocity clipping1.5*(observed_speed+.03). Unequal trainable capacity is the scope intervention, not architecture superiority. K20/no test. Native fallback may have floating point identity tolerance2e-5. 3seeds do not prove independent generalization.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'NATIVE_DECODER_GATE_FAIL'
    m.write_json(m.RESULTS/'summary.json',summary)
    (m.RESULTS/'REPORT.md').write_text('# Native generator training scope\n\n'+summary['verdict']+'\n\n'+summary['scope']+'\n\n'+json.dumps(summary['aggregate'],indent=2),encoding='utf-8')
    m.write_json(m.RESULTS/'frozen_audit.json',dict(status='PASS',policies_checked=18,scorer_exact=True,encoder_checkpoint_locked=True))
    print('COMPLETE '+summary['verdict'],flush=True)

c.preflight=preflight;c.evaluate=evaluate
if __name__=='__main__':
    try:
        if '--action' in sys.argv and sys.argv[sys.argv.index('--action')+1]=='run':
            _,_,model=prepare_backend(torch.device('cpu'));del model
        c.main()
    except Exception:
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
