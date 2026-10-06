"""Fresh endpoint-consistent pulse response: fixed vs learned center/span."""
from pathlib import Path
import sys,importlib.util,json,time,traceback
sys.dont_write_bytecode=True
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value

t=module('pulse_time_parent',ROOT/'GENERATION_TIME_RESPONSE_SERIES_20261004/run_head.py')
n,c,m,r,p=t.n,t.c,t.m,t.r,t.p
t.HERE=HERE;c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'

class PulseResponseHead(t.TimeResponseHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        self.head=nn.Sequential(nn.Linear(160,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,6))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('increment_midtime',(torch.arange(12,dtype=torch.float32)+.5)/12)
        self.register_buffer('pulse_ranges',torch.tensor([.05,.90,.05,.45],dtype=torch.float32))
    def inputs(self,observation):
        native=self.native_inputs(observation);tokens=observation[:,16:784].reshape(-1,8,96)
        query=self.query(native)
        key=self.key(tokens)+self.past_position[None]
        weight=torch.einsum('bkd,bnd->bkn',query,key).div(32**.5).softmax(-1)
        pooled=torch.einsum('bkn,bnd->bkd',weight,self.value(tokens))
        return torch.cat((native,pooled),-1),weight
    def components(self,codes,speed):
        # Only arm difference: final two codes are used or replaced by zero.
        clock=codes[...,4:6] if self.arm=='mean_std' else codes[...,4:6]*0
        center=self.pulse_ranges[0]+self.pulse_ranges[1]*clock[...,0].sigmoid()
        span=self.pulse_ranges[2]+self.pulse_ranges[3]*clock[...,1].sigmoid()
        gaussian=torch.exp(-.5*((self.increment_midtime-center[...,None])/span[...,None]).square())
        centered=gaussian-gaussian.mean(-1,keepdim=True)
        profile=centered/centered.abs().amax(-1,keepdim=True).clamp_min(1e-6)
        limit=self.strength*(speed[:,0,None,None,None]+.03)
        drift=.5*limit*codes[...,:2].tanh()[:,:,None]
        pulse=.5*limit*codes[...,2:4].tanh()[:,:,None]*profile[...,None]
        raw=drift+pulse
        residual=raw.clamp(-limit,limit)
        return residual,dict(center=center,span=span,profile=profile,drift=drift,pulse=pulse,raw=raw,limit=limit)
    def forward(self,observation,base,speed):
        if bool(self.native_fallback):return base,torch.zeros_like(base)
        feature,_=self.inputs(observation)
        residual,_=self.components(self.head(feature),speed)
        return base+residual.cumsum(2),residual

TimeResponseHead=PulseResponseHead
t.TimeResponseHead=PulseResponseHead
n.NativeGenerator=PulseResponseHead;c.Correction=PulseResponseHead;r.m.Correction=PulseResponseHead
frozen_audit=t.frozen_audit;verify=t.verify;prepared_train=t.prepared_train
n.frozen_audit=frozen_audit

def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        prior=ROOT/'F16_RESPONSE_GAP_DIAGNOSIS_20261004/old_results_manifest.json'
        files=dict(json.loads(prior.read_text(encoding='utf-8'))['files'])
        verify(files)
        for folder in ('GENERATION_TIME_RESPONSE_SERIES_20261004','F16_RESPONSE_GAP_DIAGNOSIS_20261004'):
            for source in (ROOT/folder).rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and source.suffix not in ('.lock','.tmp','.pending') and source.name!='runtime_state.json':
                    files[str(source)]=p.sha256(source)
        m.write_json(path,dict(files=files,control_files_excluded=True))
    return verify(json.loads(path.read_text(encoding='utf-8'))['files'])

def source_hashes():
    inherited=json.loads((ROOT/'F16_RESPONSE_GAP_DIAGNOSIS_20261004/preflight.json').read_text(encoding='utf-8'))['source_hashes']
    verify(inherited)
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/'README.md',HERE/'DESIGN.md',HERE/'protocol.json',HERE/'old_results_manifest.json',
        ROOT/'GENERATION_TIME_RESPONSE_SERIES_20261004/run_head.py',ROOT/'GENERATION_TIME_RESPONSE_SERIES_20261004/completion_audit.py',
        ROOT/'F16_RESPONSE_GAP_DIAGNOSIS_20261004/INTERPRETATION.md',ROOT/'F16_RESPONSE_GAP_DIAGNOSIS_20261004/completion_interpretation_audit.json']
    return {**inherited,**{str(path):p.sha256(path) for path in paths if path.is_file()}}

def preflight(device):
    old_count=old_manifest(not(HERE/'old_results_manifest.json').exists())
    _,_,original=n.prepare_backend(device);original.device=str(device)
    cache,cfg,fit,cal,values=prepared_train(device)
    parent=json.loads((ROOT/'GENERATION_CANDIDATE_EXCHANGE_SERIES/preflight.json').read_text(encoding='utf-8'))
    scale=torch.tensor(parent['fit_protection_scales'],device=device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][fit[:128]],device=device)
    common=module('pulse_replay_common',ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/common.py')
    sys.modules['common']=common
    native=module('pulse_replay_native',ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/native.py')
    raw=np.load(ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz',allow_pickle=False)
    assert np.array_equal(raw['sample_id'],cache['sample_id'])
    obs=raw['observation'][fit[:128]]
    with torch.no_grad():
        result=native.forward(original,torch.tensor(obs[:,160:304].reshape(-1,9,8,2),device=device),torch.tensor(obs[:,304],device=device))
    probe=PulseResponseHead(940,c.settings(1,'zero')).to(device).eval()
    alignment={'tokens':float((result['tokens']-o[:,16:784].reshape(-1,8,96)).abs().max()),
        'native_path':float((result['paths']-b).abs().max()),
        'context':float((result['E3']-probe.native_inputs(o)[:,0,:64]).abs().max())}
    assert max(alignment.values())<2e-5,alignment
    original_state={k:v.detach().cpu().clone() for k,v in original.state_dict().items()}
    assert not any(v.requires_grad for v in original.parameters())
    del probe
    states=[];counts={};gradients={};timings={};response={}
    for arm in c.ARMS:
        p.seed_everything(1);model=PulseResponseHead(940,c.settings(1,arm)).to(device).eval()
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        counts[arm]=sum(v.numel() for v in model.parameters() if v.requires_grad)
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
        feature,weight=model.inputs(o)
        assert torch.isfinite(feature).all() and torch.allclose(weight.sum(-1),torch.ones_like(weight.sum(-1)),atol=1e-6)
        changed=o.clone();changed[:,16:112]+=.01
        assert (model.inputs(changed)[0]-feature).abs().mean()>1e-7
        # Candidate-specific features, not a single shared/global event prediction.
        assert feature.sub(feature[:,:1]).abs().mean()>1e-6
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        torch.cuda.synchronize();start=time.perf_counter()
        for step in range(8):
            opt.zero_grad();out,res=model(o,b,s)
            loss=m.objective(out,b,gt,res,s,groups,c.settings(1,arm),'balanced_protected',scale)
            assert torch.isfinite(loss);loss.backward()
            for name,v in model.named_parameters():
                if v.requires_grad:
                    assert v.grad is not None and torch.isfinite(v.grad).all(),name
                    # Key bias cancels softmax; its gradient may be exact0/roundoff.
                    if step>=3 and name!='key.bias':assert v.grad.abs().sum()>0,name
                else:assert v.grad is None,name
            if step>=3:
                clock_grad=float(model.head[-1].weight.grad[4:].abs().sum()+model.head[-1].bias.grad[4:].abs().sum())
                assert clock_grad>0 if arm=='mean_std' else clock_grad==0
            opt.step();frozen_audit(model)
        torch.cuda.synchronize();timings[arm]=(time.perf_counter()-start)/8
        gradients[arm]={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.grad is not None}
        assert torch.isfinite(out).all() and out.shape==(128,20,12,2)
        codes=model.head(model.inputs(o)[0]);res,parts=model.components(codes,s)
        assert (res.abs()<=parts['limit']+1e-6).all()
        endpoint_error=float((res.sum(2)-12*parts['drift'].squeeze(2)).abs().max())
        profile_sum=float(parts['profile'].sum(-1).abs().max())
        assert endpoint_error<1e-5 and profile_sum<5e-6,(endpoint_error,profile_sum)
        assert (res-parts['raw']).abs().max()<1e-6
        if arm=='zero':
            assert torch.equal(parts['center'],torch.full_like(parts['center'],.5))
            assert torch.equal(parts['span'],torch.full_like(parts['span'],.275))
        response[arm]=dict(clock_gradient=clock_grad,endpoint_error=endpoint_error,profile_sum_error=profile_sum,
            center_std=float(parts['center'].std()),span_std=float(parts['span'].std()),residual_nonzero=float(res.abs().mean()))
        model.native_fallback.fill_(True);out,z=model(o,b,s);assert torch.equal(out,b) and z.abs().sum()==0
    assert counts['zero']==counts['mean_std']==39590,counts
    assert all(torch.equal(states[0][key],states[1][key]) for key in states[0])
    for name,value in original.state_dict().items():assert torch.equal(value.detach().cpu(),original_state[name]),name
    del original
    # Synthetic codes verify nontrivial same-code response/end-point separation.
    codes=torch.linspace(-2.,2.,128*20*6,device=device).reshape(128,20,6).requires_grad_()
    model.arm='mean_std';adaptive,parts=model.components(codes,s)
    adaptive.square().sum().backward()
    assert codes.grad[...,4:].abs().sum()>0
    model.arm='zero';fixed,fixedparts=model.components(codes.detach(),s)
    separation_error=float((adaptive.sum(2)-fixed.sum(2)).abs().max())
    difference=float((adaptive-fixed).abs().max())
    assert separation_error<1e-5 and difference>1e-4,(separation_error,difference)
    old_manifest()
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=counts,
        gradients=gradients,response= response,source_hashes=source_hashes(),fit_protection_scales=parent['fit_protection_scales'],
        seconds_per_128step=timings,paired_weights_buffers_exact=True,native_identity_exact=True,
        all_original_F16_decoder_scorer_frozen=True,online_cache_alignment=alignment,
        identical_codes_endpoint_max_error=separation_error,identical_codes_path_difference=difference,
        key_bias_analytic_softmax_cancellation=True,old_files=old_count,test_access=False,future_forward=False))
    print('PREFLIGHT PASS '+json.dumps(dict(counts=counts,timing=timings,alignment=alignment,response=response,old_files=old_count)),flush=True)

def pulse_diagnostics(device):
    cache=p.load_npz(p.DEV_CACHE)
    records=[];output=m.RESULTS/'pulse';output.mkdir(exist_ok=True)
    for arm in c.ARMS:
        for seed in (1,2,3):
            for policy in c.POLICIES:
                path=c.checkpoint(seed,arm,policy)
                saved=torch.load(path,map_location='cpu',weights_only=False)
                values=r.prepared(cache,False,mean=saved['mean'],std=saved['std'])
                model=PulseResponseHead(saved['input_dim'],saved['config']).to(device).eval()
                model.load_state_dict(saved['state_dict']);pieces={key:[] for key in ('center','span','drift','pulse_amplitude','profile','endpoint_error')}
                with torch.no_grad():
                    for start in range(0,len(cache['sample_id']),128):
                        o,b,s=[torch.from_numpy(v[start:start+128]).to(device) for v in values[:3]]
                        codes=model.head(model.inputs(o)[0]);res,parts=model.components(codes,s)
                        if bool(model.native_fallback):res=torch.zeros_like(res)
                        expected=12*parts['drift'].squeeze(2)
                        if bool(model.native_fallback):expected=torch.zeros_like(expected)
                        for key in ('center','span','profile'):pieces[key].append(parts[key].cpu().numpy())
                        pieces['drift'].append(expected.cpu().numpy())
                        amp=.5*parts['limit'].squeeze(2)*codes[...,2:4].tanh()
                        pieces['pulse_amplitude'].append(amp.cpu().numpy())
                        pieces['endpoint_error'].append((res.sum(2)-expected).abs().amax((1,2)).cpu().numpy())
                values2={key:np.concatenate(v) for key,v in pieces.items()}
                assert np.isfinite(values2['center']).all() and values2['endpoint_error'].max()<1e-4
                if arm=='zero':
                    assert np.allclose(values2['center'],.5,atol=1e-7)
                    assert np.allclose(values2['span'],.275,atol=1e-7)
                np.savez_compressed(output/f'seed{seed}_{arm}_{policy}_pulse.npz',sample_id=cache['sample_id'],native_fallback=bool(model.native_fallback),**values2)
                records.append(dict(seed=seed,arm=arm,policy=policy,native_fallback=bool(model.native_fallback),
                    center_std=float(values2['center'].std()),span_std=float(values2['span'].std()),
                    center_min=float(values2['center'].min()),center_max=float(values2['center'].max()),
                    span_min=float(values2['span'].min()),span_max=float(values2['span'].max()),
                    amplitude_abs_mean=float(np.abs(values2['pulse_amplitude']).mean()),
                    endpoint_error_max=float(values2['endpoint_error'].max())))
    m.write_json(output/'summary.json',dict(status='PASS_DESCRIPTIVE_ONLY',records=records,
        interpretation='Gaussian center/span are response-shape parameters, not identified GT event onset/duration. No DEV selection or causal claim.'))

def evaluate(device):
    n.evaluate(device)
    t.secondary_analysis()
    pulse_diagnostics(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text(encoding='utf-8'))
    summary['scope']='Fresh39590nominalparameter endpoint-consistent drift+zero-sum singleGaussian pulse. zero=fixed center.5/span.275; mean_std=candidate-observed learned center/span. Shared nativecontext64/mode64+candidate observation attention32. Fullpaired/nativeidentity. OriginalF16/wholedecoder/scorer fixed, same original loss/protection/cal/data/K20/nativeclip. Fixed-clock rows receive no data gradient, effective capacity/optimization differ. NotAR/newfutureobservations/GT event supervision/intent/ranking/independent generator. RepeatedDEV exploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'PULSE_RESPONSE_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# Endpoint-consistent pulse response\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    audit=json.loads((m.RESULTS/'audit.json').read_text());assert audit['status']=='PASS'
    models=json.loads((m.RESULTS/'model_lock.json').read_text())['models'];verify(models)
    for checkpoint in models:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=PulseResponseHead(saved['input_dim'],saved['config']);model.load_state_dict(saved['state_dict']);frozen_audit(model)
    source_count=verify(json.loads((HERE/'preflight.json').read_text())['source_hashes']);old_count=old_manifest()
    m.write_json(m.RESULTS/'independent_audit.json',dict(status='PASS',models=18,predictions=18,metrics_audit=audit,all_original_decoder_frozen_exact=True,
        encoder_scorer_sources_locked=True,registered_sources_unchanged=source_count,old_files_unchanged=old_count))
    print('COMPLETE '+summary['verdict'],flush=True)

c.preflight=preflight;c.evaluate=evaluate
def main():
    try:
        if '--action' in sys.argv and sys.argv[sys.argv.index('--action')+1]=='run':
            _,_,original=n.prepare_backend(torch.device('cpu'));del original
            old_manifest()
        c.main()
    except BaseException:
        m.RESULTS.mkdir(parents=True,exist_ok=True)
        (m.RESULTS/('failure_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8')
        m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
if __name__=='__main__':main()

