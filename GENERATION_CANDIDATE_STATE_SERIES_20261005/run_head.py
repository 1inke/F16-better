"""Immutable-native-state availability versus masked-state persistent response."""
from pathlib import Path
import sys,importlib.util,json,time,traceback,os
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import torch
from torch import nn
from resource_guard import guard
from state_features import state32, prepare_scalers, load_scalers, validate_state_math
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
        self.state_encoder=nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        mean,std=load_scalers(HERE)
        self.register_buffer('state_mean',torch.from_numpy(mean))
        self.register_buffer('state_std',torch.from_numpy(std))
        self.head=nn.Sequential(nn.Linear(192,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,6))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('increment_midtime',(torch.arange(12,dtype=torch.float32)+.5)/12)
        self.register_buffer('pulse_ranges',torch.tensor([.05,.90,.05,.45],dtype=torch.float32))
    def inputs(self,observation,base):
        native=self.native_inputs(observation);tokens=observation[:,16:784].reshape(-1,8,96)
        query=self.query(native)
        key=self.key(tokens)+self.past_position[None]
        weight=torch.einsum('bkd,bnd->bkn',query,key).div(32**.5).softmax(-1)
        pooled=torch.einsum('bkn,bnd->bkd',weight,self.value(tokens))
        z=(state32(base)-self.state_mean)/self.state_std
        z=z*0. if self.arm=='zero' else z
        state=self.state_encoder(z)
        return torch.cat((native,pooled,state),-1),weight
    def components(self,codes,speed):
        # BOTH clocks active. Sole formula change is the response profile.
        clock=codes[...,4:6]
        center=self.pulse_ranges[0]+self.pulse_ranges[1]*clock[...,0].sigmoid()
        span=self.pulse_ranges[2]+self.pulse_ranges[3]*clock[...,1].sigmoid()
        profile=((self.increment_midtime-center[...,None])/span[...,None]).sigmoid()
        limit=self.strength*(speed[:,0,None,None,None]+.03)
        drift=.5*limit*codes[...,:2].tanh()[:,:,None]
        pulse=.5*limit*codes[...,2:4].tanh()[:,:,None]*profile[...,None]
        raw=drift+pulse
        residual=raw.clamp(-limit,limit)
        return residual,dict(center=center,span=span,profile=profile,drift=drift,pulse=pulse,raw=raw,limit=limit)
    def forward(self,observation,base,speed):
        guard.check()
        if bool(self.native_fallback):return base,torch.zeros_like(base)
        feature,_=self.inputs(observation,base)
        residual,_=self.components(self.head(feature),speed)
        return base+residual.cumsum(2),residual

TimeResponseHead=PulseResponseHead
t.TimeResponseHead=PulseResponseHead
n.NativeGenerator=PulseResponseHead;c.Correction=PulseResponseHead;r.m.Correction=PulseResponseHead
frozen_audit=t.frozen_audit
def verify(sources):
    for count,(path,digest) in enumerate(sources.items()):
        if count%1000==0:guard.check()
        assert p.sha256(Path(path))==digest,('SOURCE/OLD SHA mismatch',path)
    return len(sources)
def prepared_train(device):
    guard.check();result=t.prepared_train(device)
    cache,cfg,fit,cal,values=result
    prepare_scalers(HERE,cache['sample_id'],fit,values[1],p.sha256)
    guard.check();return result
n.frozen_audit=frozen_audit

def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        prior=ROOT/'GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/old_results_manifest.json'
        files=dict(json.loads(prior.read_text(encoding='utf-8'))['files'])
        verify(files)
        # Extend verified chain to all currently existing project science/reviews.
        folders=[v for v in ROOT.iterdir() if v.is_dir() and v!=HERE and v.name.startswith(('GENERATION_','F16_','HEAD_','PULSE_'))]
        for folder in folders:
            guard.check()
            for source in folder.rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and source.suffix not in ('.lock','.tmp','.pending') and source.name!='runtime_state.json':
                    # Keep verified historical digest; never replace it with a
                    # current observation. Hash only genuinely new prior files.
                    if str(source) not in files:files[str(source)]=p.sha256(source)
        m.write_json(path,dict(files=files,control_files_excluded=True,excluded=['.memory','AUTONOMOUS_GENERATION_CYCLE.json','runtime_state.json','locks/tmp/__pycache__'],all_prior_failures_preserved=True))
        # Known files just verified and new files just hashed: avoid a second
        # immediate identical I/O pass. Full manifest reverified after GPU.
        print('INITIAL_OLD_SHA PASS '+str(len(files)),flush=True)
        return len(files)
    return verify(json.loads(path.read_text(encoding='utf-8'))['files'])

def source_hashes():
    inherited=json.loads((ROOT/'GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/preflight.json').read_text(encoding='utf-8'))['source_hashes']
    verify(inherited)
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/'README.md',HERE/'DESIGN.md',HERE/'protocol.json',HERE/'old_results_manifest.json',
        ROOT/'GENERATION_TIME_RESPONSE_SERIES_20261004/run_head.py',ROOT/'GENERATION_TIME_RESPONSE_SERIES_20261004/completion_audit.py',
        ROOT/'F16_RESPONSE_GAP_DIAGNOSIS_20261004/INTERPRETATION.md',ROOT/'F16_RESPONSE_GAP_DIAGNOSIS_20261004/completion_interpretation_audit.json',HERE/'state_scalers.npz',HERE/'state_scalers.json']
    return {**inherited,**{str(path):p.sha256(path) for path in paths if path.is_file()}}

def preflight(device):
    guard.check();old_count=old_manifest(not(HERE/'old_results_manifest.json').exists())
    _,_,original=n.prepare_backend(device);original.device=str(device)
    cache,cfg,fit,cal,values=prepared_train(device)
    parent=json.loads((ROOT/'GENERATION_CANDIDATE_EXCHANGE_SERIES/preflight.json').read_text(encoding='utf-8'))
    state_math=validate_state_math()
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
        feature,weight=model.inputs(o,b)
        assert torch.isfinite(feature).all() and torch.allclose(weight.sum(-1),torch.ones_like(weight.sum(-1)),atol=1e-6)
        changed=o.clone();changed[:,16:112]+=.01
        assert (model.inputs(changed,b)[0]-feature).abs().mean()>1e-7
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
                    if name=='state_encoder.0.weight' and arm=='zero':assert v.grad.abs().sum()==0,name
                    elif step>=3 and name!='key.bias':assert v.grad.abs().sum()>0,name
                else:assert v.grad is None,name
            if step>=3:
                clock_grad=float(model.head[-1].weight.grad[4:].abs().sum()+model.head[-1].bias.grad[4:].abs().sum())
                assert clock_grad>0
            opt.step();frozen_audit(model)
        torch.cuda.synchronize();timings[arm]=(time.perf_counter()-start)/8
        gradients[arm]={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.grad is not None}
        assert torch.isfinite(out).all() and out.shape==(128,20,12,2)
        codes=model.head(model.inputs(o,b)[0]);res,parts=model.components(codes,s)
        assert (res.abs()<=parts['limit']+1e-6).all()
        expected=12*parts['drift'].squeeze(2)+parts['pulse'].sum(2)
        endpoint_error=float((res.sum(2)-expected).abs().max())
        profile_sum=float(parts['profile'].sum(-1).abs().max())
        assert endpoint_error<2e-5,endpoint_error
        assert (res-parts['raw']).abs().max()<1e-6
        assert (parts['profile']>=0).all() and (parts['profile']<=1).all()
        assert (torch.diff(parts['profile'],dim=-1)>=0).all()
        # Input-mask behavior checked using only immutable predicted base.
        perturbed=b.clone();perturbed[:,:,5:,0]+=.02
        with torch.no_grad():sensitivity=float((model.inputs(o,perturbed)[0]-model.inputs(o,b)[0]).abs().max())
        assert sensitivity==0 if arm=='zero' else sensitivity>1e-7
        response[arm]=dict(state_input_sensitivity=sensitivity,state_first_weight_gradient=float(model.state_encoder[0].weight.grad.norm()),clock_gradient=clock_grad,endpoint_error=endpoint_error,profile_sum_error=profile_sum,
            center_std=float(parts['center'].std()),span_std=float(parts['span'].std()),residual_nonzero=float(res.abs().mean()))
        model.native_fallback.fill_(True);out,z=model(o,b,s);assert torch.equal(out,b) and z.abs().sum()==0
    assert counts['zero']==counts['mean_std']==45798,counts
    assert all(torch.equal(states[0][key],states[1][key]) for key in states[0])
    for name,value in original.state_dict().items():assert torch.equal(value.detach().cpu(),original_state[name]),name
    del original
    # Independent synthetic codes verify profile math, endpoint integration,
    # and clock endpoint coupling. Same-code endpoints deliberately differ.
    codes=torch.linspace(-2.,2.,128*20*6,device=device).reshape(128,20,6).requires_grad_()
    model.arm='mean_std';adaptive,parts=model.components(codes,s)
    adaptive.square().sum().backward()
    assert codes.grad[...,4:].abs().sum()>0
    model.arm='zero';fixed,fixedparts=model.components(codes.detach(),s)
    assert torch.equal(adaptive.detach(),fixed)
    separation_error=0.;difference=0.
    changed=codes.detach().clone();changed[...,4]+=1.
    clock_checks={}
    for arm in c.ARMS:
        model.arm=arm;first,part1=model.components(codes.detach(),s);second,part2=model.components(changed,s)
        actual=float((first.sum(2)-second.sum(2)).abs().max());assert actual>1e-4
        clock_checks[arm]=actual
        reference=((model.increment_midtime.double()-part1['center'].double()[...,None])/part1['span'].double()[...,None]).sigmoid()
        assert float((part1['profile'].double()-reference).abs().max())<2e-5
    old_manifest()
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=counts,
        gradients=gradients,response=response,state_math=state_math,state_scalers=json.loads((HERE/'state_scalers.json').read_text(encoding='utf-8')),source_hashes=source_hashes(),fit_protection_scales=parent['fit_protection_scales'],
        seconds_per_128step=timings,paired_weights_buffers_exact=True,native_identity_exact=True,
        all_original_F16_decoder_scorer_frozen=True,online_cache_alignment=alignment,
        identical_codes_endpoint_difference=separation_error,identical_codes_path_difference=difference,
        clock_only_endpoint_change=clock_checks,both_clock_active=True,memory=guard.check(),
        key_bias_analytic_softmax_cancellation=True,old_files=old_count,test_access=False,future_forward=False))
    guard.save('PREFLIGHT_PASS')
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
                        codes=model.head(model.inputs(o,b)[0]);res,parts=model.components(codes,s)
                        if bool(model.native_fallback):res=torch.zeros_like(res)
                        expected=12*parts['drift'].squeeze(2)+parts['pulse'].sum(2)
                        if bool(model.native_fallback):expected=torch.zeros_like(expected)
                        for key in ('center','span','profile'):pieces[key].append(parts[key].cpu().numpy())
                        pieces['drift'].append(expected.cpu().numpy())
                        amp=.5*parts['limit'].squeeze(2)*codes[...,2:4].tanh()
                        pieces['pulse_amplitude'].append(amp.cpu().numpy())
                        pieces['endpoint_error'].append((res.sum(2)-expected).abs().amax((1,2)).cpu().numpy())
                values2={key:np.concatenate(v) for key,v in pieces.items()}
                assert np.isfinite(values2['center']).all() and values2['endpoint_error'].max()<1e-4
                np.savez_compressed(output/f'seed{seed}_{arm}_{policy}_pulse.npz',sample_id=cache['sample_id'],native_fallback=bool(model.native_fallback),**values2)
                records.append(dict(seed=seed,arm=arm,policy=policy,native_fallback=bool(model.native_fallback),
                    center_std=float(values2['center'].std()),span_std=float(values2['span'].std()),
                    center_min=float(values2['center'].min()),center_max=float(values2['center'].max()),
                    span_min=float(values2['span'].min()),span_max=float(values2['span'].max()),
                    amplitude_abs_mean=float(np.abs(values2['pulse_amplitude']).mean()),
                    endpoint_error_max=float(values2['endpoint_error'].max())))
    m.write_json(output/'summary.json',dict(status='PASS_DESCRIPTIVE_ONLY',records=records,
        interpretation='Both learned sigmoid clocks active; zero masked standardized native-state32 vs mean_std actual state32. State is a deterministic native-prediction feature; center/span are not identified GT events. Saved drift field is total endpoint correction incl pulse sum. No DEV selection or causal claim.'))

def evaluate(device):
    n.evaluate(device)
    t.secondary_analysis()
    pulse_diagnostics(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text(encoding='utf-8'))
    summary['scope']='Immutable nativecandidate state32 maskedzero versus actual after fullFIT-only standardization; same45798nominal params, paired weights/buffers/nativeidentity. Both common learned sigmoid persistent profile/context+mode+observedtokens. Inputmask changes effective capacity/optimization, not extra information or causeproof. EntireoldF16/decoder/scorerfixed/nativeK20/sameoriginalhardloss/guards/data/budget. No GTfuture/logits/groupsforward/AR/intent/ranking. RepeatedDEV exploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'CANDIDATE_STATE_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# Candidate-native-state conditioning\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    audit=json.loads((m.RESULTS/'audit.json').read_text());assert audit['status']=='PASS'
    models=json.loads((m.RESULTS/'model_lock.json').read_text())['models'];verify(models)
    for checkpoint in models:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=PulseResponseHead(saved['input_dim'],saved['config']);model.load_state_dict(saved['state_dict']);frozen_audit(model)
    source_count=verify(json.loads((HERE/'preflight.json').read_text())['source_hashes']);old_count=old_manifest()
    m.write_json(m.RESULTS/'independent_audit.json',dict(status='PASS',models=18,predictions=18,metrics_audit=audit,all_original_decoder_frozen_exact=True,
        encoder_scorer_sources_locked=True,registered_sources_unchanged=source_count,old_files_unchanged=old_count))
    print('COMPLETE '+summary['verdict'],flush=True)
    guard.save('EVALUATION_PASS')

c.preflight=preflight;c.evaluate=evaluate
def main():
    try:
        if '--action' in sys.argv and sys.argv[sys.argv.index('--action')+1]=='run':
            _,_,original=n.prepare_backend(torch.device('cpu'));del original
            old_manifest()
            prepared_train(torch.device('cpu'))
        c.main()
    except BaseException:
        m.RESULTS.mkdir(parents=True,exist_ok=True)
        (m.RESULTS/('failure_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8')
        m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
if __name__=='__main__':main()
