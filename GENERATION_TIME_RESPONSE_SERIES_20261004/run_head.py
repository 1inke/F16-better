"""Common observed-token time readout, block-mean versus stepwise correction."""
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
a=module('time_response_direct',ROOT/'GENERATION_DIRECT_ANCHOR_SERIES/run_head.py')
n,c,m,r,p=a.n,a.c,a.m,a.r,a.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'

def positions(count):
    t=torch.arange(count,dtype=torch.float32)[:,None]
    frequency=torch.exp(torch.arange(0,32,2,dtype=torch.float32)*(-np.log(10000.)/32))
    result=torch.zeros(count,32);result[:,0::2]=torch.sin(t*frequency);result[:,1::2]=torch.cos(t*frequency)
    return result

class TimeResponseHead(a.DirectHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        self.query=nn.Linear(128,32);self.key=nn.Linear(96,32);self.value=nn.Linear(96,32)
        self.head=nn.Sequential(nn.Linear(192,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,2))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('past_position',positions(8))
        self.register_buffer('future_position',positions(20)[8:])
    def train(self,mode=True):
        super().train(mode);self.decoder.eval();return self
    def native_inputs(self,observation):
        d=self.decoder;past=observation[:,:16].reshape(-1,1,8,2)
        tokens=observation[:,16:784].reshape(-1,1,8,96)
        context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
        mode=d.mode_mlp(context[:,:,None]+d.mode_embedding[None,None])
        return torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0]
    def inputs(self,observation):
        native=self.native_inputs(observation);tokens=observation[:,16:784].reshape(-1,8,96)
        query=self.query(native)[:,:,None]+self.future_position[None,None]
        key=self.key(tokens)+self.past_position[None]
        weight=torch.einsum('bktd,bnd->bktn',query,key).div(32**.5).softmax(-1)
        pooled=torch.einsum('bktn,bnd->bktd',weight,self.value(tokens))
        feature=torch.cat((native[:,:,None].expand(-1,-1,12,-1),pooled,
                           self.future_position[None,None].expand(len(observation),20,-1,-1)),-1)
        return feature,weight
    def project(self,velocity):
        if self.arm=='mean_std':return velocity
        return velocity.reshape(*velocity.shape[:2],3,4,2).mean(3).repeat_interleave(4,dim=2)
    def forward(self,observation,base,speed):
        if bool(self.native_fallback):return base,torch.zeros_like(base)
        feature,_=self.inputs(observation)
        raw=self.head(feature)*(speed[:,0,None,None,None]+.03)
        limit=self.strength*(speed[:,0,None,None,None]+.03)
        residual=self.project(raw.clamp(-limit,limit))
        return base+residual.cumsum(2),residual

def frozen_audit(model):
    for name,value in model.decoder.state_dict().items():
        assert torch.equal(value.detach().cpu(),n.DECODER_STATE[name]),name
    assert not any(v.requires_grad for v in model.decoder.parameters())

n.NativeGenerator=TimeResponseHead;c.Correction=TimeResponseHead;r.m.Correction=TimeResponseHead;n.frozen_audit=frozen_audit

def verify(sources):
    for path,digest in sources.items():assert p.sha256(Path(path))==digest,path
    return len(sources)

def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        files=dict(json.loads((ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/old_results_manifest.json').read_text())['files'])
        verify(files)
        for folder in ('F16_OBSERVABLE_CUE_DIAGNOSIS_20261003','F16_DECODER_RESPONSE_ANALYSIS_20261004','F16_QUALITY_STANDARD_AUDIT_20261003'):
            for source in (ROOT/folder).rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and source.suffix not in ('.lock','.tmp','.pending') and source.name!='runtime_state.json':
                    files[str(source)]=p.sha256(source)
        m.write_json(path,dict(files=files,control_files_excluded=True))
    return verify(json.loads(path.read_text())['files'])

def source_hashes():
    inherited=json.loads((ROOT/'GENERATION_CANDIDATE_EXCHANGE_SERIES/preflight.json').read_text())['source_hashes']
    verify(inherited)
    diagnostic=module('time_response_common',ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/common.py')
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/'README.md',HERE/'DESIGN.md',HERE/'protocol.json',HERE/'old_results_manifest.json',n.HERE/'encoder_audit.json']
    paths+=diagnostic.source_paths()
    return {**inherited,**{str(path):p.sha256(path) for path in paths if path.is_file()}}

def prepared_train(device):
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1)
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert (len(fit),len(cal))==(25912,2879)
    assert not(set(cache['sample_id'][fit])&set(cache['sample_id'][cal]))
    assert not(set(cache['sample_id'])&set(n.RAW_LOAD(p.DEV_CACHE)['sample_id']))
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    assert values[0].shape[1]==940 and np.count_nonzero(values[0][:,784:])==0
    return cache,cfg,fit,cal,values

def preflight(device):
    old_count=old_manifest(not(HERE/'old_results_manifest.json').exists())
    _,_,original=n.prepare_backend(device);original.device=str(device)
    cache,cfg,fit,cal,values=prepared_train(device)
    parent=json.loads((ROOT/'GENERATION_CANDIDATE_EXCHANGE_SERIES/preflight.json').read_text())
    scale=torch.tensor(parent['fit_protection_scales'],device=device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][fit[:128]],device=device)
    # Real observation-only original F16 replay; no future to native forward.
    common=module('time_response_replay_common',ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/common.py')
    sys.modules['common']=common
    native=module('time_response_replay_native',ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/native.py')
    raw=np.load(ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz',allow_pickle=False)
    assert np.array_equal(raw['sample_id'],cache['sample_id'])
    obs=raw['observation'][fit[:128]]
    with torch.no_grad():
        result=native.forward(original,torch.tensor(obs[:,160:304].reshape(-1,9,8,2),device=device),torch.tensor(obs[:,304],device=device))
    probe=TimeResponseHead(940,c.settings(1,'zero')).to(device).eval()
    alignment={'tokens':float((result['tokens']-o[:,16:784].reshape(-1,8,96)).abs().max()),
               'native_path':float((result['paths']-b).abs().max()),
               'context':float((result['E3']-probe.native_inputs(o)[:,0,:64]).abs().max())}
    assert max(alignment.values())<2e-5,alignment
    original_state={k:v.detach().cpu().clone() for k,v in original.state_dict().items()}
    assert not any(v.requires_grad for v in original.parameters())
    del probe
    states=[];counts={};gradients={};timings={}
    for arm in c.ARMS:
        p.seed_everything(1);model=TimeResponseHead(940,c.settings(1,arm)).to(device).eval()
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        counts[arm]=sum(v.numel() for v in model.parameters() if v.requires_grad)
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
        feature,weight=model.inputs(o);assert torch.isfinite(feature).all() and torch.allclose(weight.sum(-1),torch.ones_like(weight.sum(-1)),atol=1e-6)
        changed=o.clone();changed[:,16:112]+=.01
        assert (model.inputs(changed)[0]-feature).abs().mean()>1e-7
        assert feature[:,:,1:].sub(feature[:,:,:-1]).abs().mean()>1e-5
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        torch.cuda.synchronize();start=time.perf_counter()
        for step in range(8):
            opt.zero_grad();out,res=model(o,b,s)
            loss=m.objective(out,b,gt,res,s,groups,c.settings(1,arm),'balanced_protected',scale)
            assert torch.isfinite(loss);loss.backward()
            for name,v in model.named_parameters():
                if v.requires_grad:
                    assert v.grad is not None and torch.isfinite(v.grad).all(),name
                    if step>=3:assert v.grad.abs().sum()>0,name
                else:assert v.grad is None,name
            opt.step();frozen_audit(model)
        torch.cuda.synchronize();timings[arm]=(time.perf_counter()-start)/8
        gradients[arm]={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.grad is not None}
        assert torch.isfinite(out).all() and out.shape==(128,20,12,2)
        assert (res.abs()<=1.5*(s[:,0,None,None,None]+.03)+1e-6).all()
        if arm=='zero':assert torch.equal(res.reshape(128,20,3,4,2),res.reshape(128,20,3,4,2)[:,:,:,:1].expand(-1,-1,-1,4,-1))
        model.native_fallback.fill_(True);out,z=model(o,b,s);assert torch.equal(out,b) and z.abs().sum()==0
    assert counts['zero']==counts['mean_std']==43426,counts
    assert all(torch.equal(states[0][key],states[1][key]) for key in states[0])
    for name,value in original.state_dict().items():assert torch.equal(value.detach().cpu(),original_state[name]),name
    del original
    x=torch.linspace(-1.,1.,128*20*12*2,device=device).reshape(128,20,12,2)
    model.arm='zero';projected=model.project(x)
    anchor_error=float((projected.cumsum(2)[:,:,[3,7,11]]-x.cumsum(2)[:,:,[3,7,11]]).abs().max())
    assert anchor_error<2e-6 and (projected-x).abs().max()>0
    old_manifest()
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=counts,
        gradients=gradients,source_hashes=source_hashes(),fit_protection_scales=parent['fit_protection_scales'],
        seconds_per_128step=timings,paired_weights_buffers_exact=True,native_identity_exact=True,
        all_original_F16_decoder_scorer_frozen=True,online_cache_alignment=alignment,
        identical_weights_anchor_sum_max_error=anchor_error,old_files=old_count,test_access=False,future_forward=False))
    print('PREFLIGHT PASS '+json.dumps(dict(counts=counts,timing=timings,alignment=alignment,old_files=old_count)),flush=True)

def secondary_analysis():
    common=module('time_response_event_rules',ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/common.py')
    cache=p.load_npz(p.DEV_CACHE);values=r.prepared(cache,False,mean=np.zeros((1,940),np.float32),std=np.ones((1,940),np.float32))
    past=cache['_past'];future=values[3];base=values[1]
    # Native paths/futures are displacements from forecast origin; event rule
    # requires the same absolute-local origin as observed past.
    origin=past[:,-1,None];truth=common.events(past,future+origin)
    native_error=np.linalg.norm(base-future[:,None],axis=-1);rows=[];ids=cache['sample_id']
    labels={'overall':np.ones(len(ids),bool),'clear_event':(truth['y']==1)&~truth['uncertain'],
            'no_clear_event':(truth['y']==0)&~truth['uncertain'],'uncertain':truth['uncertain']}
    for file in sorted(m.RESULTS.glob('seed*.npz')):
        saved=n.RAW_LOAD(file);assert np.array_equal(saved['sample_id'],ids)
        pred=saved['trajectory'];error=np.linalg.norm(pred-future[:,None],axis=-1)
        ade=error.mean(-1);fde=error[:,:,-1];chosen=(ade+c.settings(1)['fde_loss_weight']*fde).argmin(-1)
        ar=np.arange(len(ids));selected=error[ar,chosen]
        event=common.events(np.repeat(past,20,axis=0),(pred+origin[:,None]).reshape(-1,12,2))
        amplitude=np.abs(event['future_speed'].reshape(len(ids),20,-1)-truth['future_speed'][:,None]).mean(-1)
        kind=event['kind'].reshape(len(ids),20);onset=event['onset_step'].reshape(len(ids),20);direction=event['direction'].reshape(len(ids),20)
        for label,mask in labels.items():
            if not mask.any():continue
            row=dict(file=file.name,group=label,n=int(mask.sum()),same_candidate_ade=float(ade[ar,chosen][mask].mean()),same_candidate_fde=float(fde[ar,chosen][mask].mean()),
                     early_error=float(selected[mask,:4].mean()),middle_error=float(selected[mask,4:8].mean()),late_error=float(selected[mask,8:].mean()),
                     event_kind_match=float((kind[ar,chosen][mask]==truth['kind'][mask]).mean()),
                     event_speed_curve_mae=float(amplitude[ar,chosen][mask].mean()))
            if label=='clear_event':
                row['event_onset_step_mae']=float(np.abs(onset[ar,chosen][mask]-truth['onset_step'][mask]).mean())
                row['turn_direction_match']=float((direction[ar,chosen][mask]==truth['direction'][mask]).mean())
            for ta,tf in ((.2,.4),(.3,.5)):
                adequate=((ade<=ta)&(fde<=tf)).any(-1);old=((native_error.mean(-1)<=ta)&(native_error[:,:,-1]<=tf)).any(-1)
                key=f'adequacy_{ta}_{tf}'
                row[key]=float(adequate[mask].mean());row[key+'_rescued']=int((mask&~old&adequate).sum());row[key+'_lost']=int((mask&old&~adequate).sum())
            rows.append(row)
        directory=m.RESULTS/'temporal';directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/(file.stem+'_temporal.npz'),sample_id=ids,selected_samepath_index=chosen,early=selected[:,:4].mean(-1),middle=selected[:,4:8].mean(-1),late=selected[:,8:].mean(-1))
    assert len({x['file'] for x in rows})==18
    m.write_json(m.RESULTS/'temporal_response_analysis.json',dict(status='PASS_DESCRIPTIVE_ONLY',rows=rows,GT_forward=False,
       interpretation='Posthoc same-path joint oracle, not deployed scorer. Kind/onset99 includes missing event; direction statistic includes non-turn zero, not turn-only accuracy. No checkpoint choice or new gate.'))

def evaluate(device):
    n.evaluate(device)
    # Do not pollute generic seed*.npz glob with secondary files until the
    # inherited18-file audit has completed.
    secondary_analysis()
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='FullK20 common observed-token time-query readout; block4mean-repeat versus identity of clipped12step correction increments. Same43426nominalparams/fullpaired/nativeidentity; different function class and projection optimization geometry. Entire originalF16/decoder/scorerfixed, immutable nativeK20, original loss/protection/cal/data. NotAR/future observations/intent/ranking/independent generator; repeatedDEV exploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'TIME_RESPONSE_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# Temporal response\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    audit=json.loads((m.RESULTS/'audit.json').read_text());assert audit['status']=='PASS'
    models=json.loads((m.RESULTS/'model_lock.json').read_text())['models'];verify(models)
    for checkpoint in models:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=TimeResponseHead(saved['input_dim'],saved['config']);model.load_state_dict(saved['state_dict']);frozen_audit(model)
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
