"""Mean encoded neighbors before head versus mean per-neighbor head messages. All original modules frozen."""
from pathlib import Path
import importlib.util,sys,json,time,traceback
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('conditioning_direct',ROOT/'GENERATION_DIRECT_ANCHOR_SERIES/run_head.py')
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
n,c,m,r,p=a.n,a.c,a.m,a.r,a.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'

BASE_LOAD=p.load_npz;BASE_PREPARED=r.prepared
def load_cache(path):
    cache=BASE_LOAD(path)
    if Path(path) in (p.TRAIN_CACHE,p.DEV_CACHE):
        rawpath=ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES'/('train_observations.npz' if Path(path)==p.TRAIN_CACHE else 'dev_observations.npz')
        raw=n.RAW_LOAD(rawpath)
        assert np.array_equal(raw['sample_id'],cache['sample_id'])
        local=raw['observation'][:,160:304].reshape(-1,9,8,2);count=raw['observation'][:,304]
        assert np.isfinite(local).all() and np.array_equal(count,count.astype(np.int32)) and count.min()>=1 and count.max()<=9
        assert np.max(np.abs(local[:,0]-cache['_past']))<1e-6
        cache['_pairwise']=(local[:,1:]-local[:,:1]).reshape(-1,128)
        cache['_pairvalid']=(np.arange(1,9)[None,:]<count[:,None]).astype(np.float32)
    return cache
def prepared(cache,replace,**kwargs):
    values=list(BASE_PREPARED(cache,replace,**kwargs))
    values[0][:,784:912]=cache['_pairwise'];values[0][:,912:920]=cache['_pairvalid']
    return values
p.load_npz=load_cache;r.prepared=prepared

class PairwiseHead(a.DirectHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        self.pair_encoder=torch.nn.Sequential(torch.nn.Linear(16,64),torch.nn.GELU(),torch.nn.Linear(64,32))
        self.head=torch.nn.Sequential(torch.nn.Linear(160,128),torch.nn.GELU(),torch.nn.Linear(128,64),torch.nn.GELU(),torch.nn.Linear(64,6))
        torch.nn.init.zeros_(self.head[-1].weight);torch.nn.init.zeros_(self.head[-1].bias)
        with np.load(ROOT/'GENERATION_PAIRWISE_HISTORY_DIAGNOSIS/fit_pairwise_scalers.npz') as transform:
            self.register_buffer('pair_mean',torch.from_numpy(transform['mean'].copy()))
            self.register_buffer('pair_std',torch.from_numpy(transform['std'].copy()))
    def native_inputs(self,observation):
        d=self.decoder;past=observation[:,:16].reshape(-1,1,8,2);tokens=observation[:,16:784].reshape(-1,1,8,96)
        context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
        query=d.mode_embedding[None,None].expand(len(observation),1,-1,-1);mode=d.mode_mlp(context[:,:,None]+query)
        return torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0]
    def pair_inputs(self,observation):
        features=(observation[:,784:912].reshape(-1,8,16)-self.pair_mean)/self.pair_std
        valid=observation[:,912:920]
        values=self.pair_encoder(features)
        return (values*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
    def inputs(self,observation):
        native=self.native_inputs(observation);pair=self.pair_inputs(observation)
        return torch.cat((native,pair[:,None].expand(-1,20,-1)),-1)

    def anchor_outputs(self,observation):
        native=self.native_inputs(observation)
        if self.arm=='zero':return self.head(self.inputs(observation))
        features=(observation[:,784:912].reshape(-1,8,16)-self.pair_mean)/self.pair_std
        values=self.pair_encoder(features);valid=observation[:,912:920]
        empty=self.head(torch.cat((native,torch.zeros_like(native[...,:32])),-1))
        messages=self.head(torch.cat((native[:,:,None].expand(-1,-1,8,-1),values[:,None].expand(-1,20,-1,-1)),-1))
        increment=((messages-empty[:,:,None])*valid[:,None,:,None]).sum(2)/valid.sum(1)[:,None,None].clamp_min(1)
        return empty+increment
    def forward(self,observation,base,speed):
        if bool(self.native_fallback):return base,torch.zeros_like(base)
        anchors=self.anchor_outputs(observation).reshape(-1,20,3,2)*(speed[:,0,None,None,None]+.03)
        delta=torch.einsum('tp,bkpc->bktc',self.decoder.interpolation,anchors)
        velocity=torch.diff(delta,dim=2,prepend=torch.zeros_like(base[:,:,:1]))
        limit=self.strength*(speed[:,0,None,None,None]+.03)
        residual=velocity.clamp(-limit,limit)
        return base+residual.cumsum(2),residual

def frozen_audit(model):
    for name,value in model.decoder.state_dict().items():
        assert torch.equal(value.detach().cpu(),n.DECODER_STATE[name]),name
    assert not any(value.requires_grad for value in model.decoder.parameters())
    with np.load(ROOT/'GENERATION_PAIRWISE_HISTORY_DIAGNOSIS/fit_pairwise_scalers.npz') as transform:
        assert torch.equal(model.pair_mean.cpu(),torch.from_numpy(transform['mean']))
        assert torch.equal(model.pair_std.cpu(),torch.from_numpy(transform['std']))

n.NativeGenerator=PairwiseHead;c.Correction=PairwiseHead;r.m.Correction=PairwiseHead;n.frozen_audit=frozen_audit

def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        files=json.loads((ROOT/'GENERATION_NEIGHBOR_USAGE_DIAGNOSIS/old_results_manifest.json').read_text())['files']
        for folder in ('GENERATION_NEIGHBOR_USAGE_DIAGNOSIS',):
            for source in (ROOT/folder).rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and source.name!='runtime_state.json':files[str(source)]=p.sha256(source)
        m.write_json(path,dict(files=files))
    files=json.loads(path.read_text())['files']
    for source,digest in files.items():assert p.sha256(Path(source))==digest,source
    return len(files)

def preflight(device):
    count=old_manifest(not(HERE/'old_results_manifest.json').exists())
    _,_,original=n.prepare_backend(torch.device('cpu'));del original
    parent=json.loads((ROOT/'GENERATION_NEIGHBOR_READOUT_SERIES/preflight.json').read_text())
    for path,digest in parent['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert (len(fit),len(cal))==(25912,2879)
    assert not(set(cache['sample_id'])&set(p.RAW_LOAD(p.DEV_CACHE)['sample_id'])) if hasattr(p,'RAW_LOAD') else not(set(cache['sample_id'])&set(n.RAW_LOAD(p.DEV_CACHE)['sample_id']))
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    group=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(parent['fit_protection_scales'],device=device)
    states=[];counts={};timings={};gradients={};whitelists={}
    for arm in c.ARMS:
        p.seed_everything(1);model=PairwiseHead(940,c.settings(1,arm)).to(device).eval()
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        counts[arm]=sum(v.numel() for v in model.parameters() if v.requires_grad)
        whitelists[arm]=[key for key,v in model.named_parameters() if v.requires_grad]
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        start=time.perf_counter()
        for step in range(8):
            opt.zero_grad();out,res=model(o,b,s);loss=m.objective(out,b,gt,res,s,group,c.settings(1,arm),'balanced_protected',scale)
            assert torch.isfinite(loss);loss.backward()
            for name,v in model.named_parameters():
                if v.requires_grad:assert v.grad is not None and torch.isfinite(v.grad).all(),name
                else:assert v.grad is None,name
            if step>0:
                assert model.head[0].weight.grad.abs().sum()>0
                if step>2:
                    # Zero correction head delays B learning, and zero B then
                    # delays A learning by another step. Check after activation.
                    for name,v in model.decoder.named_parameters():
                        if v.requires_grad:assert v.grad.abs().sum()>0,name
            if step>1:
                assert model.pair_encoder[0].weight.grad.abs().sum()>0 and model.pair_encoder[2].weight.grad.abs().sum()>0
            opt.step();frozen_audit(model)
        torch.cuda.synchronize();timings[arm]=(time.perf_counter()-start)/8
        gradients[arm]={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.grad is not None}
        assert out.shape==(128,20,12,2) and torch.isfinite(out).all()
        assert (res.abs()<=1.5*(s[:,0,None,None,None]+.03)+1e-6).all()
    assert counts['zero']==counts['mean_std']==32422
    assert all(torch.equal(states[0][key],states[1][key]) for key in states[0]);old_manifest()
    p.seed_everything(13);control=PairwiseHead(940,c.settings(13,'zero')).to(device).eval()
    p.seed_everything(13);treatment=PairwiseHead(940,c.settings(13,'mean_std')).to(device).eval()
    assert torch.equal(control.native_inputs(o),treatment.native_inputs(o))
    changed=o.clone();changed[:,784:912]=o[:,784:912].reshape(-1,8,16).flip(1).reshape(-1,128);changed[:,912:920]=o[:,912:920].flip(1)
    permutation_error=float((treatment.pair_inputs(o)-treatment.pair_inputs(changed)).abs().max())
    assert permutation_error<2e-6
    changed=o.clone();invalid=o[:,912:920]==0
    pairs=changed[:,784:912].reshape(-1,8,16);pairs[invalid]=123.456
    assert torch.allclose(treatment.pair_inputs(o),treatment.pair_inputs(changed),atol=1e-6,rtol=1e-6)
    assert torch.equal(control.pair_inputs(o),treatment.pair_inputs(o))
    # Same learned weights, only placement of mean/nonlinearity changes.
    model.arm='mean_std';real=model.anchor_outputs(o)
    changed=o.clone();changed[:,784:912]=o[:,784:912].reshape(-1,8,16).flip(1).reshape(-1,128);changed[:,912:920]=o[:,912:920].flip(1)
    assert torch.allclose(real,model.anchor_outputs(changed),atol=2e-6,rtol=2e-6)
    model.arm='zero';pooled=model.anchor_outputs(o)
    assert (real-pooled).abs().mean()>1e-7
    for neighbor_count in (0,1):
        changed=o.clone();changed[:,912:920]=0
        if neighbor_count:changed[:,912]=1
        model.arm='zero';x=model.anchor_outputs(changed)
        model.arm='mean_std';y=model.anchor_outputs(changed)
        assert torch.allclose(x,y,atol=2e-6,rtol=2e-6)
    model.arm='mean_std'
    changed=o.clone();invalid=o[:,912:920]==0;changed[:,784:912].reshape(-1,8,16)[invalid]=123.456
    assert torch.allclose(real,model.anchor_outputs(changed),atol=2e-6,rtol=2e-6)

    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE/'README.md',HERE/'old_results_manifest.json',ROOT/'GENERATION_PAIRWISE_HISTORY_DIAGNOSIS/DIAGNOSIS.json',ROOT/'GENERATION_PAIRWISE_HISTORY_DIAGNOSIS/fit_pairwise_scalers.npz',ROOT/'GENERATION_PAIRWISE_HISTORY_DIAGNOSIS/README.md']
    sources={**parent['source_hashes'],**{str(path):p.sha256(path) for path in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=counts,trainable_whitelists=whitelists,gradients=gradients,fit_protection_scales=parent['fit_protection_scales'],source_hashes=sources,seconds_per_128step=timings,neighbor_permutation_max_error=permutation_error,fullpaired=True,native_identity_exact=True,old_files=count,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(counts=counts,timing=timings,old_files=count)),flush=True)

def evaluate(device):
    n.evaluate(device);oldcount=old_manifest();path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Same actual observed relativehistories/FITscalers/validmasks/3168pairencoder+29254head=32422 nominalparams. Uniformmean encoder before candidateconditioned head versus mean candidateconditioned per-neighbor anchor messages; same shared head/no new parameters/no attention. Empty-neighbor common focalhead; one-neighbor equivalent. Different nonlinear function/effectivecapacity/compute, not pooling causeproof. AlloriginalF16decoder/scorer/nativeK20fixed/originallossguardsdata unchanged. No intent/ranking/independentgenerator; repeated ETHdev exploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'NEIGHBOR_MESSAGE_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    audit=json.loads((m.RESULTS/'audit.json').read_text());assert audit['status']=='PASS'
    m.write_json(m.RESULTS/'independent_audit.json',dict(status='PASS',models=18,predictions=18,original_decoder_nonwhitelist_exact=True,encoder_source_locked=True,old_files_unchanged=oldcount,metrics_audit=audit))
c.preflight=preflight;c.evaluate=evaluate

def main():
    try:
        if '--action' in sys.argv and sys.argv[sys.argv.index('--action')+1]=='run':
            _,_,original=n.prepare_backend(torch.device('cpu'));del original
        c.main()
    except BaseException:
        m.RESULTS.mkdir(parents=True,exist_ok=True)
        path=m.RESULTS/'failure.txt'
        if path.exists():path=m.RESULTS/('failure_'+str(time.time_ns())+'.txt')
        path.write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
if __name__=='__main__':main()
