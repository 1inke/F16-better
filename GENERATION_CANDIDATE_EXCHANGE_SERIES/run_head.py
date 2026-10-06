"""Uniform versus native-candidate-query neighbor attention. All original modules frozen."""
from pathlib import Path
import importlib.util,sys,json,time,traceback
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('conditioning_direct',ROOT/'GENERATION_DIRECT_ANCHOR_SERIES/run_head.py')
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
n,c,m,r,p=a.n,a.c,a.m,a.r,a.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'

class CandidateHead(a.DirectHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        self.head=torch.nn.Sequential(torch.nn.Linear(160,128),torch.nn.GELU(),torch.nn.Linear(128,64),torch.nn.GELU(),torch.nn.Linear(64,6))
        torch.nn.init.zeros_(self.head[-1].weight);torch.nn.init.zeros_(self.head[-1].bias)
        self.query=torch.nn.Linear(128,32);self.key=torch.nn.Linear(128,32);self.value=torch.nn.Linear(128,32)
    def native_inputs(self,observation):
        d=self.decoder;past=observation[:,:16].reshape(-1,1,8,2);tokens=observation[:,16:784].reshape(-1,1,8,96)
        context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
        query=d.mode_embedding[None,None].expand(len(observation),1,-1,-1);mode=d.mode_mlp(context[:,:,None]+query)
        return torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0]
    def exchange(self,features):
        scores=torch.einsum('bkd,bnd->bkn',self.query(features),self.key(features))/(32**.5)
        if self.arm=='zero':
            scores=scores*0
            diagonal=torch.eye(20,dtype=torch.bool,device=features.device)[None]
            scores=scores.masked_fill(~diagonal,-1e4)
        weights=scores.softmax(-1);values=self.value(features)
        return torch.einsum('bkn,bnd->bkd',weights,values),weights
    def inputs(self,observation):
        native=self.native_inputs(observation);mixed,_=self.exchange(native)
        return torch.cat((native,mixed),-1)

def frozen_audit(model):
    for name,value in model.decoder.state_dict().items():
        assert torch.equal(value.detach().cpu(),n.DECODER_STATE[name]),name
    assert not any(value.requires_grad for value in model.decoder.parameters())

n.NativeGenerator=CandidateHead;c.Correction=CandidateHead;r.m.Correction=CandidateHead;n.frozen_audit=frozen_audit

def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        files=json.loads((ROOT/'GENERATION_CYCLE_10_REVIEW/all_files_manifest.json').read_text())['files']
        for folder in ('GENERATION_CYCLE_10_REVIEW',):
            for source in (ROOT/folder).rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and source.name!='runtime_state.json':files[str(source)]=p.sha256(source)
        m.write_json(path,dict(files=files))
    files=json.loads(path.read_text())['files']
    for source,digest in files.items():assert p.sha256(Path(source))==digest,source
    return len(files)

def preflight(device):
    count=old_manifest(not(HERE/'old_results_manifest.json').exists())
    _,_,original=n.prepare_backend(torch.device('cpu'));del original
    parent=json.loads((ROOT/'GENERATION_NEIGHBOR_MESSAGE_SERIES/preflight.json').read_text())
    for path,digest in parent['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert (len(fit),len(cal))==(25912,2879)
    assert not(set(cache['sample_id'])&set(p.RAW_LOAD(p.DEV_CACHE)['sample_id'])) if hasattr(p,'RAW_LOAD') else not(set(cache['sample_id'])&set(n.RAW_LOAD(p.DEV_CACHE)['sample_id']))
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    group=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(parent['fit_protection_scales'],device=device)
    states=[];counts={};timings={};gradients={};whitelists={}
    for arm in c.ARMS:
        p.seed_everything(1);model=CandidateHead(940,c.settings(1,arm)).to(device).eval()
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
                assert model.value.weight.grad.abs().sum()>0
                if arm=='zero':assert model.query.weight.grad.abs().sum()==model.key.weight.grad.abs().sum()==0
                else:assert model.query.weight.grad.abs().sum()>0 and model.key.weight.grad.abs().sum()>0
            opt.step();frozen_audit(model)
        torch.cuda.synchronize();timings[arm]=(time.perf_counter()-start)/8
        gradients[arm]={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.grad is not None}
        assert out.shape==(128,20,12,2) and torch.isfinite(out).all()
        assert (res.abs()<=1.5*(s[:,0,None,None,None]+.03)+1e-6).all()
    assert counts['zero']==counts['mean_std']==41638
    assert all(torch.equal(states[0][key],states[1][key]) for key in states[0]);old_manifest()
    p.seed_everything(13);control=CandidateHead(940,c.settings(13,'zero')).to(device).eval()
    p.seed_everything(13);treatment=CandidateHead(940,c.settings(13,'mean_std')).to(device).eval()
    assert torch.equal(control.native_inputs(o),treatment.native_inputs(o))
    native=treatment.native_inputs(o)
    assert torch.isfinite(native).all() and native.std()>1e-5
    x,weights=treatment.exchange(native);own,diag=control.exchange(native)
    assert torch.isfinite(weights).all() and torch.allclose(weights.sum(-1),torch.ones_like(weights.sum(-1)),atol=1e-6)
    assert torch.equal(diag,torch.eye(20,device=device)[None].expand(len(o),-1,-1))
    assert torch.equal(own,control.value(native))
    assert x.std()>1e-5 and (x-own).abs().mean()>1e-5
    changed=native.clone();changed[:,19]+=0.5
    own_changed,_=control.exchange(changed);x_changed,_=treatment.exchange(changed)
    assert torch.equal(own[:,:19],own_changed[:,:19])
    assert (x[:,:19]-x_changed[:,:19]).abs().mean()>1e-6
    perm=torch.arange(19,-1,-1,device=device)
    permuted,_=treatment.exchange(native[:,perm])
    permutation_error=float((permuted-x[:,perm]).abs().max())
    assert permutation_error<2e-6
    assert torch.count_nonzero(o[:,784:])==0

    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE/'README.md',HERE/'DESIGN.md',HERE/'old_results_manifest.json',ROOT/'GENERATION_CYCLE_10_REVIEW/REVIEW.json',ROOT/'GENERATION_CYCLE_10_REVIEW/CYCLE_10_REVIEW.md']
    sources={**parent['source_hashes'],**{str(path):p.sha256(path) for path in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=counts,trainable_whitelists=whitelists,gradients=gradients,fit_protection_scales=parent['fit_protection_scales'],source_hashes=sources,seconds_per_128step=timings,candidate_module_permutation_max_error=permutation_error,fullpaired=True,native_identity_exact=True,old_files=count,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(counts=counts,timing=timings,old_files=count)),flush=True)

def evaluate(device):
    n.evaluate(device);oldcount=old_manifest();path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Own-candidate value readout zero versus learned candidate-axis exchange mean_std over originalfrozencontext+mode128features; same newQKV128to32 and160to128to64to6head=41638nominalparams/allpaired/nativeoutputidentity. DiagonalQKgradzero versus fullattentionpositive not equal effectivecapacity. AlloldF16/decoder/scorerfixed/nativeK20/common3anchorclip/originallossguardsdata. Earlier nativepathmeanstd setcontext failed, not firstcandidateinformation/intent/ranking/independentgenerator. Nativecandidateorderlocked, modulefeatureperm only; repeatedETHdevexploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'CANDIDATE_EXCHANGE_GATE_FAIL'
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
