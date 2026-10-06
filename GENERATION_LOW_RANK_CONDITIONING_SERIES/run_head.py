"""Dense versus rank8 conditioning updates. Only authorized modules."""
from pathlib import Path
import importlib.util,sys,json,time,traceback
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('conditioning_direct',ROOT/'GENERATION_DIRECT_ANCHOR_SERIES/run_head.py')
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
n,c,m,r,p=a.n,a.c,a.m,a.r,a.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'

class RankLinear(torch.nn.Module):
    def __init__(self,base,dense):
        super().__init__();self.base=base;rank=8
        self.down=torch.nn.Parameter(torch.empty(rank,base.in_features))
        self.up=torch.nn.Parameter(torch.zeros(base.out_features,rank))
        torch.nn.init.kaiming_uniform_(self.down,a=5**.5)
        self.register_buffer('down_initial',self.down.detach().clone())
        self.base.weight.requires_grad_(dense)
        if self.base.bias is not None:self.base.bias.requires_grad_(True)
        self.down.requires_grad_(not dense);self.up.requires_grad_(not dense)
    def forward(self,x):
        return self.base(x)+torch.nn.functional.linear(torch.nn.functional.linear(x,self.down),self.up)

class LowRankHead(a.DirectHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        for owner in (self.decoder.context,self.decoder.mode_mlp):
            for parameter in owner.parameters():parameter.requires_grad_(True)
            for name,module in list(owner.named_modules()):
                if isinstance(module,torch.nn.Linear):
                    parts=name.split('.');container=owner
                    for part in parts[:-1]:container=getattr(container,part)
                    setattr(container,parts[-1],RankLinear(module,self.arm=='zero'))
    def inputs(self,observation):
        d=self.decoder;past=observation[:,:16].reshape(-1,1,8,2)
        tokens=observation[:,16:784].reshape(-1,1,8,96)
        context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
        query=d.mode_embedding[None,None].expand(len(observation),1,-1,-1)
        mode=d.mode_mlp(context[:,:,None]+query)
        return torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0]

def frozen_audit(model):
    for name,value in model.decoder.state_dict().items():
        if name.startswith(('context.','mode_mlp.')):
            if '.base.weight' in name and model.arm=='mean_std':
                assert torch.equal(value.detach().cpu(),n.DECODER_STATE[name.replace('.base.','.')]),name
            continue
        assert torch.equal(value.detach().cpu(),n.DECODER_STATE[name]),name
    for owner in (model.decoder.context,model.decoder.mode_mlp):
        for module in owner.modules():
            if isinstance(module,RankLinear):
                assert module.base.weight.requires_grad==(model.arm=='zero')
                assert module.down.requires_grad==module.up.requires_grad==(model.arm=='mean_std')
                if model.arm=='zero':
                    assert torch.equal(module.down,module.down_initial) and torch.count_nonzero(module.up)==0
    assert not any(v.requires_grad for name,v in model.decoder.named_parameters() if not name.startswith(('context.','mode_mlp.')))

n.NativeGenerator=LowRankHead;c.Correction=LowRankHead;r.m.Correction=LowRankHead;n.frozen_audit=frozen_audit

def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        files=json.loads((ROOT/'GENERATION_CONDITIONING_LEARNING_DIAGNOSIS/old_results_manifest.json').read_text())['files']
        for folder in ('GENERATION_CONDITIONING_LEARNING_DIAGNOSIS',):
            for source in (ROOT/folder).rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and source.name!='runtime_state.json':files[str(source)]=p.sha256(source)
        m.write_json(path,dict(files=files))
    files=json.loads(path.read_text())['files']
    for source,digest in files.items():assert p.sha256(Path(source))==digest,source
    return len(files)

def preflight(device):
    count=old_manifest(not(HERE/'old_results_manifest.json').exists())
    _,_,original=n.prepare_backend(torch.device('cpu'));del original
    parent=json.loads((ROOT/'GENERATION_CONDITIONING_ADAPTATION_SERIES/preflight.json').read_text())
    for path,digest in parent['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert (len(fit),len(cal))==(25912,2879)
    assert not(set(cache['sample_id'])&set(p.RAW_LOAD(p.DEV_CACHE)['sample_id'])) if hasattr(p,'RAW_LOAD') else not(set(cache['sample_id'])&set(n.RAW_LOAD(p.DEV_CACHE)['sample_id']))
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    group=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(parent['fit_protection_scales'],device=device)
    states=[];counts={};timings={};gradients={};whitelists={}
    for arm in c.ARMS:
        p.seed_everything(1);model=LowRankHead(940,c.settings(1,arm)).to(device).eval()
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
            opt.step();frozen_audit(model)
        torch.cuda.synchronize();timings[arm]=(time.perf_counter()-start)/8
        gradients[arm]={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.grad is not None}
        assert out.shape==(128,20,12,2) and torch.isfinite(out).all()
        assert (res.abs()<=1.5*(s[:,0,None,None,None]+.03)+1e-6).all()
    assert counts['zero']==46804 and 25158<counts['mean_std']<counts['zero']
    assert all(torch.equal(states[0][key],states[1][key]) for key in states[0]);old_manifest()
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE/'README.md',HERE/'old_results_manifest.json',ROOT/'GENERATION_CONDITIONING_LEARNING_DIAGNOSIS/RESULT.json',ROOT/'GENERATION_CONDITIONING_LEARNING_DIAGNOSIS/ANALYSIS.json',ROOT/'GENERATION_CONDITIONING_LEARNING_DIAGNOSIS/ANALYSIS.md']
    sources={**parent['source_hashes'],**{str(path):p.sha256(path) for path in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=counts,trainable_whitelists=whitelists,gradients=gradients,fit_protection_scales=parent['fit_protection_scales'],source_hashes=sources,seconds_per_128step=timings,fullpaired=True,native_identity_exact=True,old_files=count,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(counts=counts,timing=timings,old_files=count)),flush=True)

def evaluate(device):
    n.evaluate(device);oldcount=old_manifest();path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Fresh dense conditioning updates zero versus rank8 factorized weight updates mean_std; only original context/mode_mlp and new25158head. Bias and LayerNorm train in both; rank8 weightbases fixed in treatment, dense update control. Both identical original function/heads/auxiliaryinit/nativeidentity; unequal trainable capacity and factorized optimization are coupled, not pure cause proof. All F16/otherdecoder/scorer fixed/immutable nativebase. Original loss/guards/data unchanged. Repeated ETH dev exploratory, no intent/ranking/independent generator claim.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'LOW_RANK_CONDITIONING_GATE_FAIL'
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
