"""Authorized context/mode_mlp adaptation for a new correction branch only."""
from pathlib import Path
import importlib.util,sys,json,time,traceback
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('conditioning_direct',ROOT/'GENERATION_DIRECT_ANCHOR_SERIES/run_head.py')
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
n,c,m,r,p=a.n,a.c,a.m,a.r,a.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'

class ConditioningHead(a.DirectHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        for name,value in self.decoder.named_parameters():
            value.requires_grad_(self.arm=='mean_std' and name.startswith(('context.','mode_mlp.')))
    def inputs(self,observation):
        d=self.decoder;past=observation[:,:16].reshape(-1,1,8,2)
        tokens=observation[:,16:784].reshape(-1,1,8,96)
        context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
        query=d.mode_embedding[None,None].expand(len(observation),1,-1,-1)
        mode=d.mode_mlp(context[:,:,None]+query)
        return torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0]

def frozen_audit(model):
    allowed=('context.','mode_mlp.') if model.arm=='mean_std' else ()
    for name,value in model.decoder.state_dict().items():
        if not name.startswith(allowed):assert torch.equal(value.detach().cpu(),n.DECODER_STATE[name]),name
    for name,value in model.decoder.named_parameters():
        assert value.requires_grad==name.startswith(allowed),name
n.NativeGenerator=ConditioningHead;c.Correction=ConditioningHead;r.m.Correction=ConditioningHead;n.frozen_audit=frozen_audit

def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        files=json.loads((ROOT/'GENERATION_OBSERVED_SUPPORT_DIAGNOSIS/old_results_manifest.json').read_text())['files']
        for folder in ('GENERATION_OBSERVED_SUPPORT_DIAGNOSIS','GENERATION_CONDITIONING_ADAPTATION_DESIGN'):
            for source in (ROOT/folder).rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and source.name!='runtime_state.json':files[str(source)]=p.sha256(source)
        m.write_json(path,dict(files=files))
    files=json.loads(path.read_text())['files']
    for source,digest in files.items():assert p.sha256(Path(source))==digest,source
    return len(files)

def preflight(device):
    count=old_manifest(not(HERE/'old_results_manifest.json').exists())
    _,_,original=n.prepare_backend(torch.device('cpu'));del original
    parent=json.loads((ROOT/'GENERATION_PATH_MIXTURE_SERIES/preflight.json').read_text())
    for path,digest in parent['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert (len(fit),len(cal))==(25912,2879)
    assert not(set(cache['sample_id'])&set(p.RAW_LOAD(p.DEV_CACHE)['sample_id'])) if hasattr(p,'RAW_LOAD') else not(set(cache['sample_id'])&set(n.RAW_LOAD(p.DEV_CACHE)['sample_id']))
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    group=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(parent['fit_protection_scales'],device=device)
    states=[];counts={};timings={};gradients={};whitelists={}
    for arm in c.ARMS:
        p.seed_everything(1);model=ConditioningHead(940,c.settings(1,arm)).to(device).eval()
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
                if arm=='mean_std':
                    for module in (model.decoder.context,model.decoder.mode_mlp):assert all(v.grad.abs().sum()>0 for v in module.parameters())
            opt.step();frozen_audit(model)
        torch.cuda.synchronize();timings[arm]=(time.perf_counter()-start)/8
        gradients[arm]={name:float(v.grad.norm()) for name,v in model.named_parameters() if v.grad is not None}
        assert out.shape==(128,20,12,2) and torch.isfinite(out).all()
        assert (res.abs()<=1.5*(s[:,0,None,None,None]+.03)+1e-6).all()
    assert counts['zero']==25158 and counts['mean_std']>25158
    assert all(torch.equal(states[0][key],states[1][key]) for key in states[0]);old_manifest()
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE/'README.md',HERE/'old_results_manifest.json']
    sources={**parent['source_hashes'],**{str(path):p.sha256(path) for path in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=counts,trainable_whitelists=whitelists,gradients=gradients,fit_protection_scales=parent['fit_protection_scales'],source_hashes=sources,seconds_per_128step=timings,fullpaired=True,native_identity_exact=True,old_files=count,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(counts=counts,timing=timings,old_files=count)),flush=True)

def evaluate(device):
    n.evaluate(device);oldcount=old_manifest();path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Fresh frozen-conditioning zero versus ONLY decoder.context+mode_mlp adaptation mean_std for new anchor correction branch. All F16 and all other original decoder/scorer fixed; immutable native baseline. Additional trainability/capacity confounded, not equal-capacity causal proof. Original hardwinner/regularizers/protection/cal/K20/clip unchanged. Earlier whole decoder training failed. Repeated ETH dev exploratory, no intent/ranking/independent generator claim.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'CONDITIONING_ADAPTATION_GATE_FAIL'
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
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
if __name__=='__main__':main()
