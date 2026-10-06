"""Fresh full last temporal block vs attention-only, observation-only online F16."""
from pathlib import Path
import importlib.util,sys,json,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'GENERATION_UPSTREAM_TEMPORAL_SERIES'
spec=importlib.util.spec_from_file_location('subspace_online_source',SOURCE/'run_head.py')
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
n=a.n;c=a.c;m=a.m;r=a.r;p=a.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
PREFIX=a.PREFIX
def permitted(arm,name):return name.startswith(PREFIX if arm=='zero' else PREFIX+'self_attn.')
def raw_observations(split,cache):
    assert split in ('train','dev')
    saved=n.RAW_LOAD(SOURCE/(split+'_observations.npz'))
    assert np.array_equal(saved['sample_id'],cache['sample_id'])
    assert saved['observation'].shape==(len(cache['sample_id']),940) and np.isfinite(saved['observation']).all()
    return saved['observation']
a.raw_observations=raw_observations
class SubspaceHead(a.OnlineHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        for name,parameter in self.original.named_parameters():parameter.requires_grad_(permitted(self.arm,name))
    def observed_features(self,observation):
        self.original.device=str(observation.device)
        batch=len(observation);local=observation[:,160:304].reshape(batch,9,8,2)
        centered=observation[:,16:160].reshape(batch,9,8,2);valid=observation[:,304].int();captured=[]
        def capture(owner,args):captured.append(args[0][:,0])
        handle=self.original.decoder.register_forward_pre_hook(capture)
        try:prediction=self.original(centered,None,None,mode='testing',num_valid=valid,graph_loc=local)[2]
        finally:handle.remove()
        assert len(captured)==1
        tokens=captured[0];past=observation[:,:16].reshape(batch,1,8,2)
        context=self.decoder.context(torch.cat((tokens[:,-1,None],tokens.mean(1)[:,None],self.decoder.motion_features(past)),-1))
        query=self.decoder.mode_embedding[None,None].expand(batch,1,-1,-1)
        mode=self.decoder.mode_mlp(context[:,:,None]+query)
        features=torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0]
        relative=prediction[:,0]+local.mean(-2,keepdim=True)[:,0,None]-local[:,0,-1,None,None]
        return features,tokens,relative
def frozen_audit(model):
    a.h.frozen_audit(model)
    for name,value in model.original.state_dict().items():
        if not permitted(model.arm,name):assert torch.equal(value.detach().cpu(),a.ORIGINAL_STATE[name]),name
    for name,parameter in model.original.named_parameters():assert parameter.requires_grad==permitted(model.arm,name),name
n.NativeGenerator=SubspaceHead;c.Correction=SubspaceHead;r.m.Correction=SubspaceHead;n.frozen_audit=frozen_audit
def check_old():
    manifest=json.loads((HERE/'old_results_manifest.json').read_text())['files']
    for path,digest in manifest.items():assert p.sha256(Path(path))==digest,('old result changed',path)
    return len(manifest)
def preflight(device):
    _,_,source=a.prepare_backend(device);del source
    previous=json.loads((SOURCE/'preflight.json').read_text())
    for path,digest in previous['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    destination=HERE/'old_results_manifest.json'
    if not destination.exists():
        files=json.loads((SOURCE/'old_results_manifest.json').read_text())['files']
        for directory in (SOURCE/'results',SOURCE/'evaluation_repair/results'):
            for path in directory.rglob('*'):
                if path.is_file():files[str(path)]=p.sha256(path)
        m.write_json(destination,dict(files=files))
    old_count=check_old()
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),.1)
    assert len(fit)==25912 and len(cal)==2879 and cfg['epochs']==80 and cfg['seeds']==[1,2,3]
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,v,gt=[torch.from_numpy(x[fit[:128]]).to(device) for x in values[:4]]
    groups=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(previous['fit_protection_scales'],device=device)
    states=[];counts={};grads={};timing={}
    for arm in c.ARMS:
        p.seed_everything(1);model=SubspaceHead(940,c.settings(1,arm)).to(device);model.eval()
        states.append({k:x.detach().cpu().clone() for k,x in model.state_dict().items()})
        with torch.no_grad():
            features,tokens,native=model.observed_features(o);assert float(abs(native-b).max())<2e-5
            np.testing.assert_allclose(tokens.cpu().numpy(),cache['_tokens'][fit[:128]],atol=2e-5,rtol=0)
        out,res=model(o,b,v);assert torch.equal(out,b) and res.abs().sum()==0
        counts[arm]=sum(x.numel() for x in model.parameters() if x.requires_grad)
        optimizer=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        torch.cuda.synchronize();start=time.perf_counter()
        for step in range(8):
            model.train();optimizer.zero_grad();out,res=model(o,b,v)
            m.objective(out,b,gt,res,v,groups,cfg,'balanced_protected',scale).backward()
            assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters() if x.requires_grad)
            if step==7:
                assert model.head[0].weight.grad.abs().sum()>0
                norms={}
                for group in ('self_attn','linear1','linear2','norm1','norm2'):
                    norms[group]=sum(float(x.grad.norm()) for name,x in model.original.named_parameters() if name.startswith(PREFIX+group+'.') and x.grad is not None)
                assert norms['self_attn']>0
                assert all(norms[k]>0 for k in ('linear1','linear2','norm1','norm2')) if arm=='zero' else all(norms[k]==0 for k in ('linear1','linear2','norm1','norm2'))
                grads[arm]=norms
            optimizer.step();frozen_audit(model)
        torch.cuda.synchronize();timing[arm]=(time.perf_counter()-start)/8
        assert torch.isfinite(out).all() and out.shape==(128,20,12,2)
        assert (res.abs()<=model.strength*(v[:,0,None,None,None]+.03)+1e-6).all()
    assert counts==dict(zero=99942,mean_std=62406),counts
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    check_old()
    files=[Path(__file__),HERE/'protocol.json',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',destination,HERE/'DRIFT_DIAGNOSIS.json',SOURCE/'train_observations.npz',SOURCE/'dev_observations.npz']
    hashes={**previous['source_hashes'],**{str(path):p.sha256(path) for path in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',source_hashes=hashes,parameters_trainable=counts,gradient_norms=grads,seconds_per_128_training_step=timing,fit_protection_scales=previous['fit_protection_scales'],paired_all_init_exact=True,native_identity_exact=True,old_results_unchanged=old_count,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(params=counts,grads=grads,timing=timing)),flush=True)
def evaluate(device):
    n.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='zero=freshfull last temporal block99942; mean_std=attentiononly62406. OriginalotherF16/native decoder/scorerfixed/nativebaselineK20fixed; onlineobs/coreeval; original loss/cal/clip. Unequalcapacity/fullcontrolrepeat notindependentgeneralization. No intent/ranking/independentgeneratorclaim.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'TEMPORAL_SUBSPACE_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    m.write_json(m.RESULTS/'frozen_audit.json',dict(status='PASS',policies_checked=18,all_nonwhitelist_F16_exact=True,original_decoder_scorer_exact=True,encoder_all_frozen=False,whitelist=dict(zero=PREFIX,mean_std=PREFIX+'self_attn.')))
    old_count=check_old();m.write_json(m.RESULTS/'old_results_audit.json',dict(status='PASS',unchanged=old_count))
    print('COMPLETE '+summary['verdict'],flush=True)
c.preflight=preflight;c.evaluate=evaluate;a.h.preflight=preflight;a.h.evaluate=evaluate
def main():a.h.main()
if __name__=='__main__':main()
