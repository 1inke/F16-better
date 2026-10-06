"""Online frozen-versus-last-temporal-block F16 adaptation; fixed native baseline."""
from pathlib import Path
import importlib.util,sys,json,time,copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('online_direct_source',HERE.parent/'GENERATION_DIRECT_ANCHOR_SERIES/run_head.py')
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
n=h.n;c=h.c;m=h.m;r=h.r;p=h.p
h.HERE=HERE;c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
BACKEND=n.prepare_backend;BASE_LOAD=p.load_npz;BASE_PREPARED=r.prepared
ORIGINAL=None;ORIGINAL_STATE=None;BACKEND_CFG=None;TRAINER=None
PREFIX='agent_aware_trans_encoder.temporal_encoder.layers.0.'
def prepare_backend(device):
    global ORIGINAL,ORIGINAL_STATE,BACKEND_CFG,TRAINER
    cfg,trainer,model=BACKEND(device)
    assert len(model.agent_aware_trans_encoder.temporal_encoder.layers)==1 and model.num_pedestrians==9 and model.token_dim==96
    ORIGINAL=copy.deepcopy(model).cpu().eval()
    ORIGINAL_STATE={k:v.detach().cpu().clone() for k,v in ORIGINAL.state_dict().items()}
    BACKEND_CFG=cfg;TRAINER=trainer
    return cfg,trainer,model
n.prepare_backend=prepare_backend
def raw_observations(split,cache):
    destination=HERE/(split+'_observations.npz')
    if destination.exists():
        saved=n.RAW_LOAD(destination);assert np.array_equal(saved['sample_id'],cache['sample_id']);return saved['observation']
    assert ORIGINAL is not None
    dataset=TRAINER.TrajectoryDataset(*n.f.fold_data('eth',split),BACKEND_CFG)
    parts=[]
    for past,future,num_valid,*_ in DataLoader(Subset(dataset,cache['sample_id'].tolist()),batch_size=256,shuffle=False,num_workers=0):
        local,_=TRAINER.normalize_to_focal_heading(past,past)
        center=local.mean(-2,keepdim=True)
        obs=np.zeros((len(past),940),np.float32)
        obs[:,:16]=local[:,0].reshape(-1,16).numpy()
        obs[:,16:160]=(local-center).reshape(-1,144).numpy()
        obs[:,160:304]=local.reshape(-1,144).numpy()
        obs[:,304]=num_valid.reshape(-1).numpy()
        parts.append(obs)
    obs=np.concatenate(parts);assert len(obs)==len(cache['sample_id']) and np.isfinite(obs).all()
    np.savez_compressed(destination,sample_id=cache['sample_id'],observation=obs)
    return obs
def load_cache(path):
    cache=BASE_LOAD(path)
    if Path(path) in (p.TRAIN_CACHE,p.DEV_CACHE):
        cache['_online_observation']=raw_observations('train' if Path(path)==p.TRAIN_CACHE else 'dev',cache)
        np.testing.assert_allclose(cache['_online_observation'][:,:16],cache['_past'].reshape(-1,16),atol=1e-6,rtol=0)
    return cache
p.load_npz=load_cache
def prepared(cache,replace,**kwargs):
    values=list(BASE_PREPARED(cache,replace,**kwargs))
    values[0]=cache['_online_observation'];return values
r.prepared=prepared
class OnlineHead(h.DirectHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        assert dimension==940 and ORIGINAL is not None
        self.original=copy.deepcopy(ORIGINAL)
        for name,parameter in self.original.named_parameters():parameter.requires_grad_(self.arm=='mean_std' and name.startswith(PREFIX))
        self.original.eval()
    def train(self,mode=True):
        super().train(mode);self.original.eval();return self
    def observed_features(self,observation):
        # ASTRA generates positional encodings using a plain device attribute;
        # Module.to() alone cannot update it after CPU checkpoint construction.
        self.original.device=str(observation.device)
        batch=len(observation);centered=observation[:,16:160].reshape(batch,9,8,2);local=observation[:,160:304].reshape(batch,9,8,2)
        num_valid=observation[:,304].int();captured=[]
        def capture(owner,args):captured.append(args[0][:,0])
        handle=self.original.decoder.register_forward_pre_hook(capture)
        try:
            if self.arm=='zero':
                with torch.no_grad():prediction=self.original(centered,None,None,mode='testing',num_valid=num_valid,graph_loc=local)[2]
            else:prediction=self.original(centered,None,None,mode='testing',num_valid=num_valid,graph_loc=local)[2]
        finally:handle.remove()
        assert len(captured)==1
        tokens=captured[0];past=observation[:,:16].reshape(batch,1,8,2)
        context=self.decoder.context(torch.cat((tokens[:,-1,None],tokens.mean(1)[:,None],self.decoder.motion_features(past)),-1))
        query=self.decoder.mode_embedding[None,None].expand(batch,1,-1,-1)
        mode=self.decoder.mode_mlp(context[:,:,None]+query)
        features=torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0]
        relative=prediction[:,0]+local.mean(-2,keepdim=True)[:,0,None]-local[:,0,-1,None,None]
        return features,tokens,relative
    def inputs(self,observation):return self.observed_features(observation)[0]
def frozen_audit(model):
    h.frozen_audit(model)
    for name,value in model.original.state_dict().items():
        if not (model.arm=='mean_std' and name.startswith(PREFIX)):assert torch.equal(value.detach().cpu(),ORIGINAL_STATE[name]),name
    for name,parameter in model.original.named_parameters():assert parameter.requires_grad==(model.arm=='mean_std' and name.startswith(PREFIX)),name
n.NativeGenerator=OnlineHead;c.Correction=OnlineHead;r.m.Correction=OnlineHead;n.frozen_audit=frozen_audit
def preflight(device):
    _,_,source=prepare_backend(device);del source
    inspection=json.loads((HERE/'backend_inspection.json').read_text())
    assert inspection['allowed_parameters']==74784
    prior=json.loads((HERE.parent/'GENERATION_SOCIAL_RESIDUAL_SERIES/preflight.json').read_text())
    for path,digest in prior['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),.1)
    assert len(fit)==25912 and len(cal)==2879 and cfg['epochs']==80 and cfg['seeds']==[1,2,3]
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,v,gt=[torch.from_numpy(a[fit[:32]]).to(device) for a in values[:4]]
    groups=torch.tensor(cache['change_group'][fit[:32]],device=device);scale=torch.tensor(prior['fit_protection_scales'],device=device)
    counts={};states=[];times={};grads={};alignment={}
    for arm in c.ARMS:
        p.seed_everything(1);model=OnlineHead(940,c.settings(1,arm)).to(device);model.eval()
        states.append({k:a.detach().cpu().clone() for k,a in model.state_dict().items()})
        with torch.no_grad():
            features,tokens,native=model.observed_features(o)
            alignment[arm]=float(abs(native-b).max());assert alignment[arm]<2e-5
            np.testing.assert_allclose(tokens.cpu().numpy(),cache['_tokens'][fit[:32]],rtol=0,atol=2e-5)
        out,res=model(o,b,v);assert torch.equal(out,b) and res.abs().sum()==0
        counts[arm]=sum(a.numel() for a in model.parameters() if a.requires_grad)
        optimizer=torch.optim.AdamW([a for a in model.parameters() if a.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        start=time.perf_counter()
        for step in range(4):
            model.train();optimizer.zero_grad();out,res=model(o,b,v)
            m.objective(out,b,gt,res,v,groups,cfg,'balanced_protected',scale).backward()
            assert all(a.grad is not None and torch.isfinite(a.grad).all() for a in model.parameters() if a.requires_grad)
            if step==3:
                assert model.head[0].weight.grad.abs().sum()>0
                grads[arm]=sum(float(a.grad.norm()) for name,a in model.original.named_parameters() if a.requires_grad)
                assert grads[arm]>0 if arm=='mean_std' else grads[arm]==0
            optimizer.step();frozen_audit(model)
        torch.cuda.synchronize();times[arm]=(time.perf_counter()-start)/4
        assert torch.isfinite(out).all() and out.shape==(32,20,12,2)
        assert (res.abs()<=model.strength*(v[:,0,None,None,None]+.03)+1e-6).all()
        del model
    assert counts==dict(zero=25158,mean_std=99942),counts
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE/'backend_inspection.json',HERE/'inspect_backend.py',HERE/'old_results_manifest.json',HERE/'train_observations.npz']
    hashes={**prior['source_hashes'],**{str(a):p.sha256(a) for a in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',parameters_trainable=counts,paired_full_init_exact=True,native_identity_exact=True,online_observation_forward=True,allowed_module=PREFIX,encoder_gradient_norm=grads,initial_native_alignment=alignment,fit_protection_scales=prior['fit_protection_scales'],source_hashes=hashes,seconds_per_32step=times,fit_n=len(fit),cal_n=len(cal),test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(parameters=counts,encodergrads=grads,seconds_per_32step=times,alignment=alignment)),flush=True)
h.preflight=preflight;c.preflight=preflight
def evaluate(device):
    n.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Sixfresh80epoch online observations/frozen F16 vs onlylasttemporalblock trainable plus samefreshanchorhead. Nativebaseline K20 fixed/allotherF16/originaldecoder/scorer frozen. 25158 vs99942 trainableparams capacitycoupled scopeintervention/no equalcapacityproof/intent/ranking/independentgenerator claim. Originalloss/cal/bound/testisolation. RepeatedETHdev exploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'UPSTREAM_TEMPORAL_GATE_FAIL'
    m.write_json(m.RESULTS/'frozen_audit.json',dict(status='PASS',policies_checked=18,all_non_whitelist_F16_exact=True,original_decoder_scorer_exact=True,source_encoder_checkpoint_locked=True,encoder_all_frozen=False,trainable_encoder_whitelist=PREFIX))
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)
h.evaluate=evaluate;c.evaluate=evaluate
def main():h.main()
if __name__=='__main__':main()
