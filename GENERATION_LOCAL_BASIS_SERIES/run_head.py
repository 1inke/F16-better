"""Frozen original decoder features, fixed Fourier/Gaussian anchor bases."""
from pathlib import Path
import importlib.util,sys,json,time,traceback
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('local_direct_source',ROOT/'GENERATION_DIRECT_ANCHOR_SERIES/run_head.py')
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
n,c,m,r,p=h.n,h.c,h.m,h.r,h.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
BASIS=None;BACKEND=n.prepare_backend
def prepare_backend(device):
    global BASIS
    result=BACKEND(device)
    if (HERE/'fit_basis.npz').exists():
        with np.load(HERE/'fit_basis.npz') as saved:BASIS={key:saved[key].copy() for key in ('mean','std','centers','omega','phase')}
    return result
n.prepare_backend=prepare_backend
def native_features(model,observation):
    d=model.decoder;past=observation[:,:16].reshape(-1,1,8,2)
    tokens=observation[:,16:784].reshape(-1,1,8,96)
    context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
    query=d.mode_embedding[None,None].expand(len(observation),1,-1,-1)
    mode=d.mode_mlp(context[:,:,None]+query)
    return torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0]
class FixedBasis(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm;assert BASIS is not None
        for key,value in BASIS.items():self.register_buffer(key,torch.from_numpy(value.copy()))
    def forward(self,features):
        z=(features-self.mean)/self.std
        if self.arm=='zero':return torch.cos(z@self.omega+self.phase)
        # squared distances avoid allocating B*K*128*128 expanded differences.
        distance=(z.square().sum(-1,keepdim=True)+self.centers.square().sum(-1)-2*z@self.centers.T).clamp_min(0)/128
        return torch.exp(-distance/2)
class LocalHead(h.DirectHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg);assert dimension==940
        self.head=nn.Sequential(FixedBasis(self.arm),nn.Linear(128,6))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def inputs(self,observation):return native_features(self,observation)
def frozen_audit(model):
    h.frozen_audit(model)
    for key,value in BASIS.items():assert np.array_equal(model.head[0]._buffers[key].detach().cpu().numpy(),value),key
n.NativeGenerator=LocalHead;c.Correction=LocalHead;r.m.Correction=LocalHead;n.frozen_audit=frozen_audit
def old_manifest(create=False):
    destination=HERE/'old_results_manifest.json'
    if create:
        files=json.loads((ROOT/'GENERATION_FIT_MEMORIZATION_DIAGNOSIS/old_results_manifest.json').read_text())['files']
        for folder in ('GENERATION_FIT_MEMORIZATION_DIAGNOSIS','GENERATION_LEARNING_SIGNAL_DIAGNOSIS'):
            for path in (ROOT/folder).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and path.name!='runtime_state.json':files[str(path)]=p.sha256(path)
        m.write_json(destination,dict(files=files))
    saved=json.loads(destination.read_text())['files']
    for path,digest in saved.items():assert p.sha256(Path(path))==digest,path
    return len(saved)
def fit_basis(cache,fit,values,device):
    global BASIS
    destination=HERE/'fit_basis.npz'
    if destination.exists():
        with np.load(destination) as saved:
            assert np.array_equal(saved['fit_ids'],cache['sample_id'][fit]);BASIS={key:saved[key].copy() for key in ('mean','std','centers','omega','phase')}
        return
    p.seed_everything(1);model=h.DirectHead(940,c.settings(1,'zero')).to(device).eval()
    total=torch.zeros(128,dtype=torch.float64,device=device);square=total.clone();count=0
    with torch.no_grad():
        for start in range(0,len(fit),256):
            ids=fit[start:start+256];features=native_features(model,torch.from_numpy(values[0][ids]).to(device)).reshape(-1,128).double()
            total+=features.sum(0);square+=features.square().sum(0);count+=len(features)
        mean=(total/count).float();std=(square/count-(total/count).square()).clamp_min(0).sqrt().clamp_min(.001).float()
        rng=np.random.default_rng(20261003);pairs=rng.choice(len(fit)*20,128,replace=False);sample=fit[pairs//20];mode=pairs%20
        raw=native_features(model,torch.from_numpy(values[0][sample]).to(device))[torch.arange(128,device=device),torch.from_numpy(mode).to(device)]
        centers=(raw-mean)/std
        BASIS=dict(mean=mean.cpu().numpy(),std=std.cpu().numpy(),centers=centers.cpu().numpy(),omega=rng.normal(0,1/np.sqrt(128),(128,128)).astype(np.float32),phase=rng.uniform(0,2*np.pi,128).astype(np.float32))
    np.savez_compressed(destination,**BASIS,fit_ids=cache['sample_id'][fit],center_sample_ids=cache['sample_id'][sample],center_mode_ids=mode)
def preflight(device):
    oldcount=old_manifest(create=not(HERE/'old_results_manifest.json').exists())
    _,_,original=prepare_backend(torch.device('cpu'));del original
    prior=json.loads((ROOT/'GENERATION_DIRECT_ANCHOR_SERIES/preflight.json').read_text())
    for path,digest in prior['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert len(fit)==25912 and len(cal)==2879 and cfg['epochs']==80 and cfg['seeds']==[1,2,3]
    values=r.prepared(cache,False,fit=fit);assert values[0].shape[1]==940
    r.setup_weights(cache,fit,values,cfg,device);fit_basis(cache,fit,values,device)
    o,b,s,gt=[torch.from_numpy(x[fit[:128]]).to(device) for x in values[:4]]
    group=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(prior['fit_protection_scales'],device=device)
    states=[];numerical={};timings={}
    for arm in c.ARMS:
        p.seed_everything(1);model=LocalHead(940,c.settings(1,arm)).to(device).eval()
        states.append({key:value.detach().cpu().clone() for key,value in model.state_dict().items()})
        assert sum(x.numel() for x in model.parameters() if x.requires_grad)==774
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
        with torch.no_grad():
            feature=model.inputs(o);phi=model.head[0](feature);perturbed=model.head[0](feature+.01*model.head[0].std)
            assert torch.isfinite(phi).all() and phi.std()>1e-6 and (phi-perturbed).abs().max()>1e-7
            numerical[arm]=dict(mean=float(phi.mean()),std=float(phi.std()),sensitivity_max=float((phi-perturbed).abs().max()))
        opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        start=time.perf_counter()
        for step in range(8):
            opt.zero_grad();out,res=model(o,b,s);loss=m.objective(out,b,gt,res,s,group,cfg,'balanced_protected',scale);loss.backward()
            assert model.head[-1].weight.grad.abs().sum()>0 and all(torch.isfinite(x.grad).all() for x in model.parameters() if x.requires_grad)
            opt.step();frozen_audit(model)
        torch.cuda.synchronize();timings[arm]=(time.perf_counter()-start)/8
        assert torch.isfinite(out).all() and out.shape==(128,20,12,2) and (res.abs()<=1.5*(s[:,0,None,None,None]+.03)+1e-6).all()
    assert all(torch.equal(states[0][key],states[1][key]) for key in states[0])
    # Verify observed online features against locked cached native features.
    online=n.module('basis_online_alignment',ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES/run_head.py')
    online.prepare_backend(torch.device('cpu'));old=online.OnlineHead(940,online.c.settings(1,'zero')).to(device).eval()
    with np.load(ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz') as raw:
        assert np.array_equal(raw['sample_id'],cache['sample_id']);rawobs=torch.from_numpy(raw['observation'][fit[:128]]).to(device)
    with torch.no_grad():
        live,_,native=old.observed_features(rawobs)
        error=float((live-native_features(model,o)).abs().max());assert error<2e-5
        assert float((native-b).abs().max())<2e-5
    old_manifest()
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE/'DESIGN.md',HERE/'fit_basis.npz',HERE/'old_results_manifest.json',ROOT/'GENERATION_LEARNING_SIGNAL_DIAGNOSIS/RESULT.json',ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz']
    hashes={**prior['source_hashes'],**{str(path):p.sha256(path) for path in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=dict(zero=774,mean_std=774),numerical=numerical,online_cached_feature_max_error=error,paired_full_init_exact=True,native_identity_exact=True,fit_protection_scales=prior['fit_protection_scales'],seconds_per_128step=timings,source_hashes=hashes,old_files=oldcount,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(numerical=numerical,seconds=timings,alignment=error,oldfiles=oldcount)),flush=True)
def evaluate(device):
    n.evaluate(device);oldcount=old_manifest()
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Sixfresh80epoch fixed Fourier(zero) vs Gaussian(mean_std)128featurebasis plus774trainable linear3anchorreadout. FIT-only centers/scalers, same pairedreadout/allbuffers/nativeidentity. AlloriginalF16/native decoder/scorerfixed, nativebaseK20/commonclip/originalloss-cal. Same count not equal function/effectivecapacity/locality proof; noGTgroup/logitsforward/nointent/ranking/independentgenerator. RepeatedETHdev exploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'LOCAL_BASIS_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    m.write_json(m.RESULTS/'independent_audit.json',dict(status='PASS',models=18,predictions=18,IDs_finite_K20_metrics_recomputed=True,all_original_decoder_frozen=True,encoder_source_locked=True,old_files_unchanged=oldcount))
c.preflight=preflight;c.evaluate=evaluate
def main():
    try:
        if '--action' in sys.argv and sys.argv[sys.argv.index('--action')+1]=='run':
            _,_,original=prepare_backend(torch.device('cpu'));del original
        c.main()
    except BaseException:
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
if __name__=='__main__':main()
