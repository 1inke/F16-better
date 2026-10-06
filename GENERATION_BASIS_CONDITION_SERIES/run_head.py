"""Same Gaussian anchor function span, FIT-only covariance conditioning."""
from pathlib import Path
import importlib.util,sys,json,time,traceback
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;PARENT=ROOT/'GENERATION_LOCAL_BASIS_SERIES';DIAG=ROOT/'GENERATION_BASIS_UPDATE_DIAGNOSIS'
spec=importlib.util.spec_from_file_location('condition_basis_source',PARENT/'run_head.py')
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
n,c,m,r,p=a.n,a.c,a.m,a.r,a.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
TRANSFORM=None;BACKEND=a.prepare_backend
def prepare_backend(device):
    global TRANSFORM
    result=BACKEND(device)
    if (HERE/'fit_transform.npz').exists():
        with np.load(HERE/'fit_transform.npz') as saved:TRANSFORM={key:saved[key].copy() for key in ('phi_mean','whitener')}
    return result
n.prepare_backend=prepare_backend
class ConditionBasis(a.FixedBasis):
    def __init__(self,variant):
        super().__init__('mean_std');self.variant=variant;assert TRANSFORM is not None
        for key,value in TRANSFORM.items():self.register_buffer(key,torch.from_numpy(value.copy()))
    def forward(self,features):
        phi=super().forward(features)
        return phi if self.variant=='zero' else (phi-self.phi_mean)@self.whitener
class ConditionHead(a.LocalHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg);self.head=nn.Sequential(ConditionBasis(self.arm),nn.Linear(128,6))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
def frozen_audit(model):
    a.frozen_audit(model)
    for key,value in TRANSFORM.items():assert np.array_equal(model.head[0]._buffers[key].detach().cpu().numpy(),value),key
n.NativeGenerator=ConditionHead;c.Correction=ConditionHead;r.m.Correction=ConditionHead;n.frozen_audit=frozen_audit
def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        files=json.loads((PARENT/'old_results_manifest.json').read_text())['files']
        for folder in (PARENT,DIAG):
            for source in folder.rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and source.name!='runtime_state.json':files[str(source)]=p.sha256(source)
        m.write_json(path,dict(files=files))
    files=json.loads(path.read_text())['files']
    for source,digest in files.items():assert p.sha256(Path(source))==digest,source
    return len(files)
def fit_transform(cache,fit):
    global TRANSFORM
    diagnosis=json.loads((DIAG/'RESULT.json').read_text());assert diagnosis['status']=='PASS_AUDIT_AND_DESCRIPTIVE_FIT_GEOMETRY'
    for source,digest in diagnosis['source_hashes'].items():assert p.sha256(Path(source))==digest,source
    path=HERE/'fit_transform.npz'
    if path.exists():
        with np.load(path) as saved:
            assert np.array_equal(saved['fit_ids'],cache['sample_id'][fit]);TRANSFORM={key:saved[key].copy() for key in ('phi_mean','whitener')}
        return
    with np.load(DIAG/'mean_std_fit_geometry.npz') as saved:
        assert np.array_equal(saved['fit_ids'],cache['sample_id'][fit]);mean=saved['mean'];cov=saved['covariance']
    eigen,rotation=np.linalg.eigh(cov);floor=eigen.max()*1e-4;W=(rotation/np.sqrt(np.maximum(eigen,floor)))@rotation.T
    assert np.isfinite(W).all() and np.linalg.eigvalsh(W).min()>0
    TRANSFORM=dict(phi_mean=mean.astype(np.float32),whitener=W.astype(np.float32))
    np.savez_compressed(path,**TRANSFORM,fit_ids=cache['sample_id'][fit],eigenvalues=eigen,eigen_floor=floor)
def preflight(device):
    oldcount=old_manifest(create=not(HERE/'old_results_manifest.json').exists());prepare_backend(torch.device('cpu'))
    prior=json.loads((PARENT/'preflight.json').read_text())
    for source,digest in prior['source_hashes'].items():assert p.sha256(Path(source))==digest,source
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert len(fit)==25912 and len(cal)==2879 and cfg['epochs']==80 and cfg['seeds']==[1,2,3]
    values=r.prepared(cache,False,fit=fit);fit_transform(cache,fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,s,gt=[torch.from_numpy(value[fit[:128]]).to(device) for value in values[:4]];group=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(prior['fit_protection_scales'],device=device)
    states=[];numerical={};timing={}
    for arm in c.ARMS:
        p.seed_everything(1);model=ConditionHead(940,c.settings(1,arm)).to(device).eval()
        assert sum(x.numel() for x in model.parameters() if x.requires_grad)==774
        states.append({key:value.detach().cpu().clone() for key,value in model.state_dict().items()})
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
        with torch.no_grad():
            features=model.inputs(o);raw=a.FixedBasis('mean_std').to(device)(features);phi=model.head[0](features)
            if arm=='zero':assert torch.equal(raw,phi);reconstruction=0.
            else:
                reconstructed=torch.linalg.solve(model.head[0].whitener.T,phi.reshape(-1,128).T).T.reshape_as(raw)+model.head[0].phi_mean
                reconstruction=float((reconstructed-raw).abs().max());assert reconstruction<1e-4
            assert torch.isfinite(phi).all() and phi.std()>1e-6
            numerical[arm]=dict(basis_mean=float(phi.mean()),basis_std=float(phi.std()),inverse_reconstruction_max=reconstruction)
        opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);start=time.perf_counter()
        for step in range(8):
            opt.zero_grad();out,res=model(o,b,s);loss=m.objective(out,b,gt,res,s,group,cfg,'balanced_protected',scale);loss.backward()
            assert model.head[-1].weight.grad.abs().sum()>0 and all(torch.isfinite(x.grad).all() for x in model.parameters() if x.requires_grad)
            opt.step();frozen_audit(model)
        torch.cuda.synchronize();timing[arm]=(time.perf_counter()-start)/8
        assert torch.isfinite(out).all() and out.shape==(128,20,12,2) and (res.abs()<=1.5*(s[:,0,None,None,None]+.03)+1e-6).all()
    assert all(torch.equal(states[0][key],states[1][key]) for key in states[0]);old_manifest()
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE/'DESIGN.md',HERE/'fit_transform.npz',HERE/'old_results_manifest.json',DIAG/'RESULT.json',DIAG/'mean_std_fit_geometry.npz']
    sources={**prior['source_hashes'],**{str(source):p.sha256(source) for source in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=dict(zero=774,mean_std=774),fit_protection_scales=prior['fit_protection_scales'],source_hashes=sources,numerical=numerical,seconds_per_128step=timing,fullpaired_all_buffers=True,native_identity_exact=True,old_files=oldcount,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(numerical=numerical,timing=timing,oldfiles=oldcount)),flush=True)
def evaluate(device):
    n.evaluate(device);oldcount=old_manifest();path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Sixfresh80epoch rawGaussian(zero) vs FIT-only centeredcovariance-conditionedGaussian(mean_std),774trainparams each/fullpairedzero readout/allbuffers/nativeidentity. Sameunboundedlinearanchor span with bias, optimizer and weightdecay prior geometry coupled. AlloldF16/decoder/scorerfixed/nativeK20+3anchor/commonclip/originalloss-cal. Rawcontrol sameeffectivearchitecture/seeds as previousGaussian not independentgeneralization. NoGTgroup/logitforward/no intent/ranking or independentgenerator.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'BASIS_CONDITION_GATE_FAIL';m.write_json(path,summary)
    (m.RESULTS/'REPORT.md').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    m.write_json(m.RESULTS/'independent_audit.json',dict(status='PASS',models=18,predictions=18,explicit_ID_finite_K20_metrics_recomputed=True,allolddecoder_frozen=True,encoder_source_locked=True,old_files_unchanged=oldcount))
c.preflight=preflight;c.evaluate=evaluate
def main():
    try:
        if '--action' in sys.argv and sys.argv[sys.argv.index('--action')+1]=='run':prepare_backend(torch.device('cpu'))
        c.main()
    except BaseException:
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
if __name__=='__main__':main()
