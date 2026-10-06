"""Hard winner versus equal-weight ensemble energy-like set supervision."""
from pathlib import Path
import importlib.util,sys,json,time,traceback,math
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('soft_direct_source',ROOT/'GENERATION_DIRECT_ANCHOR_SERIES/run_head.py')
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
n,c,m,r,p=a.n,a.c,a.m,a.r,a.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
BACKEND=n.prepare_backend;ORIGINAL_OBJECTIVE=m.objective
def prepare_backend(device):return BACKEND(device)
n.prepare_backend=prepare_backend
class SetHead(a.DirectHead):
    def inputs(self,observation):
        d=self.decoder;past=observation[:,:16].reshape(-1,1,8,2);tokens=observation[:,16:784].reshape(-1,1,8,96)
        context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
        query=d.mode_embedding[None,None].expand(len(observation),1,-1,-1);mode=d.mode_mlp(context[:,:,None]+query)
        return torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0]
n.NativeGenerator=SetHead;c.Correction=SetHead;r.m.Correction=SetHead;n.frozen_audit=a.frozen_audit
def energy_quality(output,future,cfg):
    error=(output-future[:,None]).norm(dim=-1)
    quality=error.mean(-1)+cfg['fde_loss_weight']*error[:,:,-1]
    pair=torch.cdist(output.transpose(1,2).contiguous(),output.transpose(1,2).contiguous(),compute_mode='donot_use_mm_for_euclid_dist').mean(1)
    endpoint=torch.cdist(output[:,:,-1].contiguous(),output[:,:,-1].contiguous(),compute_mode='donot_use_mm_for_euclid_dist')
    return quality,quality.mean(-1)-.5*(pair+cfg['fde_loss_weight']*endpoint).mean((-1,-2))
def objective(output,base,future,residual,speed,groups,cfg,arm,scale):
    loss=ORIGINAL_OBJECTIVE(output,base,future,residual,speed,groups,cfg,arm,scale)
    if cfg['set_context']=='zero':return loss
    assert arm=='balanced_protected'
    quality,energy=energy_quality(output,future,cfg);hard=quality.min(-1).values
    target=r.m.labels(speed[:,1:],future[:,None,-1]/12).squeeze(1)
    return loss+(r.WEIGHTS[target]*(energy-hard)).mean()
m.objective=objective
def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        parent=ROOT/'GENERATION_SOFT_SET_SERIES';files=json.loads((parent/'old_results_manifest.json').read_text())['files']
        for source in list(parent.rglob('*'))+list((ROOT/'GENERATION_ASSIGNMENT_STABILITY_SERIES').rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts and source.name!='runtime_state.json':files[str(source)]=p.sha256(source)
        m.write_json(path,dict(files=files))
    files=json.loads(path.read_text())['files']
    for source,digest in files.items():assert p.sha256(Path(source))==digest,source
    return len(files)
def preflight(device):
    count=old_manifest(create=not(HERE/'old_results_manifest.json').exists());prepare_backend(torch.device('cpu'))
    parent=json.loads((ROOT/'GENERATION_SOFT_SET_SERIES/preflight.json').read_text())
    for path,digest in parent['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);assert(len(fit),len(cal))==(25912,2879)
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]];group=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(parent['fit_protection_scales'],device=device)
    # Equal-weight finite ensemble score: K=1 reduces to quality; candidate permutation invariant.
    q,e=energy_quality(b[:,:1],gt,cfg);assert torch.allclose(q[:,0],e,atol=1e-6)
    q,e=energy_quality(b,gt,cfg);_,perm=energy_quality(b.flip(1),gt,cfg);assert torch.allclose(e,perm,atol=1e-6)
    manual=(b[:,:,None]-b[:,None]).norm(dim=-1);ref=q.mean(-1)-.5*(manual.mean(-1)+cfg['fde_loss_weight']*manual[:,:,:,-1]).mean((-1,-2));assert torch.allclose(ref,e,atol=1e-6)
    synthetic=torch.randn(4,20,12,2,device=device,requires_grad=True);_,es=energy_quality(synthetic,torch.randn(4,12,2,device=device),cfg);es.sum().backward();assert torch.isfinite(synthetic.grad).all() and (synthetic.grad.abs().sum((-1,-2))>0).all()
    states=[];timing={};effective={}
    for arm in c.ARMS:
        p.seed_everything(1);model=SetHead(940,c.settings(1,arm)).to(device).eval();assert sum(v.numel() for v in model.parameters() if v.requires_grad)==25158
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
        original=ORIGINAL_OBJECTIVE(out,b,gt,res,s,group,c.settings(1,arm),'balanced_protected',scale)
        if arm=='zero':assert torch.equal(objective(out,b,gt,res,s,group,c.settings(1,arm),'balanced_protected',scale),original)
        with torch.no_grad():
            _,energy=energy_quality(out,gt,cfg);effective[arm]=float(energy.mean())
        opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);start=time.perf_counter()
        for step in range(8):
            opt.zero_grad();out,res=model(o,b,s);loss=objective(out,b,gt,res,s,group,c.settings(1,arm),'balanced_protected',scale);assert torch.isfinite(loss);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.head.parameters())
            assert model.head[-1].weight.grad.abs().sum()>0
            if step>0:assert model.head[0].weight.grad.abs().sum()>0
            opt.step();a.frozen_audit(model)
        torch.cuda.synchronize();timing[arm]=(time.perf_counter()-start)/8
        assert out.shape==(128,20,12,2) and torch.isfinite(out).all() and (res.abs()<=1.5*(s[:,0,None,None,None]+.03)+1e-6).all()
    assert all(torch.equal(states[0][key],states[1][key]) for key in states[0]);old_manifest()
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE/'DESIGN.md',HERE/'old_results_manifest.json']
    sources={**parent['source_hashes'],**{str(path):p.sha256(path) for path in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=dict(zero=25158,mean_std=25158),fit_protection_scales=parent['fit_protection_scales'],source_hashes=sources,initial_energy_primary_FIT128=effective,seconds_per_128step=timing,fullpaired=True,native_identity_exact=True,old_files=count,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(initial_energy=effective,timing=timing,old_files=count)),flush=True)
def evaluate(device):
    n.evaluate(device);oldcount=old_manifest();path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Sixfresh80epoch hardwinner(zero) vs equalweight candidate energy-like marginal distance score(mean_std),25158headparams each/fullpaired/sameforward/nativeidentity. MeanGTquality minus halfpairquality replaceswinner only; pairhalf intrinsic nottunedsweep. Originaldiversity/residual/smoothness/protection.1/cal unchanged. AlloldF16/decoder/scorerfixed/nativeK20+3anchor/commonclip1.5. GTlossonly/nointent/ranking/independentgenerator. MarginalADE+FDEmetric doesnotidentify fulljoint temporaldistribution; notclaimproperjoint calibration. Hardcontrol repeats previouseffectivearchitecture/seeds notindependentgeneralization.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'ENERGY_SET_GATE_FAIL';m.write_json(path,summary)
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
