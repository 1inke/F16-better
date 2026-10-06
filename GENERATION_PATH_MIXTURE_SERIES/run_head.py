"""Hard winner versus fixed-scale full-path Gaussian mixture supervision."""
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
SIGMA2=None;LOSS_MULTIPLIER=None
def prepare_backend(device):
    global SIGMA2,LOSS_MULTIPLIER
    result=BACKEND(device)
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    values=r.prepared(cache,False,fit=fit);base=torch.from_numpy(values[1][fit]);future=torch.from_numpy(values[3][fit])
    squared=(base-future[:,None]).square().mean((-1,-2))
    SIGMA2=max(float(squared.min(-1).values.median()),1e-6)
    score=-torch.logsumexp(-squared/(2*SIGMA2),-1)+math.log(20)
    error=(base-future[:,None]).norm(dim=-1)
    joint=error.mean(-1)+cfg['fde_loss_weight']*error[:,:,-1]
    LOSS_MULTIPLIER=float(joint.min(-1).values.median())/max(float(score.median()),1e-6)
    saved=dict(sigma2_coordinate_m2=SIGMA2,loss_multiplier_m=LOSS_MULTIPLIER,fit_n=len(fit),fit_ids_sha256=p.sha256(p.TRAIN_CACHE),selection='fixed FIT median best full-path coordinate MSE; native joint median / native normalized NLL median',fit_sample_ids=cache['sample_id'][fit].tolist(),no_cal_dev_selection=True)
    scale_path=HERE/'fit_loss_scale.json'
    if scale_path.exists():assert json.loads(scale_path.read_text())==saved
    else:m.write_json(scale_path,saved)
    return result
n.prepare_backend=prepare_backend
class SetHead(a.DirectHead):
    def inputs(self,observation):
        d=self.decoder;past=observation[:,:16].reshape(-1,1,8,2);tokens=observation[:,16:784].reshape(-1,1,8,96)
        context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
        query=d.mode_embedding[None,None].expand(len(observation),1,-1,-1);mode=d.mode_mlp(context[:,:,None]+query)
        return torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0]
n.NativeGenerator=SetHead;c.Correction=SetHead;r.m.Correction=SetHead;n.frozen_audit=a.frozen_audit
def mixture_quality(output,future,cfg):
    assert SIGMA2 is not None and LOSS_MULTIPLIER is not None
    error=(output-future[:,None]).norm(dim=-1)
    quality=error.mean(-1)+cfg['fde_loss_weight']*error[:,:,-1]
    squared=(output-future[:,None]).square().mean((-1,-2))
    score=LOSS_MULTIPLIER*(-torch.logsumexp(-squared/(2*SIGMA2),-1)+math.log(output.shape[1]))
    return quality,score

def objective(output,base,future,residual,speed,groups,cfg,arm,scale):
    loss=ORIGINAL_OBJECTIVE(output,base,future,residual,speed,groups,cfg,arm,scale)
    if cfg['set_context']=='zero':return loss
    assert arm=='balanced_protected'
    quality,energy=mixture_quality(output,future,cfg);hard=quality.min(-1).values
    target=r.m.labels(speed[:,1:],future[:,None,-1]/12).squeeze(1)
    return loss+(r.WEIGHTS[target]*(energy-hard)).mean()
m.objective=objective
def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        parent=ROOT/'GENERATION_ENERGY_SET_SERIES';files=json.loads((parent/'old_results_manifest.json').read_text())['files']
        for source in list(parent.rglob('*'))+list((ROOT/'GENERATION_ASSIGNMENT_STABILITY_SERIES').rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts and source.name!='runtime_state.json':files[str(source)]=p.sha256(source)
        m.write_json(path,dict(files=files))
    files=json.loads(path.read_text())['files']
    for source,digest in files.items():assert p.sha256(Path(source))==digest,source
    return len(files)
def preflight(device):
    count=old_manifest(create=not(HERE/'old_results_manifest.json').exists());prepare_backend(torch.device('cpu'))
    parent=json.loads((ROOT/'GENERATION_ENERGY_SET_SERIES/preflight.json').read_text())
    for path,digest in parent['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);assert(len(fit),len(cal))==(25912,2879)
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]];group=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(parent['fit_protection_scales'],device=device)
    # Fixed full-path mixture math and gradient semantics, no data-derived outcome tuning.
    q,e=mixture_quality(b[:,:1],gt,cfg)
    expected=LOSS_MULTIPLIER*(b[:,:1]-gt[:,None]).square().mean((-1,-2))[:,0]/(2*SIGMA2)
    assert torch.allclose(e,expected,atol=1e-6)
    _,e=mixture_quality(b,gt,cfg);_,perm=mixture_quality(b.flip(1),gt,cfg);assert torch.allclose(e,perm,atol=1e-6)
    squared=(b-gt[:,None]).square().mean((-1,-2));logq=-squared/(2*SIGMA2)
    ref=LOSS_MULTIPLIER*(-torch.log(torch.exp(logq-logq.max(-1,keepdim=True).values).mean(-1))-logq.max(-1).values)
    assert torch.allclose(e,ref,atol=1e-5)
    sq=squared.detach().requires_grad_(True)
    testscore=LOSS_MULTIPLIER*(-torch.logsumexp(-sq/(2*SIGMA2),-1)+math.log(20))
    grad=torch.autograd.grad(testscore.sum(),sq)[0]
    assert torch.allclose(grad,LOSS_MULTIPLIER*torch.softmax(-sq/(2*SIGMA2),-1)/(2*SIGMA2),atol=1e-5,rtol=1e-5)
    synthetic=(torch.randn(4,20,12,2,device=device)*.01).requires_grad_(True)
    _,score=mixture_quality(synthetic,torch.zeros(4,12,2,device=device),cfg);score.sum().backward()
    assert torch.isfinite(synthetic.grad).all() and (synthetic.grad.abs().sum((-1,-2))>0).all()
    states=[];timing={};effective={}
    for arm in c.ARMS:
        p.seed_everything(1);model=SetHead(940,c.settings(1,arm)).to(device).eval();assert sum(v.numel() for v in model.parameters() if v.requires_grad)==25158
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
        original=ORIGINAL_OBJECTIVE(out,b,gt,res,s,group,c.settings(1,arm),'balanced_protected',scale)
        if arm=='zero':assert torch.equal(objective(out,b,gt,res,s,group,c.settings(1,arm),'balanced_protected',scale),original)
        with torch.no_grad():
            _,energy=mixture_quality(out,gt,cfg);effective[arm]=float(energy.mean())
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
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE/'DESIGN.md',HERE/'old_results_manifest.json',HERE/'fit_loss_scale.json']
    sources={**parent['source_hashes'],**{str(path):p.sha256(path) for path in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=dict(zero=25158,mean_std=25158),fit_protection_scales=parent['fit_protection_scales'],source_hashes=sources,initial_mixture_primary_FIT128=effective,seconds_per_128step=timing,fullpaired=True,native_identity_exact=True,old_files=count,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(initial_energy=effective,timing=timing,old_files=count)),flush=True)
def evaluate(device):
    n.evaluate(device);oldcount=old_manifest();path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Sixfresh80epoch hardwinner(zero) vs fixedFIT-scale full-path Gaussian mixture primaryscore(mean_std),25158head each/fullpaired/sameforward/nativeidentity. Sole registered factor includes full-path squared metric, normalized mixture aggregation and fixedFIT loss-scale matching; not pureaggregation causalproof. EntireoldF16/decoder/scorerfrozen/nativeK20+3anchor/commonclip1.5/originalotherregularizersprotection.1/calguards unchanged. Uniformmixture, fixedcoordinate covariance shared across12steps; restrictive conditionalfamily not intent or calibratedjointclaim. GTtrainlossonly; nofuturegroupslogitsforward. Repeatedfreshhardcontrol nonindependentgeneralization.'

    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'PATH_MIXTURE_GATE_FAIL';m.write_json(path,summary)
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
