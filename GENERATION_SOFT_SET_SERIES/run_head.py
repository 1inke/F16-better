"""Paired hard winner versus soft minimum set supervision; frozen native inputs."""
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
TAU=None;BACKEND=n.prepare_backend;ORIGINAL_OBJECTIVE=m.objective
def prepare_backend(device):
    global TAU
    result=BACKEND(device)
    if (HERE/'fit_loss_scale.json').exists():TAU=json.loads((HERE/'fit_loss_scale.json').read_text())['temperature_m']
    return result
n.prepare_backend=prepare_backend
class SetHead(a.DirectHead):
    def inputs(self,observation):
        d=self.decoder;past=observation[:,:16].reshape(-1,1,8,2);tokens=observation[:,16:784].reshape(-1,1,8,96)
        context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
        query=d.mode_embedding[None,None].expand(len(observation),1,-1,-1);mode=d.mode_mlp(context[:,:,None]+query)
        return torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0]
n.NativeGenerator=SetHead;c.Correction=SetHead;r.m.Correction=SetHead;n.frozen_audit=a.frozen_audit
def objective(output,base,future,residual,speed,groups,cfg,arm,scale):
    loss=ORIGINAL_OBJECTIVE(output,base,future,residual,speed,groups,cfg,arm,scale)
    if cfg['set_context']=='zero':return loss
    assert TAU is not None and TAU>0 and arm=='balanced_protected'
    error=(output-future[:,None]).norm(dim=-1);quality=error.mean(-1)+cfg['fde_loss_weight']*error[:,:,-1]
    hard=quality.min(-1).values
    soft=-TAU*torch.logsumexp(-quality/TAU,dim=-1)+TAU*math.log(quality.shape[-1])
    target=r.m.labels(speed[:,1:],future[:,None,-1]/12).squeeze(1);weight=r.WEIGHTS[target]
    return loss+(weight*(soft-hard)).mean()
m.objective=objective
def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        parent=ROOT/'GENERATION_BASIS_COMPOSITION_SERIES';files=json.loads((parent/'old_results_manifest.json').read_text())['files']
        for source in parent.rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.name!='runtime_state.json':files[str(source)]=p.sha256(source)
        m.write_json(path,dict(files=files))
    files=json.loads(path.read_text())['files']
    for source,digest in files.items():assert p.sha256(Path(source))==digest,source
    return len(files)
def preflight(device):
    global TAU
    count=old_manifest(create=not(HERE/'old_results_manifest.json').exists());prepare_backend(torch.device('cpu'))
    parent=json.loads((ROOT/'GENERATION_BASIS_COMPOSITION_SERIES/preflight.json').read_text())
    for path,digest in parent['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);assert(len(fit),len(cal))==(25912,2879)
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    native_error=np.linalg.norm(values[1][fit]-values[3][fit,None],axis=-1)
    quality=native_error.mean(-1)+cfg['fde_loss_weight']*native_error[:,:,-1]
    TAU=max(float(np.median(quality.min(-1)))*.1,1e-6)
    fitids=cache['sample_id'][fit].tolist();scale_path=HERE/'fit_loss_scale.json'
    scale_data=dict(temperature_m=TAU,rule='0.1*FIT median native minimum joint ADE+fde_weight*FDE, floor1e-6',fit_ids=fitids,fit_n=len(fit),no_cal_dev_selection=True)
    if scale_path.exists():assert json.loads(scale_path.read_text())==scale_data
    else:m.write_json(scale_path,scale_data)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]];group=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(parent['fit_protection_scales'],device=device)
    # Mathematical semantics: softmin gradient reaches nonwinner candidates, tends to hard minimum.
    test=torch.tensor([[.2,.21,.3,.9]],device=device,requires_grad=True);soft=-TAU*torch.logsumexp(-test/TAU,-1)+TAU*math.log(4);soft.sum().backward()
    expected=torch.softmax(-test.detach()/TAU,-1);assert torch.allclose(test.grad,expected,atol=1e-6) and (test.grad>0).sum()>=2
    assert abs(float(-1e-6*torch.logsumexp(-test.detach()/1e-6,-1)+1e-6*math.log(4)-test.min()))<2e-6
    states=[];timing={};effective={}
    for arm in c.ARMS:
        p.seed_everything(1);model=SetHead(940,c.settings(1,arm)).to(device).eval();assert sum(v.numel() for v in model.parameters() if v.requires_grad)==25158
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
        original=ORIGINAL_OBJECTIVE(out,b,gt,res,s,group,c.settings(1,arm),'balanced_protected',scale)
        if arm=='zero':assert torch.equal(objective(out,b,gt,res,s,group,c.settings(1,arm),'balanced_protected',scale),original)
        with torch.no_grad():
            err=(out-gt[:,None]).norm(dim=-1);q=err.mean(-1)+cfg['fde_loss_weight']*err[:,:,-1];prob=torch.softmax(-q/TAU,-1)
            effective[arm]=float((1/prob.square().sum(-1)).mean())
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
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE/'DESIGN.md',scale_path,HERE/'old_results_manifest.json']
    sources={**parent['source_hashes'],**{str(path):p.sha256(path) for path in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),parameters_trainable=dict(zero=25158,mean_std=25158),fit_protection_scales=parent['fit_protection_scales'],source_hashes=sources,temperature_m=TAU,soft_effective_candidates_FIT128=effective,seconds_per_128step=timing,fullpaired=True,native_identity_exact=True,old_files=count,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(temperature_m=TAU,effective_candidates=effective,timing=timing,old_files=count)),flush=True)
def evaluate(device):
    n.evaluate(device);oldcount=old_manifest();path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Sixfresh80epoch hardwinner(zero) vs softmin(mean_std) supervision;25158newheadparams each/fullpaired/nativeidentity/sameforward. Softmin temperature0.1 FITmedian nativejointminimum fixed; regularizers/protection.1/constrainedcal unchanged. OriginalF16/decoder/scorerfixed/nativeK20+3anchor/commonclip1.5. GT trainingloss only; nointent/ranking/independentgenerator. Hardcontrol previoussamearchitecture/seeds not independentgeneralization.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'SOFT_SET_GATE_FAIL';m.write_json(path,summary)
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
