"""Single scope intervention: retain pretrained mode/context, train native geometry."""
from pathlib import Path
import sys,json,time,traceback,copy
import importlib.util
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
CONTROL=HERE.parent/'GENERATION_NATIVE_DECODER_SERIES'
spec=importlib.util.spec_from_file_location('geometry_native_source',CONTROL/'run_native.py')
n=importlib.util.module_from_spec(spec);sys.modules[spec.name]=n;spec.loader.exec_module(n)
c=n.c;m=n.m;r=n.r;p=n.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
BaseNative=n.NativeGenerator
class GeometryGenerator(BaseNative):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        if cfg['set_context']=='mean_std':
            allowed=('anchor_templates','proposal_scale.','proposal_delta.','refiner.')
            for name,parameter in self.decoder.named_parameters():parameter.requires_grad_(name.startswith(allowed))
        assert not any(x.requires_grad for x in self.decoder.context.parameters())
        assert not self.decoder.mode_embedding.requires_grad
        assert not any(x.requires_grad for x in self.decoder.mode_mlp.parameters())
c.Correction=GeometryGenerator;r.m.Correction=GeometryGenerator;n.NativeGenerator=GeometryGenerator
def checkpoint(seed,arm,tag):
    root=CONTROL if arm=='zero' else HERE
    return root/'results/models'/arm/f'seed_{seed}'/c.d.ARM/f'{tag}.pth'
c.checkpoint=checkpoint
def preflight(device):
    _,_,model=n.prepare_backend(device);del model
    prior=json.loads((CONTROL/'preflight.json').read_text())
    for path,digest in prior['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    allmodels=json.loads((CONTROL/'results/model_lock.json').read_text())['models']
    for path,digest in allmodels.items():assert p.sha256(Path(path))==digest,path
    cfg=c.settings(1);cache=p.load_npz(p.TRAIN_CACHE)
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    assert cfg['epochs']==80 and cfg['test_access']=='forbidden' and len(fit)==25912 and len(cal)==2879
    scale=np.array(prior['fit_protection_scales'],np.float32)
    r.setup_weights(cache,fit,values,cfg,device)
    o,b,s,gt=[torch.from_numpy(v[:128]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][:128],device=device)
    p.seed_everything(1);model=GeometryGenerator(o.shape[1],cfg).to(device);model.eval()
    out,res=model(o,b,s);assert torch.allclose(out,b,atol=2e-5,rtol=0)
    parameters={key for key,x in model.decoder.named_parameters() if x.requires_grad}
    assert parameters and any(key.startswith('proposal_delta.') for key in parameters)
    assert any(key.startswith('proposal_scale.') for key in parameters) and 'anchor_templates' in parameters
    frozen=copy.deepcopy(model.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    start=time.perf_counter()
    for _ in range(10):
        out,res=model(o,b,s);loss=m.objective(out,b,gt,res,s,groups,cfg,c.d.ARM,torch.tensor(scale,device=device))
        assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();optimizer.step()
    torch.cuda.synchronize();seconds=(time.perf_counter()-start)/10
    n.frozen_audit(model)
    for prefix in ('decoder.context.','decoder.mode_mlp.','decoder.mode_embedding','decoder.score_head.'):
        assert all(torch.equal(frozen[key],value) for key,value in model.state_dict().items() if key.startswith(prefix))
    assert any(not torch.equal(frozen['decoder.'+key],model.state_dict()['decoder.'+key]) for key in parameters)
    for seed in (1,2,3):
        saved=torch.load(checkpoint(seed,'zero','constrained'),map_location='cpu',weights_only=False)
        assert np.array_equal(saved['mean'],values[4]) and np.array_equal(saved['std'],values[5])
        paired_cfg=c.settings(seed)
        for key,value in saved['config'].items():
            if key not in ('protocol','cycle','experiment_in_cycle','set_context'):assert paired_cfg[key]==value,key
    sources={**prior['source_hashes'],**allmodels}
    for path in (Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE/'test_resume.py'):
        sources[str(path)]=p.sha256(path)
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),fit_protection_scales=scale.tolist(),source_hashes=sources,
        parameters_trainable=sum(x.numel() for x in model.parameters() if x.requires_grad),whitelist=sorted(parameters),seconds_per_128step=seconds,
        controls_reused_not_independent=True,new_trainings=3,test_access=False))
    print('PREFLIGHT PASS geometry only/frozencontext/mode/scorer/identity/config/controlsha/GPU '+str(seconds),flush=True)
def evaluate(device):
    n.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Single trainable-scope factor: geometry-only heads/proposal_scale/proposal_delta/anchor_templates/refiner, pretrained context/mode_embedding/mode_mlp and F16/scorer frozen. Three fresh80epoch seeds paired with VERIFIED REUSED refiner-only three80epoch controls, not independent replication. Same original nativecheckpoint/loss/fitcal/budget/velocity trustregion. Generic zero=refinercontrol mean_std=geometryhead, no poolcontext. RepeatedETHdev; no test/ranking claim. Additional prior fullgenerator comparison diagnostic, not dev checkpoint selection.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'GEOMETRY_HEAD_GATE_FAIL'
    m.write_json(path,summary)
    (m.RESULTS/'REPORT.md').write_text('# Geometry head scope\n\n'+summary['verdict']+'\n\n'+summary['scope']+'\n\n'+json.dumps(summary['aggregate'],indent=2),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)
c.preflight=preflight;c.evaluate=evaluate
def main():
    try:
        if '--action' in sys.argv and sys.argv[sys.argv.index('--action')+1]=='run':
            _,_,model=n.prepare_backend(torch.device('cpu'));del model
        c.main()
    except BaseException:
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
if __name__=='__main__':main()
