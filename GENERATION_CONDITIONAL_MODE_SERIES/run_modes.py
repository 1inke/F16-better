"""Native geometry with static versus observation-conditional mode modulation."""
from pathlib import Path
import importlib.util
import sys,json,time,traceback,copy
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
CONTROL=HERE.parent/'GENERATION_NATIVE_DECODER_SERIES'
spec=importlib.util.spec_from_file_location('conditional_native_source',CONTROL/'run_native.py')
n=importlib.util.module_from_spec(spec);sys.modules[spec.name]=n;spec.loader.exec_module(n)
c=n.c;m=n.m;r=n.r;p=n.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
BaseNative=n.NativeGenerator
class ConditionalGenerator(BaseNative):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        allowed=('anchor_templates','proposal_scale.','proposal_delta.','refiner.')
        for name,parameter in self.decoder.named_parameters():parameter.requires_grad_(name.startswith(allowed))
        self.modulation=nn.Linear(64,64)
        nn.init.zeros_(self.modulation.weight);nn.init.zeros_(self.modulation.bias)
        assert not self.decoder.mode_embedding.requires_grad
        assert not any(x.requires_grad for x in self.decoder.context.parameters())
        assert not any(x.requires_grad for x in self.decoder.mode_mlp.parameters())
    def gate(self,context):
        inputs=context if self.arm=='mean_std' else torch.ones_like(context)
        return torch.tanh(self.modulation(inputs))
    def forward(self,observation,base,speed):
        d=self.decoder
        past=observation[:,:16].reshape(-1,1,8,2)
        tokens=observation[:,16:16+8*n.TOKEN_DIM].reshape(-1,1,8,n.TOKEN_DIM)
        context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
        mode=d.mode_mlp(context[:,:,None]+(1+self.gate(context)[:,:,None])*d.mode_embedding[None,None])
        scale=torch.nn.functional.softplus(d.proposal_scale(context))+.25
        delta=.75*torch.tanh(d.proposal_delta(mode)).reshape(*mode.shape[:-1],3,2)
        anchors=scale[:,:,None,None]*(d.anchor_templates[None,None]+delta)
        corrected=anchors+d.refiner(mode,anchors)
        raw=torch.einsum('tp,bakpc->baktc',d.interpolation,corrected)[:,0]
        difference=raw-base
        velocity=torch.diff(difference,dim=2,prepend=torch.zeros_like(difference[:,:,:1]))
        limit=self.strength*(speed[:,0,None,None,None]+.03)
        residual=velocity.clamp(-limit,limit)
        return base+residual.cumsum(2),residual
c.Correction=ConditionalGenerator;r.m.Correction=ConditionalGenerator;n.NativeGenerator=ConditionalGenerator
def preflight(device):
    _,_,original=n.prepare_backend(device);del original
    previous=HERE.parent/'GENERATION_GEOMETRY_HEAD_SERIES'
    audit=json.loads((previous/'preflight.json').read_text())
    for path,digest in audit['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    locked=json.loads((previous/'results/model_lock.json').read_text())['models']
    for path,digest in locked.items():assert p.sha256(Path(path))==digest,path
    cfg=c.settings(1);cache=p.load_npz(p.TRAIN_CACHE)
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);values=r.prepared(cache,False,fit=fit)
    r.setup_weights(cache,fit,values,cfg,device)
    scale=np.array(audit['fit_protection_scales'],np.float32)
    o,b,s,gt=[torch.from_numpy(v[:128]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][:128],device=device);counts={};states=[]
    for arm in c.ARMS:
        p.seed_everything(1);model=ConditionalGenerator(o.shape[1],c.settings(1,arm)).to(device);model.eval()
        states.append(copy.deepcopy(model.state_dict()));counts[arm]=sum(x.numel() for x in model.parameters() if x.requires_grad)
        out,res=model(o,b,s);assert torch.allclose(out,b,atol=2e-5,rtol=0)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        for _ in range(5):
            out,res=model(o,b,s);loss=m.objective(out,b,gt,res,s,groups,cfg,c.d.ARM,torch.tensor(scale,device=device))
            assert torch.isfinite(loss);optimizer.zero_grad();loss.backward()
            assert model.modulation.weight.grad.abs().sum()>0
            optimizer.step()
        n.frozen_audit(model)
        context=torch.randn(8,1,64,device=device)
        with torch.no_grad():
            model.modulation.weight.normal_(std=.01)
            difference=(model.gate(context)-model.gate(context+1)).abs().max().item()
            assert difference>1e-6 if arm=='mean_std' else difference==0
        assert (res.abs()<=cfg['velocity_residual_scale']*(s[:,0,None,None,None]+.03)+1e-6).all()
    assert counts==dict(zero=31314,mean_std=31314),counts
    assert all(torch.equal(states[0][key],states[1][key]) for key in states[0])
    sources={**audit['source_hashes'],**locked}
    for path in (Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE/'test_resume.py',previous/'precision_diagnosis.json',previous/'diagnose_precision.py'):
        sources[str(path)]=p.sha256(path)
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),fit_protection_scales=scale.tolist(),parameters_trainable=counts,
        source_hashes=sources,new_trainings=6,paired_init_exact=True,test_access=False,design_basis='fit-only error diagnosis',static_modulation_effective_rank_lower=True))
    print('PREFLIGHT PASS pairedinit/parameters/identity/frozen/contextconditionalgradient/sourcehash/fitcal',flush=True)
def evaluate(device):
    n.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Sixfresh80epoch paired-order-seed originalcheckpoint trainings: original context/mode_embedding/mode_mlp/F16/scorer frozen; native geometry heads/refiner trainable BOTH arms; new31314trainableparams each, identical initialization. zero=static gate with constantones64, mean_std=observationcontext64-dependent gate, tanh64 channel modulation of native mode_embeddings. Static constant inputs imply LOWER effective functional capacity despite equal parameter count; not equal-capacity architecture proof. Original observation-dependent geometry inputs retained BOTH arms. Same loss/cal/protection/bound1.5/nativeK20; no GT/logit inference/test/newloss/hyperparametergrid. GTanchor diagnostic is not deployablemodel. Fit-only diagnosis used design, repeateddev diagnostic not selection. Score inputs change; no ranking/intent-understanding claim.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'CONDITIONAL_MODE_GATE_FAIL'
    m.write_json(path,summary)
    (m.RESULTS/'REPORT.md').write_text('# Static vs conditional mode calibration\n\n'+summary['verdict']+'\n\n'+summary['scope']+'\n\n'+json.dumps(summary['aggregate'],indent=2),encoding='utf-8')
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
