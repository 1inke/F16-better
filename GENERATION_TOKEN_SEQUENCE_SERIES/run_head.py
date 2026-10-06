"""Ordered frozen observed-token deviations vs zero; all old weights frozen."""
from pathlib import Path
import importlib.util,sys,json,time
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sequence_contrast_source',HERE.parent/'GENERATION_TOKEN_CONTRAST_SERIES/run_head.py')
z=importlib.util.module_from_spec(spec);sys.modules[spec.name]=z;spec.loader.exec_module(z)
x=z.x;h=z.h;n=z.n;c=z.c;m=z.m;r=z.r;p=z.p
z.HERE=HERE;x.HERE=HERE;h.HERE=HERE;c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
class SequenceHead(z.ContrastHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        self.temporal=nn.Linear(256,32,bias=False)
    def temporal_input(self,tokens):
        v=self.value(tokens);mean=v.mean(1);centered=v-mean[:,None]
        if self.arm=='zero':centered=centered*0
        return mean,centered.reshape(-1,256)
    def readout(self,tokens):
        mean,sequence=self.temporal_input(tokens)
        return torch.cat((mean,self.temporal(sequence)),-1)[:,None].expand(-1,20,-1)
h.DirectHead=SequenceHead;n.NativeGenerator=SequenceHead;c.Correction=SequenceHead;r.m.Correction=SequenceHead
def preflight(device):
    _,_,original=n.prepare_backend(device);del original
    prior=json.loads((HERE.parent/'GENERATION_TOKEN_CONTRAST_SERIES/preflight.json').read_text())
    for path,digest in prior['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    lockpath=HERE.parent/'GENERATION_TOKEN_CONTRAST_SERIES/results/model_lock.json'
    lock=json.loads(lockpath.read_text())['models'];assert len(lock)==18
    for path,digest in lock.items():assert p.sha256(Path(path))==digest
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1)
    assert cfg['seeds']==[1,2,3] and cfg['epochs']==80 and cfg['test_access']=='forbidden'
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert len(fit)==25912 and len(cal)==2879 and not(set(cache['sample_id'][fit])&set(cache['sample_id'][cal]))
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(prior['fit_protection_scales'],device=device)
    counts={};states=[];means=[];times={};gradnorms={};sensitivity={}
    for arm in c.ARMS:
        p.seed_everything(1);model=SequenceHead(o.shape[1],c.settings(1,arm)).to(device);model.eval()
        states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        features,tokens=model.native_features(o);mean,centered=model.temporal_input(tokens);means.append(mean.detach())
        assert torch.isfinite(centered).all()
        assert torch.allclose(centered.reshape(-1,8,32).sum(1),torch.zeros_like(mean),atol=5e-6,rtol=0)
        assert centered.abs().sum()>0 if arm=='mean_std' else centered.abs().sum()==0
        pooled=model.readout(tokens);reverse=model.readout(tokens.flip(1))
        assert torch.allclose(pooled[:,:,:32],reverse[:,:,:32],atol=2e-6,rtol=0)
        sensitivity[arm]=float((pooled[:,:,32:]-reverse[:,:,32:]).abs().max())
        assert sensitivity[arm]>1e-6 if arm=='mean_std' else sensitivity[arm]==0
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
        counts[arm]=sum(v.numel() for v in model.parameters() if v.requires_grad)
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate']);start=time.perf_counter()
        for step in range(4):
            optimizer.zero_grad();out,res=model(o,b,s)
            loss=m.objective(out,b,gt,res,s,groups,cfg,'balanced_protected',scale);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad)
            if step==3:
                assert model.value.weight.grad.abs().sum()>0 and model.head[0].weight.grad.abs().sum()>0
                gradnorms[arm]=float(model.temporal.weight.grad.norm())
                assert gradnorms[arm]>0 if arm=='mean_std' else gradnorms[arm]==0
            optimizer.step();h.frozen_audit(model)
        torch.cuda.synchronize();times[arm]=(time.perf_counter()-start)/4
        assert out.shape==(128,20,12,2) and torch.isfinite(out).all()
        assert (res.abs()<=model.strength*(s[:,0,None,None,None]+.03)+1e-6).all()
        model.native_fallback.fill_(True);out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
    assert counts==dict(zero=44646,mean_std=44646),counts
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    assert torch.equal(means[0],means[1])
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE.parent/'GENERATION_TOKEN_SEQUENCE_DESIGN.md',lockpath]
    hashes={**prior['source_hashes'],**lock,**{str(v):p.sha256(v) for v in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',parameters_trainable=counts,paired_full_init_exact=True,native_identity_exact=True,uniform_mean_equal=True,temporal_gradient_norm=gradnorms,reversal_readout_sensitivity=sensitivity,mode_order='locked original native K20 order',fit_n=len(fit),cal_n=len(cal),fit_protection_scales=prior['fit_protection_scales'],source_hashes=hashes,seconds_per_128step=times,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(parameters=counts,temporalgrads=gradnorms,timings=times)),flush=True)
h.preflight=preflight;c.preflight=preflight
def evaluate(device):
    n.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Six NEW80epoch mean+zero temporal deviations versus mean+ordered8step centeredvalue sequence readout. Originalcontext64+native-mode64 retained/value96->32/newtemporalLinear256->32 biasFalse/newhead192->128->64->6. Both44646params/fullpairedinit/nativeidentity; control temporal data gradientszero not equal effectivecapacity. Allolddecoder/F16/scorer frozen/nativebase3anchor/commonclip1.5/originalloss/cal/noGTgroup/logitforward. Noattention/halfcontrast/geometry/rawquery/slotexpert; lockednativeK20order; not intent/independentgenerator/ranking. Nativefallback not learned/repeatedETHdev exploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'TOKEN_SEQUENCE_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# Ordered observed-token sequence readout\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)
h.evaluate=evaluate;c.evaluate=evaluate
def main():h.main()
if __name__=='__main__':main()
