"""Fixed last4-minus-first4 observed-token contrast vs zero32; no attention."""
from pathlib import Path
import importlib.util,sys,json,time
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('contrast_readout_source',HERE.parent/'GENERATION_TOKEN_READOUT_SERIES/run_head.py')
x=importlib.util.module_from_spec(spec);sys.modules[spec.name]=x;spec.loader.exec_module(x)
h=x.h;n=x.n;c=x.c;m=x.m;r=x.r;p=x.p
x.HERE=HERE;h.HERE=HERE;c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
class ContrastHead(x.TokenReadout):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        del self.query;del self.key;del self.time_position
        self.head=nn.Sequential(nn.Linear(192,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,6))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def readout(self,tokens):
        v=self.value(tokens);mean=v.mean(1)
        contrast=v[:,4:].mean(1)-v[:,:4].mean(1)
        if self.arm=='zero':contrast=contrast*0
        return torch.cat((mean,contrast),-1)[:,None].expand(-1,20,-1)
    def inputs(self,observation):
        features,tokens=self.native_features(observation)
        return torch.cat((features,self.readout(tokens)),-1)
h.DirectHead=ContrastHead;n.NativeGenerator=ContrastHead;c.Correction=ContrastHead;r.m.Correction=ContrastHead
def preflight(device):
    _,_,original=n.prepare_backend(device);del original
    prior=json.loads((HERE.parent/'GENERATION_TOKEN_READOUT_SERIES/preflight.json').read_text())
    for path,digest in prior['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    lockpath=HERE.parent/'GENERATION_TOKEN_READOUT_SERIES/results/model_lock.json'
    lock=json.loads(lockpath.read_text())['models'];assert len(lock)==18
    for path,digest in lock.items():assert p.sha256(Path(path))==digest
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1)
    assert cfg['seeds']==[1,2,3] and cfg['epochs']==80 and cfg['test_access']=='forbidden'
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert len(fit)==25912 and len(cal)==2879 and not(set(cache['sample_id'][fit])&set(cache['sample_id'][cal]))
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(prior['fit_protection_scales'],device=device)
    counts={};states=[];means=[];times={};gradnorms={};differences={}
    for arm in c.ARMS:
        p.seed_everything(1);model=ContrastHead(o.shape[1],c.settings(1,arm)).to(device);model.eval()
        states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        f,t=model.native_features(o);value=model.readout(t);means.append(value[:,:,:32].detach())
        reverse=model.readout(t.flip(1))
        assert torch.allclose(reverse[:,:,:32],value[:,:,:32],atol=2e-6,rtol=0)
        assert torch.allclose(reverse[:,:,32:],-value[:,:,32:],atol=2e-6,rtol=0)
        differences[arm]=float(value[:,:,32:].abs().mean())
        assert differences[arm]>0 if arm=='mean_std' else differences[arm]==0
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
        counts[arm]=sum(v.numel() for v in model.parameters() if v.requires_grad)
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate']);start=time.perf_counter()
        for step in range(4):
            optimizer.zero_grad();out,res=model(o,b,s)
            loss=m.objective(out,b,gt,res,s,groups,cfg,'balanced_protected',scale);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad)
            if step==3:
                assert model.value.weight.grad.abs().sum()>0 and model.head[0].weight.grad.abs().sum()>0
                gradnorms[arm]=float(model.head[0].weight.grad[:,160:].norm())
                assert gradnorms[arm]>0 if arm=='mean_std' else gradnorms[arm]==0
            optimizer.step();h.frozen_audit(model)
        torch.cuda.synchronize();times[arm]=(time.perf_counter()-start)/4
        assert out.shape==(128,20,12,2) and torch.isfinite(out).all()
        assert (res.abs()<=model.strength*(s[:,0,None,None,None]+.03)+1e-6).all()
        model.native_fallback.fill_(True);out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
    assert counts==dict(zero=36454,mean_std=36454),counts
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    assert torch.equal(means[0],means[1])
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE.parent/'GENERATION_TOKEN_CONTRAST_DESIGN.md',lockpath]
    hashes={**prior['source_hashes'],**lock,**{str(v):p.sha256(v) for v in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',parameters_trainable=counts,paired_full_init_exact=True,native_identity_exact=True,uniform_mean_equal=True,token_contrast_mean_abs=differences,reverse_time_flips_contrast=True,contrast_input_column_gradient_norm=gradnorms,mode_order='locked original native K20 order',fit_n=len(fit),cal_n=len(cal),fit_protection_scales=prior['fit_protection_scales'],source_hashes=hashes,seconds_per_128step=times,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(parameters=counts,contrastgrads=gradnorms,timings=times)),flush=True)
h.preflight=preflight;c.preflight=preflight
def evaluate(device):
    n.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Six NEW80epoch uniform observationtoken mean+zero32 versus mean+fixedlast4-minusfirst4 valuecontrast32; originalcontext+native-mode128 retained/newvalue96->32/sharedhead192->128->64->6. Both36454params/fullpairedinit/nativeidentity; zero32headcolumnsdata grad0 not equal effectivecapacity. Allolddecoder/F16/scorer frozen/nativebase3anchor/commonclip1.5/originalloss/cal/noGTforward. Noattention/query/key/geometry/slotexpert; lockednativeK20order; not intent/independentgenerator/ranking. Nativefallback not learned/repeatedETHdev exploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'TOKEN_CONTRAST_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# Fixed observed-token temporal contrast\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)
h.evaluate=evaluate;c.evaluate=evaluate
def main():h.main()
if __name__=='__main__':main()
