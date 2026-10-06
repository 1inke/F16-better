"""Static vs native-candidate conditioned modulation of direct frozen social tap."""
from pathlib import Path
import importlib.util,sys,json,time
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('social_mode_source',HERE.parent/'GENERATION_SOCIAL_TAP_SERIES/run_head.py')
s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)
z=s.z;x=s.x;h=s.h;n=s.n;c=s.c;m=s.m;r=s.r;p=s.p
s.HERE=HERE;z.HERE=HERE;x.HERE=HERE;h.HERE=HERE;c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
class SocialModeHead(s.SocialHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        self.mode_gate=nn.Linear(64,32)
        nn.init.zeros_(self.mode_gate.weight);nn.init.zeros_(self.mode_gate.bias)
    def social_input(self,observation):
        return (observation[:,s.SOCIAL_START:s.SOCIAL_END]-self.social_mean)/self.social_std
    def inputs(self,observation):
        features,tokens=self.native_features(observation)
        mean=self.value(tokens).mean(1)[:,None].expand(-1,20,-1)
        social=self.social(self.social_input(observation))[:,None].expand(-1,20,-1)
        mode=features[:,:,64:]
        if self.arm=='zero':mode=mode*0
        gate=torch.tanh(self.mode_gate(mode))
        return torch.cat((features,mean,social*(1+gate)),-1)
h.DirectHead=SocialModeHead;n.NativeGenerator=SocialModeHead;c.Correction=SocialModeHead;r.m.Correction=SocialModeHead
def preflight(device):
    _,_,original=n.prepare_backend(device);del original
    prior=json.loads((HERE.parent/'GENERATION_SOCIAL_TAP_SERIES/preflight.json').read_text())
    for path,digest in prior['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    lockpath=HERE.parent/'GENERATION_SOCIAL_TAP_SERIES/results/model_lock.json'
    lock=json.loads(lockpath.read_text())['models'];assert len(lock)==18
    for path,digest in lock.items():assert p.sha256(Path(path))==digest
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert cfg['seeds']==[1,2,3] and cfg['epochs']==80 and cfg['test_access']=='forbidden'
    assert len(fit)==25912 and len(cal)==2879 and not(set(cache['sample_id'][fit])&set(cache['sample_id'][cal]))
    assert np.array_equal(s.SOCIAL_MEAN,cache['social'][fit,:96].mean(0))
    assert np.array_equal(s.SOCIAL_STD,cache['social'][fit,:96].std(0).clip(.001))
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,v,gt=[torch.from_numpy(a[fit[:128]]).to(device) for a in values[:4]]
    groups=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(prior['fit_protection_scales'],device=device)
    counts={};states=[];initialinputs=[];times={};grads={};dependence={}
    for arm in c.ARMS:
        p.seed_everything(1);model=SocialModeHead(o.shape[1],c.settings(1,arm)).to(device);model.eval()
        states.append({k:a.detach().clone() for k,a in model.state_dict().items()});initialinputs.append(model.inputs(o).detach())
        assert model.social_input(o).abs().sum()>0 and model.mode_gate.weight.abs().sum()==0
        out,res=model(o,b,v);assert torch.equal(out,b) and res.abs().sum()==0
        counts[arm]=sum(a.numel() for a in model.parameters() if a.requires_grad)
        optimizer=torch.optim.AdamW([a for a in model.parameters() if a.requires_grad],lr=cfg['learning_rate']);start=time.perf_counter()
        for step in range(4):
            optimizer.zero_grad();out,res=model(o,b,v)
            loss=m.objective(out,b,gt,res,v,groups,cfg,'balanced_protected',scale);loss.backward()
            assert all(a.grad is not None and torch.isfinite(a.grad).all() for a in model.parameters() if a.requires_grad)
            if step==3:
                assert model.value.weight.grad.abs().sum()>0 and model.social.weight.grad.abs().sum()>0 and model.head[0].weight.grad.abs().sum()>0
                grads[arm]=float(model.mode_gate.weight.grad.norm())
                assert grads[arm]>0 if arm=='mean_std' else grads[arm]==0
            optimizer.step();h.frozen_audit(model)
        torch.cuda.synchronize();times[arm]=(time.perf_counter()-start)/4
        features,_=model.native_features(o);mode=features[:,:,64:]
        gate=torch.tanh(model.mode_gate(mode if arm=='mean_std' else mode*0))
        dependence[arm]=float(gate.std(1,unbiased=False).mean())
        assert dependence[arm]>0 if arm=='mean_std' else dependence[arm]==0
        assert out.shape==(128,20,12,2) and torch.isfinite(out).all()
        assert (res.abs()<=model.strength*(v[:,0,None,None,None]+.03)+1e-6).all()
        assert torch.equal(model.social_mean,torch.from_numpy(s.SOCIAL_MEAN).to(device))
        model.native_fallback.fill_(True);out,res=model(o,b,v);assert torch.equal(out,b) and res.abs().sum()==0
    assert counts==dict(zero=41638,mean_std=41638),counts
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    assert torch.equal(initialinputs[0],initialinputs[1])
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE.parent/'GENERATION_SOCIAL_MODE_DESIGN.md',lockpath]
    hashes={**prior['source_hashes'],**lock,**{str(a):p.sha256(a) for a in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',parameters_trainable=counts,paired_full_init_exact=True,native_identity_exact=True,initial_inputs_equal=True,both_arms_full_social_input=True,mode_gate_gradient_norm=grads,learned_candidate_gate_std=dependence,social_scalers_fit_only=True,obs_schema=prior['obs_schema'],mode_order='locked original native K20 order',fit_n=len(fit),cal_n=len(cal),fit_protection_scales=prior['fit_protection_scales'],source_hashes=hashes,seconds_per_128step=times,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(parameters=counts,gategrads=grads,candidate_std=dependence,timings=times)),flush=True)
h.preflight=preflight;c.preflight=preflight
def evaluate(device):
    n.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Six NEW80epoch fullsocial BOTH static versus native-mode conditioned socialmodulation. Originalcontext64+native-mode64+uniformtokenmean32/fullsocial32 retained/newzeroLinear64->32 gate inputzero64 vs originalmode64/tanh/social*(1+gate). Both41638params/fullpairedinit/nativeoutputidentity/initialinputsEXACT; static gateweightdata gradientszero,biaslearnable, not equal effectivecapacity. Allolddecoder/F16/scorer frozen/nativebase3anchor/commonclip1.5/originalloss fitcal/K20/scalersFIT. NoGTgroup/logitforward/no rawquery/temporal/querymodification/expertloss. Tokensalready social enriched; not newdata/intent/independentgenerator/ranking; lockednativeK20order/nativefallbacknotlearned/repeatedETHdevexploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'SOCIAL_MODE_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# Native-mode conditioned frozen social readout\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)
h.evaluate=evaluate;c.evaluate=evaluate
def main():h.main()
if __name__=='__main__':main()
