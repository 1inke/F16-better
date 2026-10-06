"""Direct frozen focal social tap vs zero social; no new data or old-weight training."""
from pathlib import Path
import importlib.util,sys,json,time
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('social_contrast_source',HERE.parent/'GENERATION_TOKEN_CONTRAST_SERIES/run_head.py')
z=importlib.util.module_from_spec(spec);sys.modules[spec.name]=z;spec.loader.exec_module(z)
x=z.x;h=z.h;n=z.n;c=z.c;m=z.m;r=z.r;p=z.p
z.HERE=HERE;x.HERE=HERE;h.HERE=HERE;c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
SOCIAL_START=784;SOCIAL_END=880;SOCIAL_MEAN=None;SOCIAL_STD=None
BASE_BACKEND=n.prepare_backend;BASE_PREPARED=r.prepared
def prepare_backend(device):
    global SOCIAL_MEAN,SOCIAL_STD
    result=BASE_BACKEND(device);assert n.TOKEN_DIM==96
    raw=n.RAW_LOAD(p.TRAIN_CACHE);assert raw['social'].shape==(28791,96)
    cfg=c.settings(1);fit,cal=p.internal_split(len(raw['sample_id']),cfg['calibration_fraction'])
    SOCIAL_MEAN=raw['social'][fit].mean(0).astype(np.float32)
    SOCIAL_STD=raw['social'][fit].std(0).clip(.001).astype(np.float32)
    return result
n.prepare_backend=prepare_backend
def prepared(cache,replace,**kwargs):
    values=list(BASE_PREPARED(cache,replace,**kwargs))
    assert cache['social'].shape[1]==864 and values[0].shape[1]==940
    # social cache first96 are fusedagent; remaining768 are appended decoder tokens.
    assert np.count_nonzero(values[0][:,SOCIAL_START:])==0
    values[0][:,SOCIAL_START:SOCIAL_END]=cache['social'][:,:96]
    return values
r.prepared=prepared;n.prepared=prepared
class SocialHead(z.ContrastHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        assert SOCIAL_MEAN is not None and dimension==940
        self.social=nn.Linear(96,32)
        self.register_buffer('social_mean',torch.from_numpy(SOCIAL_MEAN.copy()))
        self.register_buffer('social_std',torch.from_numpy(SOCIAL_STD.copy()))
    def social_input(self,observation):
        v=(observation[:,SOCIAL_START:SOCIAL_END]-self.social_mean)/self.social_std
        return v*0 if self.arm=='zero' else v
    def inputs(self,observation):
        features,tokens=self.native_features(observation)
        mean=self.value(tokens).mean(1)
        social=self.social(self.social_input(observation))
        pooled=torch.cat((mean,social),-1)[:,None].expand(-1,20,-1)
        return torch.cat((features,pooled),-1)
h.DirectHead=SocialHead;n.NativeGenerator=SocialHead;c.Correction=SocialHead;r.m.Correction=SocialHead
def preflight(device):
    _,_,original=n.prepare_backend(device);del original
    prior=json.loads((HERE.parent/'GENERATION_TOKEN_SEQUENCE_SERIES/preflight.json').read_text())
    for path,digest in prior['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    proofs=[HERE.parent/'GENERATION_SOCIAL_TAP_DIAGNOSIS.json',HERE.parent/'GENERATION_SOCIAL_TAP_REPLAY.json']
    for proof in proofs:
        audit=json.loads(proof.read_text());assert audit['status'].startswith('PASS') and audit['no_gt_no_dev']
        for path,digest in audit['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    lockpath=HERE.parent/'GENERATION_TOKEN_SEQUENCE_SERIES/results/model_lock.json'
    lock=json.loads(lockpath.read_text())['models'];assert len(lock)==18
    for path,digest in lock.items():assert p.sha256(Path(path))==digest
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1)
    assert cfg['seeds']==[1,2,3] and cfg['epochs']==80 and cfg['test_access']=='forbidden'
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert len(fit)==25912 and len(cal)==2879 and not(set(cache['sample_id'][fit])&set(cache['sample_id'][cal]))
    assert np.array_equal(SOCIAL_MEAN,cache['social'][fit,:96].mean(0))
    assert np.array_equal(SOCIAL_STD,cache['social'][fit,:96].std(0).clip(.001))
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    np.testing.assert_array_equal(values[0][:,SOCIAL_START:SOCIAL_END],cache['social'][:,:96])
    np.testing.assert_array_equal(values[0][:,:16],cache['_past'].reshape(-1,16))
    np.testing.assert_array_equal(values[0][:,16:SOCIAL_START],cache['_tokens'].reshape(-1,768))
    calvalues=r.prepared(cache,False,fit=cal)
    np.testing.assert_array_equal(values[0],calvalues[0]) # observation injection independent of passed scaler fit subset
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(prior['fit_protection_scales'],device=device)
    counts={};states=[];features=[];times={};gradnorms={};sensitivity={}
    for arm in c.ARMS:
        p.seed_everything(1);model=SocialHead(o.shape[1],c.settings(1,arm)).to(device);model.eval()
        states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        native,tokens=model.native_features(o);features.append(native.detach())
        inputs=model.inputs(o);changed=o.clone();changed[:,SOCIAL_START]+=.25
        newinputs=model.inputs(changed)
        assert torch.equal(inputs[:,:,:160],newinputs[:,:,:160])
        sensitivity[arm]=float((newinputs[:,:,160:]-inputs[:,:,160:]).abs().max())
        assert sensitivity[arm]>0 if arm=='mean_std' else sensitivity[arm]==0
        social=model.social_input(o)
        assert torch.isfinite(social).all() and (social.abs().sum()>0 if arm=='mean_std' else social.abs().sum()==0)
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
        counts[arm]=sum(v.numel() for v in model.parameters() if v.requires_grad)
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate']);start=time.perf_counter()
        for step in range(4):
            optimizer.zero_grad();out,res=model(o,b,s)
            loss=m.objective(out,b,gt,res,s,groups,cfg,'balanced_protected',scale);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad)
            if step==3:
                assert model.value.weight.grad.abs().sum()>0 and model.head[0].weight.grad.abs().sum()>0
                gradnorms[arm]=float(model.social.weight.grad.norm())
                assert gradnorms[arm]>0 if arm=='mean_std' else gradnorms[arm]==0
            optimizer.step();h.frozen_audit(model)
        torch.cuda.synchronize();times[arm]=(time.perf_counter()-start)/4
        assert out.shape==(128,20,12,2) and torch.isfinite(out).all()
        assert (res.abs()<=model.strength*(s[:,0,None,None,None]+.03)+1e-6).all()
        assert torch.equal(model.social_mean,torch.from_numpy(SOCIAL_MEAN).to(device))
        model.native_fallback.fill_(True);out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
    assert counts==dict(zero=39558,mean_std=39558),counts
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    assert torch.equal(features[0],features[1])
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE.parent/'GENERATION_SOCIAL_TAP_DESIGN.md',HERE.parent/'GENERATION_CYCLE_4_REVIEW.md',lockpath]+proofs
    hashes={**prior['source_hashes'],**lock,**{str(v):p.sha256(v) for v in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',parameters_trainable=counts,paired_full_init_exact=True,native_identity_exact=True,native_features_equal=True,obs_schema=dict(past=[0,16],tokens=[16,784],social=[784,880],reserved=[880,940]),social_scalers_fit_only=True,social_mean=SOCIAL_MEAN.tolist(),social_std=SOCIAL_STD.tolist(),social_gradient_norm=gradnorms,social_input_perturbation_sensitivity=sensitivity,mode_order='locked original native K20 order',fit_n=len(fit),cal_n=len(cal),fit_protection_scales=prior['fit_protection_scales'],source_hashes=hashes,seconds_per_128step=times,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(parameters=counts,socialgrads=gradnorms,timings=times)),flush=True)
h.preflight=preflight;c.preflight=preflight
def evaluate(device):
    n.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Six NEW80epoch uniformtokenmean+constant social projection versus mean+directfrozen96socialtap projection32. Originalcontext64+native-mode64 retained/newhead192->128->64->6/newvalue96->32/social96->32. Both39558params/fullpairedinit/nativeidentity; control social weightdata gradientszero not equal effectivecapacity. Allolddecoder/F16/scorer frozen/nativebase3anchor/commonclip1.5/originalloss/cal/noGTgroup/logitforward. Social scalers fit-only savedbuffers; tokens already social enriched, not newdata/intent/independentgenerator/ranking. LockednativeK20order/nativefallbacknotlearned/repeatedETHdevexploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'SOCIAL_TAP_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# Direct frozen agent/social tap\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)
h.evaluate=evaluate;c.evaluate=evaluate
def main():h.main()
if __name__=='__main__':main()
