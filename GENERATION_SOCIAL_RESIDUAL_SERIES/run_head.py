"""Full versus FIT-linear-residual frozen social readout; no future input."""
from pathlib import Path
import importlib.util,sys,json,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('residual_social_source',HERE.parent/'GENERATION_SOCIAL_TAP_SERIES/run_head.py')
s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)
z=s.z;x=s.x;h=s.h;n=s.n;c=s.c;m=s.m;r=s.r;p=s.p
s.HERE=HERE;z.HERE=HERE;x.HERE=HERE;h.HERE=HERE;c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
TRANSFORM_PATH=HERE.parent/'GENERATION_SOCIAL_RESIDUAL_TRANSFORM.npz'
TRANSFORM=p.load_npz(TRANSFORM_PATH)
class ResidualSocialHead(s.SocialHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        # BOTH arms persist the same FIT-only fixed transformation.
        for key in ('feature_mean','feature_std','weights','residual_mean','residual_std'):
            self.register_buffer('fixed_'+key,torch.from_numpy(TRANSFORM[key].copy()))
        self.social_mean.copy_(torch.from_numpy(TRANSFORM['social_mean']))
        self.social_std.copy_(torch.from_numpy(TRANSFORM['social_std']))
    def social_input(self,observation):
        y=(observation[:,784:880]-self.social_mean)/self.social_std
        if self.arm=='zero':return y
        tokens=observation[:,16:784].reshape(-1,8,96)
        features=torch.cat((observation[:,:16],tokens[:,-1],tokens.mean(1)),1)
        xx=(features-self.fixed_feature_mean)/self.fixed_feature_std
        xx=torch.cat((xx,torch.ones_like(xx[:,:1])),1)
        residual=y-xx@self.fixed_weights
        return (residual-self.fixed_residual_mean)/self.fixed_residual_std
h.DirectHead=ResidualSocialHead;n.NativeGenerator=ResidualSocialHead;c.Correction=ResidualSocialHead;r.m.Correction=ResidualSocialHead
def preflight(device):
    _,_,original=n.prepare_backend(device);del original
    prior=json.loads((HERE.parent/'GENERATION_SOCIAL_MODE_SERIES/preflight.json').read_text())
    diagnosis=json.loads((HERE.parent/'GENERATION_SOCIAL_RESIDUAL_DIAGNOSIS.json').read_text())
    assert diagnosis['status']=='PASS_FEASIBILITY_ONLY' and diagnosis['no_gt_dev_test']
    for path,digest in {**prior['source_hashes'],**diagnosis['source_hashes']}.items():assert p.sha256(Path(path))==digest,path
    lockpath=HERE.parent/'GENERATION_SOCIAL_MODE_SERIES/results/model_lock.json'
    lock=json.loads(lockpath.read_text())['models'];assert len(lock)==18
    for path,digest in lock.items():assert p.sha256(Path(path))==digest
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1)
    assert cfg['epochs']==80 and cfg['seeds']==[1,2,3] and cfg['test_access']=='forbidden'
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert len(fit)==25912 and len(cal)==2879
    assert np.array_equal(TRANSFORM['fit_ids'],cache['sample_id'][fit]) and np.array_equal(TRANSFORM['cal_ids'],cache['sample_id'][cal])
    np.testing.assert_allclose(TRANSFORM['social_mean'],s.SOCIAL_MEAN,rtol=0,atol=1e-5)
    np.testing.assert_allclose(TRANSFORM['social_std'],s.SOCIAL_STD,rtol=0,atol=1e-5)
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    np.testing.assert_array_equal(values[0][:,784:880],cache['social'][:,:96])
    np.testing.assert_array_equal(values[0],r.prepared(cache,False,fit=cal)[0])
    counts={};states=[];features=[];times={};grads={};errors={}
    for arm in c.ARMS:
        p.seed_everything(1);model=ResidualSocialHead(940,c.settings(1,arm)).to(device);model.eval()
        states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        for name,ids in [('fit',fit[:128]),('cal',cal[:128])]:
            obs=values[0][ids];tt=obs[:,16:784].reshape(-1,8,96)
            xx=np.concatenate((obs[:,:16],tt[:,-1],tt.mean(1)),1)
            xx=np.column_stack(((xx-TRANSFORM['feature_mean'])/TRANSFORM['feature_std'],np.ones(len(obs),np.float32)))
            yy=(obs[:,784:880]-TRANSFORM['social_mean'])/TRANSFORM['social_std']
            expected=yy if arm=='zero' else (yy-xx@TRANSFORM['weights']-TRANSFORM['residual_mean'])/TRANSFORM['residual_std']
            actual=model.social_input(torch.from_numpy(obs).to(device)).detach().cpu().numpy()
            errors[arm+'_'+name]=float(abs(actual-expected).max())
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-4)
        o,b,v,gt=[torch.from_numpy(a[fit[:128]]).to(device) for a in values[:4]]
        groups=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(prior['fit_protection_scales'],device=device)
        features.append(model.native_features(o)[0].detach())
        assert model.social_input(o).abs().sum()>0
        out,res=model(o,b,v);assert torch.equal(out,b) and res.abs().sum()==0
        counts[arm]=sum(v.numel() for v in model.parameters() if v.requires_grad)
        optimizer=torch.optim.AdamW([a for a in model.parameters() if a.requires_grad],lr=cfg['learning_rate']);start=time.perf_counter()
        for step in range(4):
            optimizer.zero_grad();out,res=model(o,b,v)
            m.objective(out,b,gt,res,v,groups,cfg,'balanced_protected',scale).backward()
            assert all(a.grad is not None and torch.isfinite(a.grad).all() for a in model.parameters() if a.requires_grad)
            if step==3:
                assert model.value.weight.grad.abs().sum()>0 and model.head[0].weight.grad.abs().sum()>0
                grads[arm]=float(model.social.weight.grad.norm());assert grads[arm]>0
            optimizer.step();h.frozen_audit(model)
        torch.cuda.synchronize();times[arm]=(time.perf_counter()-start)/4
        assert out.shape==(128,20,12,2) and torch.isfinite(out).all()
        assert (res.abs()<=model.strength*(v[:,0,None,None,None]+.03)+1e-6).all()
        model.native_fallback.fill_(True);out,res=model(o,b,v);assert torch.equal(out,b) and res.abs().sum()==0
    assert counts==dict(zero=39558,mean_std=39558),counts
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    assert torch.equal(features[0],features[1])
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE.parent/'GENERATION_SOCIAL_RESIDUAL_DESIGN.md',HERE.parent/'GENERATION_SOCIAL_RESIDUAL_DIAGNOSIS.json',lockpath]
    hashes={**prior['source_hashes'],**diagnosis['source_hashes'],**lock,**{str(a):p.sha256(a) for a in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',parameters_trainable=counts,paired_full_init_exact=True,native_identity_exact=True,native_features_equal=True,fit_transform_fixed=True,reconstruction_maxerror=errors,social_gradient_norm=grads,fit_n=len(fit),cal_n=len(cal),fit_protection_scales=prior['fit_protection_scales'],source_hashes=hashes,seconds_per_128step=times,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(parameters=counts,grads=grads,reconstruction=errors,timings=times)),flush=True)
h.preflight=preflight;c.preflight=preflight
def evaluate(device):
    n.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Sixfresh80epoch fullstandardizedsocial96 vs FIT-linear-residual standardizedsocial96. Both originalcontext64/native-mode64/uniformtokenmean32/social32/newhead192-128-64-6 preserved. Same39558nominalparams/fullpaired/nativeidentity; bothsocialgradients positive; different information/distributions not equal effectivecapacity. FIT-only fixedridge.001/no refit/CALdev transform buffers. Wholeolddecoder/F16/scorer frozen/nativebase3anchor/commonclip1.5/originalK20losscal. NoGTgroup/logitforward/no intent/independentgenerator/ranking; tokens already social enriched/residualization may remove signal/amplify noise/repeatedETHdev exploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'SOCIAL_RESIDUAL_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# FIT residual social readout\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)
h.evaluate=evaluate;c.evaluate=evaluate
def main():h.main()
if __name__=='__main__':main()
