"""Shared vs native-slot-untied last-layer correction heads. Locked native order."""
from pathlib import Path
import importlib.util,sys,json,time
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('slot_direct_source',HERE.parent/'GENERATION_DIRECT_ANCHOR_SERIES/run_head.py')
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
n=h.n;c=h.c;m=h.m;r=h.r;p=h.p
h.HERE=HERE;c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
class TokenReadout(h.DirectHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        self.head=nn.Sequential(nn.Linear(160,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,6))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.query=nn.Linear(128,32);self.key=nn.Linear(n.TOKEN_DIM,32);self.value=nn.Linear(n.TOKEN_DIM,32)
        position=torch.arange(8,dtype=torch.float32)[:,None]
        frequency=torch.exp(torch.arange(0,32,2,dtype=torch.float32)*(-torch.log(torch.tensor(10000.))/32))
        pe=torch.zeros(8,32);pe[:,0::2]=torch.sin(position*frequency);pe[:,1::2]=torch.cos(position*frequency)
        self.register_buffer('time_position',pe)
    def native_features(self,observation):
        d=self.decoder;past=observation[:,:16].reshape(-1,1,8,2)
        tokens=observation[:,16:16+8*n.TOKEN_DIM].reshape(-1,1,8,n.TOKEN_DIM)
        context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
        query=d.mode_embedding[None,None].expand(len(observation),1,-1,-1)
        mode=d.mode_mlp(context[:,:,None]+query)
        return torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0],tokens[:,0]
    def readout(self,features,tokens):
        q=self.query(features);k=self.key(tokens)+self.time_position[None];v=self.value(tokens)
        if self.arm=='zero':
            weights=tokens.new_full((len(tokens),20,8),1/8)
            pooled=v.mean(1)[:,None].expand(-1,20,-1)+0*(q.sum()+k.sum())
        else:
            weights=torch.softmax(torch.einsum('bkd,btd->bkt',q,k)/(32**.5),dim=-1)
            pooled=torch.einsum('bkt,btd->bkd',weights,v)
        return pooled,weights
    def inputs(self,observation):
        features,tokens=self.native_features(observation)
        pooled,_=self.readout(features,tokens)
        return torch.cat((features,pooled),-1)
h.DirectHead=TokenReadout;n.NativeGenerator=TokenReadout;c.Correction=TokenReadout;r.m.Correction=TokenReadout
def preflight(device):
    _,_,original=n.prepare_backend(device);del original
    prior=json.loads((HERE.parent/'GENERATION_SLOT_HEAD_SERIES/preflight.json').read_text())
    for path,digest in prior['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1)
    assert cfg['seeds']==[1,2,3] and cfg['epochs']==80 and cfg['test_access']=='forbidden'
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert len(fit)==25912 and len(cal)==2879 and not(set(cache['sample_id'][fit])&set(cache['sample_id'][cal]))
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][fit[:128]],device=device);scale=torch.tensor(prior['fit_protection_scales'],device=device)
    counts={};states=[];features=[];times={};gradnorms={};sensitivity={}
    for arm in c.ARMS:
        p.seed_everything(1);model=TokenReadout(o.shape[1],c.settings(1,arm)).to(device);model.eval()
        states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        f,t=model.native_features(o);features.append(f.detach())
        variance=float(t.var(1,unbiased=False).mean());assert variance>0
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0 and torch.isfinite(out).all()
        pooled,w=model.readout(f,t);assert torch.isfinite(w).all() and torch.allclose(w.sum(-1),torch.ones_like(w[:,:,0]),atol=1e-6)
        changed=t.clone();changed[:,0]+=.25
        changed_pool,changed_w=model.readout(f,changed)
        sensitivity[arm]=float((changed_pool-pooled).abs().max())
        assert sensitivity[arm]>0
        if arm=='mean_std':assert (changed_w-w).abs().max()>1e-7 and w.std(-1).mean()>1e-6
        else:assert torch.equal(changed_w,w)
        counts[arm]=sum(v.numel() for v in model.parameters() if v.requires_grad)
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate'])
        start=time.perf_counter()
        for step in range(4):
            optimizer.zero_grad();out,res=model(o,b,s)
            loss=m.objective(out,b,gt,res,s,groups,cfg,'balanced_protected',scale);loss.backward()
            trainable=[v for v in model.parameters() if v.requires_grad]
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in trainable)
            if step==3:
                gradnorms[arm]={name:float(getattr(model,name).weight.grad.norm()) for name in ('query','key','value')}
                assert model.head[0].weight.grad.abs().sum()>0 and model.value.weight.grad.abs().sum()>0
                if arm=='mean_std':assert model.query.weight.grad.abs().sum()>0 and model.key.weight.grad.abs().sum()>0
                else:assert model.query.weight.grad.abs().sum()==0 and model.key.weight.grad.abs().sum()==0
            optimizer.step();h.frozen_audit(model)
        torch.cuda.synchronize();times[arm]=(time.perf_counter()-start)/4
        assert out.shape==(128,20,12,2) and torch.isfinite(out).all()
        assert (res.abs()<=model.strength*(s[:,0,None,None,None]+.03)+1e-6).all()
        model.native_fallback.fill_(True);out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
    assert counts['zero']==counts['mean_std']==33446+64*n.TOKEN_DIM,counts
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    assert torch.equal(features[0],features[1])
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE.parent/'GENERATION_TOKEN_READOUT_DESIGN.md',HERE.parent/'GENERATION_CYCLE_3_REVIEW.md']
    hashes={**prior['source_hashes'],**{str(v):p.sha256(v) for v in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',parameters_trainable=counts,token_dim=n.TOKEN_DIM,fit_first128_token_time_variance=variance,paired_full_init_exact=True,native_identity_exact=True,native_features_equal=True,mode_order='locked original native K20 order',real_fit_gradient_norms=gradnorms,token_perturbation_sensitivity=sensitivity,fit_n=len(fit),cal_n=len(cal),fit_protection_scales=prior['fit_protection_scales'],source_hashes=hashes,seconds_per_128step=times,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(parameters=counts,token_dim=n.TOKEN_DIM,gradient=gradnorms,timings=times)),flush=True)
h.preflight=preflight;c.preflight=preflight
def evaluate(device):
    n.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Six NEW80epoch fullfrozen8observationtoken uniformmean versus candidate-query temporal attention readout32dim trainings. Bothsame originalcontext+mode128/newhead160->128->64->6/fullpairedinit/nativeidentity/nominalparams; uniformq/key data gradszero so not equal effectivecapacity. Allolddecoder/F16/scorer frozen/nativebase3anchor/commonclip1.5/originalloss/cal; GT/group/logits absentforward. LockednativeK20order; not independentgenerator/intent/ranking; fallback not learned. RepeatedETHdev exploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'TOKEN_READOUT_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# Full frozen observation token readout\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)
h.evaluate=evaluate;c.evaluate=evaluate
def main():h.main()
if __name__=='__main__':main()
