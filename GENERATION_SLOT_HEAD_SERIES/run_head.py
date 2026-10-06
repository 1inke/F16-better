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
class SlotLinear(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.zeros(20,6,64));self.bias=nn.Parameter(torch.zeros(20,6))
    def forward(self,value):
        assert value.shape[1:]==(20,64),'Only locked original native K20 order supported'
        return torch.einsum('bkd,kod->bko',value,self.weight)+self.bias[None]
class SlotHead(h.DirectHead):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        if self.arm=='mean_std':self.head[-1]=SlotLinear()
    def inputs(self,observation):
        d=self.decoder;past=observation[:,:16].reshape(-1,1,8,2)
        tokens=observation[:,16:16+8*n.TOKEN_DIM].reshape(-1,1,8,n.TOKEN_DIM)
        context=d.context(torch.cat((tokens[:,:,-1],tokens.mean(2),d.motion_features(past)),-1))
        query=d.mode_embedding[None,None].expand(len(observation),1,-1,-1)
        mode=d.mode_mlp(context[:,:,None]+query)
        return torch.cat((context[:,:,None].expand_as(mode),mode),-1)[:,0]
h.DirectHead=SlotHead;n.NativeGenerator=SlotHead;c.Correction=SlotHead;r.m.Correction=SlotHead
def preflight(device):
    _,_,original=n.prepare_backend(device);del original
    prior=json.loads((HERE.parent/'GENERATION_ANCHOR_FEEDBACK_SERIES/preflight.json').read_text())
    for path,digest in prior['source_hashes'].items():assert p.sha256(Path(path))==digest,path
    cache=p.load_npz(p.TRAIN_CACHE);cfg=c.settings(1)
    assert cfg['seeds']==[1,2,3] and cfg['epochs']==80 and cfg['test_access']=='forbidden'
    fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    assert len(fit)==25912 and len(cal)==2879 and not(set(cache['sample_id'][fit])&set(cache['sample_id'][cal]))
    values=r.prepared(cache,False,fit=fit);r.setup_weights(cache,fit,values,cfg,device)
    o,b,s,gt=[torch.from_numpy(v[fit[:128]]).to(device) for v in values[:4]]
    groups=torch.tensor(cache['change_group'][fit[:128]],device=device)
    scale=torch.tensor(prior['fit_protection_scales'],device=device)
    counts={};front=[];features=[];times={};gradrows={}
    for arm in c.ARMS:
        p.seed_everything(1);model=SlotHead(o.shape[1],c.settings(1,arm)).to(device);model.eval()
        front.append({k:v.detach().clone() for k,v in model.head[:4].state_dict().items()})
        features.append(model.inputs(o).detach())
        counts[arm]=sum(v.numel() for v in model.parameters() if v.requires_grad)
        assert all(not v.requires_grad for v in model.decoder.parameters())
        out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0 and torch.isfinite(out).all()
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate'])
        start=time.perf_counter()
        for step in range(3):
            optimizer.zero_grad();out,res=model(o,b,s)
            loss=m.objective(out,b,gt,res,s,groups,cfg,'balanced_protected',scale)
            loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.head.parameters())
            assert model.head[-1].weight.grad.abs().sum()>0
            if step==2:assert model.head[0].weight.grad.abs().sum()>0
            if arm=='mean_std':gradrows[step]=(model.head[-1].weight.grad.abs().flatten(1).sum(1)>0).sum().item()
            optimizer.step();h.frozen_audit(model)
        torch.cuda.synchronize();times[arm]=(time.perf_counter()-start)/3
        assert out.shape==(128,20,12,2) and torch.isfinite(out).all()
        assert (res.abs()<=model.strength*(s[:,0,None,None,None]+.03)+1e-6).all()
        if arm=='mean_std':
            # Independent synthetic reachability; not claiming each slot is a FIT winner.
            optimizer.zero_grad();model.head[-1](torch.ones(2,20,64,device=device)).sum().backward()
            assert (model.head[-1].weight.grad.abs().flatten(1).sum(1)>0).all()
        model.native_fallback.fill_(True);out,res=model(o,b,s);assert torch.equal(out,b) and res.abs().sum()==0
    assert counts==dict(zero=25158,mean_std=32568),counts
    assert all(torch.equal(front[0][k],front[1][k]) for k in front[0])
    assert torch.equal(features[0],features[1])
    files=[Path(__file__),HERE/'protocol.json',HERE/'run_native.py',HERE/'test_resume.py',HERE/'independent_worker.py',HERE/'start_independent.ps1',HERE.parent/'GENERATION_SLOT_HEAD_DESIGN.md',HERE.parent/'GENERATION_ANCHOR_FEEDBACK_SERIES/shared_head_diagnosis.json']
    hashes={**prior['source_hashes'],**{str(v):p.sha256(v) for v in files}}
    m.write_json(HERE/'preflight.json',dict(status='PASS',parameters_trainable=counts,shared_front_init_exact=True,native_identity_exact=True,native_features_equal=True,mode_order='locked original native K20 IDs; arbitrary reordered inputs unsupported',real_fit_active_slot_gradient_counts=gradrows,all20_synthetic_gradient_reachable=True,fit_n=len(fit),cal_n=len(cal),fit_protection_scales=prior['fit_protection_scales'],source_hashes=hashes,seconds_per_128step=times,test_access=False))
    print('PREFLIGHT PASS '+json.dumps(dict(parameters=counts,timings=times)),flush=True)
h.preflight=preflight;c.preflight=preflight
def evaluate(device):
    n.evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['scope']='Six NEW80epoch shared versus native-slot-untied new last-layer trainings;25158 vs32568params(+29.46%),scope/capacity coupled NOT equal-capacity proof. Identical originalcontext64+native-modeMLP64 and paired new shared front/zero output/nativeidentity. Locked original native K20 order only; no GT/logit/family routing. Wholeoriginaldecoder/F16/scorer frozen. Nativeplus3anchor correction/commonvelocityclip1.5/originalloss/cal; not independentgenerator/intent/ranking; nativeepoch0fallback not learned. RepeatedETHdev exploratory.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'SLOT_HEAD_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# Untied native-slot output head\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)
h.evaluate=evaluate;c.evaluate=evaluate
def main():h.main()
if __name__=='__main__':main()
