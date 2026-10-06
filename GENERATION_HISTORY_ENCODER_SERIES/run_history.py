"""Causal history branch: fresh matched flat MLP versus GRU."""
from pathlib import Path
import importlib.util
import sys
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'GENERATION_SET_CONTEXT_SERIES/run_context.py'
spec=importlib.util.spec_from_file_location('history_source',SOURCE)
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
m=c.m;r=c.r;p=c.p
c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
FIT_NORMALIZER=None


def initialize_fit():
    global FIT_NORMALIZER
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),c.settings(1)['calibration_fraction'])
    values=r.prepared(cache,False,fit=fit);FIT_NORMALIZER=values[4:6]
    return cache,fit,cal,values


class HistoryCorrection(nn.Module):
    def __init__(self,dimension,cfg):
        super().__init__();self.mode=cfg['set_context'];self.strength=cfg['velocity_residual_scale']
        # Construct decoder FIRST so paired seeds have identical decoder initialization.
        self.network=nn.Sequential(nn.Linear(dimension+24+7+32,cfg['hidden']),nn.LayerNorm(cfg['hidden']),nn.GELU(),nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['hidden'],cfg['hidden']//2),nn.GELU(),nn.Linear(cfg['hidden']//2,24))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        if self.mode=='zero':self.history_encoder=nn.Sequential(nn.Linear(16,70),nn.GELU(),nn.Linear(70,32))
        elif self.mode=='mean_std':self.history_encoder=nn.GRU(2,32,batch_first=True)
        else:raise ValueError(self.mode)
        self.register_buffer('history_mean',torch.from_numpy(FIT_NORMALIZER[0][:,:16].copy()))
        self.register_buffer('history_std',torch.from_numpy(FIT_NORMALIZER[1][:,:16].copy()))

    def encode(self,observation,speed):
        past=(observation[:,:16]*self.history_std+self.history_mean).reshape(-1,8,2)
        increments=torch.diff(past,dim=1,prepend=past[:,:1])/(speed[:,0,None,None]+.03)
        if self.mode=='zero':return self.history_encoder(increments.flatten(1))
        return self.history_encoder(increments)[1][-1]

    def forward(self,observation,base,speed):
        history=self.encode(observation,speed)[:,None].expand(-1,base.shape[1],-1)
        family=F.one_hot(r.m.labels(speed[:,1:],base[:,:,-1]/12),7).to(base.dtype)
        features=torch.cat((observation[:,None].expand(-1,base.shape[1],-1),base.flatten(2),family,history),-1)
        residual=self.network(features).reshape_as(base).tanh()*self.strength*(speed[:,0,None,None,None]+.03)
        return base+residual.cumsum(2),residual


c.Correction=HistoryCorrection;r.m.Correction=HistoryCorrection


def preflight(device):
    cfg=c.settings(1);assert cfg['test_access']=='forbidden' and cfg['seeds']==[1,2,3]
    assert 'separate_oracle_fraction' not in cfg and 'temporal_basis' not in cfg
    prior=json.loads((c.d.CONTROL/'preflight.json').read_text())
    for path,digest in {**prior['source_hashes'],**prior['controls']}.items():assert p.sha256(Path(path))==digest,path
    cache,fit,cal,values=initialize_fit()
    assert not(set(cache['sample_id'][fit])&set(cache['sample_id'][cal]))
    r.setup_weights(cache,fit,values,cfg,device)
    scale=np.array([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32).clip(1e-6)
    assert np.array_equal(scale,np.array(prior['fit_protection_scales'],np.float32))
    old=torch.load(c.d.checkpoint(1,'control','constrained'),map_location='cpu',weights_only=False)
    for key,value in old['config'].items():
        if key!='protocol':assert cfg[key]==value,key
    assert np.array_equal(old['mean'],values[4]) and np.array_equal(old['std'],values[5])
    o,b,s,gt=[torch.from_numpy(v[:128]).to(device) for v in values[:4]];groups=torch.tensor(cache['change_group'][:128],device=device)
    counts=[];decoder=[]
    for arm in c.ARMS:
        p.seed_everything(1);model=HistoryCorrection(o.shape[1],c.settings(1,arm)).to(device);model.eval()
        # cuDNN requires RNN training-mode caches for backward; decoder stays eval (no dropout).
        model.history_encoder.train()
        counts.append(sum(x.numel() for x in model.parameters()))
        decoder.append({key:value.detach().clone() for key,value in model.network.state_dict().items()})
        out,res=model(o,b,s);assert torch.equal(out,b)
        raw=(o[:,:16]*model.history_std+model.history_mean).reshape(-1,8,2)
        expected=torch.from_numpy(cache['motion'][:128,:16].reshape(-1,8,2)).to(device)
        assert torch.allclose(raw,expected,atol=2e-5)
        loss=m.objective(out,b,gt,res,s,groups,cfg,c.d.ARM,torch.tensor(scale,device=device));loss.backward()
        assert torch.isfinite(loss) and model.network[-1].weight.grad.abs().sum()>0
        model.zero_grad()
        with torch.no_grad():model.network[-1].weight.normal_(std=.01)
        out,res=model(o,b,s);loss=m.objective(out,b,gt,res,s,groups,cfg,c.d.ARM,torch.tensor(scale,device=device));loss.backward()
        assert sum(x.grad.abs().sum().item() for x in model.history_encoder.parameters())>0
        perm=torch.randperm(20,device=device)
        with torch.no_grad():
            out_p,_=model(o,b[:,perm],s);assert torch.allclose(out_p,out[:,perm],atol=2e-6)
    assert counts==[43742,43736],counts
    assert all(torch.equal(decoder[0][key],decoder[1][key]) for key in decoder[0])
    sources={**prior['source_hashes'],str(c.d.CONTROL/'run_capacity.py'):p.sha256(c.d.CONTROL/'run_capacity.py'),str(c.d.SOURCE):p.sha256(c.d.SOURCE),str(c.SOURCE):p.sha256(c.SOURCE),str(SOURCE):p.sha256(SOURCE),str(Path(__file__)):p.sha256(Path(__file__)),str(HERE/'protocol.json'):p.sha256(HERE/'protocol.json')}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),new_trainings=6,parameters_flat=counts[0],parameters_gru=counts[1],
        fit_protection_scales=scale.tolist(),source_hashes=sources,controls={},test_access=False))
    print('PREFLIGHT PASS: near-matched params/shared decoder initialization, rawhistory reconstruction, bothencoder gradients, identity/GPU/permutation/hash/config',flush=True)


ORIGINAL_EVALUATE=c.evaluate


def evaluate(device):
    ORIGINAL_EVALUATE(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text())
    summary['variant']='history_encoder'
    summary['scope']='Repeated ETH dev; nativeK20; flat-historyMLP43742 vs GRU43736 parameters,6difference(.014%); shared decoder identical paired initialization, encoders distinct; six fresh35epoch paired-seed runs; 8observed velocity increments/observedspeed+.03,32historyfeatures; original observation/path/family also retained; originalbalancedjointwinner+regularizers+protection.1/bound1.5/fitcal/constrainedcal; no pooledsetcontext/basis/newloss/proposals/test/backbone/scorer. Local auxiliary observation encoder is trainable; original F16 backbone frozen. Generic zero/mean_std labels in NPZ/CSV represent flat/gru only.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_INDEPENDENT_VALIDATION' if all(summary['gates'].values()) else 'HISTORY_ENCODER_GATE_FAIL'
    for row in summary['aggregate']:
        row['arm']={'zero':'flat_mlp','mean_std':'gru'}[row['arm']]
    m.write_json(path,summary)
    lines=['# Causal history encoder','',summary['verdict'],'',summary['scope'],'','| Arm | Policy | Group | minADE | minFDE | Joint |','|---|---|---|---:|---:|---:|']
    for row in summary['aggregate']:
        if row['group'] in ('overall','change'):lines.append(f'| {row["arm"]} | {row["policy"]} | {row["group"]} | {row["minade20"]:.6f} | {row["minfde20"]:.6f} | {row["joint_oracle"]:.6f} |')
    lines+=['',json.dumps(summary['gates'],indent=2)];(m.RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('COMPLETE '+summary['verdict'],flush=True)


c.preflight=preflight;c.evaluate=evaluate
if __name__=='__main__':
    initialize_fit()
    try:c.main()
    except Exception:
        import traceback,time
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
