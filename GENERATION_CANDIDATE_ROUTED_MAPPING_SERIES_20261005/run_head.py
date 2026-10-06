"""Both active raw-state persistent head; sole added physical accel feedback."""
from pathlib import Path
import sys,os,json,time,traceback
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import torch
from resource_guard import guard
from state_features import state32
import route_features as route
import physical_accel as physical
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
import importlib.util
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
h=module('routed_raw_parent',ROOT/'GENERATION_CANDIDATE_STATE_SERIES_20261005/run_head.py')
n,c,m,r,p,t=h.n,h.c,h.m,h.r,h.p,h.t
h.HERE=HERE;t.HERE=HERE;c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
RawParent=h.PulseResponseHead
original_objective=m.objective;original_train=m.train;original_save=m.save_atomic
original_evaluate=h.evaluate
class PhysicalExpert(RawParent):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg)
        self.register_buffer('physical_accel_scales',torch.from_numpy(physical.load(HERE)))
    def inputs(self,observation,base):
        native=self.native_inputs(observation);tokens=observation[:,16:784].reshape(-1,8,96)
        query=self.query(native);key=self.key(tokens)+self.past_position[None]
        weight=torch.einsum('bkd,bnd->bkn',query,key).div(32**.5).softmax(-1)
        pooled=torch.einsum('bkn,bnd->bkd',weight,self.value(tokens))
        state=self.state_encoder((state32(base)-self.state_mean)/self.state_std)
        return torch.cat((native,pooled,state),-1),weight

class PhysicalHead(torch.nn.Module):
    def __init__(self,dimension,cfg):
        super().__init__();self.arm=cfg['set_context'];self.strength=cfg['velocity_residual_scale']
        self.experts=torch.nn.ModuleList([PhysicalExpert(dimension,cfg),PhysicalExpert(dimension,cfg)])
        self.experts[1].decoder=self.experts[0].decoder
        self.register_buffer('native_fallback',torch.tensor(False))
        self.register_buffer('route_median',torch.as_tensor(route.load(HERE),dtype=torch.float64))
        self.register_buffer('physical_accel_scales',torch.from_numpy(physical.load(HERE)))
    @property
    def decoder(self):return self.experts[0].decoder
    def native_inputs(self,observation):return self.experts[0].native_inputs(observation)
    def routed_components(self,observation,base,speed):
        q=route.statistic(observation,base);branch=(q.double()>self.route_median).long()
        residuals=[];parts=[]
        for expert in self.experts:
            feature,_=expert.inputs(observation,base);codes=expert.head(feature)
            residual,part=expert.components(codes,speed)
            part['amplitude']=.5*part['limit'].squeeze(2)*codes[...,2:4].tanh()
            residuals.append(residual);parts.append(part)
        stacked=torch.stack(residuals,2)
        weights=torch.ones_like(stacked[:,:,:,:,0])*.5 if self.arm=='zero' else torch.nn.functional.one_hot(branch,2).to(base.dtype)[...,None].expand(-1,-1,-1,12)
        mixed=(stacked*weights[...,None]).sum(2)
        limit=self.strength*(speed[:,0,None,None,None]+.03)
        return mixed.clamp(-limit,limit),dict(q=q,branch=branch,weights=weights,experts=parts,expert_increments=stacked,limit=limit)
    def forward(self,observation,base,speed):
        guard.check()
        if bool(self.native_fallback):return base,torch.zeros_like(base)
        residual,_=self.routed_components(observation,base,speed)
        return base+residual.cumsum(2),residual

TimeResponseHead=PhysicalHead
h.PulseResponseHead=PhysicalHead;h.TimeResponseHead=PhysicalHead;t.TimeResponseHead=PhysicalHead
n.NativeGenerator=PhysicalHead;c.Correction=PhysicalHead;r.m.Correction=PhysicalHead
verify=h.verify
def frozen_audit(model):
    for expert in model.experts:h.frozen_audit(expert)
    assert model.experts[0].decoder is model.experts[1].decoder
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==91596
    assert not any(p.requires_grad for p in model.decoder.parameters())
n.frozen_audit=frozen_audit
_TRACE=[];_ROWS=[];_LOGGING=False
def objective(output,base,future,residual,speed,groups,cfg,arm,scales):
    old=original_objective(output,base,future,residual,speed,groups,cfg,arm,scales)
    physical_scales=torch.as_tensor(physical.load(HERE),device=output.device,dtype=output.dtype)
    extra,detail=physical.penalty(output,base,groups,physical_scales,True)
    treated=True
    total=old+.1*extra if treated else old
    if _LOGGING:
        coordinate_norm=None
        if not _TRACE:
            derivative=torch.autograd.grad(extra,output,retain_graph=True,allow_unused=False)[0]
            coordinate_norm=float(derivative.norm().detach())
        _TRACE.append(dict(n=len(groups),original=float(old.detach()),P=float(extra.detach()),
            added=float((.1*extra).detach()) if treated else 0.,total=float(total.detach()),
            coordinate_P_gradient_norm=coordinate_norm,**detail))
    return total
m.objective=objective
def trace_summary(epoch):
    N=sum(x['n'] for x in _TRACE)
    assert N>0
    return dict(epoch=epoch,batches=len(_TRACE),n=N,
        **{key:sum(x['n']*x[key] for x in _TRACE)/N for key in ('original','P','added','total')},
        groups_n=[sum(x['counts'][g] for x in _TRACE) for g in range(3)],
        active_batches=[sum(x['active'][g] for x in _TRACE) for g in range(3)],
        active_present_fraction=[sum(x['active'][g] for x in _TRACE)/max(1,sum(x['counts'][g]>0 for x in _TRACE)) for g in range(3)],
        first_batch_coordinate_P_gradient_norm=_TRACE[0]['coordinate_P_gradient_norm'],
        definition='training batch population-weighted originaltotal and P; coordinate gradient only, not parameter or population gate')
def save_atomic(value,path):
    if _LOGGING and path.name.startswith('epoch_') and _TRACE:
        value={**value,'objective_diagnostic':trace_summary(value['epoch'])}
    if _LOGGING and path.name=='resume.pth':
        row=trace_summary(value['epoch']);_ROWS.append(row);_TRACE.clear()
        value={**value,'objective_diagnostics':list(_ROWS)}
        m.write_json(path.parent/'objective_history.json',dict(status='RECORDED',rows=_ROWS))
    return original_save(value,path)
m.save_atomic=save_atomic
def train(seed,arm,cache,values,fit,cal,device):
    global _LOGGING,_ROWS
    _TRACE.clear();_ROWS=[]
    resume=m.MODELS/f'seed_{seed}'/arm/'resume.pth'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        _ROWS=list(saved.get('objective_diagnostics',[]));assert len(_ROWS)==saved['epoch']
        del saved
    _LOGGING=True
    try:return original_train(seed,arm,cache,values,fit,cal,device)
    finally:_LOGGING=False;_TRACE.clear()
m.train=train
def prepared_train(device):
    result=h.prepared_train(device);cache,cfg,fit,cal,values=result
    physical.prepare(sys.modules[__name__],cache,fit,values[1]);route.prepare(sys.modules[__name__],cache,fit,values);return result

def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        prior=ROOT/'GENERATION_PHYSICAL_ACCEL_PROTECTION_SERIES_20261005/old_results_manifest.json'
        files=dict(json.loads(prior.read_text(encoding='utf-8'))['files']);verify(files)
        for folder in ROOT.iterdir():
            if not folder.is_dir() or folder==HERE or not folder.name.startswith(('GENERATION_','F16_','HEAD_','PULSE_','RESEARCH_')):continue
            for source in folder.rglob('*'):
                if source.is_file() and '__pycache__' not in source.parts and source.suffix not in ('.lock','.tmp','.pending') and source.name!='runtime_state.json':
                    if str(source) not in files:files[str(source)]=p.sha256(source)
        m.write_json(path,dict(files=files,control_files_excluded=True,all_prior_digests_preserved=True,
            inherited_files=len(json.loads(prior.read_text(encoding='utf-8'))['files'])))
        print('INITIAL_OLD_SHA PASS '+str(len(files)),flush=True);return len(files)
    return verify(json.loads(path.read_text(encoding='utf-8'))['files'])
def source_hashes():
    inherited=json.loads((ROOT/'GENERATION_PHYSICAL_ACCEL_PROTECTION_SERIES_20261005/preflight.json').read_text(encoding='utf-8'))['source_hashes']
    verify(inherited)
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/x for x in ('README.md','DESIGN.md','protocol.json','old_results_manifest.json','state_scalers.npz','state_scalers.json','physical_accel_scales.npz','physical_accel_scales.json','route_threshold.npz','route_threshold.json')]
    return {**inherited,**{str(path):p.sha256(path) for path in paths if path.is_file()}}
h.old_manifest=old_manifest;h.source_hashes=source_hashes
def preflight(device):
    import preflight_impl
    return preflight_impl.run(sys.modules[__name__],device)
c.preflight=preflight

def pulse_diagnostics(device):
    cache=p.load_npz(p.DEV_CACHE);output=m.RESULTS/'pulse';output.mkdir(exist_ok=True);records=[]
    for arm in c.ARMS:
        for seed in (1,2,3):
            for policy in c.POLICIES:
                saved=torch.load(c.checkpoint(seed,arm,policy),map_location='cpu',weights_only=False)
                values=r.prepared(cache,False,mean=saved['mean'],std=saved['std'])
                model=PhysicalHead(saved['input_dim'],saved['config']).to(device).eval();model.load_state_dict(saved['state_dict'])
                frozen_audit(model);pieces={}
                with torch.no_grad():
                    for start in range(0,len(cache['sample_id']),128):
                        o,b,s=[torch.from_numpy(v[start:start+128]).to(device) for v in values[:3]]
                        res,part=model.routed_components(o,b,s)
                        arrays={k:torch.stack([x[k] for x in part['experts']],2) for k in ('center','span','profile','amplitude')}
                        arrays.update(q=part['q'],branch=part['branch'],weights=part['weights'][...,0],expert_increments=part['expert_increments'],mixed_increments=res)
                        if bool(model.native_fallback):arrays['mixed_increments']=torch.zeros_like(res)
                        arrays['drift']=arrays['mixed_increments'].sum(2)
                        for key,value in arrays.items():pieces.setdefault(key,[]).append(value.cpu().numpy())
                columns={k:np.concatenate(v) for k,v in pieces.items()}
                np.savez_compressed(output/f'seed{seed}_{arm}_{policy}_pulse.npz',sample_id=cache['sample_id'],native_fallback=bool(model.native_fallback),route_median=float(model.route_median),**columns)
                records.append(dict(arm=arm,seed=seed,policy=policy,candidate_counts=[int((columns['branch']==i).sum()) for i in (0,1)],both_clock_active=True,nominal_parameters=91596,native_fallback=bool(model.native_fallback)))
    m.write_json(output/'summary.json',dict(status='PASS_DESCRIPTIVE_ONLY',records=records,
        interpretation='Two sigmoid experts; center/span/profile extra expert axis. zero uniform mean bounded increments, mean_std FIT fixed route; both originaltotal+.1P. q not GT intent; drift mixed endpoint.'))
h.pulse_diagnostics=pulse_diagnostics

def evaluate(device):
    original_evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text(encoding='utf-8'))
    summary['scope']='Two91596 raw persistent experts, zero uniform .5/.5 bounded increments vs mean_std FIT median predicted/lastobserved velocity deviation hard route; BOTH originaltotal+.1P. All oldF16 decoder scorer fixed/nativeK20. Nominal not equal effectivecapacity/optimization; no GT route/forward, not causal shared-interference proof.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'CANDIDATE_ROUTED_MAPPING_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# Fixed candidate route versus uniform experts\n\n'+json.dumps(summary,indent=2),encoding='utf-8')

c.evaluate=evaluate
def main():
    try:
        if '--action' in sys.argv and sys.argv[sys.argv.index('--action')+1]=='run':
            _,_,original=n.prepare_backend(torch.device('cpu'));del original
            old_manifest();prepared_train(torch.device('cpu'))
        c.main()
    except BaseException:
        m.RESULTS.mkdir(parents=True,exist_ok=True)
        (m.RESULTS/('failure_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8')
        m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
if __name__=='__main__':main()
