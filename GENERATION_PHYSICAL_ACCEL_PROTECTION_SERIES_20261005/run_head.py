"""Both active raw-state persistent head; sole added physical accel feedback."""
from pathlib import Path
import sys,os,json,time,traceback
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import torch
from resource_guard import guard
from state_features import state32
import physical_accel as physical
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
import importlib.util
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
h=module('acc_raw_parent',ROOT/'GENERATION_CANDIDATE_STATE_SERIES_20261005/run_head.py')
n,c,m,r,p,t=h.n,h.c,h.m,h.r,h.p,h.t
h.HERE=HERE;t.HERE=HERE;c.HERE=HERE;m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models'
RawParent=h.PulseResponseHead
original_objective=m.objective;original_train=m.train;original_save=m.save_atomic
original_evaluate=h.evaluate
class PhysicalHead(RawParent):
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
TimeResponseHead=PhysicalHead
h.PulseResponseHead=PhysicalHead;h.TimeResponseHead=PhysicalHead;t.TimeResponseHead=PhysicalHead
n.NativeGenerator=PhysicalHead;c.Correction=PhysicalHead;r.m.Correction=PhysicalHead
frozen_audit=h.frozen_audit;verify=h.verify
_TRACE=[];_ROWS=[];_LOGGING=False
def objective(output,base,future,residual,speed,groups,cfg,arm,scales):
    old=original_objective(output,base,future,residual,speed,groups,cfg,arm,scales)
    physical_scales=torch.as_tensor(physical.load(HERE),device=output.device,dtype=output.dtype)
    extra,detail=physical.penalty(output,base,groups,physical_scales,True)
    treated=cfg['set_context']=='mean_std'
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
    physical.prepare(sys.modules[__name__],cache,fit,values[1]);return result

def old_manifest(create=False):
    path=HERE/'old_results_manifest.json'
    if create:
        prior=ROOT/'F16_LOSS_QUALITY_CREDIT_MOTION_DIAGNOSIS_20261005/old_results_manifest.json'
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
    inherited=json.loads((ROOT/'F16_LOSS_QUALITY_CREDIT_MOTION_DIAGNOSIS_20261005/preflight.json').read_text(encoding='utf-8'))['source_hashes']
    verify(inherited)
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/x for x in ('README.md','DESIGN.md','protocol.json','old_results_manifest.json','state_scalers.npz','state_scalers.json','physical_accel_scales.npz','physical_accel_scales.json')]
    return {**inherited,**{str(path):p.sha256(path) for path in paths if path.is_file()}}
h.old_manifest=old_manifest;h.source_hashes=source_hashes
def preflight(device):
    import preflight_impl
    return preflight_impl.run(sys.modules[__name__],device)
c.preflight=preflight
def evaluate(device):
    original_evaluate(device)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text(encoding='utf-8'))
    summary['scope']='Both active raw candidate-state32 persistent45798: zero originaltotal vs mean_std originaltotal+.1 FIT-scaled batchgroup ALLK absolute-accelmean native-relative hinge; all old loss/data/calguard/frozenF16 decoder scorer unchanged. Local protection is not allgroup physical gate or causality.'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'PHYSICAL_ACCEL_FEEDBACK_GATE_FAIL'
    m.write_json(path,summary);(m.RESULTS/'REPORT.md').write_text('# Physical acceleration feedback single factor\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    pp=m.RESULTS/'pulse/summary.json';pulse=json.loads(pp.read_text(encoding='utf-8'))
    pulse['interpretation']='Both rawstate and sigmoid clocks active; sole originaltotal vs added physical acceleration feedback. Saved drift is total endpoint correction. Clock not semantic GT event, old argmax inheritedslot not new probability.'
    m.write_json(pp,pulse)
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
