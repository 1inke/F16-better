"""Bounded infrastructure extension. Original scientific AST statements unchanged."""
import ast,copy,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).resolve().parent;P=H.parent;S=P.parent;W=S.parent
sys.path.insert(0,str(P))
from epoch_recovery import EpochRecovery,atomic_json,digest,guard,verify_binding,rng_state,restore_rng
import isolated_trainer as original

class ResourceAccumulator(list):
    def __init__(self,items=()):
        super().__init__();self.count=0;self.peak_private=0.;self.peak_RSS=0.;self.min_physical=100.;self.min_commit=100.
        for x in items:self.append(x)
    def append(self,x):
        self.count+=1;self.peak_private=max(self.peak_private,x['private_GiB']);self.peak_RSS=max(self.peak_RSS,x['RSS_GiB'])
        self.min_physical=min(self.min_physical,x['physical_available_GiB']);self.min_commit=min(self.min_commit,x['commit_available_GiB'])
        super().append(x)
        if len(self)>8:del self[:-8]
    def __add__(self,other):
        r=copy.deepcopy(self)
        for x in other:r.append(x)
        if isinstance(other,ResourceAccumulator):
            r.count=self.count+other.count;r.peak_private=max(self.peak_private,other.peak_private);r.peak_RSS=max(self.peak_RSS,other.peak_RSS)
            r.min_physical=min(self.min_physical,other.min_physical);r.min_commit=min(self.min_commit,other.min_commit)
        return r
    def report(self):return dict(checks=self.count,peak_private_GiB=self.peak_private,peak_RSS_GiB=self.peak_RSS,min_physical_GiB=self.min_physical,min_commit_GiB=self.min_commit,bounded_recent_samples=8)

def fingerprint(model):
    h=hashlib.sha256()
    for n,x in model.state_dict().items():
        y=x.detach().cpu().contiguous();h.update(n.encode());h.update(str(y.dtype).encode());h.update(np.asarray(y).tobytes())
    return h.hexdigest()

tree=ast.parse(original.SOURCE.read_text(encoding='utf-8'))
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
loop=next(n for n in ast.walk(run) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and ast.unparse(n.target).startswith('(past, future'))
cut=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and ast.unparse(n)=='optimizer.zero_grad(set_to_none=True)')
REPLAY=compile(ast.fix_missing_locations(ast.Module(body=loop.body[:cut],type_ignores=[])),str(original.SOURCE)+'::unchanged-training-prefix','exec')
ParentAdapter=original.Adapter
class LifecycleAdapter(ParentAdapter):
    def transform_body(self,body):
        result=super().transform_body(body)
        for n in result:
            if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and ast.unparse(n.target).startswith('(past, future'):
                n.body=original.injected('_recovery.before_batch(locals())')+n.body
        return result
original.Adapter=LifecycleAdapter

def load(branch,recovery,subset=None):
    mod,counts=original.load_adapter(branch,recovery,subset);recovery.module=mod;return mod,counts

class ProductionRecovery(EpochRecovery):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.resources=ResourceAccumulator();self.replay_reports=[];self.input_report=None;self.replay_snapshot=None;self.frozen_snapshot=None
    def begin(self,s):
        fresh_sha=fingerprint(s['model']);self.init_sha=fresh_sha
        result=super().begin(s)
        if not isinstance(self.resources,ResourceAccumulator):self.resources=ResourceAccumulator(self.resources)
        init=s['run_dir']/'initial_fingerprint.json'
        if self.resume:
            if json.loads(init.read_text())['sha']!=fresh_sha:raise RuntimeError('STOP_INITIALIZATION_CONTRACT')
        else:atomic_json(dict(sha=fresh_sha,parameters=s['contract']['parameter_counts']),init)
        if self.fixture:self.input_report=self.check_input(s)
        self.frozen_snapshot={n:x.detach().cpu().clone() for n,x in s['model'].named_parameters() if not x.requires_grad}
        self.frozen_buffers={n:x.detach().cpu().clone() for n,x in s['model'].named_buffers()}
        return result
    def check_input(self,s):
        from input_transform import transform
        raw=original.load_adapter('full',self,513)[0].TrajectoryDataset(s['data_path'],s['num_path'],s['cfg'])
        items=[raw[int(i)] for i in s['train_indices'][:64]]
        x=torch.stack([r[0] for r in items]);n=torch.stack([r[2] for r in items]).reshape(-1)
        altered=x.clone();g=torch.Generator().manual_seed(20261009)
        if self.branch=='last2':altered[:,:,:6]=torch.randn(altered[:,:,:6].shape,generator=g)
        elif self.branch=='noNbr':altered[:,1:]=torch.randn(altered[:,1:].shape,generator=g)
        y,v=transform(x.numpy(),n.numpy(),self.branch);z,k=transform(altered.numpy(),n.numpy(),self.branch)
        if self.branch=='full':assert np.array_equal(y,x.numpy())
        else:assert np.array_equal(y,z) and np.array_equal(v,k)
        model=copy.deepcopy(s['model']).eval();before=fingerprint(model)
        def forward(a,valid):
            a=torch.from_numpy(a).to(s['device']);valid=torch.from_numpy(valid).to(s['device']).int()
            with torch.inference_mode():
                a,_=self.module.normalize_to_focal_heading(a,a[:,:,:1].repeat(1,1,12,1));c=a.mean(-2,keepdim=True)
                p=model(a-c,None,None,mode='testing',num_valid=valid,graph_loc=a)[2]+c.unsqueeze(-3)
                return p.cpu().clone(),model.last_mode_logits.cpu().clone()
        a,b=forward(y,v)
        c,d=forward(x.numpy() if self.branch=='full' else z,n.numpy() if self.branch=='full' else k)
        assert torch.equal(a,c) and torch.equal(b,d) and torch.isfinite(a).all() and fingerprint(model)==before
        actual=[s['base_dataset'][int(i)] for i in s['train_indices'][:64]]
        assert np.array_equal(torch.stack([r[0] for r in actual]).numpy(),y)
        assert np.array_equal(torch.stack([r[2] for r in actual]).reshape(-1).numpy(),v)
        return dict(status='PASS_ACTUAL_RAW_DATASET_ALL_FORWARD_PATHS',rows=64,raw_input_transform_EXACT=True,actual_prediction_and_logits_EXACT=True,GT_forward=False,weights_buffers_EXACT=True)
    def before_batch(self,s):
        self.resources.append(guard())
        if self.fixture and len(self.replay_reports)<8:
            t=dict(s);t['model']=copy.deepcopy(s['model']);t['equivariance_losses']=[]
            for key in ('past','future','num_valid','sample_scene_ids'):t[key]=s[key].clone()
            self.replay_snapshot=(t,rng_state(s['generator']))
    def observe(self,s):
        super().observe(s)
        if self.replay_snapshot is not None:
            t,state=self.replay_snapshot;now=rng_state(s['generator']);restore_rng(state,s['generator'])
            exec(REPLAY,self.module.__dict__,t);t['loss'].backward()
            assert torch.equal(s['loss'].detach(),t['loss'].detach()),'STOP_ORIGINAL_LOSS_EQUIVALENCE'
            tested=0
            for (name,p),(name2,q) in zip(s['model'].named_parameters(),t['model'].named_parameters()):
                assert name==name2
                if p.grad is None:assert q.grad is None
                else:assert q.grad is not None and torch.equal(p.grad,q.grad),('STOP_ORIGINAL_GRADIENT',name);tested+=1
            for name,x in self.frozen_snapshot.items():assert torch.equal(dict(s['model'].named_parameters())[name].detach().cpu(),x)
            if self.frozen_snapshot:
                for name,x in self.frozen_buffers.items():assert torch.equal(dict(s['model'].named_buffers())[name].detach().cpu(),x),('STOP_FROZEN_BUFFER',name)
            restore_rng(now,s['generator']);self.replay_reports.append(dict(batch=len(s['past']),loss_and_all_gradients_EXACT=True,tested_gradient_parameters=tested,frozen_old_parameters_buffers_EXACT=True))
            self.replay_snapshot=None
    def commit(self,s):
        super().commit(s)
        atomic_json(dict(stage='TRAIN_EPOCH_COMMITTED',time=time.time(),pid=os.getpid(),branch=self.branch,decoder=s['args'].decoder_experiment,seed=s['args'].seed,fold=os.environ.get('E3_FOLD','eth'),epoch=s['epoch']+1,total_epochs=s['args'].epochs,resource=self.resources.report(),fixture=self.fixture),self.root/'progress.json')
    def finish(self,s):
        if verify_binding(self.binding)!=self.binding_sha:raise RuntimeError('STOP_BINDING_FILE_MUTATION')
        extras=[]
        if s['args'].decoder_experiment=='v2_wide_endpoint_correction':
            from saved_quality import export_selected
            extras=export_selected(self,s)
        super().finish(s)
        dest=s['run_dir'];done=json.loads((dest/'done.json').read_text());done['outputs'].update({str(p):digest(p) for p in [dest/'initial_fingerprint.json',*extras]})
        done.update(resource_aggregate=self.resources.report(),input_path_audit=self.input_report,original_loss_gradient_replay=self.replay_reports,V2_posthoc_complete=bool(extras))
        atomic_json(done,dest/'done.json')
    def skip_compatible(self,s):
        super().skip_compatible(s)
        done=json.loads((s['run_dir']/'done.json').read_text())
        if s['args'].decoder_experiment=='v2_wide_endpoint_correction' and not done.get('V2_posthoc_complete'):raise RuntimeError('STOP_INCOMPLETE_SAVED_POSTHOC')
