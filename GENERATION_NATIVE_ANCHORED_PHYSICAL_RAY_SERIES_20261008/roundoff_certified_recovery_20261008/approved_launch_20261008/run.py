"""Approved paired native-anchored refinement; old modules/cache read-only."""
import sys,os,importlib.util,copy,time,gc,traceback,math
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np,torch
import physical_ray as pr
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
H=Path(__file__).resolve().parent;R=H.parents[2]
s=importlib.util.spec_from_file_location('canonical_readonly',R/'PROTOCOL_RECONCILIATION_RUN_20261007/run.py');q=importlib.util.module_from_spec(s);s.loader.exec_module(q);q.H=H
s=importlib.util.spec_from_file_location('saved_quality_readonly',R/'F16_CANONICAL_EQUAL_K20_REPLACEMENT_DIAGNOSIS_20261007/run.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v);v.H=H;v.guard=q.guard
write=q.write;read=q.read;sha=q.sha;digest=q.digest;guard=q.guard;stage=q.stage;DEVICE=q.DEVICE
SPLIT=q.PLAN[1][1];ARMS=('unprojected','physical');SEEDS=(42,43,44);PORTS=('native20','zero_native20','refined20')
torch.use_deterministic_algorithms(True)

def bernstein_matrix():
    x=np.arange(12,dtype=np.float64)/11.;B=np.stack([math.comb(11,i)*x**i*(1-x)**(11-i) for i in range(12)],axis=1)
    assert B.shape==(12,12) and (B>=0).all() and np.max(abs(B.sum(1)-1))<=1e-15 and np.linalg.matrix_rank(B)==12
    return B

class Refiner(nn.Module):
    def __init__(self,st):
        super().__init__();self.register_buffer('mean',torch.from_numpy(st[0]));self.register_buffer('std',torch.from_numpy(st[1]));self.net=nn.Sequential(nn.Linear(120,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,24));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('time_basis',torch.from_numpy(bernstein_matrix()))
    def forward(self,features,anchors,CV,arm,return_controls=False):
        assert arm in ARMS
        raw=self.net((features-self.mean)/self.std).reshape(-1,4,12,2).double();u=torch.tanh(raw);u=u/torch.linalg.vector_norm(u,dim=-1,keepdim=True).clamp_min(1.)
        w=torch.einsum('ti,bkid->bktd',self.time_basis,u)
        bound=.20*(torch.arange(1,13,device=raw.device,dtype=torch.float64)/12.)**2;residual=w*bound[None,None,:,None]
        if arm=='physical':residual=pr.project_torch(anchors.double(),residual)[0]
        result=(anchors.double()+residual,residual)
        return (*result,u) if return_controls else result

def data_load():
    ns=q.setup();data=q.load_data(ns);sp=q.validate_split(data,SPLIT);assert tuple(len(sp[k]) for k in ('grouped_train_index','outer_eval_index','inner_cal_index'))==(18909,1035,1856)
    rank=np.argsort(-data['logs'],axis=1,kind='stable');data['slots']=np.sort(rank[:,16:],axis=1);rows=np.arange(len(data['ids']))[:,None];data['anchor']=data['native'][rows,data['slots']].copy();step=data['past'][:,-1]-data['past'][:,-2];data['CV']=(step[:,None]*np.arange(1,13,dtype=np.float32)[None,:,None]).astype(np.float32)
    assert data['anchor'].shape==(28791,4,12,2) and data['CV'].shape==(28791,12,2)
    assert np.array_equal(data['slots'],np.array([sorted(sorted(range(20),key=lambda k:(-float(row[k]),k))[16:]) for row in data['logs']],dtype=np.int64))
    return ns,data,sp

def features(data,ix):return np.concatenate((np.repeat(data['p2'][ix,None],4,axis=1),data['anchor'][ix].reshape(-1,4,24)),axis=-1).astype(np.float32)

def statistics(data,tr):
    x=features(data,tr).reshape(-1,120);mu=x.mean(0,dtype=np.float64);sd=x.std(0,dtype=np.float64);sd=np.where(sd<1e-6,1.,sd);return mu.astype(np.float32),sd.astype(np.float32)

def loader(data,tr,seed,batch=256):
    ds=TensorDataset(torch.from_numpy(features(data,tr)),torch.from_numpy(data['anchor'][tr]),torch.from_numpy(data['CV'][tr]),torch.from_numpy(data['gt'][tr]),torch.from_numpy(data['groups'][tr].astype(np.int64)))
    g=torch.Generator().manual_seed(seed);return DataLoader(ds,batch_size=batch,shuffle=True,generator=g,num_workers=0,drop_last=False),g

def epoch(m,opt,ldr,arm,index):
    m.train();n=0;total=quality=diversity=0.;counts=np.zeros(4,np.int64);switch=0
    for f,a,c,gt,groups in ldr:
        guard();f,a,c,gt=(x.to(DEVICE) for x in (f,a,c,gt));out,res=m(f,a,c,arm);loss,primary,div,A,F,w=q.objective(out,gt);opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(loss) and all(x.grad is None or torch.isfinite(x.grad).all() for x in m.parameters());opt.step();N=len(f);n+=N;total+=float(loss.detach())*N;quality+=float(primary.detach())*N;diversity+=float(div.detach())*N;ass,_,_=q.assignment(A.detach().cpu().numpy(),F.detach().cpu().numpy(),groups.numpy());counts+=ass['overall']['scene_counts'];switch+=ass['overall']['switched']
    return dict(epoch=index,n=n,total=total/n,quality=quality/n,diversity=diversity/n,winner_counts=counts.tolist(),switch_count=switch,switch_rate=switch/n,denominator='population n-weighted batch pre-update, not posthoc population')

def generated(m,data,ix,arm,return_controls=False):
    m.eval();out=[];res=[];controls=[]
    with torch.inference_mode():
        for first in range(0,len(ix),128):
            guard();ind=ix[first:first+128];a,b,u=m(torch.from_numpy(features(data,ind)).to(DEVICE),torch.from_numpy(data['anchor'][ind]).to(DEVICE),torch.from_numpy(data['CV'][ind]).to(DEVICE),arm,return_controls=True);out.append(a.cpu().numpy());res.append(b.cpu().numpy());controls.append(u.cpu().numpy())
    out=np.concatenate(out);res=np.concatenate(res);bound=.20*(np.arange(1,13)/12.)**2;assert out.shape==(len(ix),4,12,2) and np.isfinite(out).all();assert (np.linalg.norm(res,axis=-1)<=bound[None,None]+2e-15).all();assert (np.linalg.norm(res,axis=-1).mean(-1)<=bound.mean()+2e-15).all();return (out,res,np.concatenate(controls)) if return_controls else (out,res)

def portfolio(data,ix,out,res,arm,controls=None):
    native=data['native'][ix].astype(np.float64);p=native.copy();zeroCV=native.copy();rows=np.arange(len(ix))[:,None];slots=data['slots'][ix];p[rows,slots]=out;zeroCV[rows,slots]=data['anchor'][ix].astype(np.float64)
    expected=data['anchor'][ix].astype(float);assert np.array_equal(out,expected+res)
    for i in range(len(ix)):
        keep=np.setdiff1d(np.arange(20),slots[i]);assert np.array_equal(p[i,keep],native[i,keep])
    z=dict(sample_id=data['ids'][ix],groups=data['groups'][ix],track=data['track'][ix],scene=data['scene'][ix],gt=data['gt'][ix],native20=native,zero_native20=zeroCV,refined20=p,refined4=out,residual=res,slots=slots,reference4=expected)
    if controls is not None:z.update(bounded_controls=controls,time_basis=bernstein_matrix(),temporal_arm=np.array(arm));proposed=np.einsum('ti,bkid->bktd',bernstein_matrix(),controls)*(.20*(np.arange(1,13)/12.)**2)[None,None,:,None];z['projection_alpha']=pr.project_numpy(expected,proposed)[1] if arm=='physical' else np.ones(expected.shape[:2])
    return z

def temporal_audit(z):
    if 'bounded_controls' not in z:return dict(status='SMOKE_WITHOUT_CONTROLS')
    u=z['bounded_controls'];B=bernstein_matrix();arm=str(z['temporal_arm']);assert arm in ARMS and np.array_equal(B,z['time_basis']);assert np.max(np.linalg.norm(u,axis=-1))<=1+2e-15
    bound=.20*(np.arange(1,13)/12.)**2;proposed=np.einsum('ti,bkid->bktd',B,u)*bound[None,None,:,None]
    expected,alpha=pr.project_numpy(z['reference4'],proposed) if arm=='physical' else (proposed,np.ones(proposed.shape[:2]))
    err=float(np.max(abs(expected-z['residual'])));assert err<=2e-12,err
    assert np.max(abs(alpha-z['projection_alpha']))<=2e-10
    physical=pr.audit_numpy(z['reference4'],z['refined4']) if arm=='physical' else dict(status='NOT_CONSTRAINED_CONTROL')
    return dict(status='PASS',independent_residual_reconstruction_max=err,alpha_min=float(alpha.min()),alpha_mean=float(alpha.mean()),alpha_quantiles=np.quantile(alpha,[0,.1,.5,.9,1]),alpha_zero_fraction=float((alpha==0).mean()),pointwise_physical=physical,stronger_than_pooled_gate=True,no_GT_in_projection=True)

def evaluate(z,ind=False):
    temporal_audit(z);paths=[z[k] for k in PORTS];raw=[v.errors(p,z['gt'],ind) for p in paths];ng=raw[0][3].any(1);cg=raw[1][3].any(1);metrics={};vectors=[];cc=[]
    for i,name in enumerate(PORTS):
        a,f,j,g,one=raw[i];good=g.any(1);cl=v.clusters(paths[i],g,ind);cc.append(cl);vectors.append(np.stack((good,a.min(1),f.min(1),j.min(1)),1));metrics[name]={}
        for group,mask in v.masks(z['groups']):
            n=int(mask.sum())
            if not n:metrics[name][group]=dict(n=0);continue
            mot=v.motion(paths[i][mask],ind);base=v.motion(paths[0][mask],ind);own=ng if np.array_equal(z['reference4'],z['native20'][np.arange(len(z['slots']))[:,None],z['slots']]) else cg
            metrics[name][group]=dict(n=n,good=int(good[mask].sum()),coverage=float(good[mask].mean()),minADE=float(a[mask].min(1).mean()),minFDE=float(f[mask].min(1).mean()),J=float(j[mask].min(1).mean()),rescued=int((good&~ng&mask).sum()),lost=int((~good&ng&mask).sum()),retained=int((good&ng&mask).sum()),own_zero_rescued=int((good&~own&mask).sum()),own_zero_lost=int((~good&own&mask).sum()),effective_distinct=float(cl[mask,0].mean()),effective_good=float(cl[mask,1].mean()),sensitivity_03_05_good=int(((a<=.3)&(f<=.5)).any(1)[mask].sum()),single_Jclosest_early_middle_late=one[mask].mean(0),motion=mot,native20_motion=base,motion_all5_1pct=all(mot[k]<=1.01*base[k] for k in mot))
    ae,fe,_,_,_=v.errors(z['refined4'],z['gt'],ind);assignment,_,_=q.assignment(ae,fe,z['groups']);return dict(metrics=metrics,assignment=assignment),dict(vectors=np.stack(vectors,1),cluster_counts=np.stack(cc,1))

def locks():
    prior=H.parent;inherited=read(prior/'source_manifest.json')['files'];out=dict(inherited);q.verify(out)
    add=[*q.INPUTS,SPLIT,*H.glob('*.py'),*H.glob('*.ps1'),H/'README.md',H/'DESIGN.md',H/'protocol.json']
    for root in (prior,):
        add.extend(p for p in root.rglob('*') if H not in p.parents and 'numeric_recovery_20261008' not in p.parts and p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.tmp','.lock','.log') and p.name!='runtime_state.json')
    for p in sorted(set(add)):
        k=str(p);h=sha(p)
        if k in out:assert out[k]==h,('INHERITED_SHA_CHANGED',k)
        else:out[k]=h
    return out,len(inherited)

def binding(data,sp,st,seed,arm,files):
    d=H/'runs'/arm/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True);sid=d/'split_ids.npz';stats=d/'statistics.npz';tr=sp['grouped_train_index']
    if not sid.exists():np.savez_compressed(sid,train_ids=data['ids'][tr],outer_ids=data['ids'][sp['outer_eval_index']],inner_ids=data['ids'][sp['inner_cal_index']],original_CAL_ids=data['ids'][25912:])
    if not stats.exists():np.savez_compressed(stats,mean=st[0],std=st[1],normalization_ids=data['ids'][tr])
    q.seed_all(seed);m=Refiner(st);protocol=read(H/'protocol.json');man=dict(protocol,arm=arm,seed=seed,source_sha=digest(files),source_files=files,split_sha=sha(SPLIT),split_ids_sha=sha(sid),statistics_sha=sha(stats),initial_parameters_sha=q.tensor_hash(m.state_dict()),parameter_names=list(dict(m.named_parameters())),parameter_count=sum(p.numel() for p in m.parameters()))
    assert man['parameter_count']==25304
    if (d/'run_manifest.json').exists():assert read(d/'run_manifest.json')==man
    else:write(d/'run_manifest.json',man)
    return d,man

def checkpoint(p,m,opt,g,history,st,man):q.save_checkpoint(p,m,opt,g,history,st,man)

def resume(p,m,opt,g,st,man):return q.resume(p,m,opt,g,st,man)

def done_valid(d,man):return q.completed_valid(d,man)

def scientific(data,sp,st,seed,arm,files,count):
    d,man=binding(data,sp,st,seed,arm,files)
    if (d/'done.json').exists():done_valid(d,man);return read(d/'metrics.json')
    q.seed_all(seed);m=Refiner(st).to(DEVICE);assert q.tensor_hash(m.state_dict())==man['initial_parameters_sha'];opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);ldr,g=loader(data,sp['grouped_train_index'],seed);ck=d/'checkpoint.pt';history=resume(ck,m,opt,g,st,man) if ck.exists() else []
    for ep in range(len(history)+1,26):
        assert datetime.now().timestamp()<datetime.fromisoformat('2026-10-08T10:10:00+08:00').timestamp(),'STOP_TIME_WINDOW_AT_SAFE_EPOCH_BOUNDARY';stage('TRAIN',arm=arm,seed=seed,epoch=ep,runs_done=count,runs_total=6);row=epoch(m,opt,ldr,arm,ep);small=sp['inner_cal_index'][:12];smallout,_=generated(m,data,small,arm);row['inner12_posthoc_Q']=float((v.errors(smallout,data['gt'][small])[0]+.5*v.errors(smallout,data['gt'][small])[1]).min(1).mean());history.append(row);checkpoint(ck,m,opt,g,history,st,man);write(d/'history.json',history)
    assert len(history)==25 and all(x['n']==18909 for x in history);metrics={};saved=[];assignment={};frozen={k:x.detach().cpu().clone() for k,x in m.state_dict().items()}
    for split,ix in [('outer',sp['outer_eval_index']),('inner',sp['inner_cal_index']),('original_CAL',np.arange(25912,28791))]:
        stage('FORWARD_SAVE',arm=arm,seed=seed,split=split,runs_done=count);out,res,controls=generated(m,data,ix,arm,True);z=portfolio(data,ix,out,res,arm,controls);path=d/f'{split}.npz';assert not path.exists(),'Partial saved output retained; exact new recovery required, no overwrite';np.savez_compressed(path,**z);reload=q.npz(path);assert all(np.array_equal(x,reload[k]) for k,x in z.items());stage('FULL_SAVED_QUALITY_CLUSTERS_MOTION',arm=arm,seed=seed,split=split);met,extra=evaluate(reload);imet,iextra=evaluate(reload,True);err=v.compare(met,imet);assert err<=1e-12 and all(np.array_equal(extra[k],iextra[k]) for k in extra);metrics[split]=met['metrics'];assignment[split]=met['assignment'];aux=d/f'{split}_metrics.npz';np.savez_compressed(aux,**extra,sample_id=z['sample_id'],groups=z['groups'],track=z['track']);saved.extend((path,aux));write(d/f'{split}_audit.json',dict(status='PASS',independent_max_error=err,all_saved_quality_full_clusters_five_motion=True,unchanged16_exact=True,bounded_residual=True,temporal_coupling=temporal_audit(reload)));saved.append(d/f'{split}_audit.json');del z,reload,extra,iextra
    assert all(torch.equal(x.detach().cpu(),frozen[k]) for k,x in m.state_dict().items());result=dict(arm=arm,seed=seed,epoch=25,splits=metrics,all_weights_bool_scaler_buffers_eval_EXACT=True)
    write(d/'metrics.json',result);write(d/'assignment_stats.json',dict(final_posthoc=assignment,training_preupdate_history=history,four_refined_candidate_quality_only=True));saved.extend([ck,d/'run_manifest.json',d/'split_ids.npz',d/'statistics.npz',d/'history.json',d/'metrics.json',d/'assignment_stats.json']);write(d/'done.json',dict(status='PASS',run_manifest_sha=digest(man),files={str(p):sha(p) for p in saved},all25epochs=True,all_frozen_old_models_not_instantiated_or_trained=True));del m,opt,ldr,frozen;gc.collect();torch.cuda.empty_cache();return result

def bootstrap():
    rng=np.random.default_rng(20261008);records=[]
    for split in ('outer','inner','original_CAL'):
        sets={a:[q.npz(H/'runs'/a/f'seed_{seed}/{split}_metrics.npz') for seed in SEEDS] for a in ARMS};z=sets['unprojected'][0]
        for seq in sets.values():
            for x in seq:assert all(np.array_equal(z[k],x[k]) for k in ('sample_id','groups','track'))
        _,inv=np.unique(z['track'],return_inverse=True);T=int(inv.max())+1
        for group,mask in v.masks(z['groups']):
            counts=np.bincount(inv[mask],minlength=T);active=np.flatnonzero(counts)
            for arm,ref in [('unprojected','native20'),('physical','native20'),('physical','unprojected')]:
                diff=np.stack([sets[arm][i]['vectors'][:,2]-(sets['unprojected'][i]['vectors'][:,2] if ref=='unprojected' else sets[arm][i]['vectors'][:,0]) for i in range(3)]);sums=np.zeros((3,T,4))
                for i in range(3):np.add.at(sums[i],inv[mask],diff[i,mask])
                draws=[]
                for b in range(2000):
                    if b%64==0:guard()
                    ss=rng.integers(0,3,size=3);tt=rng.choice(active,len(active),replace=True);draws.append(sums[ss][:,tt].sum((0,1))/(3*counts[tt].sum()))
                records.append(dict(split=split,group=group,arm=arm,reference=ref,n=int(mask.sum()),tracks=len(active),metrics=['coverage','ADE','FDE','J'],estimate=diff[:,mask].mean((0,1)),CI95=np.quantile(draws,[.025,.975],axis=0).T,exploratory=True,paired_modelseeds=True))
    return records

def main():
    guard();pre=read(H/'PREFLIGHT.json');assert pre['status']=='PASS';files=read(H/'source_manifest.json')['files'];assert digest(files)==pre['source_sha'];stage('INITIAL_SOURCE_OLD_SHA');q.verify(files);ns,data,sp=data_load();st=statistics(data,sp['grouped_train_index']);saved=q.npz(H/'statistics.npz');assert np.array_equal(st[0],saved['mean']) and np.array_equal(st[1],saved['std']);records=[]
    for arm in ARMS:
        for seed in SEEDS:records.append(scientific(data,sp,st,seed,arm,files,len(records)))
    stage('FINAL_BOOTSTRAP_SAVED_ARITHMETIC');ci=bootstrap();ci2=bootstrap();assert v.compare(ci,ci2)==0;write('BOOTSTRAP.json',dict(status='PASS',records=ci,deterministic_second_error=0));outsha={}
    for arm in ARMS:
        for seed in SEEDS:
            d=H/'runs'/arm/f'seed_{seed}';man=read(d/'run_manifest.json');done_valid(d,man);ck=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==25 and ck['run_manifest']==man and len(ck['history'])==25
            outsha.update(read(d/'done.json')['files']);outsha[str(d/'done.json')]=sha(d/'done.json')
    outsha[str(H/'BOOTSTRAP.json')]=sha(H/'BOOTSTRAP.json');stage('FINAL_SOURCE_OLD_SHA');q.verify(files);q.verify(outsha);guard();write('ARITHMETIC_AUDIT.json',dict(status='PASS',saved_jobs=18,runs=6,epochs=150,full_quality_clusters_five_motion_independent=True,all_eval_weights_bool_scaler_buffers_EXACT=True,all_fixed16_and_bounds_PASS=True,bootstrap_records=len(ci),not_all_gradients_or_models_replayed=True));write('PRESERVATION.json',dict(status='PASS',locked_files=len(files),inherited_relevant_files=pre['inherited_relevant_files'],not_all_historical_files=True,old_F16_decoder_scorer_not_instantiated_trained_or_optimizer_accessed=True));write('RESOURCE_USAGE.json',dict(status='PASS',**q.RESOURCE));write('RESULT.json',dict(status='PASS_NATIVE_ANCHORED_PHYSICAL_RAY_PAIRED_RUN',records=records,source_sha=pre['source_sha'],output_sha=outsha,resource=q.RESOURCE,runs=6,saved_jobs=18,formal_experiment_count=1,major_success_not_inferred=True));stage('SUCCESS',runs_done=6,saved_jobs=18)
if __name__=='__main__':
    try:main()
    except Exception:
        write('FAILURE_'+str(time.time_ns())+'.json',dict(status='STOP',traceback=traceback.format_exc(),resource=q.RESOURCE));raise
