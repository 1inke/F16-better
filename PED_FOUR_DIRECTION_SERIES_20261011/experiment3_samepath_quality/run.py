"""One quality objective factor on fixed position3; immutable old caches and sources."""
import sys, os, importlib.util, time, gc, traceback
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import numpy as np
import torch
from torch import nn
H=Path(__file__).resolve().parent
R=H.parents[1]
spec=importlib.util.spec_from_file_location('readonly_anchor_infrastructure',R/'GENERATION_NATIVE_ANCHORED_RESIDUAL_SERIES_20261008/run.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.H=H;b.q.H=H;b.v.H=H
q=b.q;v=b.v
write=q.write;read=q.read;sha=q.sha;digest=q.digest;guard=q.guard;stage=q.stage;DEVICE=q.DEVICE
ARMS=('sum','joint');SEEDS=(42,43,44);SPLIT=b.SPLIT
PORTS=('native20','zero_native20','refined20');b.PORTS=PORTS
data_load=b.data_load;features=b.features;statistics=b.statistics;loader=b.loader
checkpoint=b.checkpoint;resume=b.resume;done_valid=b.done_valid
torch.use_deterministic_algorithms(True)

def objective(paths,gt,arm):
    assert arm in ARMS and paths.shape[1:]==(4,12,2)
    if arm=='sum':return q.objective(paths,gt)
    err=torch.linalg.vector_norm(paths-gt[:,None],dim=-1);ade=err.mean(-1);fde=err[:,:,-1]
    scores=torch.stack((ade,.5*fde),-1).max(-1).values;win=scores.argmin(1);best=scores.min(1).values.mean()
    div=torch.zeros((),device=paths.device,dtype=paths.dtype)
    for i in range(4):
        for j in range(i+1,4):div=div+(torch.relu(.3-torch.linalg.vector_norm(paths[:,i]-paths[:,j],dim=-1).mean(-1))**2).mean()
    div=div/6;return best+.05*div,best,div,ade,fde,win

def assignment(ade,fde,groups):
    sums=ade+.5*fde;joint=np.maximum(ade,.5*fde);sw=sums.argmin(1);jw=joint.argmin(1)
    ss=np.sort(sums,1);js=np.sort(joint,1);result={}
    for name,mask in v.masks(groups):
        n=int(mask.sum())
        if not n:result[name]={'n':0};continue
        result[name]=dict(n=n,switched=int((sw[mask]!=jw[mask]).sum()),switch_rate=float((sw[mask]!=jw[mask]).mean()),sum_counts=np.bincount(sw[mask],minlength=4),joint_counts=np.bincount(jw[mask],minlength=4),confusion=np.bincount(sw[mask]*4+jw[mask],minlength=16).reshape(4,4),sum_exact_ties=int((ss[mask,1]==ss[mask,0]).sum()),joint_exact_ties=int((js[mask,1]==js[mask,0]).sum()),sum_near_ties=int((ss[mask,1]-ss[mask,0]<=1e-6).sum()),joint_near_ties=int((js[mask,1]-js[mask,0]<=1e-6).sum()),sum_gap_mean=float((ss[mask,1]-ss[mask,0]).mean()),joint_gap_mean=float((js[mask,1]-js[mask,0]).mean()),component_ADE_count=int((ade[mask]>=.5*fde[mask]).sum()),component_FDE_count=int((ade[mask]<.5*fde[mask]).sum()),component_exact_ties=int((ade[mask]==.5*fde[mask]).sum()),components_denominator=4*n)
    return result,sw,jw

def epoch(m,opt,ldr,arm,index):
    m.train();n=0;total=quality=diversity=0.;counts=np.zeros(4,np.int64);sums=np.zeros(4,np.int64);joints=np.zeros(4,np.int64);switch=0;compade=compfde=compties=sumties=jointties=0
    for f,a,c,gt,groups in ldr:
        guard();f,a,c,gt=(x.to(DEVICE) for x in (f,a,c,gt));out,res=m(f,a,c,arm);loss,primary,div,A,F,w=objective(out,gt,arm)
        opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(loss) and all(x.grad is None or torch.isfinite(x.grad).all() for x in m.parameters());opt.step()
        N=len(f);n+=N;total+=float(loss.detach())*N;quality+=float(primary.detach())*N;diversity+=float(div.detach())*N
        ass,_,_=assignment(A.detach().cpu().numpy(),F.detach().cpu().numpy(),groups.numpy());z=ass['overall'];sums+=z['sum_counts'];joints+=z['joint_counts'];counts+=z[arm+'_counts'];switch+=z['switched'];compade+=z['component_ADE_count'];compfde+=z['component_FDE_count'];compties+=z['component_exact_ties'];sumties+=z['sum_exact_ties'];jointties+=z['joint_exact_ties']
    return dict(epoch=index,n=n,total=total/n,quality=quality/n,diversity=diversity/n,winner_counts=counts.tolist(),sum_counts=sums.tolist(),joint_counts=joints.tolist(),switch_count=switch,switch_rate=switch/n,sum_exact_ties=sumties,joint_exact_ties=jointties,component_ADE_count=compade,component_FDE_count=compfde,component_exact_ties=compties,components_denominator=4*n,denominator='population n-weighted batch pre-update, not posthoc population')

def bases():
    t=np.arange(1,13,dtype=np.float64)
    position=np.stack([np.interp(t,[0,4,8,12],np.eye(4)[j+1]) for j in range(3)],1)
    acceleration=np.stack([np.interp(t,[1,6,12],np.eye(3)[j]) for j in range(3)],1)
    return position,acceleration

def decode_numpy(anchor,u,arm):
    P,A=bases();bound=.20*(np.arange(1,13)/12.)**2
    if arm in ARMS:proposal=np.einsum('ti,bkid->bktd',P,u*(.20*(np.array([4,8,12])/12.)**2)[None,None,:,None])
    else:
        assert arm=='acceleration3'
        acc=np.einsum('ti,bkid->bktd',A,u)*.05
        proposal=.4*np.cumsum(.4*np.cumsum(acc,axis=2),axis=2)
    norm=np.linalg.norm(proposal,axis=-1)
    ratio=np.where(norm>0,bound[None,None]/np.where(norm>0,norm,1.),1.)
    lam=np.minimum(1.,ratio.min(-1))
    residual=proposal*lam[:,:,None,None]
    return anchor.astype(np.float64)+residual,residual,lam

class Refiner(nn.Module):
    def __init__(self,st):
        super().__init__()
        self.register_buffer('mean',torch.from_numpy(st[0]));self.register_buffer('std',torch.from_numpy(st[1]))
        P,A=bases();self.register_buffer('position_basis',torch.from_numpy(P));self.register_buffer('acceleration_basis',torch.from_numpy(A))
        self.net=nn.Sequential(nn.Linear(120,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,6))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,f,anchor,CV,arm,return_controls=False):
        assert arm in ARMS
        raw=self.net((f-self.mean)/self.std).reshape(-1,4,3,2).double();self._last_raw=raw.detach()
        u=torch.tanh(raw);u=u/torch.linalg.vector_norm(u,dim=-1,keepdim=True).clamp_min(1.)
        bound=.20*(torch.arange(1,13,device=raw.device,dtype=torch.float64)/12.)**2
        if arm in ARMS:
            nodes=.20*(torch.tensor([4,8,12],device=raw.device,dtype=torch.float64)/12.)**2
            proposal=torch.einsum('ti,bkid->bktd',self.position_basis,u*nodes[None,None,:,None])
        else:
            acceleration=torch.einsum('ti,bkid->bktd',self.acceleration_basis,u)*.05
            proposal=.4*torch.cumsum(.4*torch.cumsum(acceleration,dim=2),dim=2)
        norm=torch.linalg.vector_norm(proposal,dim=-1)
        ratio=torch.where(norm>0,bound[None,None]/torch.where(norm>0,norm,torch.ones_like(norm)),torch.ones_like(norm))
        lam=ratio.amin(-1).clamp_max(1.)
        res=proposal*lam[:,:,None,None];out=anchor.double()+res
        return (out,res,u) if return_controls else (out,res)

def generated(m,data,ix,arm,return_controls=False):
    m.eval();out=[];res=[];controls=[];rawcontrols=[]
    with torch.inference_mode():
        for first in range(0,len(ix),128):
            guard();ind=ix[first:first+128]
            a,b,c=m(torch.from_numpy(features(data,ind)).to(DEVICE),torch.from_numpy(data['anchor'][ind]).to(DEVICE),torch.from_numpy(data['CV'][ind]).to(DEVICE),arm,True)
            out.append(a.cpu().numpy());res.append(b.cpu().numpy());controls.append(c.cpu().numpy());rawcontrols.append(m._last_raw.cpu().numpy())
    out=np.concatenate(out);res=np.concatenate(res);u=np.concatenate(controls);raw=np.concatenate(rawcontrols)
    expected,reconstructed,lam=decode_numpy(data['anchor'][ix],u,arm)
    assert np.max(abs(expected-out))<=2e-12 and np.max(abs(reconstructed-res))<=2e-12
    bound=.20*(np.arange(1,13)/12.)**2
    assert out.shape==(len(ix),4,12,2) and np.isfinite(out).all()
    assert (np.linalg.norm(res,axis=-1)<=bound[None,None]+2e-15).all()
    return (out,res,u,raw) if return_controls else (out,res)

def portfolio(data,ix,out,res,arm,controls=None,rawcontrols=None):
    native=data['native'][ix].astype(np.float64);p=native.copy();slots=data['slots'][ix];rows=np.arange(len(ix))[:,None]
    p[rows,slots]=out;expected=data['anchor'][ix].astype(np.float64)
    assert np.array_equal(out,expected+res)
    for i in range(len(ix)):
        keep=np.setdiff1d(np.arange(20),slots[i]);assert np.array_equal(p[i,keep],native[i,keep])
    z=dict(sample_id=data['ids'][ix],groups=data['groups'][ix],track=data['track'][ix],scene=data['scene'][ix],gt=data['gt'][ix],native20=native,zero_native20=native.copy(),refined20=p,refined4=out,residual=res,slots=slots,reference4=expected)
    if controls is not None:z.update(bounded_controls=controls,mapping_arm=np.array(arm))
    if rawcontrols is not None:z['raw_controls']=rawcontrols
    return z

def evaluate(z,ind=False):
    if 'bounded_controls' in z:
        p,res,lam=decode_numpy(z['reference4'],z['bounded_controls'],str(z['mapping_arm']))
        assert np.max(abs(p-z['refined4']))<=2e-12 and np.max(abs(res-z['residual']))<=2e-12
    met,extra=b.evaluate(z,ind)
    ade,fde,_,_,_=v.errors(z['refined4'],z['gt'],ind);met['assignment']=assignment(ade,fde,z['groups'])[0]
    return met,extra

def locks():
    prior=H.parent/'experiment2_native_support'
    inherited=read(prior/'source_manifest.json')['files'];out=dict(inherited);q.verify(out)
    add=[*q.INPUTS,SPLIT,H.parent/'experiment1_motion_process/run.py',*H.glob('*.py'),*H.glob('*.ps1'),H/'README.md',H/'DESIGN.md',H/'protocol.json']
    add.extend(p for p in (H/'completion_review').glob('*.py'))
    add.extend(p for p in (H/'prelaunch_review').glob('*.py'))
    add.extend(H.glob('PREFLIGHT_FAILURE*.json'))
    add.extend(p for p in (H/'failures').rglob('*') if p.is_file() and p.suffix in ('.py','.ps1','.json','.md'))
    add.extend(Path(k) for k in read(prior/'RESULT.json')['output_sha'])
    add.extend(p for p in prior.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.md') and p.name not in ('runtime_state.json','worker_state.json','exit_status.json','progress.json','halfway_check.json','completion_check.json'))
    for p in sorted(set(add)):
        key=str(p);value=sha(p)
        if key in out:assert out[key]==value,('INHERITED_SHA_CHANGED',key)
        else:out[key]=value
    return out,len(inherited)

def binding(data,sp,st,seed,arm,files):
    d=H/'runs'/arm/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True)
    sid=d/'split_ids.npz';stat=d/'statistics.npz';tr=sp['grouped_train_index']
    if not sid.exists():np.savez_compressed(sid,train_ids=data['ids'][tr],outer_ids=data['ids'][sp['outer_eval_index']],inner_ids=data['ids'][sp['inner_cal_index']],original_CAL_ids=data['ids'][25912:])
    if not stat.exists():np.savez_compressed(stat,mean=st[0],std=st[1],normalization_ids=data['ids'][tr])
    q.seed_all(seed);m=Refiner(st)
    man=dict(read(H/'protocol.json'),arm=arm,seed=seed,source_sha=digest(files),source_files=files,split_sha=sha(SPLIT),split_ids_sha=sha(sid),statistics_sha=sha(stat),initial_parameters_sha=q.tensor_hash(m.state_dict()),parameter_names=list(dict(m.named_parameters())),parameter_count=sum(x.numel() for x in m.parameters()))
    assert man['parameter_count']==24134
    if (d/'run_manifest.json').exists():assert read(d/'run_manifest.json')==man
    else:write(d/'run_manifest.json',man)
    return d,man

def scientific(data,sp,st,seed,arm,files,count):
    d,man=binding(data,sp,st,seed,arm,files)
    if (d/'done.json').exists():done_valid(d,man);return read(d/'metrics.json')
    q.seed_all(seed);m=Refiner(st).to(DEVICE);assert q.tensor_hash(m.state_dict())==man['initial_parameters_sha']
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);ldr,g=loader(data,sp['grouped_train_index'],seed)
    ck=d/'checkpoint.pt';history=resume(ck,m,opt,g,st,man) if ck.exists() else []
    for ep in range(len(history)+1,26):
        stage('TRAIN',arm=arm,seed=seed,epoch=ep,runs_done=count,runs_total=6)
        row=epoch(m,opt,ldr,arm,ep);small=sp['inner_cal_index'][:12];out,_=generated(m,data,small,arm)
        A,F,_,_,_=v.errors(out,data['gt'][small]);row['inner12_posthoc_Q']=float((A+.5*F).min(1).mean());row['inner12_posthoc_joint']=float(np.maximum(A,.5*F).min(1).mean())
        history.append(row);checkpoint(ck,m,opt,g,history,st,man);write(d/'history.json',history)
    assert len(history)==25 and all(x['n']==18909 for x in history)
    metrics={};assignment={};saved=[];frozen={k:x.detach().cpu().clone() for k,x in m.state_dict().items()}
    for split,ix in [('outer',sp['outer_eval_index']),('inner',sp['inner_cal_index']),('original_CAL',np.arange(25912,28791))]:
        stage('FORWARD_SAVE',arm=arm,seed=seed,split=split,runs_done=count)
        out,res,u,raw=generated(m,data,ix,arm,True);z=portfolio(data,ix,out,res,arm,u,raw);p=d/f'{split}.npz'
        assert not p.exists(),'Partial output must be preserved, use explicit recovery not overwrite'
        np.savez_compressed(p,**z);back=q.npz(p);assert all(np.array_equal(x,back[k]) for k,x in z.items())
        stage('FULL_SAVED_QUALITY_CLUSTERS_MOTION',arm=arm,seed=seed,split=split,runs_done=count)
        met,extra=evaluate(back);imet,iextra=evaluate(back,True);err=v.compare(met,imet)
        assert err<=1e-12 and all(np.array_equal(extra[k],iextra[k]) for k in extra)
        metrics[split]=met['metrics'];assignment[split]=met['assignment']
        aux=d/f'{split}_metrics.npz';np.savez_compressed(aux,**extra,sample_id=z['sample_id'],groups=z['groups'],track=z['track'])
        audit=d/f'{split}_audit.json';write(audit,dict(status='PASS',independent_max_error=err,controls_numpy_reconstruction_PASS=True,fixed16_exact=True,full_quality_clusters_five_motion=True))
        saved.extend((p,aux,audit));del z,back,extra,iextra
    assert all(torch.equal(x.detach().cpu(),frozen[k]) for k,x in m.state_dict().items())
    result=dict(arm=arm,seed=seed,epoch=25,splits=metrics,all_weights_bool_scaler_basis_buffers_eval_EXACT=True)
    write(d/'metrics.json',result);write(d/'assignment_stats.json',dict(final_posthoc=assignment,training_preupdate_history=history,four_refined_candidate_quality_only=True))
    saved.extend([ck,d/'run_manifest.json',d/'split_ids.npz',d/'statistics.npz',d/'history.json',d/'metrics.json',d/'assignment_stats.json'])
    write(d/'done.json',dict(status='PASS',run_manifest_sha=digest(man),files={str(p):sha(p) for p in saved},all25epochs=True,old_models_not_instantiated_or_trained=True))
    del m,opt,ldr,frozen;gc.collect();torch.cuda.empty_cache();return result

def bootstrap():
    rng=np.random.default_rng(20261011);records=[]
    for split in ('outer','inner','original_CAL'):
        sets={a:[q.npz(H/'runs'/a/f'seed_{seed}/{split}_metrics.npz') for seed in SEEDS] for a in ARMS};z=sets[ARMS[0]][0]
        for seq in sets.values():
            for x in seq:assert all(np.array_equal(z[k],x[k]) for k in ('sample_id','groups','track'))
        _,inv=np.unique(z['track'],return_inverse=True);T=int(inv.max())+1
        for group,mask in v.masks(z['groups']):
            counts=np.bincount(inv[mask],minlength=T);active=np.flatnonzero(counts)
            for arm,ref in [('sum','native20'),('joint','native20'),('joint','sum')]:
                diff=np.stack([sets[arm][i]['vectors'][:,2]-(sets['sum'][i]['vectors'][:,2] if ref=='sum' else sets[arm][i]['vectors'][:,0]) for i in range(3)])
                sums=np.zeros((3,T,4))
                for i in range(3):np.add.at(sums[i],inv[mask],diff[i,mask])
                draws=[]
                for bi in range(2000):
                    if bi%64==0:guard()
                    ss=rng.integers(0,3,size=3);tt=rng.choice(active,len(active),replace=True)
                    draws.append(sums[ss][:,tt].sum((0,1))/(3*counts[tt].sum()))
                records.append(dict(split=split,group=group,arm=arm,reference=ref,n=int(mask.sum()),tracks=len(active),metrics=['coverage','ADE','FDE','J'],estimate=diff[:,mask].mean((0,1)),CI95=np.quantile(draws,[.025,.975],axis=0).T,exploratory=True))
    return records

def main():
    guard();pre=read(H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30
    files=read(H/'source_manifest.json')['files'];assert digest(files)==pre['source_sha']
    stage('INITIAL_SOURCE_OLD_SHA');q.verify(files);ns,data,sp=data_load();st=statistics(data,sp['grouped_train_index']);stored=q.npz(H/'statistics.npz')
    assert np.array_equal(st[0],stored['mean']) and np.array_equal(st[1],stored['std'])
    records=[]
    for arm in ARMS:
        for seed in SEEDS:records.append(scientific(data,sp,st,seed,arm,files,len(records)))
    stage('FINAL_BOOTSTRAP_SAVED_ARITHMETIC');ci=bootstrap();ci2=bootstrap();assert v.compare(ci,ci2)==0
    write('BOOTSTRAP.json',dict(status='PASS',records=ci,deterministic_second_error=0));outsha={}
    for arm in ARMS:
        for seed in SEEDS:
            d=H/'runs'/arm/f'seed_{seed}';man=read(d/'run_manifest.json');done_valid(d,man)
            ck=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['epoch']==25 and ck['run_manifest']==man and len(ck['history'])==25
            outsha.update(read(d/'done.json')['files']);outsha[str(d/'done.json')]=sha(d/'done.json')
    outsha[str(H/'BOOTSTRAP.json')]=sha(H/'BOOTSTRAP.json');stage('FINAL_SOURCE_OLD_SHA');q.verify(files);q.verify(outsha);guard()
    write('ARITHMETIC_AUDIT.json',dict(status='PASS',saved_jobs=18,runs=6,epochs=150,full_quality_clusters_five_motion_independent=True,controls_numpy_reconstruction=True,all_eval_weights_bool_scaler_basis_buffers_EXACT=True,bootstrap_records=len(ci),not_all_models_gradients_replayed=True))
    write('PRESERVATION.json',dict(status='PASS',locked_files=len(files),inherited_relevant_files=pre['inherited_relevant_files'],not_all_historical_files=True,old_models_not_instantiated_or_trained=True))
    write('RESOURCE_USAGE.json',dict(status='PASS',**q.RESOURCE))
    write('RESULT.json',dict(status='PASS_SAMEPATH_QUALITY_PAIRED_RUN',records=records,source_sha=pre['source_sha'],output_sha=outsha,resource=q.RESOURCE,runs=6,saved_jobs=18,major_success_not_inferred=True))
    stage('SUCCESS',runs_done=6,saved_jobs=18)

if __name__=='__main__':
    try:main()
    except Exception:
        write('FAILURE_'+str(time.time_ns())+'.json',dict(status='STOP',traceback=traceback.format_exc(),resource=q.RESOURCE));raise
