"""Three frozen diagnostic questions, no new objective training."""
from pathlib import Path
import os,sys,ast,types,importlib.util,gc,json,time
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import spearmanr
from resource_guard import guard
from frozen_loader import build
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('parameter_readonly',ROOT/'F16_PARAMETER_TRANSFER_CONCORDANCE_20261005/diagnosis.py')
pt=importlib.util.module_from_spec(spec);spec.loader.exec_module(pt)
pt.HERE=HERE;pt.d.HERE=HERE
d=pt.d
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
GROUPS=('stable','transition','change')
TERMS=('primary','diversity','residual','smoothness','protection','total')
PHYS=('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95')
def read(p):return d.read(p)
def write(p,x):return d.write(p,x)
def sha(p):return d.sha(p)
def stage(name,**kw):return d.stage(name,**kw)
def verify(files,label):return d.verify(files,label)
def extract(path,names,env):
    selected=[n for n in ast.parse(Path(path).read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(selected)==len(names);exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),env);return env
def initialize():
    data=pt.quality_data()
    env=extract(ROOT/'GENERATION_MODE_SUPPORT_SERIES/run_mode_support.py',['labels'],dict(torch=torch,np=np,LABEL_CFG=read(ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/config.json')))
    lab=env['labels']
    data['family']=lab(torch.from_numpy(data['speed'][:,1:]),torch.from_numpy(data['base'][:,:,-1]/12)).numpy()
    data['scale']=np.asarray(read(ROOT/'GENERATION_CANDIDATE_EXCHANGE_SERIES/preflight.json')['fit_protection_scales'],np.float64)
    data['context']={}
    for split,(a,b) in d.SPLITS.items():
        fam=data['family'][a:b]
        data['context'][split]={'n':b-a,'cross':int((fam[:,:,None]!=fam[:,None,:]).sum()),'protected':int((data['group'][a:b]!=2).sum())}
    return data
def pair(x):return torch.linalg.vector_norm(x[:,:,None]-x[:,None,:],dim=-1).mean(-1)
def terms(x,base,gt,weight,family,group,scale,context):
    error=torch.linalg.vector_norm(x-gt[:,None],dim=-1);a=error.mean(-1);f=error[:,:,-1];q=a+.5*f
    residual=torch.diff(x-base,dim=2,prepend=torch.zeros_like(x[:,:,:1]))
    cross=family[:,:,None]!=family[:,None,:]
    ref=(pair(base)*.5).clamp(max=.3).detach()
    diversity=F.relu(ref-pair(x)).mul(cross).sum((1,2))*(context['n']/max(context['cross'],1))*.05
    ne=(base-gt[:,None]).norm(dim=-1)
    harm=F.relu(a.min(-1).values-ne.mean(-1).min(-1).values*1.01)/scale[0]+F.relu(f.min(-1).values-ne[:,:,-1].min(-1).values*1.01)/scale[1]
    protection=harm*(group!=2)*(context['n']/max(context['protected'],1))*.1
    out={'primary':q.min(-1).values*weight,'diversity':diversity,'residual':residual.square().mean((1,2,3))*.01,'smoothness':torch.diff(residual,dim=2).square().mean((1,2,3))*.01,'protection':protection}
    out['total']=sum(out.values());return out,a,f,q
def tensors(data,ix,device,dtype):
    vals=[torch.as_tensor(data[k][ix],device=device,dtype=dtype) for k in ('base','future','quality_weight')]
    vals += [torch.as_tensor(data[k][ix],device=device) for k in ('family','group')]
    vals += [torch.as_tensor(data['scale'],device=device,dtype=dtype)]
    return vals
def motion(x):
    v=torch.diff(x,dim=2,prepend=torch.zeros_like(x[:,:,:1]))/.4
    a=torch.diff(v,dim=2)/.4;j=torch.diff(a,dim=2)/.4
    return {'speed':v.norm(dim=-1),'accel':a.norm(dim=-1),'jerk':j.norm(dim=-1)}
def load_saved(meta,split):
    key=pt.key(meta)
    path=ROOT/'F16_MATCHED_FIT_CAL_TRANSFER_DIAGNOSIS_20261005/jobs'/(key+'_'+split+'.npz')
    done=read(path.with_suffix('.done.json'))
    for p,h in done['outputs'].items():assert sha(p)==h
    with np.load(path,allow_pickle=False) as z:return z['trajectory'].copy()
def load_witness(data):
    folder=ROOT/'F16_CURVE_REACHABILITY_DIAGNOSIS_20261004/recovery_numeric_20261004/chunks'
    slots=np.full(28791,-1,np.int8);paths=np.zeros((28791,12,2),np.float32)
    for split,(a,b) in d.SPLITS.items():
        for st in range(0,b-a,64):
            en=min(st+64,b-a);key=f'{split}_{st:05d}_{en:05d}'
            done=read(folder/(key+'.done.json'));assert done['status']=='SUCCESS'
            p=next(p for p in done['outputs'] if p.endswith('.npz'));assert sha(p)==done['outputs'][p]
            with np.load(p,allow_pickle=False) as z:
                assert np.array_equal(z['sample_id'],data['ids'][a+st:a+en])
                cand=z['pulse_paths'];valid=z['pulse_feasible_witness']
                err=np.linalg.norm(cand.astype(np.float64)-data['future'][a+st:a+en,None].astype(np.float64),axis=-1)
                good=(err.mean(-1)<=.2)&(err[:,:,-1]<=.4)
                assert np.all(good|~valid)
                score=np.where(valid,err.mean(-1)+.5*err[:,:,-1],np.inf)
                win=score.argmin(-1);ok=valid.any(-1);row=np.arange(en-st)
                slots[a+st:a+en]=np.where(ok,win,-1)
                paths[a+st:a+en]=np.where(ok[:,None,None],cand[row,win],0)
            guard.check()
    return slots,paths
def snapshot(data,paths,split,witness):
    a,b=d.SPLITS[split];n=b-a;ctx=data['context'][split]
    result={'sample_id':data['ids'][a:b],'group':data['group'][a:b]}
    # Original registered geometric clustering and complete-slot metrics.
    qm,_=d.metrics(paths,data['future'][a:b],data['logits'][a:b]);result.update(qm)
    for begin in range(0,n,128):
        guard.check();end=min(begin+128,n);ix=np.arange(a+begin,a+end);sl=slice(begin,end)
        x=torch.tensor(paths[sl],dtype=torch.float64,requires_grad=True)
        vals=tensors(data,ix,'cpu',torch.float64);loss,aa,ff,qq=terms(x,*vals,ctx)
        gq=torch.autograd.grad(loss['primary'].sum(),x,retain_graph=True)[0]
        greg=torch.autograd.grad((loss['diversity']+loss['residual']+loss['smoothness']).sum(),x,retain_graph=True)[0]
        gp=torch.autograd.grad(loss['protection'].sum(),x)[0];gtotal=gq+greg+gp
        candidate_norm=lambda v:v.norm(dim=-1).sum(-1)
        good=(aa<=.2)&(ff<=.4);winner=qq.argmin(-1);joint=(torch.maximum(aa/.2,ff/.4)).min(-1)
        row=torch.arange(end-begin);gqn=candidate_norm(gq)
        masked=gqn.clone();masked[row,winner]=0
        assert masked.max()==0
        ws,wp=witness;slot=ws[ix].astype(np.int64);ok=slot>=0;safe=np.maximum(slot,0);rr=np.arange(end-begin)
        dx=np.zeros((end-begin,20,12,2),np.float64)
        dx[rr[ok],safe[ok]]=wp[ix[ok]].astype(np.float64)-paths[sl][rr[ok],safe[ok]].astype(np.float64)
        perturbed=x.detach()+torch.from_numpy(dx)
        with torch.no_grad():new,_,_,_=terms(perturbed,*vals,ctx)
        mot=motion(x.detach())
        values={**{'L_'+k:v.detach().numpy() for k,v in loss.items()},'ADE20':aa.detach().numpy(),'FDE20':ff.detach().numpy(),'J20':torch.maximum(aa/.2,ff/.4).detach().numpy(),'J_joint':joint.values.detach().numpy(),'J_slot':joint.indices.numpy(),'Q_joint':qq.min(-1).values.detach().numpy(),'quality_winner':winner.numpy(),'winner_good':good[row,winner].numpy(),'good_present_winner_bad':(good.any(-1)&~good[row,winner]).numpy(),
            'winner_gap':torch.sort(qq.detach(),dim=-1).values[:,1].sub(qq.detach().min(-1).values).numpy(),'primary_credit20':gqn.numpy(),'regularizer_credit20':candidate_norm(greg).numpy(),'protection_credit20':candidate_norm(gp).numpy(),'total_credit20':candidate_norm(gtotal).numpy(),
            'witness_slot':slot.astype(np.int8),'witness_delta_L':np.where(ok,(new['total']-loss['total']).detach().numpy(),np.nan),'witness_delta_primary':np.where(ok,(new['primary']-loss['primary']).detach().numpy(),np.nan),
            'witness_primary_direction':np.where(ok,(gq.detach().numpy()*dx).sum((1,2,3)),np.nan),'witness_total_direction':np.where(ok,(gtotal.detach().numpy()*dx).sum((1,2,3)),np.nan),
            'witness_direct_primary_credit':np.where(ok,gqn.numpy()[rr,safe],np.nan)}
        for k,mv in mot.items():
            values[k+'_window_mean']=mv.numpy().mean((1,2))
            values[k+'_window_p95']=np.quantile(mv.numpy().reshape(len(mv),-1),.95,axis=1)
        for k,v in values.items():
            if k not in result:result[k]=np.empty((n,)+v.shape[1:],v.dtype)
            result[k][sl]=v
        del x,loss,gq,greg,gp,gtotal
    # d.metrics global motion is deliberately not substituted by window p95.
    return result
def relations(before,after):
    assert np.array_equal(before['sample_id'],after['sample_id'])
    out={'sample_id':before['sample_id'],'group':before['group']}
    names=['L_'+k for k in TERMS]+['minADE','minFDE','J_joint','Q_joint','ADE20','FDE20','J20','raw_good','effective_good','all_distinct']+[k+'_window_'+v for k in ('speed','accel','jerk') for v in ('mean','p95')]
    for k in names:out['delta_'+k]=after[k].astype(np.float64)-before[k].astype(np.float64)
    out['before_coverage']=before['coverage'];out['after_coverage']=after['coverage']
    improve=out['delta_L_total']<-1e-8;bg=before['coverage'];ag=after['coverage']
    cls=np.zeros(len(bg),np.int8)
    cls[improve&~bg&ag]=1;cls[improve&bg&~ag]=2;cls[improve&bg&ag]=3
    cls[improve&~bg&~ag&(out['delta_J_joint']<-1e-8)]=4
    cls[improve&~bg&~ag&(out['delta_J_joint']>=-1e-8)]=5
    out['loss_coverage_class']=cls;out['winner_changed']=before['quality_winner']!=after['quality_winner']
    out['J_slot_changed']=before['J_slot']!=after['J_slot']
    return out
def relationship_summary(v):
    rows=[]
    for name,mask in d.masks(v['group']):
        dl=v['delta_L_total'][mask];dj=v['delta_J_joint'][mask];de=v['delta_minADE'][mask];df=v['delta_minFDE'][mask]
        down=dl<-1e-8
        corr=lambda x:float(np.corrcoef(dl,x)[0,1]) if np.std(dl)>0 and np.std(x)>0 else None
        sp=float(spearmanr(dl,dj).statistic) if np.std(dl)>0 and np.std(dj)>0 else None
        edges=np.quantile(-dl[down],[0,.25,.5,.75,1]).tolist() if down.any() else None
        bins=[]
        if edges is not None:
            labels=np.searchsorted(np.asarray(edges[1:-1]),-dl,side='right')
            for z in range(4):
                bm=down&(labels==z)
                bins.append(dict(bin=z,n=int(bm.sum()),mean_delta_J=float(dj[bm].mean()) if bm.any() else None,rescue=int((bm&(v['loss_coverage_class'][mask]==1)).sum()),lost=int((bm&(v['loss_coverage_class'][mask]==2)).sum())))
        rows.append(dict(group=name,n=int(mask.sum()),class_counts=np.bincount(v['loss_coverage_class'][mask],minlength=6).tolist(),loss_down=int(down.sum()),loss_down_J_worse=int((down&(dj>1e-8)).sum()),loss_down_FDE_worse=int((down&(df>1e-8)).sum()),loss_down_effectivegood_falls=int((down&(v['delta_effective_good'][mask]<0)).sum()),loss_down_distinct_falls=int((down&(v['delta_all_distinct'][mask]<0)).sum()),loss_down_accelmean_rises=int((down&(v['delta_accel_window_mean'][mask]>1e-8)).sum()),loss_down_jerkmean_rises=int((down&(v['delta_jerk_window_mean'][mask]>1e-8)).sum()),pearson_L_J=corr(dj),pearson_L_ADE=corr(de),pearson_L_FDE=corr(df),spearman_L_J=sp,positive_loss_improvement_quartile_edges=edges))
        rows[-1]['loss_improvement_bins']=bins
    return rows
def ranks(paths):
    mt=motion(torch.from_numpy(paths));out={}
    for name in PHYS:
        if not name.endswith('p95'):continue
        arr=mt[name.split('_')[0]].numpy();flat=arr.ravel();rank=(len(flat)-1)*.95;lo=int(np.floor(rank));hi=int(np.ceil(rank))
        ix=np.argpartition(flat,[lo,hi])[[lo,hi]];w=np.array([hi-rank,rank-lo]) if hi!=lo else np.array([1.,0.])
        coords=np.array(np.unravel_index(ix,arr.shape)).T
        value=float((flat[ix]*w).sum())
        assert abs(value-float(np.quantile(flat,.95)))<1e-6
        out[name]={'coordinates':coords.tolist(),'weights':w.tolist(),'value':value,'rank_values':flat[ix].tolist(),'exact_ties':[int(np.count_nonzero(flat==flat[i])) for i in ix]}
    return out
def param_accumulate(model,params,data,split,paths,chunk=128,subset=None,report=False):
    a,b=d.SPLITS[split];ix=np.flatnonzero(data['group'][a:b]==2)+a
    if subset is not None:ix=ix[:subset]
    selected=paths[ix-a];quant=ranks(selected);n=len(ix);ctx=data['context'][split]
    grad={k:np.zeros(45798,np.float64) for k in TERMS+PHYS};means={k:0. for k in TERMS+PHYS};alignment=0.
    device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    for st in range(0,n,chunk):
        guard.check();end=min(n,st+chunk);ids=ix[st:end]
        obs=torch.as_tensor(data['obs'][ids],device=device,dtype=dtype);base=torch.as_tensor(data['base'][ids],device=device,dtype=dtype);speed=torch.as_tensor(data['speed'][ids],device=device,dtype=dtype)
        x,_=model(obs,base,speed);alignment=max(alignment,float((x.detach().cpu()-torch.tensor(paths[ids-a],dtype=dtype)).abs().max()))
        vals=tensors(data,ids,device,dtype);loss,_,_,_=terms(x,*vals,ctx);mt=motion(x)
        scalars={k:loss[k].sum()/n for k in TERMS}
        scalars.update(accel_mean=mt['accel'].sum()/(n*20*11),jerk_mean=mt['jerk'].sum()/(n*20*10))
        for name,r in quant.items():
            arr=mt[name.split('_')[0]];val=arr.sum()*0
            for coord,w in zip(r['coordinates'],r['weights']):
                row,k,t=coord
                if st<=row<end:val=val+arr[row-st,k,t]*w
            scalars[name]=val
        for k,s in scalars.items():
            grad[k]+=pt.vector(s,params,retain=True);means[k]+=float(s.detach().cpu())
        if report and st%2048==0:stage('CHANGE_PARAMETER_COMPONENTS',split=split,row=st,rows=n)
        del x,loss,scalars,mt
    assert alignment<=2e-5,('REPLAY_ALIGNMENT',alignment)
    return grad,means,quant,alignment
def gradient_summary(v):
    pairs={}
    for q in ('primary','total'):
        for k in ('diversity','residual','smoothness','protection')+PHYS:
            old=pt.stats(v[q],v[k])
            pairs[q+'__'+k]=dict(quality_norm=old['FITnorm'],other_norm=old['CALnorm'],dot=old['dot'],cos=old['cos'],other_derivative_along_quality_descent=old['CAL_directional_derivative'],status=old['status'].replace('LOCAL_PRIMARY','LOCAL_QUALITY'))
    return pairs
