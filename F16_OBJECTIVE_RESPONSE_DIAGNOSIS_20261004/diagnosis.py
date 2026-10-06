"""Saved-path original objective/gradient audit; no model forward or optimizer."""
from pathlib import Path
from datetime import datetime
import argparse, ast, ctypes, hashlib, json, os, struct, sys, time, types, gc
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
import psutil
import torch
from torch.nn import functional as F
torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
PREV=ROOT/'F16_CURVE_REACHABILITY_DIAGNOSIS_20261004/recovery_numeric_20261004'
GROUPS=ROOT/'F16_CUE_RESPONSE_CONCORDANCE_20261004/recovery_memory_20261004/jobs'
TASK='Trajectory-ObjectiveResponse-Diagnosis-20261004'
METHODS=('native','learned1','learned2','learned3','pulse_full','single_witness')
GIB=2**30

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);assert HERE in p.parents
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def npz(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in (keys or z.files)}
def stage(name,**kw):write(HERE/'progress.json',dict(stage=name,at=datetime.now().astimezone().isoformat(),**kw))
def verify(files,label):
    t=time.perf_counter()
    for i,(p,h) in enumerate(files.items()):
        assert sha(p)==h,('SHA mismatch',p)
        if i%4000==0:stage(label,files_done=i,files_total=len(files))
    return time.perf_counter()-t
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
class Guard:
    def __init__(self):self.rss=0;self.private=0;self.physical=2**64;self.commit=2**64
    def check(self):
        v=psutil.Process().memory_info();s=Mem();s.length=ctypes.sizeof(s)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        self.rss=max(self.rss,v.peak_wset);self.private=max(self.private,v.peak_pagefile)
        self.physical=min(self.physical,s.avail_phys);self.commit=min(self.commit,s.avail_page)
        assert max(v.rss,v.private)<6*GIB,'6GiB process budget'
        assert min(s.avail_phys,s.avail_page)>=6*GIB,'6GiB reserve'
        return dict(peak_RSS_GiB=self.rss/GIB,peak_private_GiB=self.private/GIB,min_physical_GiB=self.physical/GIB,min_commit_GiB=self.commit/GIB,process_budget_GiB=6,system_reserve_GiB=6)
guard=Guard()
def extract(path,names,env):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env
def source_paths():
    return [ROOT/'GENERATION_PROTECTED_BALANCE_SERIES/run_protected.py',ROOT/'GENERATION_DIRECTION_COVERAGE_SERIES/run_direction_coverage.py',ROOT/'GENERATION_MODE_SUPPORT_SERIES/run_mode_support.py',ROOT/'GENERATION_LOCAL_SUPPORT_SERIES/run_generation_support.py',ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/directional_regime_pilot.py',ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/config.json']
def originals(cfg,weights):
    labelcfg=read(source_paths()[-1]);env=dict(torch=torch,np=np,F=F,LABEL_CFG=labelcfg)
    extract(source_paths()[2],['labels'],env);labels=env['labels']
    pairs=extract(source_paths()[3],['pair_distance'],dict(torch=torch))['pair_distance']
    direction=dict(torch=torch,F=F,WEIGHTS=torch.tensor(weights,dtype=torch.float64),m=types.SimpleNamespace(labels=labels),g=types.SimpleNamespace(pair_distance=pairs))
    extract(source_paths()[1],['objective'],direction)
    protected=dict(torch=torch,F=F,r=types.SimpleNamespace(objective=direction['objective']))
    extract(source_paths()[0],['objective','protection'],protected)
    classifier=extract(source_paths()[4],['classify_velocity'],dict(np=np,CONTINUE=0,DECELERATE=1,STOP=2,ACCELERATE=3,LEFT=4,RIGHT=5,REVERSE=6))['classify_velocity']
    return protected['objective'],labels,pairs,classifier,labelcfg
def snapshot():
    p=HERE/'old_results_manifest.json'
    if p.exists():return read(p)['files']
    files=dict(read(PREV/'old_results_manifest.json')['files'])
    for p in PREV.parent.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.tmp','.pending') and p.name!='runtime_state.json':files[str(p)]=sha(p)
    write(HERE/'old_results_manifest.json',dict(files=files,excluded_controls=['.memory','AUTONOMOUS_GENERATION_CYCLE.json','runtime_state.json'],prior_results_and_failures_preserved=True))
    return files
def sources():
    files=dict(read(PREV/'preflight.json')['source_hashes'])
    for p in source_paths()+list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/'README.md',HERE/'DESIGN.md',HERE/'protocol.json',HERE/'old_results_manifest.json']:
        files[str(p)]=sha(p)
    # Preserve the previous resource-failed source/trace separately, unchanged.
    for p in list((HERE/'failures').rglob('*'))+[HERE/'MEMORY_FAILURE_REVIEW.json']:
        if p.is_file():files[str(p)]=sha(p)
    return files
def jobs():
    return [(pop,a,min(a+64,n)) for pop,n in [('FIT',25912),('CAL',2879)] for a in range(0,n,64)]
def name(job):return f'{job[0]}_{job[1]:05d}_{job[2]:05d}'
def getblock(job):
    done=read(PREV/'chunks'/(name(job)+'.done.json'));assert done['status']=='SUCCESS'
    paths=[p for p in done['outputs'] if p.endswith('.npz')];assert len(paths)==1
    for p,h in done['outputs'].items():assert sha(p)==h
    z=npz(paths[0],['sample_id','pulse_paths','pulse_feasible_witness','learned_paths','limit'])
    return z
def load_data():
    t=time.perf_counter();guard.check()
    cache=npz(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz',['sample_id','candidate','motion','change_group','ade','fde'])
    labels=npz(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz',['sample_id','future'])
    assert np.array_equal(cache['sample_id'],labels['sample_id']) and len(np.unique(cache['sample_id']))==28791
    past=cache['motion'][:,:16].reshape(-1,8,2);vel=(past[:,-1]-past[:,-2]).astype(np.float32)
    speed=np.concatenate((np.linalg.norm(vel,axis=-1)[:,None],vel),1).astype(np.float32)
    ck=ROOT/'GENERATION_PULSE_RESPONSE_SERIES_20261004/results/models/mean_std/seed_1/balanced_protected/constrained.pth'
    saved=torch.load(ck,map_location='cpu',weights_only=False);cfg=saved['config'];del saved
    cfgkeys=('fde_loss_weight','diversity_weight','diversity_reference_fraction','diversity_distance_cap_m','residual_weight','smoothness_weight','protection_weight','guard_fraction','velocity_residual_scale')
    assert [cfg[k] for k in cfgkeys]==[.5,.05,.5,.3,.01,.01,.1,.01,1.5]
    scale=read(ROOT/'GENERATION_CANDIDATE_EXCHANGE_SERIES/preflight.json')['fit_protection_scales']
    # Original weights derive from all FIT. Classify in the original float32 convention.
    original,lab,pair,clas,labelcfg=originals(cfg,np.ones(7))
    target=clas(vel,labels['future'][:,-1]/12,labelcfg)
    counts=np.bincount(target[:25912],minlength=7);assert (counts>0).all()
    w=np.sqrt(25912/(7*counts)).clip(.5,3);w/=w[target[:25912]].mean()
    expected=read(ROOT/'GENERATION_PROTECTED_BALANCE_SERIES/preflight.json')['weights']
    assert np.max(np.abs(w-expected))<1e-12
    # Production stores class weights as float32; retain those exact constants.
    w=w.astype(np.float32).astype(np.float64);original,lab,pair,clas,labelcfg=originals(cfg,w)
    b=cache['candidate'][...,:24].reshape(-1,20,12,2);f=labels['future']
    e=np.linalg.norm(b-f[:,None],axis=-1)
    assert np.allclose(e.mean(-1),cache['ade'],atol=2e-5,rtol=0) and np.allclose(e[:,:,-1],cache['fde'],atol=2e-5,rtol=0)
    family=clas(np.repeat(vel,20,axis=0),b[:,:,-1].reshape(-1,2)/12,labelcfg).reshape(-1,20)
    # Verify original tensor labels on all data, retaining float32 original label boundaries.
    tl=lab(torch.from_numpy(vel),torch.from_numpy(b[:,:,-1]/12)).numpy()
    assert np.array_equal(tl,family)
    assert np.array_equal(lab(torch.from_numpy(vel),torch.from_numpy(f[:,None,-1]/12)).numpy()[:,0],target)
    groups={p:npz(GROUPS/(p+'_native.npz'),['sample_id','scene_family','GT_event','GT_uncertain','GT_kind','cue_first_compatible','class_0']) for p in ('FIT','CAL')}
    for p,a,c in [('FIT',0,25912),('CAL',25912,28791)]:assert np.array_equal(groups[p]['sample_id'],cache['sample_id'][a:c])
    contexts={}
    for pop,a,c in [('FIT',0,25912),('CAL',25912,28791)]:
        cross=family[a:c,:,None]!=family[a:c,None,:]
        contexts[pop]=dict(n=c-a,cross=int(cross.sum()),protected=int((cache['change_group'][a:c]!=2).sum()))
    guard.check()
    return dict(ids=cache['sample_id'],base=b,future=f,speed=speed,regime=cache['change_group'],family=family,target=target,weights=w,scale=np.array(scale),cfg=cfg,groups=groups,contexts=contexts,original=original,pair=pair,load_seconds=time.perf_counter()-t)
def arrays_for(d,job,z):
    a=job[1]+(25912 if job[0]=='CAL' else 0);c=job[2]+(25912 if job[0]=='CAL' else 0)
    b=d['base'][a:c].astype(np.float64);y=d['future'][a:c].astype(np.float64)
    assert np.array_equal(z['sample_id'],d['ids'][a:c])
    p=z['pulse_paths'].astype(np.float64);dist=np.linalg.norm(p-y[:,None],axis=-1)
    q=dist.mean(-1)+.5*dist[:,:,-1];valid=z['pulse_feasible_witness'].astype(bool)
    assert np.all(((dist.mean(-1)<=.2)&(dist[:,:,-1]<=.4))|~valid)
    slot=np.where(valid.any(-1),np.where(valid,q,np.inf).argmin(-1),-1)
    single=b.copy();rows=np.flatnonzero(slot>=0);single[rows,slot[rows]]=p[rows,slot[rows]]
    untouched=np.ones((len(b),20),bool);untouched[rows,slot[rows]]=False
    assert np.array_equal(single[untouched],b[untouched])
    return a,c,b,y,slot,[b]+[v.astype(np.float64) for v in z['learned_paths']]+[p,single]
def components(x,b,y,speed,reg,family,target,d,ctx):
    cfg=d['cfg'];n=ctx['n'];res=torch.diff(x-b,dim=2,prepend=torch.zeros_like(x[:,:,:1]))
    err=(x-y[:,None]).norm(dim=-1);ade=err.mean(-1);fde=err[:,:,-1];q=ade+.5*fde
    cross=torch.from_numpy(family[:,:,None]!=family[:,None,:]);ref=(d['pair'](b)*.5).clamp(max=.3).detach()
    div=(F.relu(ref-d['pair'](x))*cross).sum((1,2))*(n/max(ctx['cross'],1))*.05
    quality=q.min(-1).values*torch.tensor(d['weights'][target])
    residual=res.square().mean((1,2,3))*.01;smooth=torch.diff(res,dim=2).square().mean((1,2,3))*.01
    ne=(b-y[:,None]).norm(dim=-1);na=ne.mean(-1).min(-1).values;nf=ne[:,:,-1].min(-1).values
    harm=F.relu(ade.min(-1).values-na*1.01)/d['scale'][0]+F.relu(fde.min(-1).values-nf*1.01)/d['scale'][1]
    protection=harm*torch.tensor(reg!=2)*(n/max(ctx['protected'],1))*.1
    return dict(quality=quality,diversity=div,residual=residual,smoothness=smooth,protection=protection),res,ade,fde,q
def cosine(a,b):
    a=a.flatten(1);b=b.flatten(1);den=a.norm(dim=-1)*b.norm(dim=-1)
    return torch.where(den>1e-30,(a*b).sum(-1)/den.clamp_min(1e-30),torch.zeros_like(den))
def calculate(d,job,z):
    a,c,b,y,slot,paths=arrays_for(d,job,z);ctx=d['contexts'][job[0]];out={'sample_id':z['sample_id'],'witness_slot':slot.astype(np.int8)}
    native=torch.tensor(b);gt=torch.tensor(y);delta=torch.tensor(paths[-1]-b)
    for method,p in zip(METHODS,paths):
        guard.check();x=torch.tensor(p,requires_grad=True)
        terms,res,ade,fde,q=components(x,native,gt,d['speed'][a:c],d['regime'][a:c],d['family'][a:c],d['target'][a:c],d,ctx)
        gq=torch.autograd.grad(terms['quality'].sum(),x,retain_graph=True)[0]
        greg=torch.autograd.grad((terms['diversity']+terms['residual']+terms['smoothness']).sum(),x,retain_graph=True)[0]
        gp=torch.autograd.grad(terms['protection'].sum(),x)[0];total=gq+greg+gp
        good=(ade<=.2)&(fde<=.4);winner=q.argmin(-1);rows=torch.arange(len(x));mass=total.norm(dim=-1).sum(-1)
        values={**terms,'total':sum(terms.values()),'minade':ade.min(-1).values,'minfde':fde.min(-1).values,
          'raw_good':good.sum(-1),'coverage':good.any(-1),'loose_coverage':((ade<=.3)&(fde<=.5)).any(-1),'winner_slot':winner,
          'winner_good':good[rows,winner],'good_present_winner_bad':good.any(-1)&~good[rows,winner],
          'grad_quality_norm':gq.flatten(1).norm(dim=-1),'grad_regularizer_norm':greg.flatten(1).norm(dim=-1),'grad_protection_norm':gp.flatten(1).norm(dim=-1),
          'cos_quality_regularizer':cosine(gq,greg),'cos_quality_protection':cosine(gq,gp),'cos_quality_total':cosine(gq,total),
          'total_good_gradient_mass_fraction':(mass*good).sum(-1)/mass.sum(-1).clamp_min(1e-30),
          'derivative_toward_single_quality':(gq*delta).sum((1,2,3)),'derivative_toward_single_regularizers':(greg*delta).sum((1,2,3)),
          'derivative_toward_single_protection':(gp*delta).sum((1,2,3)),'derivative_toward_single_total':(total*delta).sum((1,2,3))}
        velocity=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4
        accel=np.diff(velocity,axis=2)/.4;jerk=np.diff(accel,axis=2)/.4
        for label,v in [('speed',velocity),('accel',accel),('jerk',jerk)]:
            norm=np.linalg.norm(v,axis=-1);values[label+'_mean']=torch.tensor(norm.mean((1,2)));values[label+'_window_p95']=torch.tensor(np.quantile(norm.reshape(len(p),-1),.95,axis=-1))
        for k,v in values.items():
            array=v.detach().numpy();assert np.isfinite(array).all(),(method,k)
            out[method+'__'+k]=array
        assert (res.detach().numpy().cumsum(2)+b-p).__abs__().max()<2e-5
    return out
def numpy_terms(p,b,y,reg,family,target,d,ctx):
    err=np.sqrt(np.square(p-y[:,None]).sum(-1));ade=err.mean(-1);fde=err[:,:,-1];q=ade+.5*fde
    cross=family[:,:,None]!=family[:,None,:]
    ref=np.minimum(np.linalg.norm(b[:,:,None]-b[:,None,:],axis=-1).mean(-1)*.5,.3)
    pd=np.linalg.norm(p[:,:,None]-p[:,None,:],axis=-1).mean(-1)
    res=np.diff(p-b,axis=2,prepend=np.zeros_like(p[:,:,:1]))
    ne=np.linalg.norm(b-y[:,None],axis=-1)
    return dict(quality=q.min(-1)*d['weights'][target],diversity=np.maximum(ref-pd,0).__mul__(cross).sum((1,2))*ctx['n']/max(ctx['cross'],1)*.05,
        residual=np.square(res).mean((1,2,3))*.01,smoothness=np.square(np.diff(res,axis=2)).mean((1,2,3))*.01,
        protection=(np.maximum(ade.min(-1)-ne.mean(-1).min(-1)*1.01,0)/d['scale'][0]+np.maximum(fde.min(-1)-ne[:,:,-1].min(-1)*1.01,0)/d['scale'][1])*(reg!=2)*ctx['n']/max(ctx['protected'],1)*.1)
def audit(d,job,z,out):
    a,c,b,y,slot,paths=arrays_for(d,job,z);maximum=0.
    for method,p in zip(METHODS,paths):
        values=numpy_terms(p,b,y,d['regime'][a:c],d['family'][a:c],d['target'][a:c],d,d['contexts'][job[0]])
        values['total']=sum(values.values())
        err=np.linalg.norm(p-y[:,None],axis=-1);good=(err.mean(-1)<=.2)&(err[:,:,-1]<=.4)
        winner=(err.mean(-1)+.5*err[:,:,-1]).argmin(-1);winner_good=good[np.arange(len(p)),winner]
        values.update(minade=err.mean(-1).min(-1),minfde=err[:,:,-1].min(-1),raw_good=good.sum(-1),coverage=good.any(-1),
            loose_coverage=((err.mean(-1)<=.3)&(err[:,:,-1]<=.5)).any(-1),winner_slot=winner,winner_good=winner_good,
            good_present_winner_bad=good.any(-1)&~winner_good)
        # Separate scalar/component arithmetic and saved motion checks from gradient checks.
        velocity=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4
        accel=np.diff(velocity,axis=2)/.4;jerk=np.diff(accel,axis=2)/.4
        for label,v in [('speed',velocity),('accel',accel),('jerk',jerk)]:
            norm=np.sqrt(np.square(v).sum(-1))
            values[label+'_mean']=norm.mean((1,2))
            values[label+'_window_p95']=np.quantile(norm.reshape(len(p),-1),.95,axis=-1)
        for k,v in values.items():
            error=float(np.max(np.abs(np.asarray(v,dtype=np.float64)-np.asarray(out[method+'__'+k],dtype=np.float64))));maximum=max(maximum,error)
            assert error<1e-9,(job,method,k,error)
    return dict(status='PASS',independent_numpy_components_and_quality=True,max_error=maximum,gradient_checks='preflight original AST analytic autograd and finite differences; saved vector summaries not all model replays')
def resume_accept(done,locks):
    assert done['source_hashes']==locks,'Source identity changed'
    verify(done['outputs'],'RESUME_OUTPUT_SHA')
    return True
def masks(g,reg):
    clear=g['GT_event'].astype(bool)&~g['GT_uncertain'].astype(bool)
    r={'all':np.ones(len(reg),bool),'clear':clear,'specific_first_compatible':clear&g['cue_first_compatible'].astype(bool),'specific_coarse_path_failure':clear&g['cue_first_compatible'].astype(bool)&(g['class_0']==7)}
    r.update({p:reg==i for i,p in enumerate(('stable','transition','change'))})
    r.update({'scene_'+str(s):g['scene_family']==s for s in np.unique(g['scene_family'])})
    return r
def summarize(d,outputs):
    result={}
    for pop,a,c in [('FIT',0,25912),('CAL',25912,28791)]:
        result[pop]={}
        for group,m in masks(d['groups'][pop],d['regime'][a:c]).items():
            record={'n':int(m.sum()),'witness_available':int((outputs[pop]['witness_slot'][m]>=0).sum()),'methods':{}}
            if not m.any():result[pop][group]=record;continue
            native=outputs[pop]['native__total']
            for method in METHODS:
                v={k.split('__')[1]:a for k,a in outputs[pop].items() if k.startswith(method+'__')}
                stats={k:float(a[m].mean()) for k,a in v.items() if k!='winner_slot'}
                stats.update(coverage_count=int(v['coverage'][m].sum()),objective_lower_than_native=int((v['total'][m]<native[m]-1e-12).sum()),
                    quality_lower_total_not_lower=int(((v['quality'][m]<outputs[pop]['native__quality'][m]-1e-12)&(v['total'][m]>=native[m]-1e-12)).sum()))
                record['methods'][method]=stats
            result[pop][group]=record
    return result
def preflight():
    assert not (HERE/'preflight.json').exists(),'Do not overwrite evidence'
    t=time.perf_counter();old=snapshot();hash_seconds=verify(old,'PREFLIGHT_OLD_SHA');d=load_data();locks=sources();verify(locks,'PREFLIGHT_SOURCE_SHA')
    timings=[];original_max=0.;gradmax=0.;finite_error=0.
    for job in [('FIT',0,64),('FIT',12032,12096),('CAL',0,64),('CAL',2816,2879)]:
        z=getblock(job);start=time.perf_counter();o=calculate(d,job,z);audit(d,job,z,o);timings.append(time.perf_counter()-start)
        a,c,b,y,slot,paths=arrays_for(d,job,z);reg=d['regime'][a:c];fam=d['family'][a:c]
        ctx=dict(n=len(b),cross=int((fam[:,:,None]!=fam[:,None,:]).sum()),protected=int((reg!=2).sum()))
        for p in (paths[1],paths[-1]):
            x=torch.tensor(p,requires_grad=True);bt=torch.tensor(b);yt=torch.tensor(y);st=torch.tensor(d['speed'][a:c]);gt=torch.tensor(reg);sc=torch.tensor(d['scale'])
            terms,res,*_=components(x,bt,yt,d['speed'][a:c],reg,fam,d['target'][a:c],d,ctx)
            orig=d['original'](x,bt,yt,res,st,gt,d['cfg'],'balanced_protected',sc)
            decomposed=sum(terms.values()).mean();original_max=max(original_max,float(abs(orig-decomposed).detach()))
            g1=torch.autograd.grad(orig,x,retain_graph=True)[0];g2=torch.autograd.grad(decomposed,x)[0];gradmax=max(gradmax,float((g1-g2).abs().max()))
            assert original_max<1e-9 and gradmax<1e-9
        # Full-size input bookkeeping and reserve checked; no full pairwise allocation.
    fixture=HERE/'resume_fixture.npz';np.savez_compressed(fixture,x=np.array([1,2]))
    done=dict(source_hashes={'fixture':'one'},outputs={str(fixture):sha(fixture)})
    assert resume_accept(done,{'fixture':'one'})
    rejected=False
    try:resume_accept(done,{'fixture':'two'})
    except AssertionError:rejected=True
    assert rejected
    np.savez_compressed(fixture,x=np.array([1,3]));corrupt_rejected=False
    try:resume_accept(done,{'fixture':'one'})
    except AssertionError:corrupt_rejected=True
    assert corrupt_rejected
    np.savez_compressed(fixture,x=np.array([1,2]))
    assert resume_accept(done,{'fixture':'one'})
    # Directional derivative finite difference away from ties, on a selected scalar.
    job=('FIT',12032,12096);z=getblock(job);a,c,b,y,slot,paths=arrays_for(d,job,z)
    x=torch.tensor(paths[1],requires_grad=True);bt=torch.tensor(b);yt=torch.tensor(y);ctx=d['contexts']['FIT']
    args=(bt,yt,d['speed'][a:c],d['regime'][a:c],d['family'][a:c],d['target'][a:c],d,ctx)
    terms,*_=components(x,*args);loss=sum(terms.values()).sum();g=torch.autograd.grad(loss,x)[0]
    rng=np.random.default_rng(20261004);direction=torch.tensor(rng.normal(size=x.shape));direction/=direction.norm()
    h=1e-6
    with torch.no_grad():
        hi=sum(components(x+h*direction,*args)[0].values()).sum();lo=sum(components(x-h*direction,*args)[0].values()).sum()
    finite_error=abs(float((hi-lo)/(2*h)-(g*direction).sum()));assert finite_error<1e-5
    # Allocate and touch the worst-case planned compact result arrays for ALL rows.
    allocation=np.zeros((28791,6,40),np.float64);allocation+=1;guard.check()
    result_allocation_bytes=allocation.nbytes;del allocation;gc.collect()
    resource=guard.check();verify(old,'PREFLIGHT_FINAL_OLD_SHA');verify(locks,'PREFLIGHT_FINAL_SOURCE_SHA')
    eta_compute=float(np.median(timings))*450/60
    write(HERE/'MEMORY_PREFLIGHT.json',dict(status='PASS',**resource,input_rows=28791,chunk_rows=64,six_methods_per_block=6,full_result_allocation_test_bytes=result_allocation_bytes,largest_pairwise_float64_array_MiB=64*20*20*12*2*8/2**20,full_pairwise_not_allocated=True))
    write(HERE/'resume_check.json',dict(status='PASS',source_reject=True,corrupt_output_reject=corrupt_rejected,completed_analysis_SHA_skip=True,training_optimizer_resume=False))
    write(HERE/'preflight.json',dict(status='PASS',source_hashes=locks,old_files=len(old),source_files=len(locks),fit=25912,cal=2879,blocks=450,model_forward=False,model_training=False,path_tensor_autograd=True,
      original_loss_error=original_max,original_gradient_error=gradmax,finite_difference_error=finite_error,seconds_per_six_method_block=timings,estimated_compute_minutes=eta_compute,measured_hash_seconds=hash_seconds,
      config=d['cfg'],weights=d['weights'].tolist(),fit_protection_scales=d['scale'].tolist(),resource=resource,elapsed_seconds=time.perf_counter()-t))
    print(json.dumps(dict(status='PASS',block_seconds=timings,compute_minutes=eta_compute,hash_seconds=hash_seconds,resource=resource,original_loss_error=original_max,original_gradient_error=gradmax)),flush=True)
def run():
    pre=read(HERE/'preflight.json');assert pre['status']=='PASS';locks=pre['source_hashes'];old=read(HERE/'old_results_manifest.json')['files'];verify({**old,**locks},'INITIAL_OLD_AND_SOURCE_SHA');d=load_data()
    collected={'FIT':[],'CAL':[]};maximum=0.;done_count=0
    for job in jobs():
        stem=HERE/'blocks'/name(job);dp=stem.with_suffix('.done.json');z=getblock(job)
        stage('OBJECTIVE_PATH_GRADIENT_AUDIT',active_job=name(job),completed_jobs=done_count,total_jobs=450)
        if dp.exists():resume_accept(read(dp),locks);o=npz(stem.with_suffix('.npz'));print('[SKIP SHA] '+name(job),flush=True)
        else:
            o=calculate(d,job,z);av=audit(d,job,z,o);stem.parent.mkdir(exist_ok=True)
            np.savez_compressed(stem.with_suffix('.npz'),**o);write(stem.with_suffix('.audit.json'),av)
            write(dp,dict(status='SUCCESS',source_hashes=locks,outputs={str(stem.with_suffix(s)):sha(stem.with_suffix(s)) for s in ('.npz','.audit.json')},model_training=False,rows=len(z['sample_id'])))
        av=audit(d,job,z,o);maximum=max(maximum,av['max_error']);collected[job[0]].append(o);done_count+=1
        del z;guard.check()
    outputs={}
    for pop,parts in collected.items():
        outputs[pop]={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez_compressed(HERE/(pop+'_objective.npz'),**outputs[pop])
    summary=summarize(d,outputs);stage('FINAL_OLD_AND_SOURCE_SHA',completed_jobs=450,total_jobs=450)
    verify({**old,**locks},'FINAL_OLD_AND_SOURCE_SHA')
    for job in jobs():resume_accept(read(HERE/'blocks'/(name(job)+'.done.json')),locks)
    write(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS',blocks=450,independent_numpy_max_error=maximum,source_files=len(locks),old_files=len(old),all_900_block_outputs_SHA=True,all_saved_components_quality_counts=True,original_AST_loss_gradient_preflight=True,model_forward=False))
    write(HERE/'RESOURCE_USAGE.json',dict(status='PASS',**guard.check()))
    write(HERE/'RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_ORIGINAL_OBJECTIVE',all_summaries=summary,objective_denominators=d['contexts'],model_training=False,model_forward=False,formal_response_completed=2,
      limitations=['GT-assisted loss input not observed shared mapping','Output-space gradient not model parameter accessibility','Population denominators not actual shuffled training minibatches','Reconstructed increment residual not original float32 bitwise residual','Per-window p95 averages not formal global p95 motion gates','No effective-good reclustering for single replacement; raw counts not diversity','Only three scenes/CAL university/overlapping windows/F16 seen/CAL-selected checkpoints']))
    write(HERE/'exit_status.json',dict(status='SUCCESS',at=datetime.now().astimezone().isoformat(),worker_pid=os.getpid()))
    stage('SUCCESS',completed_jobs=450,total_jobs=450)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');a=parser.parse_args()
    try:preflight() if a.action=='preflight' else run()
    except BaseException:
        import traceback,shutil
        folder=HERE/'failures'/datetime.now().strftime('preflight_%Y%m%d_%H%M%S');folder.mkdir(parents=True,exist_ok=False)
        (folder/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        for p in HERE.glob('*.py'):shutil.copy2(p,folder/p.name)
        raise
