"""Original-primary parameter concordance; no optimizer or parameter updates."""
from pathlib import Path
from datetime import datetime
import ast, gc, hashlib, importlib.util, json, os, sys, time
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from resource_guard import guard
from frozen_loader import build
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
PREV=ROOT/'F16_MATCHED_FIT_CAL_TRANSFER_DIAGNOSIS_20261005'
spec=importlib.util.spec_from_file_location('matched_readonly_definitions',PREV/'diagnosis.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
d.HERE=HERE
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
GROUPS=('stable','transition','change')
BATCH=128
START=time.perf_counter()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);assert HERE==path.parent or HERE in path.parents
    d.write(path,value)
def sha(path):return d.sha(path)
def stage(name,**info):d.stage(name,**info)
def verify(files,label):return d.verify(files,label)
def old_snapshot():
    p=HERE/'old_results_manifest.json'
    if p.exists():return read(p)['files']
    files=dict(read(PREV/'old_results_manifest.json')['files'])
    verify(files,'INHERITED_OLD_SHA')
    for folder in ROOT.iterdir():
        if not folder.is_dir() or folder==HERE or not folder.name.startswith(('F16_','GENERATION_','HEAD_','PULSE_')):continue
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.name!='runtime_state.json' and p.suffix not in ('.lock','.tmp','.pending') and str(p) not in files:files[str(p)]=sha(p)
    write(HERE/'old_results_manifest.json',dict(files=files,prior_digests_preserved=True,control_files_excluded=True))
    return files
def sources(models):
    files=dict(read(PREV/'preflight.json')['source_hashes'])
    verify(files,'INHERITED_SOURCE_SHA')
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/x for x in ('README.md','DESIGN.md','protocol.json','old_results_manifest.json')]
    paths += [PREV/'diagnosis.py',PREV/'RESULT.json',ROOT/'GENERATION_MODE_SUPPORT_SERIES/run_mode_support.py',ROOT/'GENERATION_PROTECTED_BALANCE_SERIES/preflight.json',ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/config.json']
    for path in paths:files[str(path)]=sha(path)
    for m in models:files[m['path']]=m['sha256']
    return files
def quality_data():
    data=d.load_data()
    env=dict(torch=torch,np=np,LABEL_CFG=read(ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/config.json'))
    path=ROOT/'GENERATION_MODE_SUPPORT_SERIES/run_mode_support.py'
    node=[x for x in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(x,ast.FunctionDef) and x.name=='labels']
    assert len(node)==1
    exec(compile(ast.Module(body=node,type_ignores=[]),str(path),'exec'),env)
    with torch.no_grad():target=env['labels'](torch.from_numpy(data['speed'][:,1:]),torch.from_numpy(data['future'][:,None,-1]/12)).numpy()[:,0]
    counts=np.bincount(target[:25912],minlength=7)
    weight=np.sqrt(25912/(7*counts)).clip(.5,3)
    weight/=weight[target[:25912]].mean()
    expected=np.asarray(read(ROOT/'GENERATION_PROTECTED_BALANCE_SERIES/preflight.json')['weights'])
    assert np.max(np.abs(weight-expected))<1e-12
    data['quality_weight']=weight.astype(np.float32)[target]
    data['target']=target
    data['class_weights']=weight.astype(np.float32)
    assert len(np.unique(data['ids']))==28791
    return data
def setup(model):
    names=[];params=[];layout=[];offset=0
    for name,p in model.named_parameters():
        enabled=name.startswith(('state_encoder.','query.','key.','value.','head.'))
        p.requires_grad_(enabled)
        assert p.grad is None
        if enabled:
            names.append(name);params.append(p)
            layout.append(dict(name=name,start=offset,stop=offset+p.numel(),shape=list(p.shape)))
            offset+=p.numel()
    assert offset==45798,offset
    return names,params,layout
def unchanged(model,state):
    assert not model.training
    assert model.state_dict().keys()==state.keys()
    for k,v in model.state_dict().items():assert torch.equal(v.detach().cpu(),state[k]),('WEIGHT_BUFFER_CHANGED',k)
    for name,p in model.named_parameters():
        assert p.requires_grad==name.startswith(('state_encoder.','query.','key.','value.','head.')),name
        assert p.grad is None,name
def tensors(data,indices,device,dtype=torch.float32):
    return [torch.as_tensor(data[k][indices],device=device,dtype=dtype) for k in ('obs','base','speed','future','quality_weight')]
def quality(model,ts):
    obs,base,speed,future,weight=ts
    out,res=model(obs,base,speed)
    assert out.shape==(len(obs),20,12,2) and torch.isfinite(out).all()
    assert (res.abs()<=model.strength*(speed[:,0,None,None,None]+.03)+2e-5).all()
    err=(out-future[:,None]).norm(dim=-1)
    q=err.mean(-1)+.5*err[:,:,-1]
    return q.min(-1).values*weight,q.argmin(-1),out
def vector(loss,params,retain=False):
    grads=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=False)
    assert all(torch.isfinite(g).all() for g in grads)
    return torch.cat([g.flatten() for g in grads]).detach().cpu().double().numpy()
def accumulate(model,params,data,indices,chunk=128,report=False):
    index_groups=data['group'][indices]
    counts=np.bincount(index_groups,minlength=3).astype(np.int64)
    grad=np.zeros((3,45798),np.float64);values=np.zeros(3,np.float64)
    device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    for start in range(0,len(indices),chunk):
        guard.check()
        ix=indices[start:start+chunk];qs,winner,out=quality(model,tensors(data,ix,device,dtype))
        group=data['group'][ix]
        available=[g for g in range(3) if np.any(group==g)]
        for j,g in enumerate(available):
            mask=torch.as_tensor(group==g,device=device)
            loss=qs[mask].sum()/int(counts[g])
            grad[g]+=vector(loss,params,retain=j<len(available)-1)
            values[g]+=float(qs.detach()[mask].double().sum().cpu())/int(counts[g])
        if report and start%2048==0:stage('PARAMETER_PRIMARY_GRADIENT',row=start,rows=len(indices),checkpoint=CURRENT,split=SPLIT)
        del qs,winner,out
    overall=(grad*counts[:,None]).sum(0)/len(indices)
    overall_value=float(np.dot(values,counts)/len(indices))
    return dict(gradient=grad,overall_gradient=overall,quality=values,overall_quality=overall_value,counts=counts)
def stats(a,b):
    an=float(np.linalg.norm(a));bn=float(np.linalg.norm(b));dot=float(np.dot(a,b))
    if an<=1e-10 or bn<=1e-10:return dict(FITnorm=an,CALnorm=bn,dot=dot,cos=None,CAL_directional_derivative=None,status='UNDETERMINED_ZERO_OR_NEARZERO')
    cos=dot/(an*bn);assert abs(cos)<=1+1e-10
    return dict(FITnorm=an,CALnorm=bn,dot=dot,cos=max(-1.,min(1.,cos)),CAL_directional_derivative=-dot/an,status='LOCAL_PRIMARY_ALIGNED' if dot>0 else 'LOCAL_PRIMARY_CONFLICT' if dot<0 else 'LOCAL_PRIMARY_ORTHOGONAL')
def branch_indices(layout):
    result={}
    for prefix in ('state_encoder','query','key','value','head'):
        result[prefix]=np.concatenate([np.arange(x['start'],x['stop']) for x in layout if x['name'].startswith(prefix+'.')])
    for name,rows in [('D2',(0,1)),('A2',(2,3)),('clock2',(4,5))]:
        ix=[]
        for x in layout:
            if x['name']=='head.4.weight':
                width=x['shape'][1]
                for r in rows:ix.extend(range(x['start']+r*width,x['start']+(r+1)*width))
            elif x['name']=='head.4.bias':ix.extend(x['start']+r for r in rows)
        assert ix;result[name]=np.asarray(ix,dtype=np.int64)
    return result
def summarize(saved,layout):
    branches=branch_indices(layout);summary={}
    gf=dict(zip(GROUPS,saved['FIT_gradient']));gc_=dict(zip(GROUPS,saved['CAL_gradient']))
    gf['overall']=saved['FIT_overall_gradient'];gc_['overall']=saved['CAL_overall_gradient']
    for f,a in gf.items():
        for c,b in gc_.items():
            summary[f+'__'+c]=dict(all=stats(a,b),branches={k:stats(a[v],b[v]) for k,v in branches.items()})
    return summary
def key(meta):return f"{meta['family']}_seed{meta['seed']}_{meta['policy']}"
def complete_job(done,srchashes):
    assert done['status']=='SUCCESS' and done['source_hashes']==srchashes
    for p,h in done['outputs'].items():assert sha(p)==h
    return True
def fixtures():
    assert stats(np.array([1.,0]),np.array([2.,0]))['cos']==1
    assert stats(np.array([1.,0]),np.array([-2.,0]))['cos']==-1
    assert stats(np.array([1.,0]),np.array([0.,2.]))['cos']==0
    assert stats(np.array([0.,0]),np.array([0.,2.]))['cos'] is None
    w=torch.tensor([2.,3.],dtype=torch.float64,requires_grad=True)
    ga=torch.autograd.grad((w*torch.tensor([1.,2.])).sum(),w)[0].numpy()
    gb=torch.autograd.grad((w*torch.tensor([-1.,-2.])).sum(),w)[0].numpy()
    assert stats(ga,gb)['cos']<=-1+1e-12
    return 'PASS_KNOWN_LINEAR_PARAM_GRAD_ALIGNED_CONFLICT_ORTHOGONAL_ZERO'
def differences(model,params,data,indices):
    model.double()
    ts=tensors(data,indices,'cuda',torch.float64)
    q,winner,_=quality(model,ts);g=vector(q.mean(),params)
    norm=float(np.linalg.norm(g));assert norm>1e-10,'FD_NEARZERO'
    direction=torch.tensor(g/norm,device='cuda',dtype=torch.float64)
    initial=[p.detach().clone() for p in params]
    analytic=norm;record=[]
    def assign(sign,eps):
        off=0
        with torch.no_grad():
            for p,s in zip(params,initial):
                p.copy_(s+sign*eps*direction[off:off+p.numel()].reshape_as(p));off+=p.numel()
    for eps in (1e-4,1e-5):
        assign(1,eps)
        with torch.no_grad():plus,wp,_=quality(model,ts)
        assign(-1,eps)
        with torch.no_grad():minus,wm,_=quality(model,ts)
        derivative=float((plus.mean()-minus.mean())/(2*eps))
        switched=int(((wp!=winner)|(wm!=winner)).sum())
        tol=1e-7+1e-3*abs(analytic)
        record.append(dict(eps=eps,analytic=analytic,numeric=derivative,error=abs(derivative-analytic),tolerance=tol,winner_switch=switched,passed=switched==0 and abs(derivative-analytic)<=tol))
    assign(0,0)
    for p,s in zip(params,initial):assert torch.equal(p,s)
    assert any(x['passed'] for x in record),('FD_UNIDENTIFIABLE_STOP',record)
    return record
def preflight():
    global CURRENT,SPLIT
    guard.check();old=old_snapshot();verify(old,'INITIAL_OLD_SHA')
    models=d.checkpoints();src=sources(models)
    data=quality_data();reports=[];times=[]
    fixtures_result=fixtures()
    # Check every existing checkpoint's canonical olddecoder metadata/weights.
    for meta in models:
        model,info=build(meta['family'],Path(meta['path']))
        assert info['epoch']>=1 and not info['fallback']
        del model;gc.collect();guard.check()
    for family in ('raw','centered_raw','centered_own'):
        meta=next(x for x in models if x['family']==family and x['seed']==1 and x['policy']=='constrained')
        CURRENT=key(meta);SPLIT='FIT_SMOKE'
        model,_=build(family,Path(meta['path']));state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model=model.to('cuda');names,params,layout=setup(model)
        indices=np.arange(128)
        ts=tensors(data,indices,'cuda')
        with torch.no_grad():q,win,out=quality(model,ts)
        old_done=read(PREV/'jobs'/(CURRENT+'_FIT.done.json'))
        for p,h in old_done['outputs'].items():assert sha(p)==h
        pathfile=next(p for p in old_done['outputs'] if p.endswith('.npz') and 'paths' in p)
        with np.load(pathfile,allow_pickle=False) as z:
            aligned=float(np.max(np.abs(out.cpu().numpy()-z['paths'][:128])))
        assert aligned<=2e-5,aligned
        e=np.linalg.norm(out.cpu().numpy().astype(np.float64)-data['future'][:128,None].astype(np.float64),axis=-1)
        manual=np.min(e.mean(-1)+.5*e[:,:,-1],axis=1)*data['quality_weight'][:128]
        scalar_error=float(np.max(np.abs(q.cpu().numpy()-manual)));assert scalar_error<=2e-5
        chunks={}
        for size in (128,64,31):chunks[size]=accumulate(model,params,data,indices,size)
        checks=[]
        for size in (64,31):
            for g in range(3):
                ref=chunks[128]['gradient'][g];other=chunks[size]['gradient'][g]
                error=float(np.linalg.norm(ref-other));tol=1e-6+2e-4*float(np.linalg.norm(ref))
                assert error<=tol,('CHUNK_GRADIENT_STOP',family,size,g,error,tol)
                checks.append(dict(chunk=size,group=g,error=error,tolerance=tol))
        q2,_,_=quality(model,ts);direct=vector(q2.mean(),params)
        err=float(np.linalg.norm(direct-chunks[128]['overall_gradient']))
        assert err<=1e-6+2e-4*np.linalg.norm(direct)
        assert np.linalg.norm(direct)>1e-10
        branches=branch_indices(layout)
        for name in ('state_encoder','query','key','value','head'):assert np.linalg.norm(direct[branches[name]])>0,name
        unchanged(model,state)
        for b in range(8):
            torch.cuda.synchronize();begin=time.perf_counter()
            accumulate(model,params,data,np.arange(b*128,(b+1)*128),128)
            torch.cuda.synchronize();times.append(time.perf_counter()-begin)
        unchanged(model,state);del model,params;torch.cuda.empty_cache();gc.collect()
        # A separate double copy is a numerical audit, never a training model.
        numeric,_=build(family,Path(meta['path']));numeric=numeric.to('cuda').double()
        _,np_,_=setup(numeric);fd=differences(numeric,np_,data,np.arange(32))
        reports.append(dict(family=family,path_alignment=aligned,scalar_error=scalar_error,gradient_chunk_checks=checks,overall_gradient_error=err,numerical_float64_fd=fd))
        del numeric,np_;torch.cuda.empty_cache();gc.collect()
    with np.load(PREV/'jobs'/'native_FIT.paths.npz',allow_pickle=False) as z:assert np.array_equal(z['sample_id'],data['ids'][:25912])
    # SHA-based analysis recovery fixtures, no optimizer/RNG resume.
    temp=HERE/'preflight_fixture.txt';temp.write_text('registered fixture',encoding='utf-8')
    done=dict(status='SUCCESS',source_hashes=src,outputs={str(temp):sha(temp)})
    assert complete_job(done,src)
    rejected=[]
    for bad in ('source','output'):
        altered=json.loads(json.dumps(done))
        if bad=='source':altered['source_hashes']={'WRONG':'digest'}
        else:altered['outputs'][str(temp)]='0'*64
        try:complete_job(altered,src)
        except AssertionError:rejected.append(bad)
    assert rejected==['source','output']
    verify(src,'FINAL_SOURCE_SHA');verify(old,'FINAL_OLD_SHA')
    # Fixture created after source list intentionally not a production scientific input.
    estimate=float(np.mean(times))*sum((n+127)//128 for n in (25912,2879))*18
    write(HERE/'preflight.json',dict(status='PASS',sources=len(src),source_hashes=src,old_files=len(old),models=18,
        primary_weights=data['class_weights'].tolist(),fixtures=fixtures_result,analysis_SHA_resume='PASS_SKIP_BADSOURCE_BADOUTPUT_REJECT_NO_OPTIMIZER',actual_GPU_reports=reports,
        measured_eight_batches_per_family_seconds=times,mean_batch_seconds=float(np.mean(times)),nominal_compute_minutes=estimate/60,
        input_bytes=sum(v.nbytes for v in data.values() if isinstance(v,np.ndarray)),gradient_output_bytes=18*2*3*45798*8,
        memory=guard.values(),scope='PRIMARY_PARAMETER_DERIVATIVES_NO_OPTIMIZER_OLD_FROZEN_NO_DEV',completed_at=datetime.now().astimezone().isoformat()))
    guard.save('PREFLIGHT_PASS')
    print('PREFLIGHT PASS',estimate/60,flush=True)
def run():
    global CURRENT,SPLIT
    pf=read(HERE/'preflight.json');assert pf['status']=='PASS'
    old=read(HERE/'old_results_manifest.json')['files'];src=pf['source_hashes']
    verify(src,'INITIAL_SOURCE_SHA');verify(old,'INITIAL_OLD_SHA')
    data=quality_data();models=d.checkpoints();records=[];frozen=[];(HERE/'jobs').mkdir(exist_ok=True)
    for number,meta in enumerate(models):
        CURRENT=key(meta);job=HERE/'jobs'/CURRENT;donepath=job.with_suffix('.done.json')
        if donepath.exists():
            done=read(donepath);complete_job(done,src);records.append(done['record']);frozen.append(done['frozen']);continue
        model,info=build(meta['family'],Path(meta['path']));state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model=model.to('cuda');names,params,layout=setup(model)
        saved={};counts={}
        for split,(a,b) in d.SPLITS.items():
            SPLIT=split;result=accumulate(model,params,data,np.arange(a,b),report=True)
            counts[split]=result['counts'].tolist()
            for k,v in result.items():saved[split+'_'+k]=np.asarray(v)
            saved[split+'_sample_id']=data['ids'][a:b]
        unchanged(model,state)
        saved['class_weights']=data['class_weights'];summary=summarize(saved,layout)
        vectorpath=job.with_suffix('.npz');np.savez_compressed(vectorpath,**saved)
        with np.load(vectorpath,allow_pickle=False) as z:again={k:z[k] for k in z.files}
        for k,v in saved.items():assert np.array_equal(v,again[k]),k
        assert summarize(again,layout)==summary
        summarypath=job.with_suffix('.summary.json')
        write(summarypath,dict(layout=layout,parameter_names=names,summary=summary,model=meta,counts=counts,info=info,scope='LOCAL_PRIMARY_NOT_TOTAL_ADAMW'))
        record=dict(key=CURRENT,model=meta,info=info,counts=counts,summary=summary,outputs={str(vectorpath):sha(vectorpath),str(summarypath):sha(summarypath)})
        audit=dict(key=CURRENT,status='EXACT_ALL_WEIGHTS_BUFFERS_NO_UPDATES',parameters=45798,old_requires_grad_false=True,optimizer=False)
        write(donepath,dict(status='SUCCESS',source_hashes=src,outputs=record['outputs'],record=record,frozen=audit))
        records.append(record);frozen.append(audit)
        del model,params,state,saved,again;gc.collect();torch.cuda.empty_cache()
        stage('CHECKPOINT_COMPLETE',completed_jobs=number+1,total_jobs=18)
    stage('FINAL_SAVED_VECTOR_AUDIT',completed_jobs=18)
    for meta,rec in zip(models,records):
        complete_job(read(HERE/'jobs'/(key(meta)+'.done.json')),src)
        s=read(next(p for p in rec['outputs'] if p.endswith('.json')))
        with np.load(next(p for p in rec['outputs'] if p.endswith('.npz')),allow_pickle=False) as z:
            values={k:z[k] for k in z.files}
        assert summarize(values,s['layout'])==s['summary']
        for split,(a,b) in d.SPLITS.items():
            assert np.array_equal(values[split+'_sample_id'],data['ids'][a:b])
            counts=values[split+'_counts']
            assert np.array_equal(counts,np.bincount(data['group'][a:b],minlength=3))
            overall=(values[split+'_gradient']*counts[:,None]).sum(0)/(b-a)
            assert np.array_equal(overall,values[split+'_overall_gradient'])
    verify(src,'FINAL_SOURCE_SHA');verify(old,'FINAL_OLD_SHA')
    write(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=18,outputs=36,saved_vectors_summary_recomputed=True,not_all_gradients_replayed=True,all_old=len(old),sources=len(src)))
    write(HERE/'FROZEN_AUDIT.json',dict(status='PASS',records=frozen))
    write(HERE/'RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_PRIMARY_PARAMETER_CONCORDANCE',records=records,FIT=25912,CAL=2879,unique_IDs=28791,
        source_count=len(src),old_file_count=len(old),primary_weights=pf['primary_weights'],memory=guard.values(),elapsed_seconds=time.perf_counter()-START,
        interpretation='LOCAL_EUCLIDEAN_PRIMARY_NOT_TOTAL_ADAMW_NOT_CAUSAL_GENERALIZATION',formal_trial=False))
    guard.save('SUCCESS');write(HERE/'exit_status.json',dict(status='SUCCESS',at=datetime.now().astimezone().isoformat()))
    stage('COMPLETE',completed_jobs=18)
CURRENT='';SPLIT=''

