"""Short frozen E8, one host; no training/optimizer or scientific changes."""
import os,sys,importlib.util,time,json,inspect,traceback,gc,platform
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
H=Path(__file__).resolve().parent;P=H.parent.parent.parent
sp=importlib.util.spec_from_file_location('frozen_utils',P/'claim_evidence_followup_20261009/saved_reader_recovery/analyze_saved.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
Q=a.read(H/'protocol.json');a.Q={**Q,'bootstrap_reps':2000,'stat_seed':20261009};a.H=H;a.START=time.perf_counter()
sp=importlib.util.spec_from_file_location('stage1_existing',P/'stage1.py');s=importlib.util.module_from_spec(sp);sp.loader.exec_module(s)
def lock(p):a.lock(p)
def timecalls(fn):
    for _ in range(Q['warmup']):a.guard();fn()
    values=[]
    for _ in range(Q['measure']):
        a.guard();t=time.perf_counter();out=fn();values.append((time.perf_counter()-t)*1000)
        assert np.isfinite(out[0] if isinstance(out,tuple) else out).all()
    return np.array(values)
def summarize(x):return dict(mean_ms=float(x.mean()),p95_ms=float(np.quantile(x,.95)),median_ms=float(np.median(x)),minimum_ms=float(x.min()),n=len(x))
def forward(model,trainer,obs,valid,device):
    import torch
    past=torch.from_numpy(np.array(obs,dtype=np.float32,copy=True)).to(device);nv=torch.from_numpy(np.array(valid,dtype=np.int64,copy=True)).to(device);local,_=trainer.normalize_to_focal_heading(past,past);center=local.mean(-2,keepdim=True)
    with torch.no_grad():pred=model(local-center,None,None,mode='testing',num_valid=nv,graph_loc=local)[2];logs=model.last_mode_logits[:,0].clone()
    rel=pred[:,0]+center[:,0,None]-local[:,0,-1,None,None];d=past[:,0,-1]-past[:,0,-4];speed=torch.linalg.vector_norm(d,dim=-1)
    unit=torch.where((speed>.03)[:,None],d/speed[:,None].clamp_min(1e-6),torch.tensor([1.,0.],device=device))
    world=torch.stack((rel[...,0]*unit[:,None,None,0]-rel[...,1]*unit[:,None,None,1],rel[...,0]*unit[:,None,None,1]+rel[...,1]*unit[:,None,None,0]),-1)
    return world.cpu().numpy().astype(float),logs.cpu().numpy()
def main():
    a.guard()
    for mf in [P/'completion_review/input_manifest.json',P/'stage3_knn_memory/completion_review/input_manifest.json']:
        lock(mf);a.EXPECTED.update(a.read(mf)['files'])
    for p in [Path(__file__),H/'protocol.json',H/'DESIGN.md',P/'stage1.py',P/'stage3_knn_memory/knn.py']:lock(p)
    bank=a.load(P/'stage3_knn_memory/data/eth_bank.npz');rp=a.D/'eth_data_train.npy';npth=a.D/'eth_num_train.npy';lock(rp);lock(npth)
    raw=np.load(rp,mmap_mode='r');counts=np.load(npth,mmap_mode='r');legal=np.flatnonzero(np.minimum(counts[bank['ids']],9)==9);assert len(legal),'NO_TRAIN_VALID9'
    ID=int(bank['ids'][legal[0]]);obs=np.array(raw[ID:ID+1,:,:8,-2:],np.float32);assert np.array_equal(obs[:,0],bank['obs'][legal[0]:legal[0]+1]);a.save('benchmark_input.npz',obs=obs,source_id=np.array([ID]))
    X=np.concatenate((bank['obs'][:,-1].astype(float),(bank['obs'][:,-1]-bank['obs'][:,-2]).astype(float)),1);mu=X.mean(0);sd=X.std(0);sd=np.where(sd<1e-6,1,sd);X=(X-mu)/sd;tree=cKDTree(X)
    bankCV=(bank['obs'][:,-1]-bank['obs'][:,-2]).astype(float)[:,None]*np.arange(1,13)[None,:,None];residual=bank['gt']-bankCV
    def cv():return (obs[:,0,-1]-obs[:,0,-2]).astype(float)[:,None]*np.arange(1,13)[None,:,None]
    def knn():
        f=np.concatenate((obs[:,0,-1].astype(float),(obs[:,0,-1]-obs[:,0,-2]).astype(float)),1);q=((f-mu)/sd)[0];dist,ix=tree.query(q,k=32,workers=1);pool=np.array(tree.query_ball_point(q,np.nextafter(dist[-1]+1e-12,np.inf)),int)
        ds=np.linalg.norm(X[pool]-q,axis=1);ix=pool[np.lexsort((bank['ids'][pool],ds))[:32]];assert len(ix)==32
        return cv()+residual[ix].mean(0)[None]
    brute=np.linalg.norm(X-((np.concatenate((obs[:,0,-1].astype(float),(obs[:,0,-1]-obs[:,0,-2]).astype(float)),1)-mu)/sd)[0],axis=1)
    bix=np.lexsort((bank['ids'],brute))[:32];assert np.array_equal(knn(),cv()+residual[bix].mean(0)[None])
    times={};baseline=[]
    for name,fn in [('CV_last',cv),('position_velocity_kNN_K32',knn)]:
        values=timecalls(fn);times[name]=values;baseline.append(dict(method=name,**summarize(values),input_source_id=ID,bank_ready_in_RAM=True));print('BASELINE',name,summarize(values),flush=True)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    loader=s.module('efficiency_existing_loader',s.R/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/run_formal_regime_selection.py');loader.fold_run=lambda fold,stage='v2':s.model_paths(fold,1,stage)
    for stage in ('h3','v2'):lock(s.model_paths('eth',1,stage)/'best_selected.pth')
    evaluation=a.load(P/'data/eth_evaluation.npz');saved=a.old_job(P/'jobs/eth_seed1');rows=[];frozen=[];components=[];paramtotal=None
    for device_name in Q['devices']:
        a.guard();device=torch.device(device_name)
        if device.type=='cuda':assert torch.cuda.is_available(),'CUDA_UNAVAILABLE'
        cfg,regime,front,trainer,adapter=loader.setup_backend(device)
        for mod in [loader,regime,front,trainer,adapter,adapter.CONTROLLED]:lock(mod.__file__)
        model=loader.build_baseline('eth',cfg,front,adapter,device);model.device=str(device)
        for mod in model.modules():
            if hasattr(mod,'device'):mod.device=str(device)
            try:path=inspect.getsourcefile(type(mod))
            except TypeError:path=None
            if path and ('New project' in path or 'ASTRA_ACCURACY_Q20' in path):lock(path)
        old={k:x.detach().cpu().clone() for k,x in model.state_dict().items()};assert not any(p.requires_grad for p in model.parameters()) and not model.training
        total=sum(p.numel() for p in model.parameters());assert paramtotal is None or total==paramtotal;paramtotal=total
        if not components:
            groups={}
            for name,p in model.named_parameters():groups[name.split('.')[0]]=groups.get(name.split('.')[0],0)+p.numel()
            components=[dict(component=k,parameters=v) for k,v in groups.items()]
        reference,logs=forward(model,trainer,evaluation['obs'][:1],evaluation['valid'][:1],device);patherr=float(np.max(abs(reference-saved['model_paths'][:1])));logerr=float(np.max(abs(logs-saved['model_logits'][:1])));assert max(patherr,logerr)<=2e-5,('CACHE_ALIGN',patherr,logerr)
        for n in Q['valid_agents']:
            nv=np.array([n]);fn=lambda:forward(model,trainer,obs,nv,device)
            fn()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,with_flops=True) as prof:fn()
            operators=[dict(op=e.key,flops=int(e.flops),calls=int(e.count)) for e in prof.key_averages() if e.flops]
            reported=sum(e['flops'] for e in operators);assert reported>0
            values=timecalls(fn);times[device_name.replace(':','_')+f'_N{n}']=values
            rows.append(dict(device=device_name,valid_agents=n,padded_agents=9,**summarize(values),supported_profiler_MAC=reported/2,supported_operators=operators,NOT_full_all_ops_MAC=True,cache_path_error=patherr,cache_logit_error=logerr));print('MODEL',device_name,n,summarize(values),flush=True)
        for k,x in model.state_dict().items():assert torch.equal(x.detach().cpu(),old[k]),('FROZEN',k)
        frozen.append(dict(device=device_name,all_parameters_and_buffers_exact=True,total_state_entries=len(old),total_parameters=total,USE_PRETRAINED_UNET=bool(cfg.MODEL.USE_PRETRAINED_UNET),USE_CACHED_UNET_FEATURES=bool(cfg.MODEL.USE_CACHED_UNET_FEATURES)))
        del model,old;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    a.save('timing_samples.npz',**times)
    a.write('input_manifest.json',dict(files=a.LOCKS))
    for p,d in a.LOCKS.items():a.guard();assert a.sha(p)==d
    a.write('RESULT.json',dict(status='PASS_FROZEN_E8_SINGLE_HOST_PARTIAL_MAC',model='ETH_existing_seed1_F16_original_H3_V2_K20',total_parameters=paramtotal,components=components,baseline=baseline,measurements=rows,frozen=frozen,hardware=dict(host=platform.node(),platform=platform.platform(),CPU=platform.processor(),GPU=torch.cuda.get_device_name(0),torch=torch.__version__,CPUthreads=torch.get_num_threads()),input_files=len(a.LOCKS),initial_final_SHA_PASS=True,timings_sha=a.sha(H/'timing_samples.npz'),resource=a.RES,elapsed_seconds=time.perf_counter()-a.START,no_training=True,single_host_not_two_hardware=True,MAC_partial=True))
    print('E8 COMPLETE',round(time.perf_counter()-a.START,3),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:a.write('FAILURE.json',dict(status='STOP',traceback=traceback.format_exc(),resource=a.RES,elapsed_seconds=time.perf_counter()-a.START));raise
