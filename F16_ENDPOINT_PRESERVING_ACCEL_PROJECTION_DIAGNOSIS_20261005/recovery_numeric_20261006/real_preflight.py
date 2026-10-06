"""Six frozen final heads: actual numeric repair, not optimizer/training."""
from pathlib import Path
import os,sys,json,time,hashlib,gc,traceback,ast
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from resource_guard import guard
from frozen_loader import build,exact,nodes
from projection import project
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
ROUTED=ROOT/'GENERATION_CANDIDATE_ROUTED_MAPPING_SERIES_20261005'
TRAIN=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
ENC=ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'
LABEL=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%1000==0:guard.check()
        assert sha(p)==h,('SOURCE_SHA',p)
def load(p,keys):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in keys}
def data():
    guard.check();c=load(TRAIN,['sample_id','candidate','motion','logits','change_group','ade','fde'])
    e=load(ENC,['sample_id','past','tokens']);g=load(LABEL,['sample_id','future'])
    ids=c['sample_id'];assert len(ids)==len(np.unique(ids))==28791
    assert np.array_equal(ids,e['sample_id']) and np.array_equal(ids,g['sample_id'])
    assert not set(ids[:25912])&set(ids[25912:])
    obs=np.zeros((28791,940),np.float32);obs[:,:16]=e['past'].reshape(-1,16);obs[:,16:784]=e['tokens'].reshape(-1,768)
    past=c['motion'][:,:16].reshape(-1,8,2);v=(past[:,-1]-past[:,-2]).astype(np.float32)
    speed=np.concatenate((np.linalg.norm(v,axis=-1)[:,None],v),1).astype(np.float32)
    base=c['candidate'][...,:24].reshape(-1,20,12,2).copy();future=g['future']
    for i in range(0,len(ids),256):
        err=np.linalg.norm(base[i:i+256]-future[i:i+256,None],axis=-1)
        assert np.allclose(err.mean(-1),c['ade'][i:i+256],atol=2e-5,rtol=0)
        assert np.allclose(err[:,:,-1],c['fde'][i:i+256],atol=2e-5,rtol=0)
    for name,key in [('state_scalers.npz','fit_id'),('route_threshold.npz','fit_id'),('physical_accel_scales.npz','fit_ids')]:
        with np.load(ROUTED/name,allow_pickle=False) as z:assert np.array_equal(z[key],ids[:25912])
    result=dict(ids=ids,obs=obs,speed=speed,base=base,future=future,logits=c['logits'],group=c['change_group'])
    del c,e,g;gc.collect();guard.check();return result
def physical(x):
    v=np.diff(x.astype(np.float64),axis=2,prepend=np.zeros_like(x[:,:,:1],dtype=np.float64))/.4
    return np.linalg.norm(np.diff(v,axis=2)/.4,axis=-1).mean((1,2))
def audit(b,u,p,up,limit):
    b=b.astype(np.float64);u=u.astype(np.float64)
    mb=physical(b);mp=physical(p)
    assert np.all(mp<=1.01*mb),('STRICT_PHYSICAL',float(np.max(mp-1.01*mb)),int((mp>1.01*mb).sum()))
    assert np.all(np.abs(up)<=limit.astype(np.float64))
    endpoint=float(np.max(np.abs((p-b)[:,:,-1]-u.sum(2))))
    D=np.linalg.norm(np.diff(u,axis=2)/.4**2,axis=-1).mean((1,2))
    alpha=np.ones(len(b));nz=D!=0;alpha[nz]=np.clip(.01*mb[nz]/D[nz],0,1)
    mu=u.mean(2,keepdims=True);reference=b+np.cumsum(mu+alpha[:,None,None,None]*(u-mu),2)
    ref=float(np.max(np.abs(p-reference)))
    assert endpoint<=2e-5 and ref<=2e-5,(endpoint,ref)
    return dict(status='PASS',endpoint_error=endpoint,nominal_reference_error=ref,strict_physical_max_margin=float(np.max(mp-1.01*mb)),zero_native_count=int((mb==0).sum()))
def predict(model,d,a,b):
    values=[]
    with torch.inference_mode():
        for i in range(a,b,128):
            guard.check();stop=min(b,i+128)
            obs,base,speed=[torch.from_numpy(d[k][i:stop]).cuda() for k in ('obs','base','speed')]
            p,u=model(obs,base,speed);limit=1.5*(speed[:,0,None,None,None]+.03)
            pp,up,stats=project(base,u,limit)
            assert torch.isfinite(p).all() and torch.isfinite(pp).all()
            arrays=[x.cpu().numpy() for x in (p,u,pp,up,limit)]
            check=audit(d['base'][i:stop],arrays[1],arrays[2],arrays[3],arrays[4])
            values.append((arrays,{k:v.cpu().numpy() for k,v in stats.items()},check))
    return values
def models():
    lock=read(ROUTED/'results/model_lock.json')['models'];out=[]
    for arm in ('zero','mean_std'):
        for seed in (1,2,3):
            p=ROUTED/f'results/models/{arm}/seed_{seed}/balanced_protected/final.pth'
            assert sha(p)==lock[str(p)]
            out.append(dict(arm=arm,seed=seed,path=str(p),sha256=lock[str(p)]))
    return out
def snapshot():
    dst=HERE/'old_results_manifest.json'
    if dst.exists():return read(dst)['files']
    files=dict(read(ROUTED/'old_results_manifest.json')['files']);verify(files)
    for folder in ROOT.iterdir():
        if not folder.is_dir() or not folder.name.startswith(('GENERATION_','F16_','HEAD_','PULSE_')):continue
        for p in folder.rglob('*'):
            if HERE in p.parents or not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.lock','.tmp','.pending') or p.name=='runtime_state.json':continue
            if str(p) not in files:files[str(p)]=sha(p)
    write(dst,dict(files=files,prior_digests_preserved=True));return files
def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    own={str(p):sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.ps1') or p.name=='protocol.json'}
    inherited=read(ROUTED/'preflight.json')['source_hashes'];verify(inherited)
    old=snapshot();verify(old);initial_seconds=time.perf_counter()-start
    entries=models();d=data();checks=[]
    for entry in entries:
        model,meta=build('raw',entry['path']);state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.cuda()
        first=predict(model,d,0,128)[0];repeat=predict(model,d,0,128)[0]
        assert all(np.array_equal(x,y) for x,y in zip(first[0],repeat[0]))
        with torch.inference_mode():
            obs=torch.from_numpy(d['obs'][:128]).cuda()
            loc=obs[:,:16].reshape(-1,1,8,2);tok=obs[:,16:784].reshape(-1,1,8,96)
            p,l=model.decoder(tok,loc);p=(p[:,0]-loc[:,0,-1,None,None]).cpu().numpy();l=l[:,0].cpu().numpy()
        pe=float(np.max(np.abs(p-d['base'][:128])));le=float(np.max(np.abs(l-d['logits'][:128])))
        assert pe<2e-5 and le<2e-5
        torch.cuda.synchronize();t=time.perf_counter();predict(model,d,0,1024);torch.cuda.synchronize()
        checks.append(dict(**entry,metadata=meta,audit=first[2],native_path_error=pe,native_logit_error=le,seconds_per128=(time.perf_counter()-t)/8,all_frozen_exact=True))
        exact(model,state);del model,state,first,repeat;gc.collect();torch.cuda.empty_cache()
    verify(own);verify(inherited);verify(old)
    report=dict(status='PASS_REAL_SIX_HEAD_NUMERIC_PREFLIGHT',checks=checks,old_files=len(old),source_hashes={**inherited,**own},memory=guard.check(),initial_SHA_seconds=initial_seconds,elapsed_seconds=time.perf_counter()-start,no_optimizer=True,no_training=True,no_DEV=True)
    write(HERE/'real_preflight.json',report);guard.save('PASS');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        write(HERE/'failures/real_preflight_1/TRACEBACK.json',dict(traceback=traceback.format_exc(),source_hashes={str(p):sha(p) for p in HERE.glob('*.py')}))
        guard.save('FAILURE');raise
