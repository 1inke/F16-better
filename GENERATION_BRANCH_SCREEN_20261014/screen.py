"""One bounded saved-output branch screen. Oracle diagnosis, NOT a forecaster."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,hashlib,ctypes,traceback
from pathlib import Path
import numpy as np
import psutil
from scipy.cluster.hierarchy import linkage,fcluster

H=Path(__file__).resolve().parent
P=H.parent/'PED_FOUR_DIRECTION_SERIES_20261011/experiment3_samepath_quality'
PORTS=('native20','oracle_endpoint_native20','oracle_endpoint_line1','oracle_waypoint3_1')
RESOURCE={'peak_private_GiB':0.,'min_physical_GiB':100.,'min_commit_GiB':100.}
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
def guard():
    m=Mem();m.length=ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    p=psutil.Process().memory_info()
    RESOURCE['peak_private_GiB']=max(RESOURCE['peak_private_GiB'],p.private/2**30)
    RESOURCE['min_physical_GiB']=min(RESOURCE['min_physical_GiB'],m.avail_phys/2**30)
    RESOURCE['min_commit_GiB']=min(RESOURCE['min_commit_GiB'],m.avail_page/2**30)
    assert max(p.rss,p.private)<=6*2**30 and min(m.avail_phys,m.avail_page)>=6*2**30,'STOP_RESOURCE'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(name,a):
    with (H/name).open('x',encoding='utf8') as f:
        json.dump(a,f,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def construct(p,g):
    s=np.arange(1,13,dtype=float)/12
    endpoint=p+s[None,None,:,None]*(g[:,None,-1:, :]-p[:,:,-1:,:])
    line=s[None,:,None]*g[:,-1:, :]
    nodes=np.concatenate((np.zeros_like(g[:,:1]),g[:,[3,7,11]]),axis=1)
    basis=np.stack([np.interp(np.arange(1,13),[0,4,8,12],np.eye(4)[j]) for j in range(4)],1)
    waypoint=np.einsum('tj,bjd->btd',basis,nodes)
    return (p,endpoint,line[:,None],waypoint[:,None])
def independent_construct(p,g):
    endpoint=p.copy();line=np.zeros((len(g),1,12,2));way=np.zeros_like(line)
    nodes=np.concatenate((np.zeros_like(g[:,:1]),g[:,[3,7,11]]),1)
    for t in range(1,13):
        endpoint[:,:,t-1,:]=p[:,:,t-1,:]+t*(g[:,None,11,:]-p[:,:,11,:])/12
        line[:,0,t-1,:]=g[:,11,:]*t/12
        segment=(t-1)//4;w=(t-segment*4)/4
        way[:,0,t-1,:]=(1-w)*nodes[:,segment]+w*nodes[:,segment+1]
    return p,endpoint,line,way
def metrics(p,g,ind=False):
    err=p-g[:,None]
    e=np.sqrt((err*err).sum(-1)) if ind else np.linalg.norm(err,axis=-1)
    a=e.mean(-1);f=e[:,:,-1];j=np.maximum(a/.2,f/.4);ix=j.argmin(1)
    return a,f,j,((a<=.2)&(f<=.4)).any(1),e[np.arange(len(p)),ix].reshape(-1,3,4).mean(-1)
def motion(p,ind=False):
    if ind:
        v=np.concatenate((p[:,:,:1],p[:,:,1:]-p[:,:,:-1]),2)/.4
        a=(v[:,:,1:]-v[:,:,:-1])/.4;j=(a[:,:,1:]-a[:,:,:-1])/.4
    else:
        v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4
        a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4
    s,aa,jj=(np.sqrt((x*x).sum(-1)) for x in (v,a,j))
    return dict(speed_p95=float(np.quantile(s,.95)),accel_mean=float(aa.mean()),accel_p95=float(np.quantile(aa,.95)),jerk_mean=float(jj.mean()),jerk_p95=float(np.quantile(jj,.95)))
def cluster(p,good):
    out=[]
    for i,(x,g) in enumerate(zip(p,good)):
        if i%128==0:guard()
        if len(x)==1:out.append((1,int(g[0])));continue
        d=np.linalg.norm(x[:,None]-x[None],axis=-1)
        D=np.maximum(d.mean(-1)/.1,d[:,:,-1]/.2)
        lab=fcluster(linkage(D[np.triu_indices(len(x),1)],method='complete'),1.,criterion='distance')
        out.append((len(np.unique(lab)),len(np.unique(lab[g]))))
    return np.asarray(out)
def ci(delta,track):
    _,inv=np.unique(track,return_inverse=True);n=np.bincount(inv);num=np.bincount(inv,weights=delta)
    rng=np.random.default_rng(20261014);vals=[]
    for _ in range(2000):
        ix=rng.integers(len(n),size=len(n));vals.append(num[ix].sum()/n[ix].sum())
    return (np.quantile(vals,[.025,.975])*100).tolist()
def main():
    guard();start=time.perf_counter()
    assert not (H/'RESULT.json').exists(),'NO_REPEATED_RUN'
    source={str(p):sha(p) for p in (H/'screen.py',H/'DESIGN.md')}
    parent=read(P/'RESULT.json');records=[];checks={};output={};inputs={}
    # Exact fixtures, including zero displacement and piecewise linear reconstruction.
    t=np.arange(1,13,dtype=float);g=np.stack((t,t*0),-1)[None]
    p=np.repeat(g[:,None],20,axis=1)
    for a,b in zip(construct(p,g),independent_construct(p,g)):assert np.array_equal(a,b)
    for split,N in (('outer',1035),('inner',1856),('original_CAL',2879)):
        guard();path=P/'runs/sum/seed_42'/f'{split}.npz'
        digest=sha(path);assert digest==parent['output_sha'][str(path)]
        inputs[str(path)]=digest;z=load(path)
        assert len(z['sample_id'])==N and len(np.unique(z['sample_id']))==N
        p=z['native20'].astype(float);g=z['gt'].astype(float)
        assert p.shape==(N,20,12,2) and np.isfinite(p).all() and np.isfinite(g).all()
        ports=construct(p,g);refs=independent_construct(p,g)
        maxerr=max(float(np.max(abs(a-b))) for a,b in zip(ports,refs));assert maxerr<2e-12
        epdiff=float(np.max(abs(ports[1][:,:,-1]-g[:,None,-1])));assert epdiff<2e-12
        # Uniform displacement ramp leaves discrete acceleration/jerk unchanged (roundoff only).
        def acc(x):return np.diff(np.diff(x,axis=2,prepend=np.zeros_like(x[:,:,:1]))/.4,axis=2)/.4
        adiff=float(np.max(abs(acc(p)-acc(ports[1]))));assert adiff<2e-12
        raw=[metrics(x,g) for x in ports];ng=raw[0][3]
        counts=[]
        for pp,r in zip(ports,raw):
            counts.append(cluster(pp,(r[0]<=.2)&(r[1]<=.4)))
        for name,mask in [('overall',np.ones(N,bool)),('stable',z['groups']==0),('transition',z['groups']==1),('change',z['groups']==2)]:
            if not mask.any():continue
            baseline=motion(p[mask]);gtmotion=motion(g[mask,None])
            aa,ff=raw[0][:2]
            decomp=dict(native_endpoint_good=int((ff[mask].min(1)<=.4).sum()),native_ADE_good=int((aa[mask].min(1)<=.2).sum()),native_joint_good=int(ng[mask].sum()),separate_minima_false_success=int((((aa[mask].min(1)<=.2)&(ff[mask].min(1)<=.4))&~ng[mask]).sum()))
            for k,port in enumerate(PORTS):
                a,f,j,good,one=raw[k];mot=motion(ports[k][mask]);ind=motion(refs[k][mask],True)
                assert max(abs(mot[v]-ind[v]) for v in mot)<2e-11
                ia,iff,ij,ig,io=metrics(refs[k],g,True)
                assert np.array_equal(good,ig) and max(np.max(abs(a-ia)),np.max(abs(f-iff)),np.max(abs(j-ij)))<2e-11
                records.append(dict(split=split,group=name,port=port,n=int(mask.sum()),K=ports[k].shape[1],good=int(good[mask].sum()),coverage=float(good[mask].mean()),rescued=int((good&~ng&mask).sum()),lost=int((~good&ng&mask).sum()),minADE=float(a[mask].min(1).mean()),minFDE=float(f[mask].min(1).mean()),J=float(j[mask].min(1).mean()),same_Jclosest_segments=one[mask].mean(0),coverage_delta_pp=float((good[mask].astype(float)-ng[mask]).mean()*100),coverage_delta_track_CI95_pp=ci(good[mask].astype(float)-ng[mask],z['track'][mask]),effective_distinct=float(counts[k][mask,0].mean()),effective_good=float(counts[k][mask,1].mean()),motion=mot,native_motion=baseline,GT_motion=gtmotion,motion_failed=[v for v in mot if mot[v]>1.01*baseline[v]],**decomp))
        save={v:z[v] for v in ('sample_id','groups','track','scene','gt')}
        save.update(dict(zip(PORTS,ports)))
        out=H/f'{split}_diagnostic.npz';assert not out.exists();np.savez_compressed(out,**save)
        back=load(out);assert all(np.array_equal(v,back[k]) for k,v in save.items())
        output[str(out)]=sha(out);checks[split]=dict(IDs=N,construct_max_error=maxerr,endpoint_error=epdiff,acceleration_invariance_error=adiff,saved_exact=True,metrics_independent_PASS=True)
        print(split,'complete',flush=True)
    assert all(sha(p)==s for p,s in {**inputs,**source}.items())
    assert all(sha(p)==s for p,s in output.items())
    write('RESULT.json',dict(status='PASS_ORACLE_BRANCH_SCREEN_NOT_MODEL_IMPROVEMENT',records=records,checks=checks,input_sha=inputs,source_sha=source,output_sha=output,source_inputs_initial_final_unchanged=True,resource=RESOURCE,elapsed_seconds=time.perf_counter()-start,neural_training=False,old_models_loaded=False,formal_increment=0,independent_cluster_rebuild=False,limitations=['GT-conditioned diagnostic not deployable','only one existing native cache realization','no claim of complete model-family upper bound','track CIs exploratory and fixed-data','cluster implementation matches old definition, not second full independent cluster rebuild']))
    print(json.dumps(dict(status='PASS',elapsed_seconds=time.perf_counter()-start,resource=RESOURCE)))
if __name__=='__main__':
    try:main()
    except Exception:
        write('FAILURE_'+str(time.time_ns())+'.json',dict(traceback=traceback.format_exc(),resource=RESOURCE));raise
