"""Independent CPU saved-data completion review. No torch or model/optimizer forward."""
import os,sys,json,time,hashlib,ctypes,traceback,math
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import numpy as np,psutil
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent.parent;D=H/'completion_review'
ARMS=('unprojected','physical');SEEDS=(42,43,44);PORTS=('native20','zero_native20','refined20');GROUPS=('overall','stable','transition','change')
resource=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
def write(name,obj):
    p=D/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8');os.replace(t,p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));p=psutil.Process().memory_info()
    resource['peak_private_GiB']=max(resource['peak_private_GiB'],p.peak_pagefile/2**30);resource['peak_RSS_GiB']=max(resource['peak_RSS_GiB'],p.peak_wset/2**30)
    resource['min_physical_GiB']=min(resource['min_physical_GiB'],m.avail_phys/2**30);resource['min_commit_GiB']=min(resource['min_commit_GiB'],m.avail_page/2**30);resource['checks']+=1
    if max(p.private,p.rss)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:raise MemoryError('STOP_RESOURCE_process6_physical_and_commit_reserve6')
def verify(files):
    for i,(p,value) in enumerate(files.items()):
        if i%32==0:guard()
        assert sha(p)==value,('SHA_CHANGED',p)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),(a.keys(),b.keys());return max([compare(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,list) and a and isinstance(a[0],(dict,str)):
        assert len(a)==len(b);return max(compare(x,y) for x,y in zip(a,b))
    if isinstance(a,(str,bool,np.bool_)):assert a==b,(a,b);return 0.
    x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float);assert x.shape==y.shape;return float(np.max(abs(x-y))) if x.size else 0.
def masks(g):return [(n,np.ones(len(g),bool) if i==0 else g==i-1) for i,n in enumerate(GROUPS)]
def error(p,gt):
    e=np.sqrt(np.sum((p.astype(float)-gt[:,None].astype(float))**2,axis=-1));a=e.mean(2);f=e[:,:,11];j=np.maximum(a/.2,f/.4);g=(a<=.2)&(f<=.4);one=e[np.arange(len(p)),j.argmin(1)].reshape(-1,3,4).mean(2);return a,f,j,g,one
def motion(p):
    vv=np.concatenate([p[:,:,:1],p[:,:,1:]-p[:,:,:-1]],2)/.4;aa=(vv[:,:,1:]-vv[:,:,:-1])/.4;jj=(aa[:,:,1:]-aa[:,:,:-1])/.4
    s=np.sqrt(np.sum(vv**2,-1));a=np.sqrt(np.sum(aa**2,-1));j=np.sqrt(np.sum(jj**2,-1))
    return dict(speed_p95=float(np.quantile(s,.95)),accel_mean=float(a.mean()),accel_p95=float(np.quantile(a,.95)),jerk_mean=float(j.mean()),jerk_p95=float(np.quantile(j,.95)))
def clusters(p,g):
    ans=[]
    for row,(paths,good) in enumerate(zip(p,g)):
        if row%32==0:guard()
        d=[]
        for i in range(19):
            e=np.sqrt(np.sum((paths[i]-paths[i+1:])**2,-1));d.extend(np.maximum(e.mean(1)/.1,e[:,-1]/.2))
        lab=fcluster(linkage(np.asarray(d),method='complete'),1.,criterion='distance');ans.append([len(np.unique(lab)),len(np.unique(lab[good]))])
    return np.asarray(ans,dtype=np.int32)
def assignment(a,f,g):
    Q=a+.5*f;J=np.maximum(a/.2,f/.4);wa=Q.argmin(1);wb=J.argmin(1);qs=np.sort(Q,1);js=np.sort(J,1);out={}
    for name,mask in masks(g):
        N=int(mask.sum())
        if not N:out[name]={'n':0};continue
        out[name]=dict(n=N,switched=int((wa[mask]!=wb[mask]).sum()),switch_rate=float((wa[mask]!=wb[mask]).mean()),scene_counts=np.bincount(wa[mask],minlength=4),track_counts=np.bincount(wb[mask],minlength=4),confusion=np.bincount(wa[mask]*4+wb[mask],minlength=16).reshape(4,4),scene_exact_ties=int((qs[mask,0]==qs[mask,1]).sum()),track_exact_ties=int((js[mask,0]==js[mask,1]).sum()),scene_near_ties=int((qs[mask,1]-qs[mask,0]<=1e-6).sum()),track_near_ties=int((js[mask,1]-js[mask,0]<=1e-6).sum()),scene_gap_mean=float((qs[mask,1]-qs[mask,0]).mean()),track_gap_mean=float((js[mask,1]-js[mask,0]).mean()))
    return out

def temporal_check(z,arm):
    x=np.arange(12,dtype=float)/11.;B=np.stack([math.comb(11,i)*x**i*(1-x)**(11-i) for i in range(12)],1)
    assert np.array_equal(B,z['time_basis']) and str(z['temporal_arm'])==arm
    u=z['bounded_controls'];assert u.shape==(len(z['sample_id']),4,12,2) and np.isfinite(u).all()
    assert np.max(np.linalg.norm(u,axis=-1))<=1+2e-15
    bound=.20*(np.arange(1,13)/12.)**2
    proposed=np.einsum('ti,bkid->bktd',B,u)*bound[None,None,:,None]
    def derivatives(p):
        v=np.diff(p,axis=-2,prepend=np.zeros_like(p[...,:1,:]))/.4
        a=np.diff(v,axis=-2)/.4;j=np.diff(a,axis=-2)/.4
        return v,a,j
    native=z['reference4'];alpha=np.ones(native.shape[:2])
    if arm=='physical':
        roots=[]
        for b,d in zip(derivatives(native),derivatives(proposed)):
            A=np.sum(d*d,-1);B0=2*np.sum(b*d,-1);C=(1.01**2-1)*np.sum(b*b,-1)
            S=np.sqrt(B0*B0+4*A*C);r=np.ones_like(A);active=A>0
            pos=active&(B0>=0)&(S+B0>0);neg=active&(B0<0)
            r[active]=0;r[pos]=2*C[pos]/(S[pos]+B0[pos]);r[neg]=(-B0[neg]+S[neg])/(2*A[neg])
            roots.append(np.clip(r,0,1))
        alpha=np.concatenate(roots,-1).min(-1);alpha=np.where(alpha<1,alpha*(1-2.**-20),alpha)
    expected=proposed*alpha[...,None,None]
    err=float(abs(expected-z['residual']).max());alphaerr=float(abs(alpha-z['projection_alpha']).max())
    assert err<=2e-12 and alphaerr<=2e-10,(err,alphaerr)
    numeric=[];zero_constraints=0
    if arm=='physical':
        scale=np.abs(native)+np.abs(z['refined4']);unit=np.finfo(float).eps/2
        for order,(base,path) in enumerate(zip(derivatives(native),derivatives(z['refined4'])),1):
            prev=np.concatenate((np.zeros_like(scale[...,:1,:]),scale[...,:-1,:]),-2) if order==1 else scale[...,:-1,:]
            scale=(scale+prev)/.4 if order==1 else (scale[...,1:,:]+prev)/.4
            bn=np.linalg.norm(base,axis=-1);pn=np.linalg.norm(path,axis=-1)
            ops=8*order+16;gamma=ops*unit/(1-ops*unit);certificate=gamma*np.linalg.norm(scale,axis=-1)
            excess=pn-1.01*bn;assert np.all(excess<=certificate),('SUBSTANTIVE_EXCESS',order)
            numeric.append(dict(order=order,raw_strict_excess_count=int((excess>0).sum()),max_raw_excess=float(np.maximum(excess,0).max()),max_excess_to_certificate=float(np.max(np.divide(np.maximum(excess,0),certificate,out=np.zeros_like(excess),where=certificate>0)))))
            zero_constraints+=int((bn==0).sum())
    residual=z['residual'];d=derivatives(residual)
    return dict(status='PASS',reconstruction_max_error=err,alpha_reconstruction_max_error=alphaerr,alpha_quantiles=np.quantile(alpha,[0,.1,.5,.9,1]),alpha_zero_fraction=float((alpha==0).mean()),alpha_below_1e_6_fraction=float((alpha<1e-6).mean()),alpha_mean=float(alpha.mean()),raw_pointwise_excess=numeric,zero_native_constraints=zero_constraints,certificate_not_literal_pointwise_PASS=True,full_path_motion_separately_literal=True,residual_only_speed_mean=float(np.linalg.norm(d[0],axis=-1).mean()),residual_only_accel_mean=float(np.linalg.norm(d[1],axis=-1).mean()),residual_only_jerk_mean=float(np.linalg.norm(d[2],axis=-1).mean()),residual_norm_quantiles=np.quantile(np.linalg.norm(residual,axis=-1),[0,.5,.95,1]),proposed_norm_quantiles=np.quantile(np.linalg.norm(proposed,axis=-1),[0,.5,.95,1]))

def fresh_control_repeat_check():
    keys=('sample_id','groups','track','scene','gt','native20','refined20','refined4','residual','slots','reference4');rows=[]
    prior=H.parents[2]/'GENERATION_NATIVE_ANCHORED_BERNSTEIN_SERIES_20261008'
    for seed in SEEDS:
        for split in ('outer','inner','original_CAL'):
            guard();new=npz(H/'runs/unprojected'/f'seed_{seed}/{split}.npz');oldpath=prior/'runs/bernstein12'/f'seed_{seed}/{split}.npz';old=npz(oldpath)
            assert all(np.array_equal(new[k],old[k]) for k in keys)
            rows.append(dict(seed=seed,split=split,status='BITWISE_EXACT',fields=list(keys),old_output_sha=sha(oldpath)))
    return dict(status='PASS',jobs=9,records=rows,fresh_repeat_not_independent_generalization=True,no_model_forward=True)

def audit_job(z,arm):
    temporal=temporal_check(z,arm);N=len(z['sample_id']);assert len(np.unique(z['sample_id']))==N
    for port in PORTS:assert z[port].shape==(N,20,12,2) and np.isfinite(z[port]).all()
    slots=z['slots'];assert slots.shape==(N,4) and all(len(set(x))==4 for x in slots)
    rows=np.arange(N)[:,None];keep=np.ones((N,20),bool);keep[rows,slots]=False
    assert np.array_equal(z['refined20'][keep],z['native20'][keep]);assert np.array_equal(z['zero_native20'][keep],z['native20'][keep])
    assert np.array_equal(z['refined20'][rows,slots],z['refined4']);assert np.array_equal(z['reference4']+z['residual'],z['refined4'])
    baseline=z['native20'];assert np.array_equal(z['reference4'],baseline[rows,slots])
    bound=.20*(np.arange(1,13)/12.)**2;norm=np.sqrt(np.sum(z['residual']**2,-1));assert (norm<=bound[None,None]+2e-15).all() and (norm.mean(-1)<=bound.mean()+2e-15).all()
    raw=[error(z[p],z['gt']) for p in PORTS];nativegood=raw[0][3].any(1);owngood=raw[0][3].any(1);met={};vec=[];cc=[]
    for port,p,(a,f,j,g,one) in zip(PORTS,[z[x] for x in PORTS],raw):
        cl=clusters(p,g);cc.append(cl);good=g.any(1);vec.append(np.stack([good,a.min(1),f.min(1),j.min(1)],1));met[port]={}
        for name,mask in masks(z['groups']):
            n=int(mask.sum())
            if not n:met[port][name]={'n':0};continue
            mot=motion(p[mask]);base=motion(z['native20'][mask])
            met[port][name]=dict(n=n,good=int(good[mask].sum()),coverage=float(good[mask].mean()),minADE=float(a[mask].min(1).mean()),minFDE=float(f[mask].min(1).mean()),J=float(j[mask].min(1).mean()),rescued=int((good&~nativegood&mask).sum()),lost=int((~good&nativegood&mask).sum()),retained=int((good&nativegood&mask).sum()),own_zero_rescued=int((good&~owngood&mask).sum()),own_zero_lost=int((~good&owngood&mask).sum()),effective_distinct=float(cl[mask,0].mean()),effective_good=float(cl[mask,1].mean()),sensitivity_03_05_good=int(((a<=.3)&(f<=.5)).any(1)[mask].sum()),single_Jclosest_early_middle_late=one[mask].mean(0),motion=mot,native20_motion=base,motion_all5_1pct=all(mot[k]<=1.01*base[k] for k in mot))
    a,f,_,_,_=error(z['refined4'],z['gt']);ass=assignment(a,f,z['groups'])
    return met,ass,np.stack(vec,1),np.stack(cc,1),dict(max_endpoint_residual=float(norm[:,:,-1].max()),max_meanpath_residual=float(norm.mean(-1).max()),temporal=temporal)
def boot(saved):
    rng=np.random.default_rng(20261008);out=[]
    for split in ('outer','inner','original_CAL'):
        data={a:[saved[(a,s,split)] for s in SEEDS] for a in ARMS};z=data['unprojected'][0]
        for seq in data.values():
            for item in seq:assert all(np.array_equal(z[k],item[k]) for k in ('sample_id','groups','track'))
        _,inv=np.unique(z['track'],return_inverse=True);T=inv.max()+1
        for group,mask in masks(z['groups']):
            counts=np.bincount(inv[mask],minlength=T);active=np.flatnonzero(counts)
            for arm,ref in [('unprojected','native20'),('physical','native20'),('physical','unprojected')]:
                diff=np.stack([data[arm][i]['vectors'][:,2]-(data['unprojected'][i]['vectors'][:,2] if ref=='unprojected' else data[arm][i]['vectors'][:,0]) for i in range(3)]);sums=np.zeros((3,T,4))
                for i in range(3):np.add.at(sums[i],inv[mask],diff[i,mask])
                draws=[]
                for rep in range(2000):
                    if rep%64==0:guard()
                    ss=rng.integers(0,3,3);tt=rng.choice(active,len(active),replace=True);draws.append(sums[ss][:,tt].sum((0,1))/(3*counts[tt].sum()))
                out.append(dict(split=split,group=group,arm=arm,reference=ref,n=int(mask.sum()),tracks=len(active),metrics=['coverage','ADE','FDE','J'],estimate=diff[:,mask].mean((0,1)),CI95=np.quantile(draws,[.025,.975],axis=0).T,exploratory=True,paired_modelseeds=True))
    return out
def main():
    start=time.perf_counter();guard();result=read(H/'RESULT.json');assert result['status']=='PASS_NATIVE_ANCHORED_PHYSICAL_RAY_PAIRED_RUN';assert read(H/'worker_state.json')['status']==read(H/'exit_status.json')['status']=='SUCCESS'
    source=read(H/'source_manifest.json')['files'];verify(source);verify(result['output_sha']);saved={};records=[];err=0.;rows=0;historyrows=0;paired={};failures=[]
    cachePath=next(Path(k) for k in source if k.endswith('results\\cache\\train_dev\\eth_train.npz'))
    with np.load(cachePath,allow_pickle=False) as c:
        native=c['candidate'][...,:24].reshape(-1,20,12,2).astype(float);ids=c['sample_id'];logs=c['logits'];groups=c['change_group']
    lookup={int(x):i for i,x in enumerate(ids)};proto=read(H/'protocol.json')
    for arm in ARMS:
        for seed in SEEDS:
            guard();run=H/'runs'/arm/f'seed_{seed}';man=read(run/'run_manifest.json');done=read(run/'done.json');hist=read(run/'history.json')
            assert done['status']=='PASS' and done['run_manifest_sha']==digest(man);verify(done['files']);assert man['source_files']==source and man['source_sha']==digest(source);assert man['arm']==arm and man['seed']==seed and man['parameter_count']==25304
            assert all(man[k]==value for k,value in proto.items());assert sha(run/'statistics.npz')==man['statistics_sha'] and sha(run/'split_ids.npz')==man['split_ids_sha']
            assert len(hist)==25 and all(x['epoch']==i+1 and x['n']==18909 for i,x in enumerate(hist));assert all(np.isfinite([x['total'],x['quality'],x['diversity']]).all() for x in hist);assert all(sum(x['winner_counts'])==18909 for x in hist);historyrows+=25
            paired[(arm,seed)]=man['initial_parameters_sha'];metrics=read(run/'metrics.json');ass=read(run/'assignment_stats.json');splits=npz(run/'split_ids.npz')
            assert len(splits['train_ids'])==18909 and len(np.intersect1d(splits['train_ids'],splits['outer_ids']))==0
            for split,N in [('outer',1035),('inner',1856),('original_CAL',2879)]:
                z=npz(run/f'{split}.npz');expected=splits[{'outer':'outer_ids','inner':'inner_ids','original_CAL':'original_CAL_ids'}[split]]
                assert len(z['sample_id'])==N and np.array_equal(expected,z['sample_id']);ix=np.asarray([lookup[int(x)] for x in z['sample_id']]);assert np.array_equal(z['native20'],native[ix]) and np.array_equal(z['groups'],groups[ix])
                slots=np.asarray([sorted(sorted(range(20),key=lambda k:(-float(logs[i,k]),k))[16:]) for i in ix]);assert np.array_equal(slots,z['slots'])
                met,assignment_,vectors,cluster,bounds=audit_job(z,arm);error=max(compare(met,metrics['splits'][split]),compare(assignment_,ass['final_posthoc'][split]));aux=npz(run/f'{split}_metrics.npz');assert np.array_equal(vectors,aux['vectors']) and np.array_equal(cluster,aux['cluster_counts']);assert error<=1e-12,error;err=max(err,error)
                saved[(arm,seed,split)]=dict(vectors=vectors,**{k:z[k] for k in ('sample_id','groups','track')});rows+=N
                for group,value in met['refined20'].items():
                    bad={k:100*(value['motion'][k]/value['native20_motion'][k]-1) for k in value['motion'] if value['motion'][k]>value['native20_motion'][k]*1.01}
                    if bad:failures.append(dict(arm=arm,seed=seed,split=split,group=group,percent_above_native=bad))
                records.append(dict(arm=arm,seed=seed,split=split,metrics=met,bounds=bounds,independent_error=error));print('AUDITED',arm,seed,split,error,flush=True)
    assert historyrows==150 and len(records)==18 and all(paired[('unprojected',s)]==paired[('physical',s)] for s in SEEDS)
    ci=boot(saved);cierr=compare(ci,read(H/'BOOTSTRAP.json')['records']);assert cierr<=1e-12 and len(ci)==36
    repeat=fresh_control_repeat_check();verify(source);verify(result['output_sha']);guard();write('CONTROL_REPEAT.json',repeat)
    write('SUMMARY.json',dict(status='PASS',records=records,motion_failures=failures,bootstrap=ci,resource=resource))
    write('REVIEW.json',dict(status='PASS',jobs=18,replay_rows=rows,not_unique_independent_rows=True,full_original_SciPy_GTfree_complete_link=True,all_saved_Bernstein_controls_basis_physical_alpha_and_roundoff_reconstructed=True,all_saved_metrics_clusters_bounds_fixed16_motion_assignment_recomputed=True,max_error=err,bootstrap_records=36,scalar_CIs=144,bootstrap_max_error=cierr,checkpoint_history_rows=150,six_completed_manifests_and_output_SHA_verified_not_checkpoint_model_replay=True,paired_initial_parameter_SHAs_exact=True,source_count=len(source),output_SHA_count=len(result['output_sha']),source_output_SHA_initial_final_PASS=True,no_torch_model_forward_backward_optimizer=True,not_external_review=True,resource=resource,elapsed_seconds=time.perf_counter()-start))
    print('REVIEW_PASS',err,cierr,rows,len(failures),resource,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        write('FAILURE_'+str(time.time_ns())+'.json',dict(status='STOP',traceback=traceback.format_exc(),resource=resource));raise


