"""Fixed GT-free native mixtures; offline saved-data audit only, no torch/model."""
import os,sys,ast,json,math,time,hashlib,ctypes,traceback,gc
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from datetime import datetime
import numpy as np,psutil
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent
P=H.parent.parent/'GENERATION_NATIVE_SIMILARITY_FEASIBLE_SERIES_20261008'
SPLITS=('outer','inner','original_CAL');GROUPS=('overall','stable','transition','change')
resource=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));x=psutil.Process().memory_info();resource['checks']+=1
    resource['peak_private_GiB']=max(resource['peak_private_GiB'],x.peak_pagefile/2**30);resource['peak_RSS_GiB']=max(resource['peak_RSS_GiB'],x.peak_wset/2**30)
    resource['min_physical_GiB']=min(resource['min_physical_GiB'],m.avail_phys/2**30);resource['min_commit_GiB']=min(resource['min_commit_GiB'],m.avail_page/2**30)
    if max(x.private,x.rss)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:raise MemoryError('STOP_RESOURCE_process6_physical_and_commit_eachreserve6')
def now():return datetime.now().astimezone().isoformat()
def write(name,value):
    p=H/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8');os.replace(tmp,p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify(files):
    for i,(p,v) in enumerate(files.items()):
        if i%32==0:guard()
        assert sha(p)==v,('SOURCE_OR_OUTPUT_SHA_CHANGED',p)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def stage(name,**kw):write('progress.json',dict(stage=name,at=now(),**kw))
def ast_functions(path,names,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'));nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert {x.name for x in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace);return namespace
RAY=ast_functions(P/'physical_ray.py',('differences_numpy','project_numpy','audit_numpy'),dict(np=np,MARGIN=1.01,NUMERICAL_INTERIOR=1.-2.**-20))
INDEP=ast_functions(P/'completion_review/audit_saved.py',('error','motion','clusters'),dict(np=np,guard=guard,linkage=linkage,fcluster=fcluster))
def inputs(split):
    guard();path=P/f'runs/ray/seed_42/{split}.npz';z=npz(path)
    keep=('sample_id','groups','track','scene','gt','native20','slots','reference4');z={k:z[k] for k in keep};N=len(z['sample_id'])
    assert N==dict(outer=1035,inner=1856,original_CAL=2879)[split] and len(np.unique(z['sample_id']))==N
    assert z['native20'].shape==(N,20,12,2) and np.isfinite(z['native20']).all() and np.array_equal(z['reference4'],z['native20'][np.arange(N)[:,None],z['slots']])
    return z
def identity_check():
    rows=[]
    for split in SPLITS:
        base=inputs(split)
        for arm in ('ray','similarity'):
            for seed in (42,43,44):
                guard();z=npz(P/f'runs/{arm}/seed_{seed}/{split}.npz')
                assert all(np.array_equal(base[k],z[k]) for k in base),('ID_NATIVE_GROUNDTRUTH_SLOT_MISMATCH',arm,seed,split)
                rows.append(dict(split=split,arm=arm,seed=seed,status='EXACT'));del z
    return rows
def derivatives(p):
    v=np.concatenate((p[...,:1,:],p[...,1:,:]-p[...,:-1,:]),axis=-2)/.4
    a=(v[...,1:,:]-v[...,:-1,:])/.4;j=(a[...,1:,:]-a[...,:-1,:])/.4;return v,a,j
def roots_independent(native,residual):
    out=[]
    for b,d in zip(derivatives(native),derivatives(residual)):
        aa=(d*d).sum(-1);bb=2*(b*d).sum(-1);budget=(1.01**2-1)*(b*b).sum(-1);s=np.sqrt(bb*bb+4*aa*budget)
        root=np.ones_like(aa);act=aa>0;pos=act&(bb>=0)&(s+bb>0);neg=act&(bb<0);root[act]=0.;root[pos]=2*budget[pos]/(s[pos]+bb[pos]);root[neg]=(-bb[neg]+s[neg])/(2*aa[neg]);out.append(np.clip(root,0,1))
    alpha=np.concatenate(out,-1).min(-1);return np.where(alpha<1,alpha*(1-2.**-20),alpha)
def bounded_proposal(native,slots,partner):
    N=len(native);anchor=native[np.arange(N)[:,None],slots];raw=.5*(native[:,partner,None]-anchor);norm=np.sqrt(np.sum(raw*raw,-1));bound=.20*(np.arange(1,13)/12.)**2
    ratio=np.divide(bound,norm,out=np.ones_like(norm),where=norm>0);lam=np.minimum(1,ratio.min(-1));lam=np.where(lam<1,lam*(1-2.**-20),lam)
    return anchor,raw*lam[...,None,None],lam
def generation(native,slots,partner):
    # Contract intentionally exposes NO groundtruth, group, IDs, context or logits.
    anchor,proposal,lam=bounded_proposal(native,slots,partner);res,alpha=RAY['project_numpy'](anchor,proposal);out=anchor+res
    return dict(refined4=out,residual=res,position_lambda=lam,alpha=alpha)
def whole(native,slots,refined4):
    p=native.copy();p[np.arange(len(p))[:,None],slots]=refined4
    mask=np.ones(p.shape[:2],bool);mask[np.arange(len(p))[:,None],slots]=False;assert np.array_equal(p[mask],native[mask]);return p
def errors(p,gt):
    e=np.sqrt(np.sum((p-gt[:,None].astype(float))**2,-1));a=e.mean(-1);f=e[...,-1];j=np.maximum(a/.2,f/.4);good=(a<=.2)&(f<=.4);one=e[np.arange(len(p)),j.argmin(1)].reshape(-1,3,4).mean(-1);return a,f,j,good,one
def mot(p):
    v,a,j=derivatives(p);v=np.sqrt((v*v).sum(-1));a=np.sqrt((a*a).sum(-1));j=np.sqrt((j*j).sum(-1));return dict(speed_p95=float(np.quantile(v,.95)),accel_mean=float(a.mean()),accel_p95=float(np.quantile(a,.95)),jerk_mean=float(j.mean()),jerk_p95=float(np.quantile(j,.95)))
def clus(p,good):
    out=[];ix=np.triu_indices(20,1)
    for i,(x,g) in enumerate(zip(p,good)):
        if i%32==0:guard()
        # Full matrix versus prior auditor's per-anchor condensed-distance construction.
        e=np.sqrt(np.sum((x[:,None]-x[None,:])**2,-1));d=np.maximum(e.mean(-1)/.1,e[...,-1]/.2)[ix];lab=fcluster(linkage(d,method='complete'),1.,criterion='distance');out.append((len(np.unique(lab)),len(np.unique(lab[g]))))
    return np.asarray(out,dtype=np.int32)
def masks(g):return [(k,np.ones(len(g),bool) if i==0 else g==i-1) for i,k in enumerate(GROUPS)]
def shape_diagnostic(anchor,res):
    E=np.sum(anchor*anchor,(-1,-2));den=np.where(E>0,E,1);a=np.sum(anchor*res,(-1,-2))/den;b=np.sum(anchor[...,0]*res[...,1]-anchor[...,1]*res[...,0],-1)/den
    rot=np.stack((-anchor[...,1],anchor[...,0]),-1);fit=a[...,None,None]*anchor+b[...,None,None]*rot;rms=np.sqrt(np.mean(np.sum((res-fit)**2,-1),-1));return rms
def metric(z,p,ind=False,native_stats=None):
    err=INDEP['error'](p,z['gt']) if ind else errors(p,z['gt']);A,F,J,g,one=err;clusters=INDEP['clusters'](p,g) if ind else clus(p,g);ng=errors(z['native20'],z['gt'])[3].any(1);gg=g.any(1);out={}
    for group,mask in masks(z['groups']):
        motion=INDEP['motion'](p[mask]) if ind else mot(p[mask]);native_motion=INDEP['motion'](z['native20'][mask]) if ind else mot(z['native20'][mask])
        out[group]=dict(n=int(mask.sum()),good=int(gg[mask].sum()),coverage=float(gg[mask].mean()),minADE=float(A[mask].min(-1).mean()),minFDE=float(F[mask].min(-1).mean()),J=float(J[mask].min(-1).mean()),rescued=int((gg&~ng&mask).sum()),lost=int((~gg&ng&mask).sum()),retained=int((gg&ng&mask).sum()),effective_distinct=float(clusters[mask,0].mean()),effective_good=float(clusters[mask,1].mean()),motion=motion,native_motion=native_motion,all5_literal_PASS=all(motion[k]<=1.01*native_motion[k] for k in motion),single_Jclosest_early_middle_late=one[mask].mean(0),sensitivity03_05_good=int(((A<=.3)&(F<=.5)).any(-1)[mask].sum()))
    return out,dict(vectors=np.stack((gg,A.min(-1),F.min(-1),J.min(-1)),1),cluster_counts=clusters,candidate_good=g)
def compare(a,b):
    if isinstance(a,dict):assert a.keys()==b.keys();return max([compare(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,(str,bool)):assert a==b;return 0.
    x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float);assert x.shape==y.shape;return float(np.max(abs(x-y))) if x.size else 0.
def physical_review(z,j,g):
    anchor,proposal,lam=bounded_proposal(z['native20'],z['slots'],j);alpha=roots_independent(anchor,proposal);err=max(float(np.max(abs(alpha-g['alpha']))),float(np.max(abs(lam-g['position_lambda']))),float(np.max(abs(anchor+proposal*alpha[...,None,None]-g['refined4']))))
    assert err<=2e-12,('GEOMETRY_REBUILD',err);audit=RAY['audit_numpy'](anchor,g['refined4']);bound=.20*(np.arange(1,13)/12)**2
    assert np.all(np.linalg.norm(g['residual'],axis=-1)<=bound+2e-15);return err,audit
def locks():
    old=read(H.parent/'source_manifest.json')['files'];files=dict(old)
    add=[*H.glob('*.py'),*H.glob('*.ps1'),H/'README.md',H/'DESIGN.md',H/'protocol.json',P/'RESULT_ANALYSIS.md',P/'RESULT_ANALYSIS.json',P/'completion_review/REVIEW.json',P/'completion_review/SUMMARY.json',P/'completion_review/CONTROL_REPEAT.json',P/'source_manifest.json']
    add.extend(Path(k) for k in read(P/'RESULT.json')['output_sha'])
    add.extend(H.parent/x for x in ('source_manifest.json','PREFLIGHT.json','startup_check.json','exit_status.json','worker_state.json','progress.json','failures/production_save_field_collision/README.md'))
    for path in sorted(set(add)):
        key=str(path);value=sha(path)
        if key in files:assert files[key]==value,('INHERITED_SHA_CHANGED',key)
        else:files[key]=value
    return files,len(old)
def done_valid(d,source_sha):
    x=read(d/'done.json');assert x['source_sha']==source_sha;verify(x['files']);return x
def job(split,j,z,source_sha):
    d=H/'jobs'/split/f'partner_{j:02d}';d.mkdir(parents=True,exist_ok=True)
    if (d/'done.json').exists():done_valid(d,source_sha);return read(d/'metrics.json')
    guard();g=generation(z['native20'],z['slots'],j);p=whole(z['native20'],z['slots'],g['refined4']);m,x=metric(z,p)
    # Groundtruth first used here, after construction; never used for alpha or partner.
    physical_error,audit=physical_review(z,j,g);rms=shape_diagnostic(z['reference4'],g['residual']);cgood=errors(g['refined4'],z['gt'])[3]
    path=d/'paths.npz';assert not path.exists(),'Partial output retained; explicit recovery required'
    np.savez_compressed(path,**g,sample_id=z['sample_id'],partner=j,slots=z['slots'],shape_RMS_beyond_global_similarity=rms,generated_candidate_good=cgood,**x)
    saved=npz(path);assert all(np.array_equal(saved[k],v) for k,v in g.items());p2=whole(z['native20'],z['slots'],saved['refined4']);im,ix=metric(z,p2,True);err=compare(m,im);assert err<=1e-12 and all(np.array_equal(ix[k],x[k]) for k in x)
    physical_error2,audit2=physical_review(z,j,saved);assert compare(audit,audit2)==0 and physical_error2<=2e-12
    for group,mask in masks(z['groups']):
        m[group]['candidate_nonzero_ge1mm_fraction']=float((np.linalg.norm(g['residual'][mask],axis=-1).max(-1)>=.001).mean());m[group]['candidate_temporal_shape_ge1mm_fraction']=float((rms[mask]>=.001).mean());m[group]['alpha_zero_fraction']=float((g['alpha'][mask]==0).mean());m[group]['alpha_median']=float(np.median(g['alpha'][mask]));m[group]['residual_norm_quantiles']=np.quantile(np.linalg.norm(g['residual'][mask],axis=-1),[0,.5,.95,1]);m[group]['rescued_ids']=z['sample_id'][mask&(x['vectors'][:,0]>0)&~errors(z['native20'],z['gt'])[3].any(1)].tolist();m[group]['lost_ids']=z['sample_id'][mask&~(x['vectors'][:,0]>0)&errors(z['native20'],z['gt'])[3].any(1)].tolist()
    result=dict(split=split,partner=j,metrics=m,independent_metrics_clusters_motion_max_error=err,geometry_max_error=physical_error,pointwise_roundoff_not_scientific_epsilon=audit)
    write(f'jobs/{split}/partner_{j:02d}/metrics.json',result);files={str(path):sha(path),str(d/'metrics.json'):sha(d/'metrics.json')};write(f'jobs/{split}/partner_{j:02d}/done.json',dict(status='PASS',source_sha=source_sha,files=files));del p,p2,saved,g,x,ix;gc.collect();return result
def main():
    guard();pre=read(H/'PREFLIGHT.json');assert pre['status']=='PASS';lock=read(H/'source_manifest.json');assert digest(lock['files'])==pre['source_sha'];files=lock['files'];stage('INITIAL_SOURCE_INPUT_SHA',done=0,total=60);verify(files);records=[];witness=[]
    for split in SPLITS:
        z=inputs(split);union=errors(z['native20'],z['gt'])[3].any(1);ng=union.copy();shapeunion=np.zeros(len(union),bool)
        for j in range(20):
            stage('FIXED_K20_POLICY_SAVED_ARITHMETIC',split=split,partner=j,done=len(records),total=60);records.append(job(split,j,z,digest(files)));saved=npz(H/f'jobs/{split}/partner_{j:02d}/paths.npz');cg=saved['generated_candidate_good'];union|=cg.any(1);shapeunion|=(cg&(saved['shape_RMS_beyond_global_similarity']>=.001)).any(1)
        for group,mask in masks(z['groups']):witness.append(dict(split=split,group=group,n=int(mask.sum()),native_good=int((ng&mask).sum()),finite_bank_union_good=int((union&mask).sum()),finite_bank_rescued_ids=z['sample_id'][mask&union&~ng].tolist(),beyond_global_similarity_bank_rescued_ids=z['sample_id'][mask&shapeunion&~ng].tolist(),not_K20_deployment_or_major_gain=True,not_universal_reachability_bound=True))
        del z,saved;gc.collect()
    stage('FINAL_SOURCE_INPUT_OUTPUT_SHA',done=60,total=60);verify(files);outsha={}
    for split in SPLITS:
        for j in range(20):outsha.update(done_valid(H/f'jobs/{split}/partner_{j:02d}',digest(files))['files'])
    assert len(records)==60 and len(outsha)==120;verify(outsha);write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=60,all_saved_samepath_full_original_SciPy_GTfree_K20_clusters_five_motion_geometry_rebuilt=True,no_model_forward_backward_optimizer=True,max_error=max(r['independent_metrics_clusters_motion_max_error'] for r in records)))
    write('RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_NATIVE_TEMPORAL_MIX_REACHABILITY',records=records,bank_witness=witness,source_sha=digest(files),output_sha=outsha,not_K20_union_gain=True,formal_count_increment=0));write('RESOURCE_USAGE.json',resource);write('PRESERVATION.json',dict(status='PASS',source_count=len(files),output_count=len(outsha),initial_final_SHA_unchanged=True));stage('SUCCESS',done=60,total=60)
if __name__=='__main__':main()
