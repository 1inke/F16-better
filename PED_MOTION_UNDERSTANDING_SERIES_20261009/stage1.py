"""Frozen five-fold, three-seed motion diagnostics; all writes isolated here."""
import os, sys
sys.dont_write_bytecode = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
from pathlib import Path
import json, hashlib, importlib.util, time, ctypes, gc, traceback
import numpy as np
import psutil
from scipy.stats import spearmanr
H = Path(__file__).resolve().parent
R = H.parent
A = Path('C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20')
D = A/'datasets/eth_ucy/processed_data_diverse'
PROTOCOL = json.loads((H/'protocol.json').read_text(encoding='utf-8'))
FOLDS = PROTOCOL['folds']
SEEDS = PROTOCOL['model_seeds']
RESOURCE = dict(peak_private_GiB=0., peak_RSS_GiB=0., min_physical_GiB=100., min_commit_GiB=100., checks=0)

def write(name, obj):
    p=H/name; p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8'); os.replace(tmp,p)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def save(name, **arr):
    p=H/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arr);os.replace(tmp,p)
def stage(name, **kw):
    write('progress.json',dict(stage=name,at=time.time(),**kw));print(name,kw,flush=True)
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));p=psutil.Process().memory_info()
    RESOURCE['peak_private_GiB']=max(RESOURCE['peak_private_GiB'],p.peak_pagefile/2**30);RESOURCE['peak_RSS_GiB']=max(RESOURCE['peak_RSS_GiB'],p.peak_wset/2**30)
    RESOURCE['min_physical_GiB']=min(RESOURCE['min_physical_GiB'],m.avail_phys/2**30);RESOURCE['min_commit_GiB']=min(RESOURCE['min_commit_GiB'],m.avail_page/2**30);RESOURCE['checks']+=1
    if max(p.private,p.rss)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:
        write('RESOURCE_USAGE.json',dict(status='STOP_RESOURCE',**RESOURCE));raise MemoryError('Process6/physical+commit reserve6GiB')
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%32==0: guard()
        assert sha(p)==h,('SOURCE_MISMATCH',p)
def module(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def model_paths(fold,seed,stage_name='v2'):
    root=(R/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER' if seed==1 else R/'OFFICIAL_ETH_UCY_FINAL_3SEED_SERIES/EXP_164_FINAL_V2_OFFICIAL_3SEED')
    return root/'results'/fold/stage_name/'f16_temporal_social_cross'/f'seed_{seed}'/'val_scene_-1'
def inputs():
    paths=[Path(PROTOCOL['source_plan'])]
    paths+=list(H.glob('*.py'))+list(H.glob('*.ps1'))+list(H.glob('*.md'))+[H/'protocol.json']
    for fold in FOLDS:
        paths += [D/f'{fold}_{kind}_{split}.npy' for split in ('train','test') for kind in ('data','num')]
        paths += [R/f'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/{fold}_train.npz']
        paths += list((A/f'datasets/eth_ucy/labels/{fold}').glob('*.txt'))
        for seed in SEEDS:
            paths += [model_paths(fold,seed,s)/'best_selected.pth' for s in ('h3','v2')]
            paths += [model_paths(fold,seed)/'metrics.json']
    paths += [R/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/run_formal_regime_selection.py',R/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/motif_rbf_official_4p8.yaml',R/'FRONT_ENCODER_SERIES/front_encoder_models.py',R/'FRONT_ENCODER_SERIES/train_front_variant.py',R/'E2_F16_END_TO_END_COMPARISON_SERIES/joint_v2_adapter.py',R/'CONTROLLED_ANCHOR_CORRECTION_SERIES/controlled_anchor_correction_models.py']
    return paths
def load_raw(fold,split,ix=None):
    guard();raw=np.load(D/f'{fold}_data_{split}.npy',mmap_mode='r');counts=np.load(D/f'{fold}_num_{split}.npy',mmap_mode='r')
    if ix is None: ix=np.arange(len(raw))
    assert raw.shape[1:]==(9,20,6)
    pos=np.array(raw[ix,:,:,-2:],dtype=np.float32);valid=np.minimum(np.array(counts[ix],dtype=np.int64),9)
    assert np.isfinite(pos).all() and (valid>=1).all() and (valid<=9).all()
    return dict(ids=np.array(ix,dtype=np.int64),obs=pos[:,:,:8].copy(),gt=(pos[:,0,8:]-pos[:,0,7,None]).astype(np.float64),position=pos[:,0,7].copy(),valid=valid)
def heading(obs):
    v=obs[:,0,-1]-obs[:,0,-2];s=np.linalg.norm(v,axis=1);u=np.zeros_like(v,dtype=np.float64);u[:,0]=1.;ok=s>=.02;u[ok]=v[ok]/s[ok,None];return u,ok,s
def cv(obs,k=1):
    v=(obs[:,0,-1]-obs[:,0,-1-k])/k;return v[:,None].astype(np.float64)*np.arange(1,13)[None,:,None]
def ca(obs,cap):
    v=np.diff(obs[:,0].astype(np.float64),axis=1);a=v[:,-1]-v[:,-2];a*=np.minimum(1.,cap/np.maximum(np.linalg.norm(a,axis=1),1e-300))[:,None]
    velocity=v[:,-1,None]+np.cumsum(.8**np.arange(1,13)[None,:,None]*a[:,None],axis=1);return np.cumsum(velocity,axis=1)
def kalman(obs,q,r):
    y=obs[:,0].astype(np.float64);T=np.array([[1.,1.],[0.,1.]]);Q=q*np.array([[.25,.5],[.5,1.]])
    x=np.stack((y[:,0],y[:,1]-y[:,0]),axis=-1);P=np.tile(np.eye(2)[None],(len(y),1,1))
    for t in range(1,8):
        x=x@T.T;P=T@P@T.T+Q;S=P[:,0,0]+r;K=P[:,:,0]/S[:,None];x+=(y[:,t]-x[:,:,0])[:,:,None]*K[:,None,:];P-=K[:,:,None]*P[:,None,0,:]
    return x[:,:,0][:,None]+x[:,:,1][:,None]*np.arange(1,13)[None,:,None]-y[:,-1,None]
def errors(p,gt):
    e=np.linalg.norm(p-gt[:,None] if p.ndim==4 else p-gt,axis=-1);return e.mean(-1),e[...,-1]
def bins(obs,valid,edges):
    s=np.linalg.norm(obs[:,0,-1]-obs[:,0,-2],axis=1);return np.searchsorted(edges,s,side='right')*4+np.searchsorted([1,2,4],valid-1,side='right')
def cues(obs,valid):
    v=np.diff(obs.astype(np.float64),axis=2);speed=np.linalg.norm(v[:,0],axis=-1);first=speed[:,:3].mean(1);last=speed[:,-3:].mean(1)
    ratio=np.divide(last,first,out=np.ones_like(last),where=first>0);c1=(ratio<.75)|(ratio>1.33)|((first==0)&(last>0))
    a=v[:,0,:3].mean(1);b=v[:,0,-3:].mean(1);norm=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);angle=np.degrees(np.arctan2(a[:,0]*b[:,1]-a[:,1]*b[:,0],(a*b).sum(1)))
    c2=(np.linalg.norm(a,axis=1)>=.02)&(np.linalg.norm(b,axis=1)>=.02)&(abs(angle)>15)
    pred=obs[:,:,-1,None]+v[:,:,-1,None]*np.arange(1,13)[None,None,:,None];dist=np.linalg.norm(pred[:,1:]-pred[:,:1],axis=-1);mask=np.arange(1,9)[None]<valid[:,None];dca=np.where(mask[:,:,None],dist,np.inf).min((1,2));c3=dca<1.
    return c1,c2,c3
def calibrate(fold):
    path=H/'calibration'/f'{fold}.json'
    if path.exists():
        done=read(path.with_suffix('.done.json'));assert sha(path)==done['sha'];assert sha(R/f'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/{fold}_train.npz')==done['train_ids_source_sha'];return read(path)
    cache=npz(R/f'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/{fold}_train.npz');ix=cache['sample_id'].astype(int);assert len(np.unique(ix))==len(ix);data=load_raw(fold,'train',ix);obs=data['obs'];gt=data['gt'];last=cv(obs);_,F=errors(last,gt)
    v=np.diff(obs[:,0].astype(np.float64),axis=1);cap=float(np.quantile(np.linalg.norm(v[:,-1]-v[:,-2],axis=1),.95));edges=np.quantile(np.linalg.norm(v[:,-1],axis=1),[.25,.5,.75])
    trials=[]
    for q in (.001,.01,.1):
        for r in (.01,.1,1.):
            aa,ff=errors(kalman(obs,q,r),gt);trials.append(dict(q=q,r=r,score=float((aa+.5*ff).mean())))
    best=min(trials,key=lambda z:z['score']);baselines=dict(CV_last=last,CV_avg3=cv(obs,3),CA=ca(obs,cap),Kalman=kalman(obs,best['q'],best['r']));scores={k:float(sum(w*e for w,e in zip((1.,.5),errors(p,gt))).mean()) for k,p in baselines.items()}
    residual=gt-last;delta=float(np.quantile(np.linalg.norm(residual,axis=-1).mean(1),.75));u,ok,speed=heading(obs);lateral=np.stack((-u[:,1],u[:,0]),1);bank=np.stack(((residual*u[:,None]).sum(-1),(residual*lateral[:,None]).sum(-1)),axis=-1)
    speedbins=np.searchsorted(edges,speed,side='right');speedbins[~ok]=4
    # Stationary rows may exhaust a low-speed quartile: an unused bin is valid.
    assert len(speedbins)==len(ix) and np.isin(speedbins,np.arange(5)).all()
    save(f'calibration/{fold}_bank.npz',sample_id=ix,residual_heading=bank,speed_bin=speedbins)
    rec=dict(fold=fold,n_train=len(ix),id_sha=hashlib.sha256(ix.tobytes()).hexdigest(),cv_fde_quantiles=np.quantile(F,[.5,.75,.9]).tolist(),speed_edges=edges.tolist(),ca_cap=cap,kalman=best,kalman_train_grid=trials,strongest_simple=min(scores,key=scores.get),train_baseline_scores=scores,delta_nonCV_candidate=delta,all_train_only=True)
    write(f'calibration/{fold}.json',rec);write(f'calibration/{fold}.done.json',dict(sha=sha(path),bank_sha=sha(H/f'calibration/{fold}_bank.npz'),train_ids_source_sha=sha(R/f'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/{fold}_train.npz')));return rec
def fingerprint(x):return np.round(x.astype(np.float32),4).tobytes()
def metadata(fold,data):
    # Exact raw-coordinate/source-label match; no track inference from window index.
    lookup={};raw=np.load(D/f'{fold}_data_test.npy',mmap_mode='r');wanted={fingerprint(x) for x in np.array(raw[:,0,:,-2:],dtype=np.float32)}
    for path in sorted((A/f'datasets/eth_ucy/labels/{fold}').glob('*.txt')):
        if path.stem.endswith(('_train','_val')):continue
        t=np.loadtxt(path,dtype=str);frames=t[:,0].astype(float).astype(int);ped=t[:,1].astype(float).astype(int);xy=t[:,[13,15]].astype(np.float32)
        for person in np.unique(ped):
            ix=np.flatnonzero(ped==person);ix=ix[np.argsort(frames[ix])];f=frames[ix];p=xy[ix]
            for j in range(len(ix)-19):
                if not np.all(np.diff(f[j:j+20])==1):continue
                k=fingerprint(p[j:j+20])
                if k in wanted:lookup.setdefault(k,[]).append((path.stem,int(person),int(f[j])))
    tracks=[];starts=[];recordings=[];amb=0
    for x in raw[:,0,:,-2:]:
        candidates=lookup.get(fingerprint(x),[]);identities={(s,p) for s,p,f in candidates}
        assert len(identities)==1,('STOP_IDENTITY',fold,len(identities),len(tracks))
        s,p=next(iter(identities));tracks.append(s+'::'+str(p));recordings.append(s);times={f for s,p,f in candidates};starts.append(next(iter(times)) if len(times)==1 else -1);amb+=len(times)!=1
    return dict(track=np.array(tracks),frame=np.array(starts),recording=np.array(recordings),ambiguous_time_n=amb)
def build(fold,seed):
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    loader=module('motion_understanding_frozen_backend',R/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/run_formal_regime_selection.py')
    cfg,_,front,trainer,adapter=loader.setup_backend(torch.device('cuda:0'));loader.fold_run=lambda f,stage='v2':model_paths(f,seed,stage)
    model=loader.build_baseline(fold,cfg,front,adapter,torch.device('cuda:0'));model.device='cuda:0'
    assert not model.training and not any(p.requires_grad for p in model.parameters());assert float(cfg.PREDICTION.TRAJ_SCALE)==1.
    return model,trainer,cfg
def forward(model,trainer,obs,valid):
    import torch
    past=torch.from_numpy(np.array(obs,dtype=np.float32,copy=True)).cuda();nv=torch.from_numpy(np.array(valid,dtype=np.int64,copy=True)).cuda();local,_=trainer.normalize_to_focal_heading(past,past);center=local.mean(-2,keepdim=True)
    with torch.no_grad():pred=model(local-center,None,None,mode='testing',num_valid=nv,graph_loc=local)[2];logs=model.last_mode_logits[:,0].clone()
    rel=pred[:,0]+center[:,0,None]-local[:,0,-1,None,None]
    d=past[:,0,-1]-past[:,0,-4];speed=torch.linalg.vector_norm(d,dim=-1);unit=torch.where((speed>.03)[:,None],d/speed[:,None].clamp_min(1e-6),torch.tensor([1.,0.],device='cuda'))
    world=torch.stack((rel[...,0]*unit[:,None,None,0]-rel[...,1]*unit[:,None,None,1],rel[...,0]*unit[:,None,None,1]+rel[...,1]*unit[:,None,None,0]),-1)
    p=world.cpu().numpy().astype(np.float64);lg=logs.cpu().numpy();assert p.shape==(len(obs),20,12,2) and lg.shape==(len(obs),20) and np.isfinite(p).all() and np.isfinite(lg).all();return p,lg
def snapshot(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def exact(model,old):
    import torch
    assert set(old)==set(model.state_dict());assert all(torch.equal(old[k],v.detach().cpu()) for k,v in model.state_dict().items());assert all(p.grad is None and not p.requires_grad for p in model.parameters())
def masks(data,cal):
    last=cv(data['obs']);_,F=errors(last,data['gt']);c1,c2,c3=cues(data['obs'],data['valid']);cue=c1|c2|c3;p50,p75,p90=cal['cv_fde_quantiles'];b=bins(data['obs'],data['valid'],cal['speed_edges']);non=F>p75;success=F<=p50;matched=np.zeros(len(F),bool);deficits={}
    for k in np.unique(b):
        target=int(((b==k)&non&cue).sum());pool=np.flatnonzero((b==k)&success);pool=sorted(pool,key=lambda i:hashlib.sha256(f'20261009:{int(data["ids"][i])}'.encode()).hexdigest());matched[pool[:target]]=True;deficits[str(k)]=max(0,target-len(pool))
    result=dict(all=np.ones(len(F),bool),nonCV=non,strong_nonCV=F>p90,nonCV_cue_positive=non&cue,nonCV_cue_negative=non&~cue,cue_positive=cue,cue_negative=~cue,CV_success=success,matched_CV_success=matched,stationary=np.linalg.norm(data['obs'][:,0,-1]-data['obs'][:,0,-2],axis=1)<.02)
    return result,b,deficits
def bootstrap_ratio(nums,dens,track,seed=20261009):
    uniq,inv=np.unique(track,return_inverse=True);num=np.bincount(inv,weights=nums);den=np.bincount(inv,weights=dens);rng=np.random.default_rng(seed);values=[]
    for _ in range(PROTOCOL['bootstrap']):
        ix=rng.integers(0,len(uniq),len(uniq));d=den[ix].sum()
        if d>0:values.append(float(1-num[ix].sum()/d))
    return np.quantile(values,[.025,.5,.975]).tolist() if values else None
def paired_ci(values,track):
    uniq,inv=np.unique(track,return_inverse=True);a=np.bincount(inv,weights=values);n=np.bincount(inv);rng=np.random.default_rng(PROTOCOL['stat_seed']);out=[]
    for _ in range(PROTOCOL['bootstrap']):
        ix=rng.integers(0,len(uniq),len(uniq));out.append(a[ix].sum()/n[ix].sum())
    return np.quantile(out,[.025,.5,.975]).tolist()
def permute(rm,rg,b):
    den=(rg*rg).sum()
    if den==0:return None
    groups=[np.flatnonzero(b==k) for k in np.unique(b)];rng=np.random.default_rng(PROTOCOL['stat_seed']);out=[]
    for _ in range(PROTOCOL['permutations']):
        ix=np.arange(len(rm))
        for group in groups:ix[group]=rng.permutation(group)
        out.append(float(1-((rg-rm[ix])**2).sum()/den))
    return np.array(out)
def correlation(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    r,p=spearmanr(x,y);return dict(rho=float(r),p_unadjusted_exploratory=float(p),n=len(x))
def candidate_summary(p,gt,cvpath,delta):
    aa,ff=errors(p,gt);J=np.maximum(aa/.2,ff/.4).min(1);deviation=np.linalg.norm(p-cvpath[:,None],axis=-1).mean(-1)>delta
    return dict(minADE=aa.min(1),minFDE=ff.min(1),J=J,coverage=(J<=1).astype(float),nonCV_hit=((aa<=.2)&(ff<=.4)&deviation).any(1).astype(float),nonCV_candidate_count=deviation.sum(1),spread=np.linalg.norm(p[:,:,-1]-p[:,:,-1].mean(1,keepdims=True),axis=-1).mean(1),**{f'coverage_{a:.2f}':((aa<=a)&(ff<=2*a)).any(1).astype(float) for a in (.1,.15,.2,.3,.4)})
def summarize(data,cal,paths,logs):
    group,b,deficits=masks(data,cal);last=cv(data['obs']);u,defined,_=heading(data['obs']);normal=np.stack((-u[:,1],u[:,0]),1);gt=data['gt'];rg=gt-last;den=(rg*rg).sum((1,2));base=dict(CV_last=last,CV_avg3=cv(data['obs'],3),CA=ca(data['obs'],cal['ca_cap']),Kalman=kalman(data['obs'],cal['kalman']['q'],cal['kalman']['r']));strong=base[cal['strongest_simple']];ba,bf=errors(strong,gt)
    reads=dict(top1=paths[np.arange(len(paths)),logs.argmax(1)],meanK=paths.mean(1));records=[];nulls={};vectors=dict(sample_id=data['ids'],track=data['track'],bin=b,GT=gt,CV=last,model_paths=paths,model_logits=logs)
    for name,p in {**base,**reads}.items():
        aa,ff=errors(p,gt);rm=p-last;num=((gt-p)**2).sum((1,2));dots=(rm*rg).sum((1,2));norm=np.sqrt((rm*rm).sum((1,2))*den);cos=np.divide(dots,norm,out=np.full(len(p),np.nan),where=norm>0);vectors[name+'_ADE']=aa;vectors[name+'_FDE']=ff
        for g,m in group.items():
            n=int(m.sum());rec=dict(method=name,subset=g,n=n,tracks=len(np.unique(data['track'][m])),ADE=float(aa[m].mean()) if n else None,FDE=float(ff[m].mean()) if n else None,skill=float(1-num[m].sum()/den[m].sum()) if den[m].sum()>0 else None,cos_mean=float(np.nanmean(cos[m])) if np.isfinite(cos[m]).any() else None,cos_defined_n=int(np.isfinite(cos[m]).sum()))
            if n:
                rec['ADE_delta_strongest_simple']=float((aa-ba)[m].mean());rec['FDE_delta_strongest_simple']=float((ff-bf)[m].mean())
            if name in reads and n:
                rec['skill_track_CI']=bootstrap_ratio(num[m],den[m],data['track'][m]);rec['ADE_delta_track_CI']=paired_ci((aa-ba)[m],data['track'][m]);rec['FDE_delta_track_CI']=paired_ci((ff-bf)[m],data['track'][m]);null=permute(rm[m],rg[m],b[m]);key=name+'__'+g
                if null is not None:
                    nulls[key]=null;rec['null95']=float(np.quantile(null,.95));rec['p_permutation_plus1']=float((1+(null>=rec['skill']).sum())/(len(null)+1));rec['permutation_window_conditional_not_track_independent']=True
                for axis,vec in [('longitudinal',u),('lateral',normal)]:
                    gs=(rg[:,-1]*vec).sum(1);ps=(rm[:,-1]*vec).sum(1);ok=m&defined&(abs(gs)>1e-12);count=int(ok.sum());rec[axis+'_sign_n']=count;rec[axis+'_sign_accuracy']=float((np.sign(gs[ok])==np.sign(ps[ok])).mean()) if count else None;rec[axis+'_GT_majority_baseline']=float(max((gs[ok]>0).mean(),(gs[ok]<0).mean())) if count else None
            records.append(rec)
    candidates=candidate_summary(paths,gt,last,cal['delta_nonCV_candidate']);vectors.update({'candidate_'+k:v for k,v in candidates.items()});candidate_records=[]
    for g,m in group.items():
        vectors['mask_'+g]=m
        if not m.any():continue
        candidate_records.append(dict(subset=g,n=int(m.sum()),tracks=len(np.unique(data['track'][m])),**{k:float(v[m].mean()) for k,v in candidates.items()},spread_vs_CV_FDE=correlation(candidates['spread'][m],errors(last,gt)[1][m]),spread_vs_top1_FDE=correlation(candidates['spread'][m],vectors['top1_FDE'][m])))
    return dict(point_records=records,candidate_records=candidate_records,matching_deficits=deficits,strongest_simple=cal['strongest_simple']),vectors,nulls
def random_candidates(data,cal,fold,seed):
    bank=npz(H/f'calibration/{fold}_bank.npz');assert sha(H/f'calibration/{fold}_bank.npz')==read(H/f'calibration/{fold}.done.json')['bank_sha'];u,ok,s=heading(data['obs']);b=np.searchsorted(cal['speed_edges'],s,side='right');b[~ok]=4;rg=np.random.default_rng(seed);out=np.empty((len(s),20,12,2));normal=np.stack((-u[:,1],u[:,0]),1)
    for group in range(5):
        ix=np.flatnonzero(b==group)
        if not len(ix):continue
        pool=np.flatnonzero(bank['speed_bin']==group);assert len(pool)>0,('UNSUPPORTED_EMPIRICAL_CONTROL_BIN',group)
        sel=rg.choice(pool,size=(len(ix),20),replace=True);res=bank['residual_heading'][sel];out[ix]=cv(data['obs'])[ix,None]+res[...,0,None]*u[ix,None,None]+res[...,1,None]*normal[ix,None,None]
    return out
def job_valid(path,files):
    done=read(path);assert done['source_manifest_sha']==sha(H/'source_manifest.json');verify(done['output_sha']);return done
def main():
    t=time.perf_counter();guard();files=read(H/'source_manifest.json')['files'];stage('INITIAL_SOURCE_SHA');verify(files)
    for fold in FOLDS:calibrate(fold)
    verify(read(H/'PREFLIGHT.json')['calibration_sha']);records=[];jobs=[];frozen=[]
    for fold in FOLDS:
        data=load_raw(fold,'test');md=metadata(fold,data);data.update(md);cal=calibrate(fold);save(f'data/{fold}_evaluation.npz',**{k:v for k,v in data.items() if isinstance(v,np.ndarray)})
        write(f'data/{fold}_identity.json',dict(status='PASS',n=len(data['ids']),tracks=len(np.unique(data['track'])),ambiguous_time_n=md['ambiguous_time_n'],identity_source='full20 world XY rounded4 exact label fingerprint',never_window_ID_cluster=True))
        for seed in SEEDS:
            name=f'{fold}_seed{seed}';done=H/'jobs'/name/'done.json'
            if done.exists():job_valid(done,files);summary=read(done.parent/'summary.json');records.append(dict(fold=fold,seed=seed,**summary));jobs.append(str(done));continue
            stage('FROZEN_NATIVE_FORWARD',fold=fold,seed=seed,jobs_done=len(jobs));m,tr,cfg=build(fold,seed);old=snapshot(m);pp=[];ll=[]
            for start in range(0,len(data['ids']),128):
                guard();p,l=forward(m,tr,data['obs'][start:start+128],data['valid'][start:start+128]);pp.append(p);ll.append(l)
            paths=np.concatenate(pp);logs=np.concatenate(ll);exact(m,old);frozen.append(dict(fold=fold,seed=seed,status='ALL_PARAMETERS_BOOL_BUFFERS_EXACT'));del m,old;gc.collect()
            stage('RESIDUAL_BOOTSTRAP_PERMUTATION',fold=fold,seed=seed,jobs_done=len(jobs));summary,vectors,nulls=summarize(data,cal,paths,logs);save(f'jobs/{name}/outputs.npz',**vectors);save(f'jobs/{name}/nulls.npz',**nulls);write(f'jobs/{name}/summary.json',summary)
            outputs={str(done.parent/x):sha(done.parent/x) for x in ('outputs.npz','nulls.npz','summary.json')};write(f'jobs/{name}/done.json',dict(status='SUCCESS',fold=fold,seed=seed,checkpoint=str(model_paths(fold,seed)/'best_selected.pth'),checkpoint_sha=sha(model_paths(fold,seed)/'best_selected.pth'),source_manifest_sha=sha(H/'source_manifest.json'),output_sha=outputs,n=len(data['ids']),frozen_weights_buffers_EXACT=True));records.append(dict(fold=fold,seed=seed,**summary));jobs.append(str(done))
        # Equal-budget random controls are distinct RNG controls, not model repeats.
        group,_,_=masks(data,cal)
        for seed in PROTOCOL['random_control_seeds']:
            name=f'{fold}_random{seed}';done=H/'jobs'/name/'done.json'
            if done.exists():job_valid(done,files);summary=read(done.parent/'summary.json');records.append(dict(fold=fold,control_seed=seed,**summary));jobs.append(str(done));continue
            stage('EQUAL_K_CV_EMPIRICAL_CONTROL',fold=fold,seed=seed,jobs_done=len(jobs));p=random_candidates(data,cal,fold,seed);cs=candidate_summary(p,data['gt'],cv(data['obs']),cal['delta_nonCV_candidate']);summary=dict(candidate_records=[dict(subset=g,n=int(mask.sum()),tracks=len(np.unique(data['track'][mask])),**{k:float(v[mask].mean()) for k,v in cs.items()}) for g,mask in group.items() if mask.any()]);save(f'jobs/{name}/outputs.npz',sample_id=data['ids'],paths=p,**cs);write(f'jobs/{name}/summary.json',summary);write(f'jobs/{name}/done.json',dict(status='SUCCESS',source_manifest_sha=sha(H/'source_manifest.json'),output_sha={str(done.parent/x):sha(done.parent/x) for x in ('outputs.npz','summary.json')},n=len(data['ids'])));records.append(dict(fold=fold,control_seed=seed,**summary));jobs.append(str(done))
    stage('SAVED_OUTPUT_AUDIT',jobs_done=len(jobs));audit_saved(jobs);stage('FINAL_SOURCE_SHA');verify(files);verify(read(H/'PREFLIGHT.json')['calibration_sha']);write('FROZEN_AUDIT.json',dict(status='PASS',models=15,all_done_records_frozen_exact=True,records=frozen));write('RESOURCE_USAGE.json',dict(status='PASS',**RESOURCE));write('RESULT.json',dict(status='PASS_STAGE1_E0_E1_E2_E7',records=records,jobs=jobs,elapsed_seconds=time.perf_counter()-t,whole_series_complete=False,not_training=True));stage('SUCCESS',jobs_done=len(jobs),jobs_total=30)
def audit_saved(jobs):
    maximum=0.
    for path in jobs:
        guard();done=job_valid(Path(path),None);z=npz(Path(path).parent/'outputs.npz');summary=read(Path(path).parent/'summary.json')
        if 'model_paths' in z:
            gt=z['GT'];p=z['model_paths'];aa,ff=errors(p,gt);maximum=max(maximum,float(abs(aa.min(1)-z['candidate_minADE']).max()),float(abs(ff.min(1)-z['candidate_minFDE']).max()));assert np.array_equal(np.maximum(aa/.2,ff/.4).min(1)<=1,z['candidate_coverage'].astype(bool))
            for rec in summary['point_records']:
                if rec['method'] not in ('top1','meanK'):continue
                point=p[np.arange(len(p)),z['model_logits'].argmax(1)] if rec['method']=='top1' else p.mean(1);mask=z['mask_'+rec['subset']];den=((gt-z['CV'])[mask]**2).sum();num=((gt-point)[mask]**2).sum()
                if den>0:maximum=max(maximum,abs(float(1-num/den)-rec['skill']))
        else:
            fold=Path(path).parent.name.split('_random')[0];data=npz(H/f'data/{fold}_evaluation.npz');cs=candidate_summary(z['paths'],data['gt'],cv(data['obs']),calibrate(fold)['delta_nonCV_candidate']);maximum=max(maximum,max(float(abs(cs[k]-z[k]).max()) for k in cs))
    assert maximum<1e-10,maximum;write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=len(jobs),max_error=maximum,scope='full saved candidate metrics and top1/meanK skill identities; bootstrap/null full reproduction deferred completion review',no_second_model_forward=True))
if __name__=='__main__':main()
