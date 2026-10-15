"""Frozen existing-checkpoint replay. Only new-directory outputs are written."""
import os,sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from datetime import datetime
import ast,json,hashlib,time,ctypes,gc,traceback
import numpy as np,torch,psutil
from torch import nn
from torch.nn import functional as F
from scipy.cluster.hierarchy import linkage,fcluster
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
NEW=ROOT/'CALIBRATED_TOP3_RESEARCH_20261007';SEL=ROOT/'S1L3_SELECTION_RESEARCH_20261007'
CONF=ROOT/'AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007'
PROGRAM=Path('C:/Users/linke/Downloads/F16_MOTION_PROGRAM_LONG_RUN_20261007/src')
DEVICE=torch.device('cuda:0')
SEEDS=(42,43,44)
PROTOCOLS=[('scene_family',ROOT/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/identity_audit/splits.npz',CONF/'baseline_reconciliation/canonical_intent_classifier.pth',ROOT/'AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007/models'),('target_track',ROOT/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz',CONF/'target_track/models/isolated_intent_classifier.pth',CONF/'target_track/models'),('recording_time',ROOT/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time/identity_audit/splits.npz',CONF/'recording_time/models/isolated_intent_classifier.pth',CONF/'recording_time/models')]
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
RESOURCE=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
def write(name,obj):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8');os.replace(q,p)
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));p=psutil.Process().memory_info()
    for key,val in [('peak_private_GiB',p.peak_pagefile/2**30),('peak_RSS_GiB',p.peak_wset/2**30)]:RESOURCE[key]=max(RESOURCE[key],val)
    for key,val in [('min_physical_GiB',m.avail_phys/2**30),('min_commit_GiB',m.avail_page/2**30)]:RESOURCE[key]=min(RESOURCE[key],val)
    RESOURCE['checks']+=1
    if max(p.rss,p.private)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:
        write('RESOURCE_USAGE.json',dict(status='STOP_RESOURCE',**RESOURCE));raise MemoryError('Process6GiB/systemreserve6GiB')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def verify(d):
    for i,(p,h) in enumerate(d.items()):
        if i%64==0:guard()
        assert sha(p)==h,('SOURCE_OR_OUTPUT_MISMATCH',p)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def stage(name,**kw):write('progress.json',dict(stage=name,at=datetime.now().astimezone().isoformat(),**kw));print(name,kw,flush=True)
def definitions(path,names,ns):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
def setup():
    ns=dict(np=np,torch=torch,nn=nn,F=F,Path=Path,CLASS_NAMES=('continue','decelerate','stop','accelerate','left','right','reverse'),RAW_TRAIN=Path('C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy/processed_data_diverse/eth_data_train.npy'))
    definitions(ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/directional_regime_pilot.py',['IntentClassifier','rotate_to_focal_heading','raw_focal_trajectories'],ns)
    definitions(NEW/'M1_probability_calibration/run_m1_calibration.py',['BestOf4Generator','CandidateAwareScorer','Top3ProbabilityCalibrator','extract_features','compute_ece'],ns)
    definitions(PROGRAM/'motion_program.py',['_sigmoid','generate_motion_program','default_params'],ns)
    definitions(PROGRAM/'oracle_optimizer.py',['_profile'],ns)
    return ns
def manifest(ns):
    files=set([HERE/'audit.py',HERE/'DESIGN.md',ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/directional_regime_pilot.py',PROGRAM/'motion_program.py',PROGRAM/'oracle_optimizer.py',ns['RAW_TRAIN']])
    for tree in (NEW,SEL,CONF,ROOT/'AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007'):
        files.update(p for p in tree.rglob('*') if p.is_file() and p.suffix in ('.py','.pth','.json','.md') and '__pycache__' not in p.parts)
    files.update(p for p in HERE.glob('*') if p.suffix in ('.py','.ps1','.md'))
    files.update([ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz',ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz',ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz'])
    files.update(p[1] for p in PROTOCOLS)
    return {str(p):sha(p) for p in sorted(files)}
def frozen(cls,path):
    m=cls.to(DEVICE);ck=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(ck.get('state_dict',ck));m.eval();m.requires_grad_(False)
    return m,{k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def exact(m,before):assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in m.state_dict().items())
def infer(m,arrays):
    pieces=[]
    with torch.inference_mode():
        for i in range(0,len(arrays[0]),128):
            guard();out=m(*[torch.from_numpy(x[i:i+128]).to(DEVICE) for x in arrays]);out=out[-1] if isinstance(out,tuple) else out;pieces.append(out.cpu().numpy())
    return np.concatenate(pieces)
def norm(x,axes=0):
    mu=x.mean(axis=axes,keepdims=True);sd=x.std(axis=axes,keepdims=True);sd=np.where(sd<1e-6,1.,sd).astype(np.float32);return mu,sd
def apply(x,s):return ((x-s[0])/s[1]).astype(np.float32)
def entropy(query,reference,kinds,selfoffset=False):
    out=[];ref=torch.from_numpy(reference).to(DEVICE)
    with torch.inference_mode():
        for i in range(0,len(query),128):
            guard();d=torch.cdist(torch.from_numpy(query[i:i+128]).to(DEVICE),ref)
            # Reproduce the original M1 rule, which drops nearest for training.
            ix=d.topk(21 if selfoffset else 20,largest=False).indices.cpu().numpy();ix=ix[:,1:] if selfoffset else ix
            for row in ix:
                p=np.bincount(kinds[row],minlength=5)/20.;p=p[p>0];out.append(float(-(p*np.log2(p)).sum()))
    return np.asarray(out,np.float32)
def make_static(ns,past,prob):
    classes=['decelerate','stop','accelerate','mild_left','mild_right','reverse'];labels=np.argsort(-prob[:,1:7],axis=1)[:,0]
    result=np.zeros((len(past),12,2),np.float32)
    for c in range(6):
        mask=labels==c
        if mask.any():result[mask]=ns['generate_motion_program'](past[mask],ns['default_params'](classes[c],int(mask.sum())))
    return result
def generate(ns,model,p2,past):
    pieces=[]
    with torch.inference_mode():
        for i in range(0,len(p2),128):
            guard();x=torch.from_numpy(p2[i:i+128]).to(DEVICE);p=torch.from_numpy(past[i:i+128]).to(DEVICE);r=model(x)
            out,_=ns['_profile'](p.repeat_interleave(4,dim=0),r.reshape(-1,8),'general');pieces.append(out.reshape(-1,4,12,2).cpu().numpy())
    return np.concatenate(pieces)
def r2score(model,p2,cand,logs,stats,ns):
    cf,sf=ns['extract_features'](cand);pieces=[]
    with torch.inference_mode():
        for i in range(0,len(p2),128):
            guard();values=[p2[i:i+128],apply(cf[i:i+128],stats[0]),apply(sf[i:i+128],stats[1]),logs[i:i+128]]
            out,_=model(*[torch.from_numpy(v).to(DEVICE) for v in values]);pieces.append(out.cpu().numpy())
    return np.concatenate(pieces)
def features(logs,scores,probs,ent,trajs):
    n=logs.argmax(1);row=np.arange(len(logs));sort=np.sort(logs,axis=1)[:,::-1];a=scores[row,n];b=scores[:,20];c=scores[:,21]
    return np.stack([logs[row,n],sort[:,0]-sort[:,1],a,b,c,c-a,b-a,(n!=scores.argmax(1)).astype(np.float32),probs[:,1:7].sum(1),ent,np.linalg.norm(trajs[:,0]-trajs[:,1],axis=-1).mean(1),np.linalg.norm(trajs[:,0]-trajs[:,2],axis=-1).mean(1)],1)
def errors(paths,gt):
    e=np.linalg.norm(paths.astype(np.float64)-gt[:,None].astype(np.float64),axis=-1);a=e.mean(-1);f=e[:,:,-1];return a,f,np.maximum(a/.2,f/.4),(a<=.2)&(f<=.4)
def motion(paths):
    v=np.diff(paths.astype(np.float64),axis=2,prepend=np.zeros_like(paths[:,:,:1]))/.4;s=np.linalg.norm(v,axis=-1);a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4
    return dict(speed_p95=float(np.quantile(s,.95)),accel_mean=float(np.linalg.norm(a,axis=-1).mean()),accel_p95=float(np.quantile(np.linalg.norm(a,axis=-1),.95)),jerk_mean=float(np.linalg.norm(j,axis=-1).mean()),jerk_p95=float(np.quantile(np.linalg.norm(j,axis=-1),.95)))
def metrics(z,groups,slots):
    paths=z['paths'][:,slots];a,f,j,g=errors(paths,z['gt']);native=z['paths'][:,:20];ref=motion(native);out=[]
    for name,mask in [('overall',np.ones(len(groups),bool)),('stable',groups==0),('transition',groups==1),('change',groups==2)]:
        if not mask.any():continue
        mot=motion(paths[mask]);base=motion(native[mask]);out.append(dict(group=name,n=int(mask.sum()),good=int(g[mask].any(1).sum()),coverage=float(g[mask].any(1).mean()),minADE=float(a[mask].min(1).mean()),minFDE=float(f[mask].min(1).mean()),J=float(j[mask].min(1).mean()),motion=mot,native20_motion=base,motion_all5_1pct=all(mot[k]<=base[k]*1.01 for k in mot)))
    return out
def clusters(paths,good):
    ans=[]
    for p,g in zip(paths,good):
        guard();K=len(p);d=np.linalg.norm(p[:,None]-p[None,:],axis=-1);D=np.maximum(d.mean(-1)/.1,d[:,:,-1]/.2);ix=np.triu_indices(K,1);lab=fcluster(linkage(D[ix],method='complete'),1.,criterion='distance') if K>1 else np.ones(1,int);ans.append((len(set(lab)),len(set(lab[g]))))
    return np.asarray(ans,np.int32)
def evaluation(z):
    groups=z['groups'];native=z['native_rank'];N=len(groups);row=np.arange(N)
    ports={'native1':native[:,:1],'native3':native[:,:3],'native20':np.tile(np.arange(20),(N,1)),'C1':np.c_[native[:,0],np.full(N,20),np.full(N,21)],'C2':np.c_[native[:,0],np.full(N,21),np.full(N,22)],'C3':np.c_[native[:,0],np.full(N,21),np.full(N,23)],'full24':np.tile(np.arange(24),(N,1))}
    result={'portfolios':{},'preprocessing':{}}
    a,f,j,g=errors(z['paths'],z['gt'])
    for name,ix in ports.items():
        p=z['paths'][row[:,None],ix];local=dict(z,paths=np.concatenate([z['paths'][:,:20],p],1));rec=metrics(local,groups,list(range(20,20+ix.shape[1])));cl=clusters(p,g[row[:,None],ix]);result['portfolios'][name]=dict(groups=rec,effective_distinct_mean=float(cl[:,0].mean()),effective_good_mean=float(cl[:,1].mean()))
    ix=ports['C1'];aa=a[row[:,None],ix];ff=f[row[:,None],ix];jj=j[row[:,None],ix];gg=g[row[:,None],ix];utility=-2*jj+3*gg;target=np.exp(utility-utility.max(1,keepdims=True));target/=target.sum(1,keepdims=True);unif=np.full_like(target,1/3)
    nativegood=g[row,native[:,0]];setgood=gg.any(1)
    for mode in ('A','B','C'):
        p=z['prob_'+mode].astype(np.float64);pred=p.argmax(1);conf=p.max(1);truth=gg[row,pred].astype(float);ece=0.;bins=[]
        for low in np.arange(10)/10:
            mask=(conf>low)&(conf<=low+.1)
            if mask.any():ece+=mask.mean()*abs(conf[mask].mean()-truth[mask].mean());bins.append(dict(n=int(mask.sum()),confidence=float(conf[mask].mean()),good_rate=float(truth[mask].mean())))
        tier=z['tier_'+mode];ret=tier!=3;success=np.where(tier==1,nativegood,np.where(tier==2,setgood,False));selade=aa[row,pred]
        result['preprocessing'][mode]=dict(utility_crossentropy=float(-(target*np.log(np.clip(p,1e-12,1))).sum(1).mean()),utility_uniform_crossentropy=float(-(target*np.log(unif)).sum(1).mean()),utility_squared_error=float(((p-target)**2).sum(1).mean()),utility_uniform_squared_error=float(((unif-target)**2).sum(1).mean()),selected_good_ECE=float(ece),selected_good_Brier=float(((conf-truth)**2).mean()),selected_good_NLL=float(-(truth*np.log(np.clip(conf,1e-12,1))+(1-truth)*np.log(np.clip(1-conf,1e-12,1))).mean()),bins=bins,probability_weighted_ADE=float((p*aa).sum(1).mean()),probability_weighted_FDE=float((p*ff).sum(1).mean()),native_top1_ADE=float(aa[:,0].mean()),argmax_ADE=float(selade.mean()),all_window_success=float(success.mean()),retained_fraction=float(ret.mean()),retained_success=float(success[ret].mean()) if ret.any() else None,native1_same_retained_good=float(nativegood[ret].mean()) if ret.any() else None,native3_same_retained_good=float(g[row[:,None],ports['native3']].any(1)[ret].mean()) if ret.any() else None,expected_hypotheses=float(np.where(tier==1,1,np.where(tier==2,3,0)).mean()),tier_counts={str(k):int((tier==k).sum()) for k in (1,2,3)})
    return result
def job(ns,data,protocol,seed,smoke=False):
    tag,split,clfp,modeldir=protocol;stage('FROZEN_REPLAY',protocol=tag,seed=seed,smoke=smoke);t0=time.perf_counter();sp=load(split);tr=sp['grouped_train_index'];ot=sp['outer_eval_index'];ot=ot[:128] if smoke else ot
    ids=data['ids'];assert len(np.unique(ids[tr]))==len(tr) and not np.intersect1d(ids[tr],ids[ot]).size and max(np.r_[tr,ot])<25912
    p2=data['p2'];ps=norm(p2[tr]);pt=apply(p2[tr],ps);po=apply(p2[ot],ps);n=min(5000,len(tr))
    ck=torch.load(clfp,map_location='cpu',weights_only=False);clf,cb=frozen(ns['IntentClassifier'](data['obs'].shape[1],.1),clfp)
    probtr=infer(clf,[apply(data['obs'][tr],(ck['mean'],ck['std']))]);probot=infer(clf,[apply(data['obs'][ot],(ck['mean'],ck['std']))]);exact(clf,cb);del clf,cb,ck
    stattr=make_static(ns,data['past'][tr],probtr);statot=make_static(ns,data['past'][ot],probot)
    et=entropy(pt[:n],pt,data['kind'][tr],True);eo=entropy(po,pt,data['kind'][tr])
    bo,bb=frozen(ns['BestOf4Generator'](),modeldir/f'seed_{seed}/model.pth');bot=generate(ns,bo,pt,data['past'][tr]);boo=generate(ns,bo,po,data['past'][ot]);exact(bo,bb);del bo,bb
    kt=np.concatenate([data['native'][tr],stattr[:,None],bot[:,[3,0,1]]],1);ko=np.concatenate([data['native'][ot],statot[:,None],boo[:,[3,0,1]]],1)
    lt=data['logs'][tr];lo=data['logs'][ot]
    def prior(l):return np.c_[l,np.repeat(np.median(l,axis=1,keepdims=True)-l.std(1,keepdims=True),4,1)].astype(np.float32)
    ft,st=ns['extract_features'](kt);fo,so=ns['extract_features'](ko);statsfull=(norm(ft,(0,1)),norm(st));statssub=(norm(ft[:n],(0,1)),norm(st[:n]));statsouter=(norm(fo,(0,1)),norm(so))
    r2,rb=frozen(ns['CandidateAwareScorer'](),SEL/f'R2_candidate_aware_scorer/models/{tag}/seed_{seed}/scorer.pth')
    cal,mb=frozen(ns['Top3ProbabilityCalibrator'](),NEW/f'M1_probability_calibration/models/{tag}/seed_{seed}/calibrator.pth')
    ranks=np.argsort(-lo,axis=1);rt=lt.argmax(1);ro=lo.argmax(1);rowt=np.arange(n);rowo=np.arange(len(ot));t3t=np.stack([kt[rowt,rt[:n]],kt[:n,20],kt[:n,21]],1);t3o=np.stack([ko[rowo,ro],ko[:,20],ko[:,21]],1)
    result=dict(sample_id=ids[ot],train_ids=ids[tr],normalization_ids=ids[tr[:n]],paths=ko,gt=data['future'][ot],groups=data['groups'][ot],native_rank=ranks,entropy_outer=eo,entropy_train=et)
    for mode,stats in [('A',statsouter),('B',statssub),('C',statsfull)]:
        # A reproduces M3; B M1's subset preprocessing; C R2's trained contract.
        scores=r2score(r2,po,ko,prior(lo),stats,ns)
        xo=features(lo,scores,probot,eo,t3o)
        if mode=='A':xs=norm(xo);thresholds=np.quantile(eo,[.5,.9])
        else:
            ts=r2score(r2,pt[:n],kt[:n],prior(lt[:n]),stats,ns);xt=features(lt[:n],ts,probtr[:n],et,t3t);xs=norm(xt);thresholds=np.quantile(et,[.5,.9])
        p=infer(cal,[apply(xo,xs)]);assert np.isfinite(p).all() and np.max(abs(p.sum(1)-1))<1e-6
        tier=np.full(len(ot),2,np.int8);tier[(p[:,0]>=.6)&(eo<thresholds[0])]=1;tier[eo>=thresholds[1]]=3
        result['prob_'+mode]=p;result['tier_'+mode]=tier;result['r2_'+mode]=scores
        for prefix,s in [('candidate',stats[0]),('set',stats[1]),('calibrator',xs)]:result[mode+'_'+prefix+'_mean']=s[0];result[mode+'_'+prefix+'_std']=s[1]
        result[mode+'_entropy_thresholds']=thresholds
    exact(r2,rb);exact(cal,mb);del r2,rb,cal,mb,kt,ft,bot
    directory=HERE/('smoke' if smoke else 'saved');directory.mkdir(exist_ok=True);file=directory/f'{tag}_{seed}.npz';assert not file.exists(),'Do not overwrite completed outputs';np.savez_compressed(file,**result)
    saved=load(file);assert all(np.array_equal(v,saved[k]) for k,v in result.items());summary=evaluation(saved);stamp=dict(status='PASS_FROZEN_SAVED_REPLAY',protocol=tag,seed=seed,rows=len(ot),elapsed_seconds=time.perf_counter()-t0,output_SHA=sha(file),frozen_allparams_buffers_EXACT=True)
    write(str(directory.relative_to(HERE)/f'{tag}_{seed}_done.json'),dict(stamp,summary=summary));gc.collect();torch.cuda.empty_cache();return stamp,summary
def main(smoke=False):
    guard();ns=setup();assert torch.cuda.is_available();stage('INITIAL_RELEVANT_INPUT_SOURCE_SHA');locks=manifest(ns);write('smoke_manifest.json' if smoke else 'source_manifest.json',dict(files=locks,scope='all relevant sources/checkpoints/reports/inputs, not all historical files'))
    if not smoke:
        pre=json.loads((HERE/'PREFLIGHT.json').read_text(encoding='utf-8'));assert pre['status']=='PASS';verify(pre['files'])
    cache=load(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz');labels=load(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz');enc=load(ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz');ids=cache['sample_id'];assert np.array_equal(ids,labels['sample_id']) and np.array_equal(ids,enc['sample_id']) and len(np.unique(ids))==len(ids)
    past,_=ns['raw_focal_trajectories'](ids,cache['motion']);data=dict(ids=ids,past=past,p2=enc['tokens'].mean(1).astype(np.float32),obs=np.concatenate([cache['motion'],cache['social']],1).astype(np.float32),kind=labels['kind'],future=labels['future'],native=cache['candidate'][...,:24].reshape(-1,20,12,2).astype(np.float32),logs=cache['logits'].astype(np.float32),groups=cache['change_group'])
    del cache,labels,enc
    # Core samepath and duplication fixtures.
    q=np.zeros((1,20,12,2));assert clusters(q,np.ones((1,20),bool))[0].tolist()==[1,1]
    stamps=[];summaries={};plan=[(PROTOCOLS[0],42)] if smoke else [(p,s) for p in PROTOCOLS for s in SEEDS]
    for p,s in plan:
        guard();stamp,summary=job(ns,data,p,s,smoke);stamps.append(stamp);summaries[p[0]+'_'+str(s)]=summary;stage('JOB_DONE',done=len(stamps),total=len(plan))
    verify(locks);guard()
    if smoke:write('PREFLIGHT.json',dict(status='PASS',files=locks,job=stamps[0],estimated9job_minutes=stamps[0]['elapsed_seconds']*9/60,resource=RESOURCE,no_optimizer=True,not_production_guarantee=True))
    else:write('RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_PORTFOLIO_CALIBRATION_REPLAY',jobs=stamps,summaries=summaries,sources_unchanged=True,resource=RESOURCE,no_training=True,formal_count_unchanged=True));write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=len(stamps),saved_arrays_exact=True,all_metrics_recomputed_from_saved=True))
    write('RESOURCE_USAGE.json',dict(status='PASS',**RESOURCE));stage('SUCCESS',jobs=len(stamps))
if __name__=='__main__':
    try:main('--smoke' in sys.argv)
    except Exception:
        write('FAILURE_'+str(time.time_ns())+'.json',dict(status='STOP',traceback=traceback.format_exc(),resource=RESOURCE));raise
