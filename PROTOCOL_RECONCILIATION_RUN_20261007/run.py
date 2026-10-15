"""Human-approved 9-run canonical reconciliation; no legacy top-level execution."""
import sys,os
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from datetime import datetime
import ast,copy,json,hashlib,time,random,ctypes,gc,traceback
import numpy as np,torch,psutil
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from scipy.cluster.hierarchy import linkage,fcluster
H=Path(__file__).resolve().parent;R=H.parent;DEVICE=torch.device('cuda:0')
CANON=R/'AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007/run_best_of_4_3seed.py'
PILOT=R/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/directional_regime_pilot.py'
PROGRAM=Path('C:/Users/linke/Downloads/F16_MOTION_PROGRAM_LONG_RUN_20261007/src/oracle_optimizer.py')
RAW=Path('C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy/processed_data_diverse/eth_data_train.npy')
CONF=R/'AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007'
PLAN=[('scene_family',R/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/identity_audit/splits.npz',R/'AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007/models'),('target_track',R/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz',CONF/'target_track/models'),('recording_time',R/'GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment2_recording_time/identity_audit/splits.npz',CONF/'recording_time/models')]
SEEDS=(42,43,44)
INPUTS=[R/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz',R/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz',R/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz',R/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz',RAW]
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
RESOURCE=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
def write(path,obj):
    p=Path(path);p=p if p.is_absolute() else H/p;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8');os.replace(q,p)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def now():return datetime.now().astimezone().isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()).encode()).hexdigest()
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));p=psutil.Process().memory_info()
    RESOURCE['peak_private_GiB']=max(RESOURCE['peak_private_GiB'],p.peak_pagefile/2**30);RESOURCE['peak_RSS_GiB']=max(RESOURCE['peak_RSS_GiB'],p.peak_wset/2**30)
    RESOURCE['min_physical_GiB']=min(RESOURCE['min_physical_GiB'],m.avail_phys/2**30);RESOURCE['min_commit_GiB']=min(RESOURCE['min_commit_GiB'],m.avail_page/2**30);RESOURCE['checks']+=1
    if max(p.private,p.rss)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:
        write('RESOURCE_USAGE.json',dict(status='STOP_RESOURCE',**RESOURCE));raise MemoryError('Process6GiB/physical-and-commit-reserve6GiB')
def stage(name,**kw):write('progress.json',dict(stage=name,at=now(),**kw));print(name,kw,flush=True)
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%64==0:guard()
        assert sha(p)==h,('SOURCE_OR_INPUT_MISMATCH',p)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def definitions(path,names,ns):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
def setup():
    ns=dict(np=np,torch=torch,nn=nn,F=F,Path=Path,RAW_TRAIN=RAW)
    definitions(CANON,['BestOf4Generator'],ns);definitions(PILOT,['rotate_to_focal_heading','raw_focal_trajectories'],ns);definitions(PROGRAM,['_profile'],ns)
    # Extract the actual scene quality/diversity AST, not a hand-labelled replica.
    tree=ast.parse(CANON.read_text(encoding='utf-8-sig'));loop=next(n for n in tree.body if isinstance(n,ast.For) and ast.unparse(n.target)=='s')
    epochloop=next(n for n in loop.body if isinstance(n,ast.For) and ast.unparse(n.target)=='epoch');batchloop=next(n for n in epochloop.body if isinstance(n,ast.For));body=batchloop.body
    start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='diff');end=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='loss')
    fn=ast.parse('def reference_loss(trajs_4,b_gt):\n pass').body[0];fn.body=copy.deepcopy(body[start:end+1])+[ast.parse('return loss').body[0]]
    ns.update(device=DEVICE,DELTA_DIV=.3,LAMBDA_DIV=.05);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(CANON),'exec'),ns)
    assert not any('clip_grad' in ast.unparse(n) for n in body)
    return ns
def manifest():
    # Preserve the last completed replay's hashes, then extend; never replace old digests.
    prev=read(R/'F16_PORTFOLIO_CALIBRATION_REPLAY_AUDIT_20261007/calibration_contract_recovery/source_manifest.json')['files'];verify(prev)
    files=set([CANON,PILOT,PROGRAM,*INPUTS,*[x[1] for x in PLAN]])
    files.update(p for p in H.glob('*') if p.suffix in ('.py','.ps1','.md') or p.name=='protocol.json')
    for directory in (R/'AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007',CONF,R/'F16_K3_SINGLE_REPLACEMENT_AUDIT_20261007',R/'EXPERIMENT_PROTOCOL_RECONCILIATION_20261007'):
        files.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.pth','.pt','.npz','.json','.md'))
    locks=dict(prev)
    for p in sorted(files):
        key=str(p);actual=sha(p)
        if key in locks:assert locks[key]==actual,key
        else:locks[key]=actual
    return locks
def load_data(ns):
    guard();c=npz(INPUTS[0]);l=npz(INPUTS[1]);e=npz(INPUTS[2]);ids=c['sample_id'];assert np.array_equal(ids,l['sample_id']) and np.array_equal(ids,e['sample_id']) and len(np.unique(ids))==28791
    past,rawfuture=ns['raw_focal_trajectories'](ids,c['motion']);assert np.max(abs(rawfuture-l['future']))<5e-5
    data=dict(ids=ids,past=past,p2=e['tokens'].mean(1).astype(np.float32),gt=l['future'].astype(np.float32),native=c['candidate'][...,:24].reshape(-1,20,12,2).astype(np.float32),groups=c['change_group'],logs=c['logits'].astype(np.float32))
    md=npz(INPUTS[3]);lookup={int(x):i for i,x in enumerate(md['sample_id'])};order=np.array([lookup[int(x)] for x in ids]);data['track']=np.array([str(md['sequence'][i])+'::'+str(md['pedestrian_id'][i]) for i in order]);data['scene']=md['scene_family'][order]
    for k in ('past','p2','gt','native'):assert np.isfinite(data[k]).all(),k
    assert data['p2'].shape==(28791,96) and data['native'].shape==(28791,20,12,2);guard();return data
def stats(data,tr):
    x=data['p2'][tr];mu=x.mean(0,keepdims=True);sd=x.std(0,keepdims=True);sd=np.where(sd<1e-6,1.,sd).astype(np.float32);return mu,sd
def validate_split(data,p):
    sp=npz(p);assert np.array_equal(sp['sample_id'],data['ids'][:25912]);tr=sp['grouped_train_index'];ot=sp['outer_eval_index'];inner=sp['inner_cal_index']
    for ix in (tr,ot,inner):assert len(np.unique(ix))==len(ix) and ix.min()>=0 and ix.max()<25912
    assert not np.intersect1d(tr,ot).size and not np.intersect1d(tr,inner).size and not np.intersect1d(ot,inner).size
    for key,ix in [('grouped_train_ids',tr),('outer_eval_ids',ot),('inner_cal_ids',inner)]:assert np.array_equal(sp[key],data['ids'][ix])
    return sp
def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def tensor_hash(sd):
    h=hashlib.sha256()
    for k,v in sd.items():h.update(k.encode());h.update(str(v.dtype).encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def trajectory(ns,m,x,p):
    raw=m(x);out,_=ns['_profile'](p.repeat_interleave(4,0),raw.reshape(-1,8),'general');return out.reshape(-1,4,12,2)
def objective(paths,gt):
    err=torch.linalg.vector_norm(paths-gt[:,None],dim=-1);ade=err.mean(-1);fde=err[:,:,-1];q=ade+.5*fde;win=q.argmin(1);best=q.min(1).values.mean();div=torch.zeros((),device=paths.device,dtype=paths.dtype)
    for i in range(4):
        for j in range(i+1,4):div=div+(F.relu(.3-torch.linalg.vector_norm(paths[:,i]-paths[:,j],dim=-1).mean(-1))**2).mean()
    div=div/6;return best+.05*div,best,div,ade,fde,win
def assignment(ade,fde,groups):
    q=ade+.5*fde;j=np.maximum(ade/.2,fde/.4);a=q.argmin(1);b=j.argmin(1);qs=np.sort(q,1);js=np.sort(j,1);out={}
    for name,mask in [('overall',np.ones(len(a),bool)),('stable',groups==0),('transition',groups==1),('change',groups==2)]:
        n=int(mask.sum())
        if not n:out[name]={'n':0};continue
        cm=np.bincount(a[mask]*4+b[mask],minlength=16).reshape(4,4)
        out[name]=dict(n=n,switched=int((a[mask]!=b[mask]).sum()),switch_rate=float((a[mask]!=b[mask]).mean()),scene_counts=np.bincount(a[mask],minlength=4),track_counts=np.bincount(b[mask],minlength=4),confusion=cm,scene_exact_ties=int((qs[mask,1]==qs[mask,0]).sum()),track_exact_ties=int((js[mask,1]==js[mask,0]).sum()),scene_near_ties=int((qs[mask,1]-qs[mask,0]<=1e-6).sum()),track_near_ties=int((js[mask,1]-js[mask,0]<=1e-6).sum()),scene_gap_mean=float((qs[mask,1]-qs[mask,0]).mean()),track_gap_mean=float((js[mask,1]-js[mask,0]).mean()))
    return out,a,b
def loader(data,tr,st,s,batch=256):
    x=((data['p2'][tr]-st[0])/st[1]).astype(np.float32);g=torch.Generator().manual_seed(s)
    ds=TensorDataset(torch.from_numpy(x),torch.from_numpy(data['past'][tr]),torch.from_numpy(data['gt'][tr]),torch.from_numpy(data['groups'][tr].astype(np.int64)))
    return DataLoader(ds,batch_size=batch,shuffle=True,generator=g,num_workers=0,drop_last=False),g
def epoch(ns,m,opt,ldr,index):
    m.train();n=0;loss_sum=0.;primary_sum=0.;div_sum=0.;win_counts=np.zeros(4,np.int64);switch=0;nearq=0;nearj=0;cm=np.zeros((4,4),np.int64)
    for x,p,gt,group in ldr:
        guard();x,p,gt=x.to(DEVICE),p.to(DEVICE),gt.to(DEVICE);out=trajectory(ns,m,x,p);loss,primary,div,a,f,w=objective(out,gt)
        opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters());opt.step()
        N=len(x);n+=N;loss_sum+=float(loss)*N;primary_sum+=float(primary)*N;div_sum+=float(div)*N
        assign,_,_=assignment(a.detach().cpu().numpy(),f.detach().cpu().numpy(),group.numpy());z=assign['overall'];win_counts+=z['scene_counts'];switch+=z['switched'];nearq+=z['scene_near_ties'];nearj+=z['track_near_ties'];cm+=z['confusion']
    return dict(epoch=index,n=n,total=loss_sum/n,quality=primary_sum/n,diversity=div_sum/n,winner_counts=win_counts.tolist(),switch_count=switch,switch_rate=switch/n,scene_near_ties=nearq,track_near_ties=nearj,confusion=cm.tolist(),denominator='all samples once, pre-update batch output; population posthoc is different')
def rng(g):return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),loader=g.get_state())
def restore_rng(z,g):random.setstate(z['python']);np.random.set_state(z['numpy']);torch.set_rng_state(z['torch']);torch.cuda.set_rng_state_all(z['cuda']);g.set_state(z['loader'])
def save_checkpoint(p,m,opt,g,history,st,manifest):
    obj=dict(state_dict=m.state_dict(),optimizer=opt.state_dict(),rng=rng(g),epoch=len(history),history=history,statistics=dict(mean=torch.from_numpy(st[0]),std=torch.from_numpy(st[1])),run_manifest=manifest,run_manifest_sha=digest(manifest),parameter_names=list(dict(m.named_parameters())))
    temp=p.with_suffix('.tmp');torch.save(obj,temp);os.replace(temp,p)
def resume(p,m,opt,g,st,manifest):
    z=torch.load(p,map_location='cpu',weights_only=False);assert z['run_manifest_sha']==digest(manifest) and z['run_manifest']==manifest,'SOURCE_SPLIT_STAT_PROTOCOL_REJECT'
    assert torch.equal(z['statistics']['mean'],torch.from_numpy(st[0])) and torch.equal(z['statistics']['std'],torch.from_numpy(st[1]));m.load_state_dict(z['state_dict']);opt.load_state_dict(z['optimizer']);restore_rng(z['rng'],g);return z['history']
def generated(ns,m,data,ix,st):
    pieces=[];m.eval()
    with torch.inference_mode():
        for i in range(0,len(ix),128):
            guard();ind=ix[i:i+128];x=((data['p2'][ind]-st[0])/st[1]).astype(np.float32);out=trajectory(ns,m,torch.from_numpy(x).to(DEVICE),torch.from_numpy(data['past'][ind]).to(DEVICE));pieces.append(out.cpu().numpy())
    result=np.concatenate(pieces);assert result.shape==(len(ix),4,12,2) and np.isfinite(result).all();return result
def errors(paths,gt):
    e=np.linalg.norm(paths.astype(np.float64)-gt[:,None].astype(np.float64),axis=-1);a=e.mean(-1);f=e[:,:,-1];j=np.maximum(a/.2,f/.4);return a,f,j,(a<=.2)&(f<=.4)
def motion(p):
    v=np.diff(p.astype(np.float64),axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4;s=np.linalg.norm(v,axis=-1);a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4;aa=np.linalg.norm(a,axis=-1);jj=np.linalg.norm(j,axis=-1)
    return dict(speed_p95=float(np.quantile(s,.95)),accel_mean=float(aa.mean()),accel_p95=float(np.quantile(aa,.95)),jerk_mean=float(jj.mean()),jerk_p95=float(np.quantile(jj,.95)))
def clusters(paths,good):
    ans=[]
    for p,g in zip(paths,good):
        guard();K=len(p);d=np.linalg.norm(p[:,None].astype(np.float64)-p[None,:].astype(np.float64),axis=-1);D=np.maximum(d.mean(-1)/.1,d[:,:,-1]/.2);ix=np.triu_indices(K,1);lab=fcluster(linkage(D[ix],method='complete'),1.,criterion='distance')
        ans.append((len(set(lab)),len(set(lab[g]))))
    return np.asarray(ans,np.int32)
def evaluate(z):
    result={};native=z['paths'][:,:20];_,_,_,ng=errors(native,z['gt']);refgood=ng.any(1)
    for name,paths in [('native20',native),('full24',z['paths']),('learned4',z['paths'][:,20:])]:
        a,f,j,good=errors(paths,z['gt']);cl=clusters(paths,good);port={}
        for group,mask in [('overall',np.ones(len(a),bool)),('stable',z['groups']==0),('transition',z['groups']==1),('change',z['groups']==2)]:
            if not mask.any():port[group]={'n':0};continue
            mot=motion(paths[mask]);base=motion(native[mask]);gg=good.any(1)
            port[group]=dict(n=int(mask.sum()),good=int(gg[mask].sum()),coverage=float(gg[mask].mean()),minADE=float(a[mask].min(1).mean()),minFDE=float(f[mask].min(1).mean()),J=float(j[mask].min(1).mean()),rescued=int((gg&~refgood&mask).sum()),lost=int((~gg&refgood&mask).sum()),retained=int((gg&refgood&mask).sum()),effective_distinct=float(cl[mask,0].mean()),effective_good=float(cl[mask,1].mean()),motion=mot,native20_motion=base,motion_all5_1pct=all(mot[k]<=base[k]*1.01 for k in mot))
        result[name]=port
    a,f,_,_=errors(z['paths'][:,20:],z['gt']);assign,_,_=assignment(a,f,z['groups']);result['assignment']=assign
    return result
def frozen_winner_switch(ns,data):
    records=[];directory=H/'old_winner_switch';directory.mkdir(exist_ok=True)
    for tag,split,modeldir in PLAN:
        sp=validate_split(data,split);st=stats(data,sp['grouped_train_index'])
        for s in SEEDS:
            guard();file=modeldir/f'seed_{s}/model.pth';mh=sha(file);m=ns['BestOf4Generator']().to(DEVICE);z=torch.load(file,map_location='cpu',weights_only=False);m.load_state_dict(z.get('state_dict',z));m.eval();m.requires_grad_(False);before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};jobs=[]
            for name,ix in [('train',sp['grouped_train_index']),('outer',sp['outer_eval_index']),('original_CAL',np.arange(25912,28791))]:
                stage('OLD_FULL4_WINNER_SWITCH',grouping=tag,seed=s,split=name);p=generated(ns,m,data,ix,st);a,f,_,_=errors(p,data['gt'][ix]);summary,winA,winB=assignment(a,f,data['groups'][ix]);output=directory/f'{tag}_{s}_{name}.npz';assert not output.exists(),'Do not overwrite old-switch output'
                np.savez_compressed(output,sample_id=data['ids'][ix],groups=data['groups'][ix],ADE=a,FDE=f,winner_scene=winA,winner_track=winB)
                saved=npz(output);replay,wa,wb=assignment(saved['ADE'],saved['FDE'],saved['groups']);assert digest(summary)==digest(replay) and np.array_equal(wa,saved['winner_scene']) and np.array_equal(wb,saved['winner_track']);jobs.append(dict(split=name,summary=summary,output=str(output),sha=sha(output)))
            assert sha(file)==mh and all(torch.equal(v.detach().cpu(),before[k]) for k,v in m.state_dict().items());records.append(dict(grouping=tag,seed=s,checkpoint=str(file),checkpoint_sha=mh,frozen_allweights_buffers_EXACT=True,jobs=jobs));del m,z,before;gc.collect();torch.cuda.empty_cache()
    write('OLD_WINNER_SWITCH.json',dict(status='PASS_FROZEN_FULL4_DIAGNOSTIC',records=records,no_optimizer=True,no_new_training=True,limits='Existing bare checkpoint cannot prove actual old training protocol. Float64 saved-error argmins; ties reported. This is same-output rule comparison, not retraining counterfactual.'))
    return records
def bind_run(data,tag,split,sp,st,seed,source):
    d=H/'runs'/tag/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True)
    sid=d/'split_ids.npz';stat=d/'statistics.npz'
    if not sid.exists():np.savez_compressed(sid,train_ids=data['ids'][sp['grouped_train_index']],outer_ids=data['ids'][sp['outer_eval_index']],inner_ids=data['ids'][sp['inner_cal_index']],original_CAL_ids=data['ids'][25912:])
    if not stat.exists():np.savez_compressed(stat,mean=st[0],std=st[1],normalization_ids=data['ids'][sp['grouped_train_index']])
    saved=npz(stat);assert np.array_equal(saved['mean'],st[0]) and np.array_equal(saved['std'],st[1]) and np.array_equal(saved['normalization_ids'],data['ids'][sp['grouped_train_index']])
    seed_all(seed);initial=setup()['BestOf4Generator']();protocol=read(H/'protocol.json');man=dict(protocol,grouping=tag,seed=seed,source_sha=digest(source),source_files=source,split_sha=sha(split),split_file=str(split),split_ids_file=str(sid),split_ids_sha=sha(sid),statistics_file=str(stat),statistics_sha=sha(stat),statistics_mean=st[0].tolist(),statistics_std=st[1].tolist(),input_sha={str(p):source[str(p)] for p in INPUTS},train_n=len(sp['grouped_train_index']),outer_n=len(sp['outer_eval_index']),inner_n=len(sp['inner_cal_index']),initial_parameters_sha=tensor_hash(initial.state_dict()),parameter_count=sum(v.numel() for v in initial.parameters()),parameter_names=list(dict(initial.named_parameters())),normalization_value_differences='grouping-derived training data only; fixed algorithm not identical statistics')
    p=d/'run_manifest.json'
    if p.exists():assert read(p)==man,'Existing manifest contract mismatch'
    else:write(p,man)
    return d,man
def completed_valid(d,manifest):
    done=read(d/'done.json');assert done['status']=='PASS' and done['run_manifest_sha']==digest(manifest),'Done contract rejected';verify(done['files']);return done
def scientific_run(ns,data,tag,split,source,done_count):
    sp=validate_split(data,split);st=stats(data,sp['grouped_train_index']);records=[]
    for seed in SEEDS:
        d,man=bind_run(data,tag,split,sp,st,seed,source)
        if (d/'done.json').exists():completed_valid(d,man);records.append(read(d/'metrics.json'));continue
        seed_all(seed);m=ns['BestOf4Generator']().to(DEVICE);assert tensor_hash(m.state_dict())==man['initial_parameters_sha'];opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);ldr,g=loader(data,sp['grouped_train_index'],st,seed)
        history=[];ck=d/'checkpoint.pt'
        if ck.exists():history=resume(ck,m,opt,g,st,man)
        for ep in range(len(history)+1,26):
            stage('TRAIN',grouping=tag,seed=seed,epoch=ep,total_epochs=25,runs_completed=done_count+len(records));history.append(epoch(ns,m,opt,ldr,ep));save_checkpoint(ck,m,opt,g,history,st,man);write(d/'history.json',history)
        assert len(history)==25 and all(z['n']==man['train_n'] for z in history);result={};outputs=[]
        for name,ix in [('outer',sp['outer_eval_index']),('inner',sp['inner_cal_index']),('original_CAL',np.arange(25912,28791))]:
            stage('SAVE_EVAL',grouping=tag,seed=seed,split=name);path=generated(ns,m,data,ix,st);z=dict(sample_id=data['ids'][ix],groups=data['groups'][ix],track=data['track'][ix],scene=data['scene'][ix],gt=data['gt'][ix],paths=np.concatenate([data['native'][ix],path],1));assert np.array_equal(z['paths'][:,:20],data['native'][ix]);file=d/f'{name}.npz';assert not file.exists(),'Partial evaluation retained; no overwrite'
            np.savez_compressed(file,**z);saved=npz(file);assert all(np.array_equal(v,saved[k]) for k,v in z.items());stage('SAVED_QUALITY_CLUSTER_MOTION',grouping=tag,seed=seed,split=name);result[name]=evaluate(saved);outputs.append(file)
        assignment_stats=dict(training_preupdate_history=history,final_posthoc={k:v['assignment'] for k,v in result.items()},four_generated_slots_only=True,first_index_tie_rule=True)
        write(d/'assignment_stats.json',assignment_stats);write(d/'metrics.json',dict(grouping=tag,seed=seed,epoch=25,splits=result,larger_K_not_equal_budget=True));files=[ck,d/'run_manifest.json',d/'split_ids.npz',d/'statistics.npz',d/'history.json',d/'metrics.json',d/'assignment_stats.json',*outputs];stamp=dict(status='PASS',run_manifest_sha=digest(man),files={str(p):sha(p) for p in files},all25epochs=True,native20_exact_unchanged=True);write(d/'done.json',stamp);records.append(read(d/'metrics.json'));del m,opt,ldr;gc.collect();torch.cuda.empty_cache();stage('RUN_DONE',grouping=tag,seed=seed,runs_completed=done_count+len(records),runs_total=9)
    return records
def saved_audit():
    checks=[]
    for tag,_,_ in PLAN:
        for seed in SEEDS:
            d=H/'runs'/tag/f'seed_{seed}';man=read(d/'run_manifest.json');completed_valid(d,man);met=read(d/'metrics.json');z=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False)
            assert z['epoch']==25 and z['run_manifest']==man and z['run_manifest_sha']==digest(man) and len(z['history'])==25
            for split in ('outer','inner','original_CAL'):
                recompute=evaluate(npz(d/f'{split}.npz'));assert digest(recompute)==digest(met['splits'][split]),('Saved audit mismatch',tag,seed,split)
            checks.append(dict(grouping=tag,seed=seed,checkpoint_sha=sha(d/'checkpoint.pt'),three_saved_metrics_quality_clusters_motion_exact=True))
    write('ARITHMETIC_AUDIT.json',dict(status='PASS',records=checks,all27saved_metrics_recomputed=True,not_all_models_or_gradients_replayed=True))
def main():
    guard();pre=read(H/'PREFLIGHT.json');assert pre['status']=='PASS';source=pre['files'];stage('INITIAL_SOURCE_INPUT_SHA');verify(source);ns=setup();data=load_data(ns);allrecords=[]
    for tag,split,_ in PLAN:allrecords.extend(scientific_run(ns,data,tag,split,source,len(allrecords)))
    stage('FINAL_SAVED_ARITHMETIC_AUDIT');saved_audit();stage('FINAL_SOURCE_OLD_SHA');verify(source);guard();write('RESOURCE_USAGE.json',dict(status='PASS',**RESOURCE));write('RESULT.json',dict(status='PASS_PROTOCOL_RECONCILIATION_RUN',runs=9,records=allrecords,source_sha=digest(source),old_sources_models_results_unchanged=True,old_F16_decoder_scorer_not_instantiated_or_updated=True,native_cache_exact=True,resource=RESOURCE,major_success_not_inferred=True));stage('SUCCESS',runs=9)
if __name__=='__main__':
    try:main()
    except Exception:
        write('FAILURE_'+str(time.time_ns())+'.json',dict(status='STOP',traceback=traceback.format_exc(),resource=RESOURCE));raise
