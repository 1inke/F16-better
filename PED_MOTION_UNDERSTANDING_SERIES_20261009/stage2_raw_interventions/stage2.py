"""Raw-observation interventions only. All old model values fixed."""
import os,sys,time,json,hashlib,importlib.util,gc,traceback,inspect
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
H=Path(__file__).resolve().parent;P=H.parent
def mod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
s=mod('rawmotion_frozen_loader',P/'stage1.py')
a=mod('rawmotion_saved_math',P/'completion_review/audit_saved.py')
s.H=H
def read(path):return a.read(path)
def sha(path):return a.digest(path)
def guard():a.guard()
s.guard=guard
def write(name,obj):
 path=H/name;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');a.write(tmp,obj);os.replace(tmp,path)
s.write=write
def save(name,**arr):
 path=H/name;path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arr);os.replace(tmp,path)
def stage(name,**kw):write('progress.json',dict(stage=name,at=time.time(),**kw));print(name,kw,flush=True)
s.stage=stage
SPEC=[f'L{x}' for x in (2,3,4,6)]+['reverse_prefix','shuffle_prefix','noNbr','farNbr','replaceNbr']+[f'leave{j}' for j in range(1,9)]+['rotate','mirror','translate','scale08','scale12']+[f'syn_{side}_{d}' for side in ('left','right') for d in ('03','06','10','15','25')]
assert len(SPEC)==32
def verify(files):
 for i,(path,d) in enumerate(files.items()):
  if i%32==0:guard()
  assert sha(path)==d,('SOURCE',path)
def selected(fold):
 data=a.load(P/f'data/{fold}_evaluation.npz');order=sorted(range(len(data['ids'])),key=lambda i:hashlib.sha256(f'20261009-stage2:{fold}:{int(data["ids"][i])}'.encode()).hexdigest());ix=np.array(order[:1024])
 out={k:v[ix].copy() for k,v in data.items()};base=a.load(P/f'jobs/{fold}_seed1/outputs.npz')
 out.update({'mask_'+g:base['mask_'+g][ix] for g in ('all','nonCV','nonCV_cue_positive','stationary')})
 _,_,c3=s.cues(out['obs'],out['valid']);out['mask_C3']=c3;out['mask_noC3']=~c3;out['full_row']=ix
 return out
def bank(fold):
 c=a.load(s.R/f'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/{fold}_train.npz');raw=s.load_raw(fold,'train',c['sample_id']);v=(raw['obs'][:,:,-1]-raw['obs'][:,:,-2]).astype(float);valid=np.arange(9)[None]<raw['valid'][:,None]
 pool=v[:,1:][valid[:,1:]];assert len(pool)>0 and np.isfinite(pool).all()
 save(f'data/{fold}_neighbor_velocity_bank.npz',velocity=pool,train_ids=c['sample_id']);return pool
def synthetic(data,pool,fold):
 o=data['obs'];v=(o[:,0,-1]-o[:,0,-2]).astype(float);speed=np.linalg.norm(v,axis=1);rows=[];vel=[];normals=[]
 for i in range(len(o)):
  if speed[i]<.02:continue
  left=np.array([-v[i,1],v[i,0]])/speed[i]
  for j in range(32):
   k=int(hashlib.sha256(f'20261009-synthetic:{fold}:{int(data["ids"][i])}:{j}'.encode()).hexdigest()[:16],16)%len(pool);w=pool[k]-v[i];q=np.linalg.norm(w)
   if q<.02:continue
   normal=np.array([-w[1],w[0]])/q
   if abs(normal@left)<.1:continue
   if normal@left<0:normal=-normal
   rows.append(i);vel.append(pool[k]);normals.append(normal);break
  if len(rows)==128:break
 assert len(rows)>0
 return dict(index=np.array(rows),velocity=np.array(vel),leftnormal=np.array(normals))
def intervention(data,name,synth=None):
 o=data['obs'].copy();valid=data['valid'].copy();ix=np.arange(len(o));params={}
 if name.startswith('syn_'):
  ix=synth['index'];o=o[ix];valid=np.full(len(ix),2,dtype=int);side,d=name.split('_')[1:];dist=dict(zip(('03','06','10','15','25'),(.3,.6,1.,1.5,2.5)))[d];normal=synth['leftnormal']*(1 if side=='left' else -1);vn=synth['velocity'];vt=(o[:,0,-1]-o[:,0,-2]).astype(float);x=o[:,0,-1].astype(float);w=vn-vt
  xN=x-5*w+dist*normal;o[:,1]=(xN[:,None]+vn[:,None]*np.arange(-7,1)[None,:,None]).astype(np.float32);o[:,2:]=o[:,0,None]
  actual=(o[:,1,-1]-o[:,0,-1]).astype(float);wa=(o[:,1,-1]-o[:,1,-2]).astype(float)-vt
  tca=-(actual*wa).sum(1)/(wa*wa).sum(1);closest=np.linalg.norm(actual+tca[:,None]*wa,axis=1)
  assert np.max(abs(tca-5))<=.002 and np.max(abs(closest-dist))<=2e-5,('SYNTHETIC_CLOSEST',np.max(abs(tca-5)),np.max(abs(closest-dist)))
  params=dict(away=-normal,tca=tca,closest=closest,distance=np.full(len(ix),dist),side=np.full(len(ix),1 if side=='left' else -1))
 elif name.startswith('L'):
  L=int(name[1:]);o[:,:,:8-L]=o[:,:,8-L:8-L+1]
 elif name=='reverse_prefix':o[:,:,:6]=o[:,:,:6][:,:,::-1].copy()
 elif name=='shuffle_prefix':
  for i,ID in enumerate(data['ids']):
   rng=np.random.default_rng(int(hashlib.sha256(f'20261009-history:{int(ID)}'.encode()).hexdigest()[:16],16));o[i,:,:6]=o[i,:,rng.permutation(6)].transpose(1,0,2)
 elif name=='noNbr':valid[:]=1;o[:,1:]=o[:,0,None]
 elif name=='farNbr':
  vt=o[:,0,-1]-o[:,0,-2];speed=np.linalg.norm(vt,axis=1);u=np.tile([1.,0.],(len(o),1));ok=speed>=.02;u[ok]=vt[ok]/speed[ok,None];o[:,1:]+=10*np.stack((-u[:,1],u[:,0]),1)[:,None,None].astype(np.float32)
 elif name=='replaceNbr':
  for i in range(len(o)):
   j=(i+1)%len(o);o[i,1:]=data['obs'][j,1:]-data['obs'][j,0,-1]+data['obs'][i,0,-1];valid[i]=data['valid'][j]
 elif name.startswith('leave'):
  j=int(name[5:])
  for i in range(len(o)):
   if j<valid[i]:o[i,j:8]=o[i,j+1:9];valid[i]-=1;o[i,valid[i]:]=o[i,0]
 elif name in ('rotate','mirror','translate','scale08','scale12'):
  if name=='rotate':
   angle=np.array([2*np.pi*int(hashlib.sha256(f'20261009-rotation:{int(ID)}'.encode()).hexdigest()[:12],16)/16**12 for ID in data['ids']]);Q=np.stack((np.cos(angle),-np.sin(angle),np.sin(angle),np.cos(angle)),1).reshape(-1,2,2);o=np.einsum('bij,bntj->bnti',Q,o).astype(np.float32);params['Q']=Q
  elif name=='mirror':o[...,0]*=-1
  elif name=='translate':o+=np.array([10.,-7.],np.float32)
  else:o*=.8 if name=='scale08' else 1.2
 else:raise ValueError(name)
 assert np.isfinite(o).all() and (valid>=1).all() and (valid<=9).all()
 return o,valid,ix,params
def invert(p,name,params):
 if name=='rotate':return np.einsum('bji,bktj->bkti',params['Q'],p)
 if name=='mirror':p=p.copy();p[...,0]*=-1
 if name=='scale08':p=p/.8
 if name=='scale12':p=p/1.2
 return p
def forward(model,trainer,obs,valid,attention=False):
 import torch
 past=torch.from_numpy(np.array(obs,dtype=np.float32,copy=True)).cuda();nv=torch.from_numpy(np.array(valid,dtype=np.int64,copy=True)).cuda();local,_=trainer.normalize_to_focal_heading(past,past);center=local.mean(-2,keepdim=True);captured=[];orig=torch.softmax
 def soft(x,*args,**kw):
  y=orig(x,*args,**kw)
  if attention and inspect.currentframe().f_back.f_code.co_name=='temporal_social_attention' and x.ndim==4 and x.shape[2:]==(9,72):captured.append(y.reshape(len(obs),-1,9,9,8)[:,:,0].sum(-1).mean(1).detach().cpu().numpy())
  return y
 try:
  if attention:torch.softmax=soft
  with torch.no_grad():pred=model(local-center,None,None,mode='testing',num_valid=nv,graph_loc=local)[2];logs=model.last_mode_logits.clone()
 finally:torch.softmax=orig
 rel=pred+center[:, :,None]-local[:,0,-1,None,None,None]
 d=past[:,0,-1]-past[:,0,-4];speed=torch.linalg.vector_norm(d,dim=-1);unit=torch.where((speed>.03)[:,None],d/speed[:,None].clamp_min(1e-6),torch.tensor([1.,0.],device='cuda'))
 world=torch.stack((rel[...,0]*unit[:,None,None,None,0]-rel[...,1]*unit[:,None,None,None,1],rel[...,0]*unit[:,None,None,None,1]+rel[...,1]*unit[:,None,None,None,0]),-1)
 p=world.cpu().numpy().astype(float);lg=logs.cpu().numpy();assert p.shape==(len(obs),9,20,12,2) and lg.shape==(len(obs),9,20)
 assert np.isfinite(p).all() and np.isfinite(lg).all()
 att=np.array(captured[0],dtype=float) if captured else np.empty((len(obs),0));assert len(captured)<=1
 return p[:,0],lg[:,0],p[:,:2],lg[:,:2],att
def summary(z):
 p=z['paths'];g=z['GT'];base=z['base_paths'];lg=z['logits'];bl=z['base_logits'];masks={k[5:]:v for k,v in z.items() if k.startswith('mask_')};records=[];aa,ff=a.errors(p,g)
 for name in ('top1','meanK'):
  point=p[np.arange(len(p)),lg.argmax(1)] if name=='top1' else p.mean(1);bpoint=base[np.arange(len(p)),bl.argmax(1)] if name=='top1' else base.mean(1)
  ad,fd=a.errors(point,g);bad,bfd=a.errors(bpoint,g);resp=np.linalg.norm(point-bpoint,axis=-1).mean(1);eqend=np.linalg.norm(point[:,-1]-bpoint[:,-1],axis=-1)
  if 'away' in z:
   # Synthetic reference is no-neighbor, not original multiagent baseline.
   nb=z['noNbr_paths'];nl=z['noNbr_logits'];ref=nb[np.arange(len(nb)),nl.argmax(1)] if name=='top1' else nb.mean(1);signed=((point[:,-1]-ref[:,-1])*z['away']).sum(1)
  for group,m in masks.items():
   if not m.any():continue
   rec=dict(method=name,subset=group,n=int(m.sum()),tracks=len(np.unique(z['track'][m])),ADE=float(ad[m].mean()),FDE=float(fd[m].mean()),delta_ADE=float((ad-bad)[m].mean()),delta_FDE=float((fd-bfd)[m].mean()),response_meanpath=float(resp[m].mean()),response_endpoint=float(eqend[m].mean()),delta_ADE_track_CI=a.ci((ad-bad)[m],z['track'][m]),delta_FDE_track_CI=a.ci((fd-bfd)[m],z['track'][m]))
   if 'away' in z:rec.update(away_endpoint=float(signed[m].mean()),away_fraction=float((signed[m]>0).mean()),away_track_CI=a.ci(signed[m],z['track'][m]),away_fraction_track_CI=a.ci((signed[m]>0).astype(float),z['track'][m]))
   records.append(rec)
 dc,ec=a.cluster(p,g);good=((aa<=.2)&(ff<=.4)).any(1);ba,bf=a.errors(base,g);bgood=((ba<=.2)&(bf<=.4)).any(1);quality=[]
 for group,m in masks.items():
  if m.any():quality.append(dict(subset=group,n=int(m.sum()),minADE=float(aa[m].min(1).mean()),minFDE=float(ff[m].min(1).mean()),J=float(np.maximum(aa[m]/.2,ff[m]/.4).min(1).mean()),coverage=int(good[m].sum()),base_coverage=int(bgood[m].sum()),rescue=int((m&good&~bgood).sum()),lost=int((m&~good&bgood).sum()),distinct=float(dc[m].mean()),effectivegood=float(ec[m].mean()),motion=a.motion(p,m)))
 result=dict(point_records=records,quality=quality)
 if 'away' in z:
  jp=z['joint_first2'];jl=z['joint_logits_first2'];tp=jp[np.arange(len(jp)),0,jl[:,0].argmax(1)];npred=jp[np.arange(len(jp)),1,jl[:,1].argmax(1)];near=np.linalg.norm(tp-npred,axis=-1).min(1)
  result['synthetic_joint_top1']=dict(n=len(jp),min_distance_mean=float(near.mean()),collision_fraction_below_02=float((near<.2).mean()),not_joint_probability=True,tca_max_error=float(abs(z['tca']-5).max()),distance_max_error=float(abs(z['closest']-z['distance']).max()))
 return result,dc,ec
def valid_done(done):
 d=read(done);assert d['status']=='SUCCESS' and d['source_sha']==sha(H/'source_manifest.json');verify(d['outputs']);return d
def process_job(fold,seed,name,data,model,trainer,base,synth):
 done=H/f'jobs/{fold}_seed{seed}/{name}/done.json'
 if done.exists():
  return valid_done(done)
 o,n,ix,params=intervention(data,name,synth);parts=[];logs=[];joint=[];jointlogs=[]
 for i in range(0,len(o),128):
  guard();p,l,j,jl,_=forward(model,trainer,o[i:i+128],n[i:i+128]);parts.append(p);logs.append(l);joint.append(j);jointlogs.append(jl)
 p=invert(np.concatenate(parts),name,params);z=dict(sample_id=data['ids'][ix],track=data['track'][ix],GT=data['gt'][ix],paths=p,logits=np.concatenate(logs),base_paths=base['paths'][ix],base_logits=base['logits'][ix],joint_first2=np.concatenate(joint),joint_logits_first2=np.concatenate(jointlogs),valid=n,obs=o,**params)
 z.update({k:v[ix] for k,v in data.items() if k.startswith('mask_')})
 if name.startswith('syn_'):
  nb=a.load(H/f'jobs/{fold}_seed{seed}/noNbr/outputs.npz');z['noNbr_paths']=nb['paths'][ix];z['noNbr_logits']=nb['logits'][ix]
 if name.startswith('leave'):z['baseline_attention']=base['attention'][ix,int(name[5:])] if base['attention'].shape[1] else np.full(len(ix),np.nan)
 save(str(done.parent.relative_to(H)/'outputs.npz'),**z);r,dc,ec=summary(z);save(str(done.parent.relative_to(H)/'clusters.npz'),sample_id=z['sample_id'],distinct=dc,effectivegood=ec);write(str(done.parent.relative_to(H)/'metrics.json'),r)
 outputs={str(done.parent/f):sha(done.parent/f) for f in ('outputs.npz','clusters.npz','metrics.json')};rec=dict(status='SUCCESS',fold=fold,seed=seed,intervention=name,n=len(ix),source_sha=sha(H/'source_manifest.json'),outputs=outputs);write(str(done.relative_to(H)),rec);return rec
def main():
 t=time.perf_counter();guard();verify(read(H/'source_manifest.json')['files']);jobs=[];frozen=[]
 for fold in s.FOLDS:
  data=a.load(H/f'data/{fold}_selected.npz');synth=a.load(H/f'data/{fold}_synthetic.npz')
  for seed in s.SEEDS:
   stage('FROZEN_MODEL',fold=fold,seed=seed,jobs_done=len(jobs));model,tr,cfg=s.build(fold,seed);old=s.snapshot(model);baseparts=[];baselogs=[];atts=[]
   for i in range(0,len(data['ids']),128):
    guard();p,l,j,jl,att=forward(model,tr,data['obs'][i:i+128],data['valid'][i:i+128],True);baseparts.append(p);baselogs.append(l);atts.append(att)
   base=dict(paths=np.concatenate(baseparts),logits=np.concatenate(baselogs),attention=np.concatenate(atts))
   ref=a.load(P/f'jobs/{fold}_seed{seed}/outputs.npz');ix=data['full_row'];assert max(float(abs(base['paths']-ref['model_paths'][ix]).max()),float(abs(base['logits']-ref['model_logits'][ix]).max()))<=2e-5,'BASELINE_ALIGNMENT'
   save(f'jobs/{fold}_seed{seed}/baseline.npz',sample_id=data['ids'],**base)
   for name in SPEC:
    stage('RAW_INTERVENTION_SAVED_QUALITY',fold=fold,seed=seed,intervention=name,jobs_done=len(jobs),jobs_total=480);jobs.append(process_job(fold,seed,name,data,model,tr,base,synth))
   s.exact(model,old);frozen.append(dict(fold=fold,seed=seed,weights_bool_buffers_EXACT=True,attention_capture_no_value_change=True));del model,old,base,ref;gc.collect()
 stage('INDEPENDENT_SAVED_AUDIT',jobs_done=len(jobs));maxerr=0.
 for job in jobs:
  guard();verify(job['outputs']);p=Path(next(k for k in job['outputs'] if k.endswith('outputs.npz')));z=a.load(p);r,dc,ec=summary(z);saved=read(p.parent/'metrics.json');assert r==saved
  cls=a.load(p.parent/'clusters.npz');assert np.array_equal(dc,cls['distinct']) and np.array_equal(ec,cls['effectivegood'])
  # Synthetic joint nearest separation uses each agent's own original argmax; not a joint-probability inference.
 stage('FINAL_SOURCE_SHA');verify(read(H/'source_manifest.json')['files'])
 write('FROZEN_AUDIT.json',dict(status='PASS',models=frozen));write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=len(jobs),max_error=maxerr,all_saved_quality_cluster_motion_point_CI_recomputed=True,no_second_model_forward=True));write('RESOURCE_USAGE.json',dict(status='PASS',**a.RES))
 write('RESULT.json',dict(status='PASS_STAGE2_RAW_HISTORY_NEIGHBOR_GEOMETRY',jobs=jobs,elapsed_seconds=time.perf_counter()-t,whole_series_complete=False,not_training=True,attention_scope='Actual F16 temporal_social_attention softmax, summed time and averaged heads; observational association only'))
 stage('SUCCESS',jobs_done=len(jobs),jobs_total=480)
if __name__=='__main__':main()

