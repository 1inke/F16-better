"""E6 deterministic train-only residual kNN; isolated writes, no torch."""
import os,sys,time,json,hashlib,importlib.util,gc,traceback
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
H=Path(__file__).resolve().parent;P=H.parent;R=P.parent
spec=importlib.util.spec_from_file_location('e6_saved_cpu_math',P/'completion_review/audit_saved.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
D=Path('C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20/datasets/eth_ucy/processed_data_diverse')
L=D.parent/'labels'
Q=json.loads((H/'protocol.json').read_text())
def read(p):return a.read(p)
def sha(p):return a.digest(p)
def guard():a.guard()
def write(name,o):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');a.write(tmp,o);os.replace(tmp,p)
def save(name,**kw):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,**kw);os.replace(tmp,p)
def stage(name,**kw):write('progress.json',dict(stage=name,at=time.time(),**kw));print(name,kw,flush=True)
def verify(locks):
 for i,(p,h) in enumerate(locks.items()):
  if i%32==0:guard()
  assert sha(p)==h,('SOURCE',p)
def fp(x):return np.round(x.astype(np.float32),4).tobytes()
def train_identity(fold,full):
 wanted={fp(x) for x in full};lookup={}
 for path in sorted((L/fold).glob('*.txt')):
  if not path.stem.endswith(('_train','_val')):continue
  tab=np.loadtxt(path,usecols=(0,1,13,15));frame=tab[:,0].astype(int);ped=tab[:,1].astype(int);xy=tab[:,2:].astype(np.float32)
  scene=path.stem.rsplit('_',1)[0]
  for person in np.unique(ped):
   row=np.flatnonzero(ped==person);row=row[np.argsort(frame[row],kind='stable')];t=frame[row];p=xy[row]
   for j in range(len(row)-19):
    if not np.all(np.diff(t[j:j+20])==1):continue
    key=fp(p[j:j+20])
    if key in wanted:lookup.setdefault(key,[]).append((scene,int(person),int(t[j]),path.name))
  guard()
 scene=[];track=[];start=[];amb=0
 for x in full:
  candidates=lookup.get(fp(x),[]);ident={(sc,pe) for sc,pe,t,f in candidates}
  assert len(ident)==1,('STOP_TRAIN_IDENTITY',fold,len(ident),len(scene))
  sc,pe=next(iter(ident));times={t for sc,pe,t,f in candidates}
  scene.append(sc);track.append(sc+'::'+str(pe));start.append(next(iter(times)) if len(times)==1 else -1);amb+=len(times)!=1
 return dict(scene=np.array(scene),track=np.array(track),start=np.array(start),ambiguous_time_n=amb)
def features(obs,kind):
 v=(obs[:,-1]-obs[:,-2]).astype(float)
 return np.concatenate((obs[:,-1].astype(float),v),1) if kind=='position_velocity' else v
def cv(obs):return (obs[:,-1]-obs[:,-2]).astype(float)[:,None]*np.arange(1,13)[None,:,None]
def tree_neighbors(tree,trainX,query,ids):
 out=[];ds=[]
 d,ix=tree.query(query,k=32,workers=1)
 for i in range(len(query)):
  radius=np.nextafter(d[i,-1]+1e-12,np.inf);pool=np.array(tree.query_ball_point(query[i],radius),int)
  dist=np.linalg.norm(trainX[pool]-query[i],axis=1);order=np.lexsort((ids[pool],dist))[:32];chosen=pool[order];assert len(chosen)==32
  out.append(chosen);ds.append(dist[order])
 return np.array(out),np.array(ds)
def predict(bank,test,kind,subset,limit=None,brute=False):
 use=np.ones(len(bank['ids']),bool) if subset=='all_train' else bank['scene']!=Q['sibling_exclusion'][test['fold']]
 assert use.sum()>=32
 ids=bank['ids'][use];X=features(bank['obs'][use],kind);mean=X.mean(0);std=X.std(0);std=np.where(std<1e-6,1.,std);X=(X-mean)/std
 obs=test['obs'][:,0] if limit is None else test['obs'][:limit,0];queries=(features(obs,kind)-mean)/std
 tree=cKDTree(X);parts=[];indices=[];distances=[];t=time.perf_counter()
 for start in range(0,len(obs),128):
  guard();batch=queries[start:start+128]
  if brute:
   ix=[];d=[]
   for row in batch:
    dist=np.linalg.norm(X-row,axis=1);order=np.lexsort((ids,dist))[:32];ix.append(order);d.append(dist[order])
   ix=np.array(ix);d=np.array(d)
  else:ix,d=tree_neighbors(tree,X,batch,ids)
  residual=(bank['gt'][use]-cv(bank['obs'][use]))[ix].mean(1)
  parts.append(cv(obs[start:start+128])+residual);indices.append(ids[ix]);distances.append(d)
 return dict(pred=np.concatenate(parts),neighbor_ids=np.concatenate(indices),distances=np.concatenate(distances),mean=mean,std=std,train_ids=ids,elapsed=time.perf_counter()-t)
def summary(pred,test,fold):
 g=test['gt'];last=cv(test['obs'][:,0]);den=((g-last)**2).sum((1,2));num=((g-pred)**2).sum((1,2));aa,ff=a.errors(pred,g);good=(aa<=.2)&(ff<=.4)
 ref=a.load(P/f'jobs/{fold}_seed1/outputs.npz');masks={k[5:]:v for k,v in ref.items() if k.startswith('mask_')}
 comparisons={}
 for seed in (1,2,3):
  z=a.load(P/f'jobs/{fold}_seed{seed}/outputs.npz')
  for method in ('top1','meanK'):
   p=z['model_paths'][np.arange(len(pred)),z['model_logits'].argmax(1)] if method=='top1' else z['model_paths'].mean(1)
   comparisons[f'model_seed{seed}_{method}']=a.errors(p,g)
 cal=read(P/f'calibration/{fold}.json');simple=dict(CV_last=last,CV_avg3=a.cv(test['obs'],3),CA=a.ca(test['obs'],cal['ca_cap']),Kalman=a.kalman(test['obs'],cal['kalman']['q'],cal['kalman']['r']))
 comparisons['strongest_simple_'+cal['strongest_simple']]=a.errors(simple[cal['strongest_simple']],g)
 records=[]
 for sub,m in masks.items():
  if not m.any():continue
  rec=dict(subset=sub,n=int(m.sum()),tracks=len(np.unique(test['track'][m])),ADE=float(aa[m].mean()),FDE=float(ff[m].mean()),single_path_coverage=int(good[m].sum()),skill=float(1-num[m].sum()/den[m].sum()) if den[m].sum()>0 else None,skill_track_CI=a.ci(num[m],test['track'][m],den[m],True) if den[m].sum()>0 else None,comparisons=[])
  for method,(ba,bf) in comparisons.items():
   rec['comparisons'].append(dict(method=method,delta_ADE=float((aa-ba)[m].mean()),delta_FDE=float((ff-bf)[m].mean()),ADE_track_CI=a.ci((aa-ba)[m],test['track'][m]),FDE_track_CI=a.ci((ff-bf)[m],test['track'][m])))
  v=(test['obs'][:,0,-1]-test['obs'][:,0,-2]).astype(float);sp=np.linalg.norm(v,axis=1);ok=m&(sp>=.02);u=np.divide(v,sp[:,None],out=np.zeros_like(v),where=sp[:,None]>0);normal=np.stack((-u[:,1],u[:,0]),1)
  for axis,vec in [('longitudinal',u),('lateral',normal)]:
   truth=((g[:,-1]-last[:,-1])*vec).sum(1);model=((pred[:,-1]-last[:,-1])*vec).sum(1);valid=ok&(abs(truth)>1e-12)
   rec[axis+'_n']=int(valid.sum());rec[axis+'_sign_accuracy']=float((np.sign(truth[valid])==np.sign(model[valid])).mean()) if valid.any() else None
  records.append(rec)
 return records
def fixtures():
 ids=np.array([3,2,1,0]+list(range(4,40)));X=np.zeros((40,2));tree=cKDTree(X);ix,d=tree_neighbors(tree,X,np.zeros((1,2)),ids);assert np.array_equal(ids[ix[0]],np.arange(32)) and np.max(d)==0
def preflight():
 t=time.perf_counter();guard();fixtures()
 locks=read(P/'stage2_raw_interventions/source_manifest.json')['files'].copy()
 for path in (P/'stage2_raw_interventions').glob('completion_review/*'):
  if path.is_file():locks[str(path)]=sha(path)
 for path in H.glob('*'):
  if path.is_file() and path.name not in ('runtime_state.json','preflight_stdout.log','preflight_stderr.log'):locks[str(path)]=sha(path)
 verify(locks);recs=[];times=[]
 for fold in Q['folds']:
  stage('IDENTITY_PREFLIGHT',fold=fold)
  path=R/f'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/{fold}_train.npz';ids=a.load(path)['sample_id'].astype(int)
  raw=np.load(D/f'{fold}_data_train.npy',mmap_mode='r');full=np.array(raw[ids,0,:,-2:],np.float32)
  assert len(np.unique(ids))==len(ids);meta=train_identity(fold,full);test=a.load(P/f'data/{fold}_evaluation.npz')
  bank=dict(ids=ids,obs=full[:,:8],gt=(full[:,8:]-full[:,7,None]).astype(float),scene=meta['scene'],track=meta['track'],start=meta['start'])
  test['fold']=fold
  save(f'data/{fold}_bank.npz',**bank)
  composition={sc:int((meta['scene']==sc).sum()) for sc in np.unique(meta['scene'])}
  assert not np.intersect1d(meta['track'],test['track']).size,('TRAIN_TEST_TRACK_OVERLAP',fold)
  variants=[('position_velocity','all_train'),('velocity_only','all_train')]
  if fold in Q['sibling_exclusion']:
   assert (meta['scene']==Q['sibling_exclusion'][fold]).any()
   variants += [('position_velocity','no_sibling'),('velocity_only','no_sibling')]
  for kind,subset in variants:
   small=predict(bank,test,kind,subset,limit=32);brute=predict(bank,test,kind,subset,limit=32,brute=True)
   assert np.array_equal(small['neighbor_ids'],brute['neighbor_ids']) and np.max(abs(small['pred']-brute['pred']))<=1e-10
   use=np.isin(bank['ids'],small['train_ids']);X=features(bank['obs'][use],kind)
   S=np.zeros(X.shape[1]);S2=np.zeros(X.shape[1]);n=0
   for start in range(0,len(X),32):S+=X[start:start+32].sum(0);S2+=(X[start:start+32]**2).sum(0);n+=len(X[start:start+32])
   expectedS=np.zeros_like(S);expectedS2=np.zeros_like(S2)
   for start in range(0,len(X),256):expectedS+=X[start:start+256].sum(0);expectedS2+=(X[start:start+256]**2).sum(0)
   assert max(abs(S/n-expectedS/n).max(),abs(S2/n-expectedS2/n).max())<=1e-11
   bench=predict(bank,test,kind,subset,limit=min(1024,len(test['ids'])));times.append(dict(fold=fold,features=kind,subset=subset,query_seconds=bench['elapsed'],query_n=len(bench['pred']),total_n=len(test['ids'])))
   recs.append(dict(fold=fold,features=kind,subset=subset,train_n=len(small['train_ids']),test_n=len(test['ids']),composition=composition,ambiguous_train_time=meta['ambiguous_time_n'],exact_actual32_brute=True,no_train_test_track_overlap=True))
  del bank,full,raw;gc.collect()
 for path in (H/'data').glob('*.npz'):locks[str(path)]=sha(path)
 verify(locks);write('source_manifest.json',dict(files=locks))
 compute=sum(v['query_seconds']/v['query_n']*v['total_n'] for v in times)
 write('PREFLIGHT.json',dict(status='PASS',jobs=len(recs),records=recs,actual_query_benchmark=times,nominal_query_seconds=compute,source_count=len(locks),source_initial_final_PASS=True,no_torch=True,no_training=True,resource=a.RES,elapsed_seconds=time.perf_counter()-t))
 stage('PREFLIGHT_SUCCESS',source_count=len(locks));print(json.dumps(read(H/'PREFLIGHT.json')),flush=True)
def main():
 t=time.perf_counter();guard();locks=read(H/'source_manifest.json')['files'];verify(locks);jobs=[]
 for fold in Q['folds']:
  bank=a.load(H/f'data/{fold}_bank.npz');test=a.load(P/f'data/{fold}_evaluation.npz');test['fold']=fold
  variants=[('position_velocity','all_train'),('velocity_only','all_train')]
  if fold in Q['sibling_exclusion']:variants += [('position_velocity','no_sibling'),('velocity_only','no_sibling')]
  for kind,subset in variants:
   stage('KNN_QUERY_AND_TRACK_METRICS',fold=fold,features=kind,subset=subset,jobs_done=len(jobs),jobs_total=14);tag=f'{fold}_{kind}_{subset}'
   out=predict(bank,test,kind,subset);rec=summary(out['pred'],test,fold)
   save(f'jobs/{tag}/outputs.npz',sample_id=test['ids'],track=test['track'],GT=test['gt'],pred=out['pred'],CV=cv(test['obs'][:,0]),neighbor_ids=out['neighbor_ids'],distances=out['distances'],train_ids=out['train_ids'],stats_mean=out['mean'],stats_std=out['std'])
   write(f'jobs/{tag}/metrics.json',dict(fold=fold,features=kind,subset=subset,records=rec))
   files={str(H/f'jobs/{tag}/{n}'):sha(H/f'jobs/{tag}/{n}') for n in ('outputs.npz','metrics.json')}
   done=dict(status='SUCCESS',fold=fold,features=kind,subset=subset,source_sha=sha(H/'source_manifest.json'),outputs=files);write(f'jobs/{tag}/done.json',done);jobs.append(done)
  del bank,test;gc.collect()
 stage('SAVED_ARITHMETIC_AUDIT',jobs_done=len(jobs))
 for d in jobs:
  verify(d['outputs']);path=Path(next(k for k in d['outputs'] if k.endswith('.npz')));z=a.load(path);bank=a.load(H/f'data/{d["fold"]}_bank.npz');test=a.load(P/f'data/{d["fold"]}_evaluation.npz')
  lookup={int(ID):i for i,ID in enumerate(bank['ids'])};ix=np.array([[lookup[int(ID)] for ID in row] for row in z['neighbor_ids']]);pred=cv(test['obs'][:,0])+(bank['gt']-cv(bank['obs']))[ix].mean(1)
  assert np.array_equal(pred,z['pred']);assert summary(pred,test,d['fold'])==read(path.parent/'metrics.json')['records']
 verify(locks);write('ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=len(jobs),all_saved_predictions_metrics_CI_recomputed=True,no_nearest_search_second_pass=True));write('RESOURCE_USAGE.json',dict(status='PASS',**a.RES))
 write('RESULT.json',dict(status='PASS_STAGE3_TRAIN_ONLY_KNN_MEMORY',jobs=jobs,elapsed_seconds=time.perf_counter()-t,no_training=True,whole_series_complete=False))
 stage('SUCCESS',jobs_done=len(jobs),jobs_total=14)
if __name__=='__main__':
 try:preflight() if '--preflight' in sys.argv else main()
 except Exception:
  write('FAILURE.json',dict(status='FAILURE',traceback=traceback.format_exc(),resource=a.RES));raise

