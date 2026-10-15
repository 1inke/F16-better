import os,sys,json,time,hashlib,ctypes,random
from pathlib import Path
from datetime import datetime,timezone,timedelta
sys.dont_write_bytecode=True
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
H=Path(__file__).resolve().parent;S=H.parent;R=S.parent
OLD=R/"PED_REPAIRABLE_ANCHOR_SERIES_20261011/R5_TEMPORAL_MATCHING_DESIGN_20261013/production_20261013"
import numpy as np,psutil
DATA=None
INPUTS=[R/"REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz",R/"F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz",R/"GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz",R/"GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz",R/"F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz",R/"GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz"]
RESOURCE=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
class Mem(ctypes.Structure):
 _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","avail_extended")]
def guard():
 m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));p=psutil.Process().memory_info()
 RESOURCE["peak_private_GiB"]=max(RESOURCE["peak_private_GiB"],p.peak_pagefile/2**30);RESOURCE["peak_RSS_GiB"]=max(RESOURCE["peak_RSS_GiB"],p.peak_wset/2**30);RESOURCE["min_physical_GiB"]=min(RESOURCE["min_physical_GiB"],m.avail_phys/2**30);RESOURCE["min_commit_GiB"]=min(RESOURCE["min_commit_GiB"],m.avail_page/2**30);RESOURCE["checks"]+=1
 assert max(p.private,p.rss)<=6*2**30 and min(m.avail_phys,m.avail_page)>=6*2**30,"STOP_RESOURCE_PROCESS6_BOTHRESERVE6"
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def safe(x):
 if isinstance(x,np.ndarray):return safe(x.tolist())
 if isinstance(x,np.generic):return safe(x.item())
 if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def write(p,x):
 p=Path(p);p=p if p.is_absolute() else H/p;p.parent.mkdir(parents=True,exist_ok=True);payload=json.dumps(safe(x),ensure_ascii=False,allow_nan=False,indent=2);tmp=p.with_name(p.name+"."+str(os.getpid())+"."+str(time.time_ns())+".tmp")
 with tmp.open("x",encoding="utf8") as f:f.write(payload);f.flush();os.fsync(f.fileno())
 for wait in (0.,.05,.1,.2,.4,.8):
  if wait:time.sleep(wait)
  try:os.replace(tmp,p);return
  except PermissionError:
   if wait==.8:raise
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for v in iter(lambda:f.read(2**20),b""):h.update(v)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(safe(x),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(p,**x):
 p=H/p;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),("NO_OVERWRITE",str(p));np.savez_compressed(p,**x);back=npz(p);assert all(np.array_equal(v,back[k]) for k,v in x.items())
def verify(files):
 for i,(p,v) in enumerate(files.items()):
  if i%32==0:guard()
  assert sha(p)==v,("STOP_SOURCE_OUTPUT_SHA",p)
def locks():
 prior=read(OLD/"source_manifest.json");files=prior["files"].copy();verify(files);out=read(OLD/"RESULT.json")["output_sha"];verify(out);assert read(OLD/"completion_review/REVIEW.json")["status"]=="PASS"
 add=list(INPUTS)+[p for p in OLD.iterdir() if p.is_file()]+list((OLD/"completion_review").rglob("*"))+[p for p in S.iterdir() if p.is_file()]+list(H.glob("*.py"))+list(H.glob("*.ps1"))+[H/"DESIGN.md",H/"README.md",H/"protocol.json"]
 for p,v in out.items():
  if p in files:assert files[p]==v
  else:files[p]=v
 for p in add:
  if not p.is_file():continue
  k=str(p);v=sha(p)
  if k in files:assert files[k]==v,("OLD_DIGEST_CHANGE",k)
  else:files[k]=v
 return files,dict(inherited_R5_sources=len(prior["files"]),R5_outputs=len(out),not_all_historical_files=True)
def stage(name,**kw):guard();write("progress.json",dict(stage=name,at=now(),**kw));print(name,kw,flush=True)
def load():
 global DATA
 if DATA is not None:return DATA
 guard();cache,labels,encoder,obs,metadata,split=INPUTS
 with np.load(cache,allow_pickle=False) as z:ids=z["sample_id"].copy();native=z["candidate"][...,:24].reshape(-1,20,12,2).astype(np.float32);groups=z["change_group"].astype(int)
 with np.load(labels,allow_pickle=False) as z:assert np.array_equal(ids,z["sample_id"]);gt=z["future"].astype(np.float32)
 with np.load(encoder,allow_pickle=False) as z:assert np.array_equal(ids,z["sample_id"]);past=z["past"].copy();p2=z["tokens"].mean(1).astype(np.float32)
 with np.load(obs,allow_pickle=False) as z:assert np.array_equal(ids,z["sample_id"]);a=z["observation"];assert np.array_equal(past.reshape(-1,16),a[:,:16]) and np.array_equal(past,a[:,160:176].reshape(-1,8,2))
 md=npz(metadata);lookup={int(v):i for i,v in enumerate(md["sample_id"])};order=np.array([lookup[int(v)] for v in ids]);track=np.array([json.dumps([str(a),str(b)],separators=(",",":")) for a,b in zip(md["sequence"][order],md["pedestrian_id"][order])]);scene=md["scene_family"][order].astype(str)
 assert len(ids)==28791 and len(np.unique(ids))==28791 and past.shape==(28791,8,2) and not np.count_nonzero(past[:,-1]);assert all(np.isfinite(a).all() for a in (past,p2,native,gt))
 sp=npz(split);assert np.array_equal(sp["sample_id"],ids[:25912]);ix={name:sp[key] for name,key in (("FIT","grouped_train_index"),("outer","outer_eval_index"),("inner","inner_cal_index"))};ix["original_CAL"]=np.arange(25912,28791)
 assert tuple(len(ix[k]) for k in ix)==(18909,1035,1856,2879)
 for x,y in (("FIT","outer"),("FIT","inner"),("inner","outer")):assert not np.intersect1d(ix[x],ix[y]).size and not np.intersect1d(track[ix[x]],track[ix[y]]).size
 for name,key in (("FIT","grouped_train_ids"),("outer","outer_eval_ids"),("inner","inner_cal_ids")):assert np.array_equal(ids[ix[name]],sp[key])
 vel=(past[:,-1].astype(float)-past[:,-2].astype(float))/.4;DATA=dict(ids=ids,native=native,groups=groups,gt=gt,past=past,p2=p2,track=track,scene=scene,vel=vel,split=ix);guard();return DATA
def stats(data):
 tr=data["split"]["FIT"];x=data["p2"][tr].astype(float);mean=x.mean(0);sd=x.std(0);vs=data["vel"][tr].std(0);vm=data["vel"][tr].mean(0)
 from sklearn.cluster import KMeans
 from threadpoolctl import threadpool_limits
 residual=data["gt"][tr,-1].astype(float)-4.8*data["vel"][tr]
 with threadpool_limits(1):km=KMeans(n_clusters=20,init="k-means++",n_init=1,random_state=20261014,max_iter=100,tol=1e-4,algorithm="lloyd").fit(residual)
 centers=km.cluster_centers_.copy();assert len(np.unique(centers,axis=0))==20
 return dict(mean=mean.astype(np.float32),std=np.where(sd<1e-6,1.,sd).astype(np.float32),centers=centers,velocity_mean=vm,velocity_scale=np.where(vs<1e-6,1.,vs),training_ids=data["ids"][tr],kmeans_labels=km.labels_,kmeans_iterations=np.array(km.n_iter_))
def done_valid(done,source_sha):
 assert done["status"]=="COMPLETE" and done["source_sha"]==source_sha,"BAD_DONE_SOURCE";verify(done["files"]);return True
def eq(a,b,tol=1e-12):
 if isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:eq(a[k],b[k],tol)
 elif isinstance(a,(list,tuple,np.ndarray)):
  assert len(a)==len(b)
  for x,y in zip(a,b):eq(x,y,tol)
 elif isinstance(a,(float,np.floating)):assert abs(a-b)<=tol,(a,b)
 else:assert a==b,(a,b)
