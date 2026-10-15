import os,sys,json,time,hashlib,ctypes
from pathlib import Path
from datetime import datetime,timezone,timedelta
sys.dont_write_bytecode=True
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
H=Path(__file__).resolve().parent;D=H.parent;R=Path("C:/Users/linke/Documents/New project")
R3=D.parent/"R3_WITHIN_WINDOW_RANKING_DESIGN_20261013";R2=D.parent/"R2_NONLINEAR_MATCHING_DESIGN_20261012/production_20261013/report_recovery_20261013"
OLD=D.parent/"R4_CROSS_TRACK_NEIGHBORHOOD_DESIGN_20261013/production_20261013"
sys.path.insert(0,str(R3))
import numpy as np,psutil
import core_r5 as k,frozen_inputs as fi,evaluation_r5 as ev
REFERENCES=ev.REFERENCES
RESOURCE=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0);DATA=None
class Mem(ctypes.Structure):
 _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","avail_extended")]
def guard():
 v=Mem();v.length=ctypes.sizeof(v);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(v));p=psutil.Process().memory_info()
 RESOURCE["peak_private_GiB"]=max(RESOURCE["peak_private_GiB"],p.peak_pagefile/2**30);RESOURCE["peak_RSS_GiB"]=max(RESOURCE["peak_RSS_GiB"],p.peak_wset/2**30)
 RESOURCE["min_physical_GiB"]=min(RESOURCE["min_physical_GiB"],v.avail_phys/2**30);RESOURCE["min_commit_GiB"]=min(RESOURCE["min_commit_GiB"],v.avail_page/2**30);RESOURCE["checks"]+=1
 assert max(p.private,p.rss)<=6*2**30 and min(v.avail_phys,v.avail_page)>=6*2**30,"STOP_RESOURCE_PROCESS6_BOTHRESERVE6"
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p,a):
 p=Path(p);p=p if p.is_absolute() else H/p;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+"."+str(os.getpid())+"."+str(time.time_ns())+".tmp")
 payload=json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())
 with tmp.open("x",encoding="utf8") as f:f.write(payload);f.flush();os.fsync(f.fileno())
 for wait in (0.,.05,.1,.2,.4,.8):
  if wait:time.sleep(wait)
  try:os.replace(tmp,p);return
  except PermissionError:
   if wait==.8:raise
def sha(p):return fi.digest(p)
def digest(a):return hashlib.sha256(json.dumps(a,sort_keys=True,separators=(",",":"),default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()).encode()).hexdigest()
def npz(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def save(p,**a):
 p=H/p;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),("NO_OVERWRITE",str(p));np.savez_compressed(p,**a);b=npz(p);assert all(np.array_equal(v,b[k]) for k,v in a.items())
def verify(files):
 for i,(p,s) in enumerate(files.items()):
  if i%32==0:guard()
  assert sha(p)==s,("STOP_SOURCE_OUTPUT_SHA",p)
def sources():
 files=read(OLD/"source_manifest.json")["files"].copy();verify(files);inherited=len(files);r=read(OLD/"RESULT.json");verify(r["output_sha"]);assert read(OLD/"completion_review/REVIEW.json")["status"]=="PASS"
 for p,s in r["output_sha"].items():
  if p in files:assert files[p]==s
  else:files[p]=s
 add=[p for p in OLD.iterdir() if p.is_file()]+list((OLD/"completion_review").glob("*"))+[p for p in D.iterdir() if p.is_file()]+list(H.glob("*.py"))+list(H.glob("*.ps1"))+[H/"README.md",H/"protocol.json"]+RAW_PATHS
 for p in add:
  if not p.is_file():continue
  s=sha(p);name=str(p)
  if name in files:assert files[name]==s,("OLD_DIGEST_CHANGE",name)
  else:files[name]=s
 return files,dict(inherited_R4_sources=inherited,R4_outputs=len(r["output_sha"]),not_all_historical_files=True)
def inputs():
 global DATA
 if DATA is None:guard();DATA=fi.load_inputs(R,R3/"protocol.json",guard=guard)
 return DATA
def stage(name,**kw):guard();write("progress.json",dict(stage=name,at=now(),**kw));print(name,kw,flush=True)
def maps_arrays(mapping):return {name:getattr(mapping,name).copy() for name in ("mean","std","centers","center_keys","rbf_mean","rbf_std")}
def empty_report(scores):
 a=dict(windows=0,knownP_windows=0,hits=0,Hit4=None,all_window_lower=None,all_window_unknown_upper=None,selected_U_fraction=None,knownP_miss=0,miss_all_N=0,miss_contains_U=0)
 if scores:a.update(best_P_rank_mean=None,best_P_rank_median=None,P_greater_N_fraction=None,PN_tie_fraction=None,PN_pairs=0,pair_fraction_is_pooled_descriptive_not_PU_AUC=True,descriptive_HitK={str(v):None for v in (1,2,4,8)})
 return a
def reports(d,slots,scores=None):
 masks={"overall":np.ones(len(d["ids"]),bool),"stable":d["groups"]==0,"transition":d["groups"]==1,"change":d["groups"]==2}
 masks.update({"scene_"+str(s):d["scene"]==s for s in np.unique(inputs()[0]["scene"])})
 masks.update({"fold_"+str(f):d["fold"]==f for f in range(3)})
 return {name:ev.describe(slots[mask],d["labels"][mask],None if scores is None else scores[mask]) if mask.any() else empty_report(scores is not None) for name,mask in masks.items()}
def versions():
 import sklearn,scipy,threadpoolctl
 return dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,threadpoolctl=threadpoolctl.__version__)
def done_valid(done,source_sha):
 assert done["status"]=="COMPLETE" and done["source_sha"]==source_sha,"BAD_DONE_SOURCE"
 verify(done["files"]);return True


RAW_PATHS=[R/"GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz",R/"GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz",R/"REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz",D.parent/"stage1b_history_interaction_20261012/features.py"]
def subset(d,mask):return {name:v[mask] for name,v in d.items() if isinstance(v,np.ndarray) and v.ndim and len(v)==384}
def base_fold(fold):
 d,m,core,con=inputs();return fi.fold_features(d,m,core,con,fold,guard=guard)
