import os,sys,json,time,hashlib,ctypes
from pathlib import Path
from datetime import datetime,timezone,timedelta
sys.dont_write_bytecode=True
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
H=Path(__file__).resolve().parent;D=H.parent;R=Path("C:/Users/linke/Documents/New project")
OLD=D.parent/"R3_WITHIN_WINDOW_RANKING_DESIGN_20261013/production_20261013"
sys.path.insert(0,str(D))
import numpy as np,psutil
import core_neighbors as k,fold_diagnostic as f,summary_stats as s,input_adapter as ia
DATA=None
RESOURCE=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
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
 p=Path(p);p=p if p.is_absolute() else H/p;p.parent.mkdir(parents=True,exist_ok=True)
 payload=json.dumps(s.json_safe(a),ensure_ascii=False,indent=2,allow_nan=False);tmp=p.with_name(p.name+"."+str(os.getpid())+"."+str(time.time_ns())+".tmp")
 with tmp.open("x",encoding="utf8") as q:q.write(payload);q.flush();os.fsync(q.fileno())
 for wait in (0.,.05,.1,.2,.4,.8):
  if wait:time.sleep(wait)
  try:os.replace(tmp,p);return
  except PermissionError:
   if wait==.8:raise
def sha(p):return ia.sha256(p)
def digest(a):return hashlib.sha256(json.dumps(s.json_safe(a),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def npz(p):
 with np.load(p,allow_pickle=False) as q:return {key:q[key].copy() for key in q.files}
def eq(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b),(set(a),set(b))
  for key in a:eq(a[key],b[key])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):eq(x,y)
 elif isinstance(a,np.ndarray):
  if a.dtype.kind in "fc":assert np.allclose(a,b,rtol=0,atol=1e-12,equal_nan=True)
  else:assert np.array_equal(a,b)
 elif a is None:assert b is None
 elif isinstance(a,(float,np.floating)):assert abs(a-b)<=1e-12,(a,b)
 else:assert a==b,(a,b)
def save(p,**a):
 p=H/p;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),("NO_OVERWRITE",str(p));np.savez_compressed(p,**a);b=npz(p)
 for key,v in a.items():
  assert np.array_equal(v,b[key],equal_nan=True) if np.asarray(v).dtype.kind in "fc" else np.array_equal(v,b[key])
def verify(files):
 for i,(p,v) in enumerate(files.items()):
  if i%32==0:guard()
  assert sha(p)==v,("STOP_SOURCE_SHA",p)
def sources():
 lock=read(OLD/"source_manifest.json");files=lock["files"].copy();verify(files);r=read(OLD/"RESULT.json");verify(r["output_sha"])
 assert read(OLD/"completion_review/REVIEW.json")["status"]=="PASS"
 for p,v in r["output_sha"].items():
  if p in files:assert files[p]==v
  else:files[p]=v
 add=[p for p in OLD.iterdir() if p.is_file()]+[p for p in (OLD/"completion_review").rglob("*") if p.is_file()]+[p for p in D.iterdir() if p.is_file()]+list(H.glob("*.py"))+list(H.glob("*.ps1"))+[H/"README.md",H/"protocol.json"]
 for p in add:
  if not p.is_file():continue
  name=str(p);v=sha(p)
  if name in files:assert files[name]==v,("OLD_DIGEST_CHANGE",name)
  else:files[name]=v
 return files,dict(inherited_R3_sources=len(lock["files"]),R3_outputs=len(r["output_sha"]),not_all_historical_files=True)
def inputs():
 global DATA
 if DATA is None:guard();DATA=ia.load_frozen(R,D/"protocol.json",guard=guard)
 return DATA
def fold(fold):
 d,m,core,con,fn=inputs();return fn(d,m,core,con,fold,guard=guard)
def stage(name,**kw):guard();write("progress.json",dict(stage=name,at=now(),**kw));print(name,kw,flush=True)
def done_valid(done,source_sha):
 assert done["status"]=="COMPLETE" and done["source_sha"]==source_sha,"BAD_DONE_SOURCE";verify(done["files"]);return True
def unit_payload(z,ids,tracks,bank,ix,dist):
 return dict(features=z,sample_id=ids,track=tracks,slot=np.tile(np.arange(20),len(ids)),query_id=np.repeat(ids,20),query_track=np.repeat(tracks,20),indices=ix,distances=dist,neighbor_id=bank.ids[ix],neighbor_slot=bank.slots[ix],neighbor_track=bank.tracks[ix],components=k.distance_components(z.reshape(-1,200),bank,ix))
def posthoc(geometries,d):
 # This is reached only after all SIX geometry units have finished.
 fields={}
 for fold,g in enumerate(geometries):
  tr,va=g["train_mask"],g["valid_mask"];y=d["labels"][tr];bank=k.make_bank(g["train_z"],d["ids"][tr],d["track"][tr]);v=k.label_support(g["indices"],bank,y)
  v.update(k.geometry_support(g["distances"],g["radius"]));v["supported"]=v["near"]&v["known_sufficient"];v["same_scene"]=d["scene"][tr][bank.window[g["indices"]]]==d["scene"][va].repeat(20)[:,None]
  for key,a in v.items():
   a=a.reshape((int(va.sum()),20)+a.shape[1:])
   if key not in fields:fields[key]=np.empty((384,20)+a.shape[2:],dtype=a.dtype)
   fields[key][va]=a
 p=f.primary_rows(d["pairwise_scores"],d["labels"],fields);i=p["window_index"]
 p.update(sample_id=d["ids"][i],track=d["track"][i],groups=d["groups"][i],scene=d["scene"][i],fold=d["fold"][i])
 return fields,p
def descriptions(d,fields,p):
 masks={"overall":np.ones(len(p["rank"]),bool)}
 masks.update({"group_"+str(v):p["groups"]==v for v in range(3)})
 masks.update({"fold_"+str(v):p["fold"]==v for v in range(3)})
 masks.update({"scene_"+str(v):p["scene"]==v for v in np.unique(d["scene"])})
 out={}
 for name,mask in masks.items():
  out[name]={}
  for band in range(3):
   sel=mask&(p["rank_bin"]==band);n=int(sel.sum());ok=sel&p["supported"];finite=sel&np.isfinite(p["enrichment"])
   out[name][str(band)]=dict(windows=n,tracks=len(np.unique(p["track"][sel])),near=int((sel&p["near"]).sum()),known_sufficient=int((sel&p["known_sufficient"]).sum()),supported=int(ok.sum()),supported_tracks=len(np.unique(p["track"][ok])),enrichment_all_finite=float(p["enrichment"][finite].mean()) if finite.any() else None,enrichment_supported=float(p["enrichment"][ok].mean()) if ok.any() else None,mean_d1=float(p["d1"][sel].mean()) if n else None,mean_d5=float(p["d5"][sel].mean()) if n else None,mean_nU=float(p["nU"][sel].mean()) if n else None,neighbor_same_scene_fraction=float(p["same_scene"][sel].mean()) if n else None,observed_P=float(np.nanmean(p["observed_P"][sel])) if finite.any() else None,expected_P=float(np.nanmean(p["expected_P"][sel])) if finite.any() else None,observed_anyP=float(np.nanmean(p["observed_anyP"][sel])) if finite.any() else None,expected_anyP=float(np.nanmean(p["expected_anyP"][sel])) if finite.any() else None)
 macro={}
 for name,mask in {"overall":np.ones(384,bool),**{"group_"+str(v):d["groups"]==v for v in range(3)},**{"fold_"+str(v):d["fold"]==v for v in range(3)},**{"scene_"+str(v):d["scene"]==v for v in np.unique(d["scene"])} }.items():
  macro[name]={key:k.class_window_macro(fields[key][mask].astype(float),d["labels"][mask]) for key in ("enrichment","near","known_sufficient","supported","nU","observed_P","expected_P","observed_anyP","expected_anyP","d1","d5")}
 old=k.best_positive(d["pointwise_scores"],d["labels"]);new=k.best_positive(d["pairwise_scores"],d["labels"]);has=new["hasP"];matrix=np.zeros((3,3),int)
 for a,b in zip(old["bin"][has],new["bin"][has]):matrix[a,b]+=1
 gained=d["ids"][has&(old["rank"]>4)&(new["rank"]<=4)];lost=d["ids"][has&(old["rank"]<=4)&(new["rank"]>4)]
 return dict(primary=out,all_P_N_U_window_macro=macro,rank_transition=matrix,gained_hit_ids=gained,lost_hit_ids=lost)
