import os,sys,json,time,hashlib,ctypes
from pathlib import Path
from datetime import datetime,timezone,timedelta
sys.dont_write_bytecode=True
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[name]="1"
H=Path(__file__).resolve().parent;P=H.parent;D=P.parent;R=Path("C:/Users/linke/Documents/New project")
OLD=R/"PED_REPAIRABLE_ANCHOR_SERIES_20261011/R1_safe_repairability_supervision_20261012/record_recovery_20261012"
sys.path.insert(0,str(D))
import numpy as np,psutil
import core_matching as m,input_contract as ic,evaluation as ev
RESOURCE=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
class Mem(ctypes.Structure):
 _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","avail_extended")]
def guard():
 z=Mem();z.length=ctypes.sizeof(z);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(z));v=psutil.Process().memory_info()
 RESOURCE["peak_private_GiB"]=max(RESOURCE["peak_private_GiB"],v.peak_pagefile/2**30);RESOURCE["peak_RSS_GiB"]=max(RESOURCE["peak_RSS_GiB"],v.peak_wset/2**30)
 RESOURCE["min_physical_GiB"]=min(RESOURCE["min_physical_GiB"],z.avail_phys/2**30);RESOURCE["min_commit_GiB"]=min(RESOURCE["min_commit_GiB"],z.avail_page/2**30);RESOURCE["checks"]+=1
 assert max(v.private,v.rss)<=6*2**30 and min(z.avail_phys,z.avail_page)>=6*2**30,"STOP_RESOURCE_PROCESS6_BOTHRESERVE6"
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p,v):
 p=Path(p);p=p if p.is_absolute() else H/p;p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+"."+str(os.getpid())+"."+str(time.time_ns())+".tmp")
 with t.open("x",encoding="utf8") as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()));f.flush();os.fsync(f.fileno())
 for wait in (0.,.05,.1,.2,.4,.8):
  if wait:time.sleep(wait)
  try:os.replace(t,p);return
  except PermissionError:
   if wait==.8:raise
def sha(p):return ic.sha256(p)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()).encode()).hexdigest()
def npz(p):
 with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def save(p,**a):
 p=H/p;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),("NO_OVERWRITE",str(p));np.savez_compressed(p,**a)
 assert all(np.array_equal(v,npz(p)[k]) for k,v in a.items())
def verify(files):
 for i,(p,v) in enumerate(files.items()):
  if i%32==0:guard()
  assert sha(p)==v,("STOP_SOURCE_OUTPUT_SHA",p)
def inputs():
 guard();d=ic.load_r1(OLD,D/"protocol.json");d["reference_slots"]["old_S1b"]=npz(OLD/"policy_slots.npz")["old_S1b"];return d
def stage(name,**kw):guard();write("progress.json",dict(stage=name,at=now(),**kw));print(name,kw,flush=True)
def original_sources():
 old=read(P/"source_manifest.json")["files"].copy();verify(old);inherited=len(old)
 for p in P.rglob("*"):
  if p.is_file() and not p.is_relative_to(H):
   k=str(p);v=sha(p)
   if k in old:assert old[k]==v
   else:old[k]=v
 for p in list(H.glob("*.py"))+list(H.glob("*.ps1"))+[H/"README.md",H/"protocol.json"]:
  k=str(p);v=sha(p)
  if k in old:assert old[k]==v
  else:old[k]=v
 return old,dict(inherited_parent_scientific_sources=inherited,parent_all_artifacts_locked=True,no_old_digest_replacement=True)
def load_locked():
 z=read(H/"source_manifest.json");verify(z["files"]);return z
def version():
 import sklearn,scipy,threadpoolctl
 return dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,threadpoolctl=threadpoolctl.__version__)
def reports(d,slots,scores=None):
 masks={"overall":np.ones(len(d["ids"]),bool),"stable":d["groups"]==0,"transition":d["groups"]==1,"change":d["groups"]==2}
 masks.update({"scene_"+str(s):d["scene"]==s for s in np.unique(d["scene"])})
 masks.update({"fold_"+str(f):d["fold"]==f for f in range(3)})
 return {k:ev.describe(slots[v],d["labels"][v],None if scores is None else scores[v]) for k,v in masks.items() if v.any()}
def done_valid(done,source_sha,files):
 assert done["status"]=="COMPLETE" and done["source_sha"]==source_sha,"BAD_DONE_SOURCE"
 verify(done["files"]);assert done["files"]==files,"BAD_DONE_OUTPUTS"
 return True
