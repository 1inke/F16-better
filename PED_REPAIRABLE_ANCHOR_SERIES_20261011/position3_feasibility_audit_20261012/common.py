import os,sys,json,time,hashlib,ctypes
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="1"
import numpy as np,psutil
H=Path(__file__).resolve().parent; R=H.parents[1]
P=H.parent/"stage1b_history_interaction_20261012"
RESOURCE=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
class Mem(ctypes.Structure):
    _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","avail_extended")]
def now(): return datetime.now().astimezone().isoformat()
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    p=psutil.Process().memory_info()
    RESOURCE["peak_private_GiB"]=max(RESOURCE["peak_private_GiB"],p.peak_pagefile/2**30)
    RESOURCE["peak_RSS_GiB"]=max(RESOURCE["peak_RSS_GiB"],p.peak_wset/2**30)
    RESOURCE["min_physical_GiB"]=min(RESOURCE["min_physical_GiB"],m.avail_phys/2**30)
    RESOURCE["min_commit_GiB"]=min(RESOURCE["min_commit_GiB"],m.avail_page/2**30);RESOURCE["checks"]+=1
    if max(p.private,p.rss)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30: raise MemoryError("STOP_RESOURCE_PROCESS6_BOTHRESERVE6")
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p,o):
    p=Path(p);p=p if p.is_absolute() else H/p;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding="utf-8");os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for a in iter(lambda:f.read(2**20),b""): h.update(a)
    return h.hexdigest()
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%32==0: guard()
        assert sha(p)==h,("SOURCE_SHA",p)
def npz(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def save(p,**z):
    p=H/p;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),("NO_OVERWRITE",str(p))
    np.savez_compressed(p,**z)
    a=npz(p);assert all(np.array_equal(a[k],v) for k,v in z.items())
def stage(s,**kw):
    guard();write("progress.json",dict(stage=s,at=now(),**kw));print(s,kw,flush=True)
def load():
    guard();reg=npz(P/"split_registry.npz");lab=npz(P/"labels_offline.npz")
    with np.load(R/"REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz",allow_pickle=False) as z:
        ids=z["sample_id"];native=z["candidate"][...,:24].reshape(-1,20,12,2).copy()
    with np.load(R/"F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz",allow_pickle=False) as z:
        assert np.array_equal(ids,z["sample_id"]);gt=z["future"].copy()
    assert np.array_equal(ids,reg["sample_id"]) and np.array_equal(ids,lab["sample_id"])
    a=reg["A_index"];assert len(a)==7334
    # Only A side eligible; no B/outer/inner/CAL analysis or optimization.
    eligible=a[(~lab["native_good"][a]) & lab["potential"][a].any(1)]
    chosen=[]
    def hh(i):return hashlib.sha256(("position3-feasibility-v1|"+str(int(ids[i]))).encode()).digest()
    for g,n in [(2,32),(1,16),(0,16)]:
        seen=set();selected=[]
        for i in sorted(eligible[reg["groups"][eligible]==g].tolist(),key=hh):
            tr=str(reg["track"][i])
            if tr in seen:continue
            seen.add(tr);selected.append(i)
            if len(selected)==n:break
        assert len(selected)==n,("STOP_FIXED_SAMPLE_SCALE",g,len(selected));chosen.extend(selected)
    chosen=np.array(chosen,dtype=np.int64);assert len(np.unique(chosen))==64
    return reg,lab,ids,native,gt,chosen
