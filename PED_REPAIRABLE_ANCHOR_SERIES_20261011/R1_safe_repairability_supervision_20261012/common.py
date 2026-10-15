import os,sys,json,time,hashlib,ctypes
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="1"
import numpy as np,psutil
H=Path(__file__).resolve().parent; R=H.parents[1]
P=H.parent/"stage1b_history_interaction_20261012"
F=H.parent/"position3_feasibility_audit_20261012"
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
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def select(reg,lab,discovery):
    excluded=np.unique(discovery["track"]);A=reg["A_index"]
    eligible=A[(~lab["native_good"][A])&lab["potential"][A].any(1)&~np.isin(reg["track"][A],excluded)]
    chosen=[];scale=[]
    for g in range(3):
        pop=eligible[reg["groups"][eligible]==g];count={};sel=[]
        for i in sorted(pop.tolist(),key=lambda k:hashlib.sha256(("repairability-supervision-v1|"+str(int(reg["sample_id"][k]))).encode()).digest()):
            tr=str(reg["track"][i])
            if count.get(tr,0)>=4:continue
            count[tr]=count.get(tr,0)+1;sel.append(i)
            if len(sel)==128:break
        scale.append(dict(group=g,eligible=len(pop),selected=len(sel),tracks=len(count)))
        assert len(sel)==128,("STOP_FIXED_QUOTA",scale);chosen.extend(sel)
    ix=np.array(chosen,int);assert len(np.unique(ix))==384 and not np.intersect1d(reg["track"][ix],excluded).size
    return ix,scale,excluded
def module(name,path):
    import importlib.util
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def load():
    guard();reg=npz(P/"split_registry.npz");lab=npz(P/"labels_offline.npz");ix,scale,excluded=select(reg,lab,npz(F/"sample.npz"))
    with np.load(R/"REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz",allow_pickle=False) as z:
        assert np.array_equal(z["sample_id"],reg["sample_id"]);native=z["candidate"][ix,...,:24].reshape(-1,20,12,2).copy();logs=z["logits"][ix].copy()
    with np.load(R/"F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz",allow_pickle=False) as z:
        assert np.array_equal(z["sample_id"],reg["sample_id"]);gt=z["future"][ix].copy()
    with np.load(R/"GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz",allow_pickle=False) as z:
        assert np.array_equal(z["sample_id"],reg["sample_id"]);past=z["past"][ix].copy();p2=z["tokens"][ix].mean(1).astype(np.float32)
    with np.load(R/"GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz",allow_pickle=False) as z:
        assert np.array_equal(z["sample_id"],reg["sample_id"]);assert np.array_equal(past.reshape(-1,16),z["observation"][ix,:16])
    assert np.count_nonzero(past[:,-1])==0
    feature=module("locked_features",P/"features.py");X=feature.build(p2,past,native,"interaction");assert np.array_equal(X,feature.independent(p2,past,native,"interaction"))
    prior=npz(P/"interaction/OOF_predictions.npz");lookup={int(v):j for j,v in enumerate(prior["sample_id"])};order=np.array([lookup[int(v)] for v in reg["sample_id"][ix]])
    return dict(index=ix,ids=reg["sample_id"][ix],track=reg["track"][ix],groups=reg["groups"][ix],fold=reg["fold"][ix],scene=reg["scene"][ix],native=native,gt=gt,logs=logs,past=past,p2=p2,X=X,envelope=lab["potential"][ix],old_scores=prior["scores"][order],excluded_tracks=excluded,scale=scale)
def locks():
    files=read(F/"source_manifest.json")["files"].copy();verify(files);inherited=len(files)
    add=[*H.glob("*.py"),*H.glob("*.ps1"),H/"README.md",H/"DESIGN.md",H/"protocol.json",H/"sample_design.npz",H/"SCALE_CHECK.json",F/"source_manifest.json",F/"RESULT.json",F/"RESULT_ANALYSIS.md",F/"RESULT_ANALYSIS.json",*list((F/"completion_review").glob("*")),Path("C:/Users/linke/.codex/attachments/aaf2969d-c3f4-45a5-8016-cfe10fc09284/已粘贴的文本.txt")]
    add.extend(Path(p) for p in read(F/"RESULT.json")["output_sha"])
    add.extend(p for p in (H/"failures").rglob("*") if p.is_file())
    add.extend(H.glob("PREFLIGHT_FAILURE_*.json"))
    add.append(H/"REPAIR_PROVENANCE.md")
    for p in add:
        if not p.is_file():continue
        h=sha(p);key=str(p)
        if key in files:assert files[key]==h,("OLD_DIGEST_CHANGE",key)
        else:files[key]=h
    return files,inherited
