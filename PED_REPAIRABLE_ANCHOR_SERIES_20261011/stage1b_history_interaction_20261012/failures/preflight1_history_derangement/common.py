"""S1 CPU-only common I/O and resource gates. Never import old executable modules."""
import os,sys,json,hashlib,time,ctypes,gc,importlib.util
sys.dont_write_bytecode=True
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
from pathlib import Path
from datetime import datetime
import numpy as np,psutil
H=Path(__file__).resolve().parent; R=H.parents[1]
PLAN=R/"PED_FOUR_DIRECTION_SERIES_20261011/NEXT_EXPERIMENT_DESIGN_20261011"
spc=importlib.util.spec_from_file_location("registered_core",PLAN/"core_design.py")
core=importlib.util.module_from_spec(spc);spc.loader.exec_module(core)
INPUTS=dict(cache=R/"REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz",labels=R/"F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz",encoder=R/"GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz",metadata=R/"F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz",split=R/"GENERATION_GROUPED_TRANSFER_SERIES_20261006/identity_audit/splits.npz")
RESOURCE=dict(peak_private_GiB=0.,peak_RSS_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
class Mem(ctypes.Structure):
    _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","avail_extended")]
def now():return datetime.now().astimezone().isoformat()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def write(p,obj):
    p=Path(p);p=p if p.is_absolute() else H/p;p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding="utf-8");os.replace(t,p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    p=psutil.Process().memory_info()
    RESOURCE["peak_private_GiB"]=max(RESOURCE["peak_private_GiB"],p.peak_pagefile/2**30)
    RESOURCE["peak_RSS_GiB"]=max(RESOURCE["peak_RSS_GiB"],p.peak_wset/2**30)
    RESOURCE["min_physical_GiB"]=min(RESOURCE["min_physical_GiB"],m.avail_phys/2**30)
    RESOURCE["min_commit_GiB"]=min(RESOURCE["min_commit_GiB"],m.avail_page/2**30);RESOURCE["checks"]+=1
    if max(p.private,p.rss)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:raise MemoryError("STOP_RESOURCE_PROCESS6_RESERVE_BOTH6")
def stage(name,**kw):guard();write("progress.json",dict(stage=name,at=now(),**kw));print(name,kw,flush=True)
def verify(files):
    for i,(p,v) in enumerate(files.items()):
        if i%32==0:guard()
        assert sha(p)==v,("SOURCE_CHANGED",p)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(p,**z):
    p=Path(p);p=p if p.is_absolute() else H/p
    assert not p.exists(),("NO_OVERWRITE",str(p));p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**z);rz=npz(p)
    assert all(np.array_equal(v,rz[k]) for k,v in z.items()),"SAVED_ROUNDTRIP";return p
def load():
    guard()
    with np.load(INPUTS["cache"],allow_pickle=False) as c:
        ids=c["sample_id"];native=c["candidate"][...,:24].reshape(-1,20,12,2).astype(np.float32);groups=c["change_group"].astype(np.int64);logs=c["logits"].astype(np.float32)
    with np.load(INPUTS["labels"],allow_pickle=False) as z:
        assert np.array_equal(ids,z["sample_id"]);gt=z["future"].astype(np.float32)
    with np.load(INPUTS["encoder"],allow_pickle=False) as z:
        assert np.array_equal(ids,z["sample_id"]);tokens=z["tokens"];assert tokens.shape==(28791,8,96);p2=tokens.mean(1).astype(np.float32)
    md=npz(INPUTS["metadata"]);assert len(np.unique(md["sample_id"]))==len(md["sample_id"])
    lu={int(x):i for i,x in enumerate(md["sample_id"])};order=np.array([lu[int(x)] for x in ids])
    seq=md["sequence"][order].astype(str);ped=md["pedestrian_id"][order].astype(str);track=np.array([json.dumps([a,b],ensure_ascii=False,separators=(",",":")) for a,b in zip(seq,ped)])
    assert all(a and b and b.lower() not in ("nan","none","") for a,b in zip(seq,ped))
    d=dict(ids=ids,native=native,gt=gt,p2=p2,groups=groups,logs=logs,sequence=seq,ped=ped,track=track,scene=md["scene_family"][order].astype(str),start=md["start_frame"][order],ambiguous=md["ambiguous_track_time"][order].astype(bool))
    assert len(ids)==28791 and len(np.unique(ids))==28791 and all(np.isfinite(d[k]).all() for k in ("native","gt","p2","logs"))
    sp=npz(INPUTS["split"]);assert np.array_equal(sp["sample_id"],ids[:25912])
    for a,b in (("grouped_train_index","grouped_train_ids"),("outer_eval_index","outer_eval_ids"),("inner_cal_index","inner_cal_ids")):
        ix=sp[a];assert np.array_equal(ids[ix],sp[b]) and len(np.unique(ix))==len(ix) and np.all((ix>=0)&(ix<25912))
    tr=sp["grouped_train_index"];outer=sp["outer_eval_index"];inner=sp["inner_cal_index"]
    assert (len(tr),len(outer),len(inner))==(18909,1035,1856)
    for x,y in ((tr,outer),(tr,inner),(outer,inner)):assert not np.intersect1d(x,y).size and not np.intersect1d(track[x],track[y]).size
    bucket=np.array([core.identity_bucket(a,b) for a,b in zip(seq,ped)])
    fold=np.array([core.identity_bucket(a,b,"repair-oof-v1",3) for a,b in zip(seq,ped)])
    A=tr[bucket[tr]<2];B=tr[bucket[tr]>=2]
    assert len(A)>=5000 and len(B)>=8000 and not np.intersect1d(track[A],track[B]).size
    assert all(np.bincount(groups[ix],minlength=3).min()>0 for ix in (A,B))
    with np.load(INPUTS["encoder"],allow_pickle=False) as z:
        past=z["past"].copy()
    with np.load(R/"GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz",allow_pickle=False) as z:
        assert np.array_equal(ids,z["sample_id"]), "OBS_ID_ALIGNMENT"
        assert np.array_equal(past.reshape(-1,16),z["observation"][:,:16]), "OBS_COORDINATE_ALIGNMENT"
        assert np.array_equal(past,z["observation"][:,160:176].reshape(-1,8,2)), "OBS_LOCAL_ALIGNMENT"
    assert past.shape==(28791,8,2) and np.isfinite(past).all() and np.count_nonzero(past[:,-1])==0
    d.update(bucket=bucket,fold=fold,past=past)
    guard();return d,dict(A=A,B=B,train=tr,outer=outer,inner=inner,original_CAL=np.arange(25912,28791))
def label(d,ix,independent=False):
    n=d["native"][ix].astype(np.float64);g=d["gt"][ix].astype(np.float64)
    if not independent:return core.envelope_targets(n,g)
    e=np.sqrt(((n-g[:,None])**2).sum(3));r=.20*(np.arange(1,13,dtype=float)/12)**2;l=np.clip(e-r[None,None],0,None)
    good=(e.mean(2)<=.2)&(e[:,:,11]<=.4)
    return dict(candidate_good=good,native_good=good.any(1),potential=(~good)&(l.mean(2)<=.2)&(l[:,:,11]<=.4),j=np.maximum(e.mean(2)/.2,e[:,:,11]/.4),lower_j=np.maximum(l.mean(2)/.2,l[:,:,11]/.4))
def x(d,ix,arm,history_index=None):
    from features import build
    guard();hi=ix if history_index is None else history_index
    return build(d["p2"][hi],d["past"][hi],d["native"][ix],arm)
def normalizer(d,ix,arm):
    a=x(d,ix,arm).reshape(-1,124 if arm=="additive" else 136).astype(np.float64);mu=a.mean(0);sd=a.std(0);sd=np.where(sd<1e-6,1.,sd);del a;gc.collect();return mu,sd
def infer(d,ix,model,arm,history_index=None):
    from numerical_audit import sigmoid
    pieces=[]
    for begin in range(0,len(ix),256):
        guard();batch=ix[begin:begin+256];hi=None if history_index is None else history_index[begin:begin+256]
        z=x(d,batch,arm,hi).astype(float);t=((z-model["mean"])/model["std"])@model["coef"]+float(model["intercept"][0])
        pieces.append(sigmoid(t))
    ans=np.concatenate(pieces);assert ans.shape==(len(ix),20) and np.isfinite(ans).all();return ans
def summaries(d,ix,l,scores):
    slots={"selected":core.select4(scores),"bottom":core.bottom4(d["logs"][ix]),"random":core.fixed_random4(d["ids"][ix])}
    hit={k:core.potential_hit(l["potential"],v) for k,v in slots.items()};groups=d["groups"][ix];out={}
    masks={"overall":np.ones(len(ix),bool),"stable":groups==0,"transition":groups==1,"change":groups==2}
    masks.update({"scene_"+str(s):d["scene"][ix]==s for s in np.unique(d["scene"][ix])})
    for name,m in masks.items():
        bad=m&~l["native_good"];n=int(bad.sum());poss=l["potential"].any(1)
        out[name]=dict(n=int(m.sum()),native_failed=n,opportunity_windows=int((bad&poss).sum()),opportunity_tracks=len(np.unique(d["track"][ix[bad&poss]])),policies={})
        for k,v in slots.items():
            count=int(hit[k][bad].sum());chosen=l["potential"][np.arange(len(ix))[:,None],v]
            out[name]["policies"][k]=dict(hits=count,H4=count/n if n else None,opportunity_recall=count/int((bad&poss).sum()) if (bad&poss).any() else None,selected_positive_candidates=int(chosen[bad].sum()),physical_slot_hist=np.bincount(v[bad].ravel(),minlength=20).tolist())
    return out,slots,hit
def bootstrap(d,ix,l,hit):
    records=[]
    for gi,group in enumerate(("overall","change")):
        mask=~l["native_good"]
        if group=="change":mask=mask&(d["groups"][ix]==2)
        keys,inv=np.unique(d["track"][ix[mask]],return_inverse=True);counts=np.bincount(inv)
        assert len(keys)>0
        for pi,ref in enumerate(("bottom","random")):
            diff=hit["selected"][mask].astype(float)-hit[ref][mask].astype(float);sums=np.bincount(inv,weights=diff,minlength=len(keys))
            rng=np.random.default_rng(20261011+gi*10+pi);draw=[]
            for it in range(2000):
                if it%64==0:guard()
                tt=rng.integers(0,len(keys),size=len(keys));draw.append(sums[tt].sum()/counts[tt].sum())
            records.append(dict(group=group,reference=ref,n=int(mask.sum()),tracks=len(keys),estimate=float(diff.mean()),CI95=np.quantile(draw,[.025,.975]),CI975=np.quantile(draw,[.0125,.9875]),rng=20261011+gi*10+pi,iterations=2000,paired_tracks=True,not_model_seed_replications=True))
    return records
def locks():
    prior=H.parent/"stage1_selector_production_20261011";inherited=read(prior/"source_manifest.json")["files"];verify(inherited);out=dict(inherited)
    add=[*H.glob("*.py"),*H.glob("*.ps1"),H/"README.md",H/"DESIGN.md",H/"protocol.json",R/"GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz",R/"GENERATION_NATIVE_DECODER_SERIES/run_native.py",R/"GENERATION_NATIVE_DECODER_SERIES/encoder_audit.json",R/"F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/protocol.json"]
    add.extend(prior/p for p in ("source_manifest.json","RESULT.json","RESULT_ANALYSIS.md","RESULT_ANALYSIS.json","OOF_RESULT.json","OOF_predictions.npz","split_registry.npz","labels_offline.npz","completion_review/REVIEW.json","completion_review/LINEAR_RANK_INTERPRETATION.json"))
    add.extend(Path(p) for p in read(prior/"RESULT.json")["output_sha"])
    for p in sorted(set(add)):
        k=str(p);h=sha(p)
        if k in out:assert h==out[k],("INHERITED_SHA_CHANGED",k)
        else:out[k]=h
    return out,len(inherited)
