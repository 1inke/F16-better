"""Metadata-only fixed third design scale. No model, label future, quality or optimizer."""
from pathlib import Path
import json,hashlib,ctypes,sys
sys.dont_write_bytecode=True
import numpy as np,psutil
HERE=Path(__file__).resolve().parent
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
p=psutil.Process().memory_info();assert max(p.rss,p.private)<6*2**30 and min(m.avail_phys,m.avail_page)>=6*2**30,'STOP_RESOURCE'
proto=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'));tag=proto['tag']
src=HERE.parent/'identity_audit/splits.npz'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
locked={str(f):sha(f) for f in (src,HERE/'DESIGN.md',HERE/'protocol.json',Path(__file__))}
with np.load(src,allow_pickle=False) as z:
    ids=z['sample_id'];seq=z['sequence'];scene=z['scene_family'];ped=z['pedestrian_id'];g=z['loss_change_group'];t=z['start_frame'];amb=z['ambiguous_track_time']
assert len(ids)==len(np.unique(ids))==25912 and not (ped<0).any()
def rank(purpose,v):return hashlib.sha256((tag+'|'+purpose+'|'+str(v)).encode()).hexdigest()
keys=np.asarray([str(s)+'|'+str(v) for s,v in zip(seq,ped)])
outer=[];inner=[]
for s in sorted(np.unique(seq)):
    rr=np.flatnonzero(seq==s)
    if np.all(scene[rr]=='zara'):
        ordered=sorted(rr,key=lambda i:(rank('outer-anchor',int(ids[i])),int(i)))
        outer.extend(ordered[:max(1,round(.2*len(rr)))])
    else:
        assert not (scene[rr]=='zara').any()
        tracks=sorted(set(keys[rr]),key=lambda k:(rank('inner-track',k),k))
        chosen=tracks[:max(1,round(.1*len(tracks)))]
        inner.extend(rr[np.isin(keys[rr],chosen)].tolist())
outer=np.sort(outer);inner=np.sort(inner)
grouped=np.flatnonzero((scene!='zara')&~np.isin(np.arange(len(ids)),inner))
eligible=np.flatnonzero(~np.isin(np.arange(len(ids)),np.r_[outer,inner]))
random=[]
for j in (0,1,2):
    nn=int((g[grouped]==j).sum());rr=eligible[g[eligible]==j];assert len(rr)>=nn
    random.extend(sorted(rr,key=lambda i:(rank('random-training',int(ids[i])),int(i)))[:nn])
random=np.sort(random);common=np.intersect1d(random,grouped)
def counts(ix):
    return dict(n=len(ix),scene={str(s):int((scene[ix]==s).sum()) for s in np.unique(scene)},sequence={str(s):int((seq[ix]==s).sum()) for s in np.unique(seq)},loss_group={str(j):int((g[ix]==j).sum()) for j in (0,1,2)})
assert len(random)==len(grouped)>=10000 and len(common)>=5000 and len(outer)>=1000 and (g[outer]==2).sum()>=200,'STOP_METADATA_SCALE'
assert all((g[inner]==j).any() for j in (0,1,2)) and not (scene[grouped]=='zara').any() and (scene[random]=='zara').any()
assert counts(random)['loss_group']==counts(grouped)['loss_group']
assert not np.isin(keys[inner],keys[random]).any() and not np.isin(keys[inner],keys[grouped]).any()
assert all(sha(Path(f))==v for f,v in locked.items())
report=dict(status='PASS_METADATA_SCALE_ONLY_NOT_ID_SOURCE_GPU_RESUME_PREFLIGHT',formal=False,no_forward=True,no_optimizer=True,train_each=len(random),common_train=len(common),outer=len(outer),outer_change=int((g[outer]==2).sum()),inner=len(inner),outer_tracks=len(set(keys[outer])),outer_recordings=len(np.unique(seq[outer])),unknown_time_registry=int(((t<0)|amb).sum()),unknown_outer=int(((t[outer]<0)|amb[outer]).sum()),partitions={n:counts(ix) for n,ix in [('random',random),('grouped',grouped),('common',common),('outer',outer),('inner',inner)]},input_current_source_SHA=locked,unchanged_after=True,source_locks_fullnotdone=True)
(HERE/'METADATA_SCALE_CHECK.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False),flush=True)
