"""Posthoc saved-data necessary reachability bound; no model, training or deployment selector."""
import json,hashlib,ctypes,time
from pathlib import Path
import numpy as np,psutil
H=Path(__file__).resolve().parent.parent
D=Path(__file__).resolve().parent
class Mem(ctypes.Structure):
    _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","avail_extended")]
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    p=psutil.Process().memory_info()
    assert max(p.rss,p.private)<6*2**30 and min(m.avail_phys,m.avail_page)>=6*2**30
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
    guard();start=time.perf_counter();result=json.loads((H/"RESULT.json").read_text(encoding="utf-8"))
    before={};rows=[];masksaved={};maxerr=0.
    # Scalar fixtures: triangle inequality bound, zero distance, threshold, unreachable endpoint.
    assert max(.7-.2,0)>.4
    assert max(.1-.2,0)==0
    for split in ("outer","inner","original_CAL"):
        pth=H/"runs/sum/seed_42"/(split+".npz");z=load(pth)
        for arm in ("sum","joint"):
            for seed in (42,43,44):
                q=H/"runs"/arm/("seed_"+str(seed))/(split+".npz")
                before[str(q)]=sha(q);assert before[str(q)]==result["output_sha"][str(q)]
                zz=load(q)
                assert all(np.array_equal(z[k],zz[k]) for k in ("sample_id","native20","gt","groups","slots"))
        p=z["native20"].astype(float);gt=z["gt"].astype(float);slots=z["slots"];e=np.linalg.norm(p-gt[:,None],axis=-1)
        radius=.2*(np.arange(1,13)/12.)**2
        lower=np.maximum(e-radius,0)
        # Independent geometric realization in the enlarged per-time ball; no temporal/motion constraints.
        move=gt[:,None]-p
        ideal=p+move*np.minimum(1,radius/np.where(e>0,e,1))[...,None]
        err=np.linalg.norm(ideal-gt[:,None],axis=-1);maxerr=max(maxerr,float(np.max(abs(err-lower))))
        assert maxerr<1e-12
        native=((e.mean(-1)<=.2)&(e[:,:,-1]<=.4)).any(1)
        possible=(lower.mean(-1)<=.2)&(lower[:,:,-1]<=.4)
        selected=possible[np.arange(len(p))[:,None],slots].any(1);anyslot=possible.any(1)
        # Every actually rescued case must be inside the necessary bound.
        for arm in ("sum","joint"):
            for seed in (42,43,44):
                zz=load(H/"runs"/arm/("seed_"+str(seed))/(split+".npz"))
                actual=np.linalg.norm(zz["refined20"]-gt[:,None],axis=-1)
                ag=((actual.mean(-1)<=.2)&(actual[:,:,-1]<=.4)).any(1)
                assert not np.any(ag&~native&~selected)
        for name,mask in (("overall",np.ones(len(p),bool)),("change",z["groups"]==2)):
            fail=mask&~native;allowed=fail&selected
            rows.append(dict(split=split,group=name,n=int(mask.sum()),native_good=int((mask&native).sum()),native_fail=int(fail.sum()),selected4_possible_rescues_upper_bound=int(allowed.sum()),selected4_provably_outside=int((fail&~selected).sum()),selected4_good_count_ceiling_no_losses=int((mask&native).sum()+allowed.sum()),any20_anchor_possible_rescues_upper_bound=int((fail&anyslot).sum())))
            masksaved[split+"_"+name]=dict(failed_ids=z["sample_id"][fail].tolist(),selected4_possible_ids=z["sample_id"][allowed].tolist(),selected4_outside_ids=z["sample_id"][fail&~selected].tolist())
        guard()
    assert all(sha(p)==v for p,v in before.items())
    data=dict(status="PASS_POSTHOC_NECESSARY_ENVELOPE_BOUND",records=rows,ids=masksaved,input_sha=before,input_sha_initial_final_pass=True,script_sha=sha(__file__),independent_radial_realization_max_error=maxerr,actual_rescues_subset_of_bound=True,interpretation="GT diagnostic only. Necessary, not sufficient: removes basis, motion and learnability restrictions. Any20 changes addressable anchors and is not an achieved policy or a trained comparison. No new model forward, optimizer, training, or candidate deployment.",elapsed_seconds=time.perf_counter()-start)
    (D/"REACHABILITY_ENVELOPE.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in data.items() if k not in ("ids","input_sha")}))
if __name__=="__main__":main()
