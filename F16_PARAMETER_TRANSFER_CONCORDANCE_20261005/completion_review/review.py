"""CPU saved-vector completion audit; no torch/model/optimizer imports."""
from pathlib import Path
import json, hashlib, time, os
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
peak=0
def guard():
    global peak
    proc=psutil.Process(); m=proc.memory_info()
    private=getattr(m,"private",m.rss); peak=max(peak,private)
    vm=psutil.virtual_memory(); sm=psutil.swap_memory()
    assert private<6*2**30 and vm.available>=6*2**30
    if os.name=="nt":
        import ctypes
        class PI(ctypes.Structure):
            _fields_=[("cb",ctypes.c_uint32)]+[(n,ctypes.c_size_t) for n in ("CommitTotal","CommitLimit","CommitPeak","PhysicalTotal","PhysicalAvailable","SystemCache","KernelTotal","KernelPaged","KernelNonpaged")]+[("PageSize",ctypes.c_size_t)]+[(n,ctypes.c_uint32) for n in ("HandleCount","ProcessCount","ThreadCount")]
        pi=PI();pi.cb=ctypes.sizeof(pi)
        assert ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi),pi.cb)
        assert (pi.CommitLimit-pi.CommitTotal)*pi.PageSize>=6*2**30
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%256==0:guard()
        assert digest(p)==h,("SHA",p)
result=read(ROOT/"RESULT.json");pf=read(ROOT/"preflight.json")
old=read(ROOT/"old_results_manifest.json")["files"];src=pf["source_hashes"]
assert result["status"]=="PASS_DIAGNOSTIC_ONLY_PRIMARY_PARAMETER_CONCORDANCE"
verify(old);verify(src)
assert read(ROOT/"worker_state.json")["status"]=="SUCCESS"
assert read(ROOT/"exit_status.json")["status"]=="SUCCESS"
assert read(ROOT/"ARITHMETIC_AUDIT.json")["status"]=="PASS"
assert read(ROOT/"FROZEN_AUDIT.json")["status"]=="PASS"
expected={}
for split in ("FIT","CAL"):
    with np.load(ROOT.parent/"F16_MATCHED_FIT_CAL_TRANSFER_DIAGNOSIS_20261005"/"jobs"/("native_"+split+".npz"),allow_pickle=False) as z:expected[split]=z["sample_id"]
groups=["stable","transition","change","overall"]
rows=[];maxerr=0;outputs={}
for rec in result["records"]:
    guard()
    done=read(ROOT/"jobs"/(rec["key"]+".done.json"))
    assert done["status"]=="SUCCESS" and done["source_hashes"]==src
    verify(done["outputs"]);outputs.update(done["outputs"])
    summary=read(next(p for p in rec["outputs"] if p.endswith(".json")))
    layout=summary["layout"]
    assert layout[0]["start"]==0 and layout[-1]["stop"]==45798
    assert all(a["stop"]==b["start"] for a,b in zip(layout,layout[1:]))
    branches={}
    for name in ("state_encoder","query","key","value","head"):
        branches[name]=np.concatenate([np.arange(x["start"],x["stop"]) for x in layout if x["name"].startswith(name+".")])
    for name,rg in (("D2",range(2)),("A2",range(2,4)),("clock2",range(4,6))):
        ix=[]
        for x in layout:
            if x["name"]=="head.4.weight":
                for r in rg:ix.extend(range(x["start"]+r*x["shape"][1],x["start"]+(r+1)*x["shape"][1]))
            if x["name"]=="head.4.bias":ix.extend(x["start"]+r for r in rg)
        branches[name]=np.array(ix)
    with np.load(next(p for p in rec["outputs"] if p.endswith(".npz")),allow_pickle=False) as z:v={k:z[k] for k in z.files}
    vectors={}
    for split,n in (("FIT",25912),("CAL",2879)):
        assert np.array_equal(v[split+"_sample_id"],expected[split])
        count=v[split+"_counts"];assert int(count.sum())==n
        assert np.isfinite(v[split+"_gradient"]).all() and v[split+"_gradient"].shape==(3,45798)
        ov=(v[split+"_gradient"]*count[:,None]).sum(axis=0)/n
        assert np.array_equal(ov,v[split+"_overall_gradient"])
        assert abs(np.dot(v[split+"_quality"],count)/n-v[split+"_overall_quality"])<1e-12
        vectors[split]=list(v[split+"_gradient"])+[ov]
    for i,f in enumerate(groups):
        for j,c in enumerate(groups):
            saved=rec["summary"][f+"__"+c]
            for name,ix in [("all",np.arange(45798))]+list(branches.items()):
                a=vectors["FIT"][i][ix];b=vectors["CAL"][j][ix]
                an=float(np.sqrt(np.sum(a*a)));bn=float(np.sqrt(np.sum(b*b)));dot=float(np.sum(a*b))
                new={"FITnorm":an,"CALnorm":bn,"dot":dot,"cos":dot/an/bn if min(an,bn)>1e-10 else None,"CAL_directional_derivative":-dot/an if min(an,bn)>1e-10 else None}
                target=saved["all"] if name=="all" else saved["branches"][name]
                for k,x in new.items():
                    if x is None:assert target[k] is None
                    else:
                        er=abs(x-target[k]);maxerr=max(maxerr,er);assert er<=1e-12+1e-10*abs(x),(rec["key"],f,c,name,k,er)
    rows.append({"key":rec["key"],"model":rec["model"],"epoch":rec["info"]["epoch"],"pairs":rec["summary"],"quality":{"FIT":v["FIT_quality"].tolist(),"CAL":v["CAL_quality"].tolist()}})
    del v,vectors
verify(outputs);verify(src);verify(old)
stats={}
for policy in ("constrained","final","all"):
    subset=[r for r in rows if policy=="all" or r["model"]["policy"]==policy]
    stats[policy]={}
    for pair in result["records"][0]["summary"]:
        stats[policy][pair]={}
        for branch in ["all"]+list(branches):
            xs=[r["pairs"][pair]["all"] if branch=="all" else r["pairs"][pair]["branches"][branch] for r in subset]
            cs=[x["cos"] for x in xs if x["cos"] is not None]
            stats[policy][pair][branch]={"n":len(xs),"aligned":sum(x["dot"]>0 and x["cos"] is not None for x in xs),"conflict":sum(x["dot"]<0 and x["cos"] is not None for x in xs),"undetermined":sum(x["cos"] is None for x in xs),"cos_range":[min(cs),max(cs)] if cs else None}
review={"status":"PASS_CPU_SAVED_VECTORS_RECOMPUTE","models":18,"outputs":len(outputs),"sources":len(src),"old_files":len(old),"max_numeric_error":maxerr,"before_after_SHA":"PASS","no_model_forward":True,"no_optimizer":True,"not_all_gradients_replayed":True,"private_peak_GiB":peak/2**30,"summary_counts":stats,"rows":rows}
(OUT/"REVIEW.json").write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in review.items() if k not in ("rows","summary_counts")}))
for pol in ("constrained","final"):
    print(pol,json.dumps({p:{b:stats[pol][p][b] for b in ("all","head","D2","A2","clock2")} for p in ("overall__overall","change__change","overall__change")}))

