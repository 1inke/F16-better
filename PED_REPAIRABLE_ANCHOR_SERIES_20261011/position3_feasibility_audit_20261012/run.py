import common as c
import numerics as n
import numpy as np,time,ast
from scipy.cluster.hierarchy import linkage,fcluster
def quality_banks(p,gt,ind=False):
    e=np.sqrt(((p-gt[:,None])**2).sum(-1)) if ind else np.linalg.norm(p-gt[:,None],axis=-1)
    a=e.mean(-1);f=e[:,:,-1];j=np.maximum(a/.2,f/.4);good=(a<=.2)&(f<=.4);counts=[]
    for paths,g in zip(p,good):
        if ind:
            d=[]
            for i in range(19):
                err=np.sqrt(((paths[i]-paths[i+1:])**2).sum(-1));d.extend(np.maximum(err.mean(-1)/.1,err[:,-1]/.2).tolist())
        else:
            err=np.linalg.norm(paths[:,None]-paths[None,:],axis=-1);d=np.maximum(err.mean(-1)/.1,err[:,:,-1]/.2)[np.triu_indices(20,1)]
        lab=fcluster(linkage(d,method="complete"),1.,criterion="distance")
        counts.append([len(np.unique(lab)),len(np.unique(lab[g]))])
    return dict(ADE=a.min(1),FDE=f.min(1),J=j.min(1),good=good.any(1),clusters=np.array(counts))
def compact(x):
    return {k:v for k,v in x.items() if k not in ("raw32","unit_controls")}
def inherited():
    files=c.read(c.P/"source_manifest.json")["files"].copy()
    old=c.read(c.P/"completion_review/REVIEW.json")["scientific_outputs_sha"]
    for p,h in old.items():
        assert p not in files or files[p]==h;files[p]=h
    for p in [c.P/"source_manifest.json",c.P/"completion_review/REVIEW.json",c.P/"RESULT_ANALYSIS.md",c.P/"RESULT_ANALYSIS.json"]:
        h=c.sha(p);assert str(p) not in files or files[str(p)]==h;files[str(p)]=h
    for p in c.H.iterdir():
        if p.suffix in (".py",".ps1") or p.name in ("README.md","DESIGN.md","protocol.json"):files[str(p)]=c.sha(p)
    return files
def cases(lab,chosen):
    return [(int(i),int(s)) for i in chosen for s in np.flatnonzero(lab["potential"][i])]
def checked(native,gt,s,mode):
    z,path,bank=n.solve(native,gt,s,mode,c.guard)
    a,q,b=n.independent_verify(native,gt,s,z["raw32"])
    err=float(np.max(np.abs(q-path)));assert err<=2e-12,("RECONSTRUCT",err)
    z["reconstruction_max"]=err;z["independent_witness"]=a["witness"]
    z["safe_witness"]=bool(z["witness"] and a["witness"])
    assert np.array_equal(bank[np.arange(20)!=s],native.astype(float)[np.arange(20)!=s])
    return z,path
def main():
    started=time.perf_counter();files=c.read(c.H/"source_manifest.json")["files"]
    c.stage("INITIAL_SOURCE_SHA");c.verify(files)
    reg,lab,ids,native,gt,chosen=c.load();todo=cases(lab,chosen)
    pre=c.npz(c.H/"preflight_sample.npz");assert np.array_equal(chosen,pre["index"])
    c.save("sample.npz",index=chosen,sample_id=ids[chosen],track=reg["track"][chosen],groups=reg["groups"][chosen],native=native[chosen],gt=gt[chosen],potential=lab["potential"][chosen])
    records=[];paths=[];raw=[];unit=[]
    for i,s in todo:
        for mode in ("shape","pointwise"):
            c.stage("PER_CASE_NUMERICAL_DIAGNOSTIC",done=len(records),total=2*len(todo),sample_id=int(ids[i]),slot=s,mode=mode)
            t=time.perf_counter();z,p=checked(native[i],gt[i],s,mode)
            records.append(dict(index=i,sample_id=int(ids[i]),slot=s,group=int(reg["groups"][i]),elapsed=time.perf_counter()-t,**compact(z)))
            paths.append(p);raw.append(z["raw32"]);unit.append(z["unit_controls"])
            c.write("partial_records.json",records)
    c.save("solutions.npz",paths=np.array(paths),raw32=np.array(raw),unit_controls=np.array(unit),index=np.array([z["index"] for z in records]),slot=np.array([z["slot"] for z in records]),mode=np.array([z["mode"] for z in records]))
    c.write("records.json",records)
    oracle=native[chosen].astype(float).copy();which=np.full(64,-1,dtype=int)
    for w,i in enumerate(chosen):
        found=[k for k,z in enumerate(records) if z["index"]==int(i) and z["safe_witness"]]
        if found:
            k=found[0];which[w]=k;oracle[w,records[k]["slot"]]=paths[k]
    c.save("oracle_diagnostic.npz",paths=oracle,selected_record=which)
    c.stage("SAVED_METRICS");groups={}
    old=quality_banks(native[chosen].astype(float),gt[chosen]);new=quality_banks(oracle,gt[chosen])
    assert not old["good"].any()
    for name,mask in [("overall",np.ones(64,bool))]+[(v,reg["groups"][chosen]==g) for g,v in enumerate(("stable","transition","change"))]:
        m=n.motion(oracle[mask]);b=n.motion(native[chosen][mask]);good=new["good"][mask]
        groups[name]=dict(n=int(mask.sum()),safe_windows=int(good.sum()),native_motion=b,oracle_motion=m,pooled_allfive_PASS=bool(np.all(m<=1.01*b)),minADE_mean=float(new["ADE"][mask].mean()),minFDE_mean=float(new["FDE"][mask].mean()),J_mean=float(new["J"][mask].mean()),effective_good_mean=float(new["clusters"][mask,1].mean()),distinct_mean=float(new["clusters"][mask,0].mean()))
    outputs={str(p):c.sha(p) for p in [c.H/x for x in ("sample.npz","solutions.npz","records.json","oracle_diagnostic.npz")]}
    c.stage("FINAL_SOURCE_SHA");c.verify(files)
    c.write("RESULT.json",dict(status="PASS_DIAGNOSTIC_EXECUTION_ONLY",windows=64,anchors=len(todo),solves=len(records),groups=groups,output_sha=outputs,source_count=len(files),elapsed_seconds=time.perf_counter()-started,resource=c.RESOURCE,no_neural_training=True,unknown_is_not_infeasible=True))
if __name__=="__main__":main()
