import common as c
import numerics as n
import run as r
import numpy as np,time
def main():
    started=time.perf_counter();c.guard();files=c.read(c.H/"source_manifest.json")["files"];c.verify(files)
    result=c.read(c.H/"RESULT.json");c.verify(result["output_sha"])
    sample=c.npz(c.H/"sample.npz");sol=c.npz(c.H/"solutions.npz");records=c.read(c.H/"records.json");saved=c.npz(c.H/"oracle_diagnostic.npz")
    reg,lab,ids,native,gt,chosen=c.load();assert np.array_equal(chosen,sample["index"])
    assert np.array_equal(sample["native"],native[chosen]) and np.array_equal(sample["gt"],gt[chosen])
    todo=[(i,s,m) for i,s in r.cases(lab,chosen) for m in ("shape","pointwise")]
    assert len(todo)==len(records)==len(sol["paths"])
    maxerror=0.;safe=[];quality=[]
    for k,(i,s,mode) in enumerate(todo):
        c.guard();z=records[k];assert (z["index"],z["slot"],z["mode"])==(i,s,mode)
        check,q,bank=n.independent_verify(native[i],gt[i],s,sol["raw32"][k])
        err=float(np.max(np.abs(q-sol["paths"][k])));maxerror=max(maxerror,err);assert err<=2e-12
        first,path,_=n.validate(native[i],gt[i],s,sol["raw32"][k])
        assert np.array_equal(path,sol["paths"][k])
        assert z["safe_witness"]==bool(check["witness"] and first["witness"])
        assert abs(z["ADE"]-check["ADE"])<=2e-12 and abs(z["FDE"]-check["FDE"])<=2e-12
        safe.append(z["safe_witness"]);quality.append(first["quality_good"] and first["bound_literal"])
    oracle=sample["native"].astype(float).copy();which=np.full(64,-1,dtype=int);rows=[]
    oof=c.npz(c.P/"interaction/OOF_predictions.npz");lookup={int(x):j for j,x in enumerate(oof["index"])}
    for w,i in enumerate(chosen):
        ix=[k for k,z in enumerate(records) if z["index"]==int(i)]
        found=[k for k in ix if safe[k]];qw=[k for k in ix if quality[k]]
        if found:
            k=found[0];which[w]=k;oracle[w,records[k]["slot"]]=sol["paths"][k]
        slots=sorted(set(records[k]["slot"] for k in found));j=lookup[int(i)]
        rows.append(dict(index=int(i),sample_id=int(ids[i]),group=int(reg["groups"][i]),potential_slots=np.flatnonzero(lab["potential"][i]).tolist(),safe_slots=slots,quality_slots=sorted(set(records[k]["slot"] for k in qw)),status="SAFE_WITNESS" if found else "QUALITY_WITNESS_ONLY" if qw else "UNKNOWN",interaction_hit=bool(set(slots)&set(oof["selected_slots"][j])),random_hit=bool(set(slots)&set(oof["random_slots"][j])),bottom_hit=bool(set(slots)&set(oof["bottom_slots"][j]))))
    assert np.array_equal(oracle,saved["paths"]) and np.array_equal(which,saved["selected_record"])
    old=r.quality_banks(sample["native"].astype(float),sample["gt"],True);new=r.quality_banks(oracle,sample["gt"],True)
    direct=r.quality_banks(oracle,sample["gt"]);assert all(np.array_equal(new[k],direct[k]) for k in new)
    groups={}
    for name,mask in [("overall",np.ones(64,bool))]+[(v,sample["groups"]==g) for g,v in enumerate(("stable","transition","change"))]:
        rowsg=[z for z,m in zip(rows,mask) if m];v=result["groups"][name]
        assert v["safe_windows"]==int(new["good"][mask].sum())
        assert np.array_equal(v["oracle_motion"],n.motion(oracle[mask]))
        groups[name]=dict(n=len(rowsg),safe=sum(z["status"]=="SAFE_WITNESS" for z in rowsg),quality_only=sum(z["status"]=="QUALITY_WITNESS_ONLY" for z in rowsg),unknown=sum(z["status"]=="UNKNOWN" for z in rowsg),interaction_hit=sum(z["interaction_hit"] for z in rowsg),random_hit=sum(z["random_hit"] for z in rowsg),bottom_hit=sum(z["bottom_hit"] for z in rowsg),pooled_allfive_PASS=v["pooled_allfive_PASS"],motion_native=v["native_motion"],motion_oracle=v["oracle_motion"])
    c.verify(files);c.verify(result["output_sha"])
    c.write("completion_review/CASES.json",rows)
    c.write("completion_review/REVIEW.json",dict(status="PASS",groups=groups,cases=len(records),independent_reconstruct_max=maxerror,source_count=len(files),all_outputs_SHA_PASS=True,source_SHA_PASS=True,elapsed_seconds=time.perf_counter()-started,resource=c.RESOURCE,no_solve_no_model=True,not_external_review=True))
    print(groups,flush=True)
if __name__=="__main__":main()
