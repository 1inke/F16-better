import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as c
import numpy as np,ast
from collections import Counter
def main():
    c.guard();files=c.read(c.H/"source_manifest.json")["files"];res=c.read(c.H/"RESULT.json")
    initial={str(p):c.sha(p) for p in [Path(__file__),c.H/"completion_review/REVIEW.json",c.H/"completion_review/CASES.json"]}
    c.verify(files);c.verify(res["output_sha"])
    sample=c.npz(c.H/"sample.npz");sol=c.npz(c.H/"solutions.npz");oracle=c.npz(c.H/"oracle_diagnostic.npz");records=c.read(c.H/"records.json");cases=c.read(c.H/"completion_review/CASES.json")
    src=c.R/"F16_CANONICAL_EQUAL_K20_REPLACEMENT_DIAGNOSIS_20261007/run.py"
    tree=ast.parse(src.read_text(encoding="utf-8-sig"));nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ("errors","motion")]
    assert len(nodes)==2
    env=dict(np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),"exec"),env)
    motion=env["motion"];errors=env["errors"];maxerr=0.;groups={}
    for group,mask in [("overall",np.ones(64,bool))]+[(v,sample["groups"]==g) for g,v in enumerate(("stable","transition","change"))]:
        original=sample["native"][mask].astype(float);new=oracle["paths"][mask];gt=sample["gt"][mask].astype(float)
        bm=motion(original,True);nm=motion(new,True);bmarr=np.array(list(bm.values()));nmarr=np.array(list(nm.values()))
        assert np.array_equal(bmarr,res["groups"][group]["native_motion"])
        assert np.array_equal(nmarr,res["groups"][group]["oracle_motion"])
        before=errors(original,gt,True);after=errors(new,gt,True)
        assert int(after[3].any(1).sum())==res["groups"][group]["safe_windows"] and not before[3].any(1).any()
        groups[group]=dict(n=int(mask.sum()),native_minADE=float(before[0].min(1).mean()),oracle_minADE=float(after[0].min(1).mean()),native_minFDE=float(before[1].min(1).mean()),oracle_minFDE=float(after[1].min(1).mean()),native_J=float(before[2].min(1).mean()),oracle_J=float(after[2].min(1).mean()),motion_percent=100*(nmarr/bmarr-1),pooled_five_PASS=bool(np.all(nmarr<=bmarr*1.01)))
    lookup={int(v):j for j,v in enumerate(sample["index"])}
    status={};qualityonly={}
    for mode in ("shape","pointwise"):
        zz=[z for z in records if z["mode"]==mode]
        status[mode]=dict(solves=len(zz),success=sum(z["solver_success"] for z in zz),codes=dict(Counter(str(z["solver_status"]) for z in zz)),safe_records=sum(z["safe_witness"] for z in zz),safe_windows=len(set(z["index"] for z in zz if z["safe_witness"])),quality_bound_windows=len(set(z["index"] for z in zz if z["quality_good"] and z["bound_literal"])),nonconverged_safe=sum(z["safe_witness"] and not z["solver_success"] for z in zz))
    for case in cases:
        if case["status"]!="QUALITY_WITNESS_ONLY":continue
        zz=[z for z in records if z["index"]==case["index"] and z["quality_good"] and z["bound_literal"]]
        qualityonly[str(case["sample_id"])]=dict(quality_records=len(zz),all_fail_window_motion=all(not z["window_allfive_PASS"] for z in zz),arithmetic_disagreement=any(z["witness"]!=z["independent_witness"] for z in zz),failed_motion=sorted(set(k for z in zz for k,x,y in zip(("speed_p95","accel_mean","accel_p95","jerk_mean","jerk_p95"),z["motion"],z["native_motion"]) if x>1.01*y)))
    # Full saved per-case motion from the separate historical concatenation implementation.
    for k,z in enumerate(records):
        j=lookup[z["index"]];bank=sample["native"][j:j+1].astype(float).copy();bank[0,z["slot"]]=sol["paths"][k]
        measured=np.array(list(motion(bank,True).values()));err=float(np.max(np.abs(measured-np.array(z["motion"]))));maxerr=max(maxerr,err);assert err<=2e-12
    c.verify(files);c.verify(res["output_sha"]);c.verify(initial)
    out=dict(status="PASS",groups=groups,solver=status,quality_only=qualityonly,all176_motion_maxerror=maxerr,source_SHA_unchanged=True,output_SHA_unchanged=True,supplement_SHA=initial,no_solve_no_model=True)
    c.write("completion_review/SUPPLEMENT.json",out);print(out,flush=True)
if __name__=="__main__":main()
