"""S1 actual CPU-only preflight. No GPU/epoch-resume claims."""
import common as c
import selector as sel
import ast,math,time,gc,traceback,inspect
from numerical_audit import reference_audit
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from threadpoolctl import threadpool_limits
def preflight():
    t=time.perf_counter();c.guard();assert sel.sklearn.__version__=="1.4.1.post1";files,old=c.locks();c.verify(files);c.write("source_manifest.json",dict(files=files))
    d,s=c.load();reg=c.npz(c.H/"split_registry.npz");saved=c.npz(c.H/"labels_offline.npz")
    assert np.array_equal(reg["sample_id"],d["ids"]) and np.array_equal(saved["sample_id"],d["ids"])
    for k in s:assert np.array_equal(reg[k+"_index"],s[k])
    for start in range(0,len(d["ids"]),256):
        ix=np.arange(start,min(start+256,len(d["ids"])));a=c.label(d,ix);b=c.label(d,ix,True)
        for k in a:
            if a[k].dtype==bool:assert np.array_equal(a[k],b[k])
            else:assert np.max(abs(a[k]-b[k]))<=1e-12
        assert all(np.array_equal(a[k],saved[k][ix]) for k in a)
    A=s["A"];tr=A[d["fold"][A]!=0][:513];va=A[d["fold"][A]==0][:128]
    m1,r1=sel.fit(d,tr);m2,r2=sel.fit(d,tr)
    numerical=reference_audit(d,tr,m1);c.write("NUMERICAL_REPAIR_AUDIT.json",numerical)
    assert all(np.array_equal(m1[k],m2[k]) for k in m1),"REPEATED_FIT_NOT_EXACT"
    sc1=sel.predict(d,va,m1);sc2=sel.predict(d,va,m2);assert np.array_equal(sc1,sc2)
    z=c.core.select4(sc1);assert all(np.array_equal(a,np.sort(np.argsort(-b,kind="stable")[:4])) for a,b in zip(z,sc1))
    assert np.array_equal(c.core.select4(np.zeros((3,20))),np.tile(np.arange(4),(3,1)))
    assert np.array_equal(c.core.bottom4(np.zeros((3,20))),np.tile(np.arange(16,20),(3,1)))
    changed=dict(d);changed["gt"]=d["gt"]+1000;changed["groups"]=d["groups"][::-1].copy();changed["logs"]=d["logs"]+123
    assert np.array_equal(sc1,sel.predict(changed,va,m1)),"PREDICTION_USES_GT_GROUP_LOGIT"
    c.save("preflight_smoke/model1.npz",**m1);c.save("preflight_smoke/model2.npz",**m2)
    p=c.save("preflight_smoke/saved128.npz",sample_id=d["ids"][va],index=va,native20=d["native"][va],gt=d["gt"][va],groups=d["groups"][va],scores=sc1,slots=z)
    reload=c.npz(p);assert np.array_equal(reload["scores"],sel.predict(d,reload["index"],c.npz(c.H/"preflight_smoke/model1.npz")))
    # Read-only AST, no legacy top-level module execution, original SciPy complete-link.
    path=c.R/"PED_FOUR_DIRECTION_SERIES_20261011/experiment3_samepath_quality/completion_review/audit_saved.py"
    nodes=[n for n in ast.parse(path.read_text(encoding="utf-8-sig")).body if isinstance(n,ast.FunctionDef) and n.name in ("error","motion","clusters")]
    ns=dict(np=np,guard=c.guard,linkage=linkage,fcluster=fcluster);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),ns)
    # Original motion/cluster functions expect float64 caller input; saved cache remains float32 unchanged.
    reload["native20"]=reload["native20"].astype(np.float64);reload["gt"]=reload["gt"].astype(np.float64)
    a,ff,j,g,one=ns["error"](reload["native20"],reload["gt"]);lab=c.label(d,va)
    assert np.array_equal(g,lab["candidate_good"]) and np.max(abs(j-lab["j"]))<=1e-12
    cc=ns["clusters"](reload["native20"],g);motion=ns["motion"](reload["native20"])
    c.save("preflight_smoke/native_quality.npz",sample_id=reload["sample_id"],ADE=a,FDE=ff,J=j,clusters=cc,early_middle_late=one)
    # Independent direct pairwise complete-link and five motion arithmetic.
    other=[]
    for paths,good in zip(reload["native20"].astype(float),g):
        DD=np.linalg.norm(paths[:,None]-paths[None],axis=-1);D=np.maximum(DD.mean(2)/.1,DD[:,:,11]/.2);ii=np.triu_indices(20,1)
        cl=fcluster(linkage(D[ii],method="complete"),1,criterion="distance");other.append([len(set(cl)),len(set(cl[good]))])
    assert np.array_equal(cc,np.asarray(other))
    pp=reload["native20"].astype(float);v=np.diff(pp,axis=2,prepend=np.zeros_like(pp[:,:,:1]))/.4;acc=np.diff(v,axis=2)/.4;jerk=np.diff(acc,axis=2)/.4
    vm=np.linalg.norm(v,axis=-1);am=np.linalg.norm(acc,axis=-1);jm=np.linalg.norm(jerk,axis=-1)
    mm=dict(speed_p95=float(np.quantile(vm,.95)),accel_mean=float(am.mean()),accel_p95=float(np.quantile(am,.95)),jerk_mean=float(jm.mean()),jerk_p95=float(np.quantile(jm,.95)))
    assert all(abs(motion[k]-mm[k])<=1e-12 for k in motion)
    # Saved model binding and deliberate bad-source/output rejection, no overwrite.
    expected={str(p):c.sha(p)};c.verify(expected)
    try:c.verify({str(p):"0"*64})
    except AssertionError:pass
    else:raise AssertionError("BAD_OUTPUT_NOT_REJECTED")
    try:c.verify({str(c.H/"common.py"):"0"*64})
    except AssertionError:pass
    else:raise AssertionError("BAD_SOURCE_NOT_REJECTED")
    timings=[];batch=s["B"][:256]
    with threadpool_limits(limits=1):
        for i in range(36):
            b=time.perf_counter();sel.predict(d,batch,m1)
            if i>=4:timings.append(time.perf_counter()-b)
    # One fixed larger TRAIN-only runtime probe, never validation outcome/model selection.
    btr=A[d["fold"][A]!=0][:1033];bm,br=sel.fit(d,btr);del bm;gc.collect()
    fit_plan=4*br["elapsed_seconds"]*(len(A)/len(btr))/60
    # Empirical planning with3x whole-fit safety +3min save/audit/SHA overhead.
    # max_iter500 remains a numerical stopping cap, not an assertion of500 observed Newton steps.
    planning=math.ceil(max(5.,fit_plan*3.+3.))
    c.write("BENCHMARK.json",dict(actual1033_fit=br,fixture513_reports=[r1,r2],full4fit_scaled_minutes=fit_plan,planning_minutes=planning,method="actual1033 wholefit linear rowscale x4 x3 safety plus3min IO/audit",old_max500_formula_NOT_actual_ETA=True,no_saved_benchmark_model_or_H4_outcome_selection=True))
    assert planning<=30,("STOP_PLANNING_OVER30",planning)
    c.verify(files);c.guard()
    result=dict(status="PASS",numerical_repair=numerical,source_sha=c.digest(files),locked_files=len(files),inherited=old,actual_A=len(A),actual_B=len(s["B"]),repeat_actual513_fit_params_stats_probabilities_BITWISE=True,no_GPU_torch_optimizer_epoch_RNG_claim=True,all28791_labels_independent_and_cache_ID_EXACT=True,actual128_saved_selection_full_original_SciPy_quality_five_motion_PASS=True,GT_group_logit_prediction_invariance=True,bad_source_output_rejected=True,fit_fixture_reports=[r1,r2],infer256_4warm32mean_seconds=float(np.mean(timings)),actual1033_benchmark_report=br,empirical_four_fit_extrapolation_minutes=fit_plan,planning_minutes=planning,ETA_minutes=[max(2,planning*.6),planning*1.8],resource=c.RESOURCE,disk_free_GiB=__import__("shutil").disk_usage(c.H).free/2**30,elapsed_seconds=time.perf_counter()-t,production_peak_not_measured=True)
    assert result["disk_free_GiB"]>=2
    c.write("PREFLIGHT.json",result);c.write("prelaunch_review/REVIEW.json",dict(status="PASS",source_sha=result["source_sha"],actualCPU_fit_and_saved128=True,no_torch=True,ARS_checkpoint1="PASS_INLINE_NOT_EXTERNAL; optimistic labels not physical feasibility; A/B identity separation; stop linear scheme on gate failure"))
    print(result,flush=True)
if __name__=="__main__":
    try:preflight()
    except Exception:c.write("PREFLIGHT_FAILURE_"+str(time.time_ns())+".json",dict(traceback=traceback.format_exc(),resource=c.RESOURCE));raise
