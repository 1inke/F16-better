"""Actual fixed budget/source/ID/weighted numerical preflight."""
import common as c
c.guard()
import selector as sel,numerics as n,labels as lab
import numpy as np,time,traceback,ast
def checked(native,gt,slot,mode):
    rec,path,bank=n.solve(native,gt,slot,mode,c.guard);other,q,b=n.independent_verify(native,gt,slot,rec["raw32"])
    err=float(abs(q-path).max());assert err<=2e-12,("DECODE_ERROR",err)
    assert np.array_equal(bank[np.arange(20)!=slot],native.astype(float)[np.arange(20)!=slot])
    rec["independent_witness"]=other["witness"];rec["safe_witness"]=bool(rec["witness"] and other["witness"]);rec["reconstruction_max"]=err
    return rec,path
def main():
    started=time.perf_counter();c.stage("PREFLIGHT_SOURCE_ID")
    for k in range(32):c.write("io_fixture.json",dict(n=k))
    assert c.read(c.H/"io_fixture.json")==dict(n=31)
    assert c.digest({"motion":np.array([1.,2.])})==c.digest({"motion":[1.,2.]})
    # Actual first preserved prefix record is reconstructed before production with allnonelapsed fields exact.
    oldprefix=c.read(c.PARENT/"partial_records.json");assert len(oldprefix)==19
    inherited=c.read(c.PARENT/"io_recovery_20261012/source_manifest.json")["files"];t=time.perf_counter();c.verify(inherited);sha_seconds=time.perf_counter()-t
    d=c.load();assert d["native"].shape==(384,20,12,2) and d["X"].shape==(384,20,136)
    assert c.sha(c.H/"numerics.py")==c.sha(c.F/"numerics.py"),"OLD_SOLVER_EQUATIONS_CHANGED"
    for path in c.H.glob("*.py"):ast.parse(path.read_text(encoding="utf8"))
    if (c.H/"sample_design.npz").exists():
        prior=c.npz(c.H/"sample_design.npz");assert all(np.array_equal(prior[k],v) for k,v in d.items() if isinstance(v,np.ndarray))
    else:c.save("sample_design.npz",**{k:v for k,v in d.items() if isinstance(v,np.ndarray)})
    c.write("SCALE_CHECK.json",dict(status="PASS_FIXED_QUOTA_ID_FEATURE_ONLY",groups=d["scale"],windows=384,potential_anchors=int(d["envelope"].sum()),excluded_discovery_tracks=len(d["excluded_tracks"]),source_input_scope="Aonlydataanalysis; full registry onlyidentitychecks",native_encoder_IDpast_features_exact=True))
    timings=[];recs=[];todo=[(i,int(s)) for i in range(384) for s in np.flatnonzero(d["envelope"][i])]
    for i,s in todo[:4]:
        c.stage("PREFLIGHT_FIXED_DOUBLE_SOLVER",window=i,slot=s);t=time.perf_counter()
        for mode in ("shape","pointwise"):
            rec,p=checked(d["native"][i],d["gt"][i],s,mode)
            prefix_record=dict(window=i,index=int(d["index"][i]),sample_id=int(d["ids"][i]),slot=int(s),**{k:v for k,v in rec.items() if k not in ("raw32","unit_controls")})
            j=len(recs);assert c.digest(prefix_record)==c.digest({k:v for k,v in oldprefix[j].items() if k!="elapsed"}),"PRELAUNCH_PREFIX_METADATA_MISMATCH"
            recs.append(dict(window=i,slot=s,**{k:v for k,v in rec.items() if k not in ("raw32","unit_controls")}))
        timings.append(time.perf_counter()-t)
    # Conservative endpoint certificate on all384 without declaring unsolved anchors negative.
    y,cert=lab.build(d["native"],d["gt"],d["envelope"],np.zeros_like(d["envelope"]))
    assert not lab.endpoint_excluded(np.array([.6000000005,0]),np.zeros(2)) and lab.endpoint_excluded(np.array([.61,0]),np.zeros(2))
    assert not np.any(d["envelope"]&(y==0))
    # Fixed synthetic labels on actual32 features: numerical proof only, no actual witness claim/H4.
    X=d["X"][:32];fy=np.zeros((32,20),np.int8);fy[:,0]=1;fy[:,1]=-1;e=np.zeros_like(fy,bool);e[:,:2]=True;mean=X.astype(float).mean((0,1));sd=X.astype(float).std((0,1));sd=np.where(sd<1e-6,1.,sd)
    audits=sel.numerical_fixture(X,e,fy,d["track"][:32],mean,sd)
    for mode in ("envelope","witness"):
        _,known,w=lab.supervision(e,fy,d["track"][:32],mode);assert abs(w.sum()-640)<1e-8
    # Fixed selectors match original ranking policies exactly, including original-logit ties.
    oldcore=c.module("registered_reference",c.R/"PED_FOUR_DIRECTION_SERIES_20261011/NEXT_EXPERIMENT_DESIGN_20261011/core_design.py")
    assert np.array_equal(lab.random4(d["ids"]),oldcore.fixed_random4(d["ids"])) and np.array_equal(lab.bottom4(d["logs"]),oldcore.bottom4(d["logs"]))
    try:c.verify({str(c.F/"solutions.npz"):"0"*64})
    except AssertionError:pass
    else:raise AssertionError("BAD_OUTPUT_ACCEPTED")
    try:c.verify({str(c.H/"labels.py"):"0"*64})
    except AssertionError:pass
    else:raise AssertionError("BAD_SOURCE_ACCEPTED")
    planning=(2*max(timings)*len(todo)+2*sha_seconds+240)/60
    c.write("NUMERICAL_PREFLIGHT.json",dict(status="PASS",fixed_four_anchor_seconds=timings,records=recs,endpoint_certificate=cert,weighted_fixtures=audits,fixture_not_scientific_training_labels=True))
    if planning>30:
        c.write("PREFLIGHT.json",dict(status="STOP_PLANNING_EXCEEDS30",planning_minutes=planning,windows=384,anchors=len(todo),timings=timings,resource=c.RESOURCE));return
    files,inherit=c.locks();c.verify(files);c.write("source_manifest.json",dict(files=files,inherited=inherit,digest=c.digest(files)))
    c.write("PREFLIGHT.json",dict(status="PASS",planning_minutes=planning,source_count=len(files),source_sha=c.digest(files),inherited=inherit,windows=384,anchors=len(todo),solves=2*len(todo),fixed_four_anchor_seconds=timings,sha_seconds=sha_seconds,elapsed_seconds=time.perf_counter()-started,resource=c.RESOURCE,no_neural_training=True,no_full_label_or_OOF_performance_claim=True))
    c.stage("PREFLIGHT_PASS",planning_minutes=planning)
if __name__=="__main__":
    try:main()
    except Exception:c.write("PREFLIGHT_FAILURE_"+str(time.time_ns())+".json",dict(traceback=traceback.format_exc(),resource=c.RESOURCE));raise
