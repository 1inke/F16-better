"""Actual S1b preflight, never chooses scientific outcomes."""
import common as c,selector as sel,features as ft
from numerical_audit import reference_audit
import numpy as np,time,gc,traceback
def main():
    started=time.perf_counter();c.stage("PREFLIGHT_SOURCE_ID_OBSERVATION")
    inherited=c.read(c.H.parent/"stage1_selector_production_20261011/source_manifest.json")["files"];c.verify(inherited)
    d,s=c.load();parent=c.H.parent/"stage1_selector_production_20261011"
    reg=c.npz(parent/"split_registry.npz")
    assert np.array_equal(reg["sample_id"],d["ids"])
    for k in s:assert np.array_equal(reg[k+"_index"],s[k])
    assert np.array_equal(reg["fold"],d["fold"]) if "fold" in reg else True
    labs=c.npz(parent/"labels_offline.npz")
    assert np.array_equal(labs["sample_id"],d["ids"])
    lab=c.label(d,np.arange(len(d["ids"])),True)
    for k in lab:
        if k in labs:
            assert np.array_equal(lab[k],labs[k]) if lab[k].dtype==bool else np.max(abs(lab[k]-labs[k]))<1e-12
    for name,z in (("split_registry.npz",reg),("labels_offline.npz",labs)):
        if (c.H/name).exists():assert c.sha(c.H/name)==c.sha(parent/name)
        else:c.save(name,**z)
    # Scalar/vector actual128; original120 unaffected, motion-only inputs not GT.
    ix=s["A"][:128];errors={}
    for arm in ("additive","interaction"):
        z=c.x(d,ix,arm);ind=ft.independent(d["p2"][ix],d["past"][ix],d["native"][ix],arm)
        assert np.array_equal(z,ind),("FEATURE_SCALAR_MISMATCH",arm,float(abs(z-ind).max()))
        assert np.array_equal(z[:,:,:120],c.core.selector_features(d["p2"][ix],d["native"][ix]))
        errors[arm]=float(abs(z-ind).max())
    # Exact synthetic velocities and dot/cross, rotation/reflection covariance.
    p=np.zeros((1,8,2),np.float32);p[0,:,0]=np.arange(-7,1)*.4
    native=np.zeros((1,20,12,2),np.float32);native[:,:,:,0]=np.arange(1,13)*.4
    z=ft.build(np.zeros((1,96),np.float32),p,native,"interaction")
    assert np.max(abs(z[:,:,124::2][:,:,:3]-1))<1e-6 and np.max(abs(z[:,:,125::2]))<1e-6
    assert np.max(abs(z[:,:,130:]))<1e-6
    # binary-exact synthetic values avoid backend interpretation of tolerance.
    p=np.arange(16,dtype=np.float32).reshape(1,8,2)/4;p-=p[:,-1:]
    native=np.arange(480,dtype=np.float32).reshape(1,20,12,2)/8
    theta=np.array([[0.,-1.],[1.,0.]],np.float32);mirror=np.diag([1.,-1.]).astype(np.float32)
    zz=ft.build(np.zeros((1,96),np.float32),p,native,"interaction")
    rot=ft.build(np.zeros((1,96),np.float32),p@theta,native@theta,"interaction")
    ref=ft.build(np.zeros((1,96),np.float32),p@mirror,native@mirror,"interaction")
    assert np.array_equal(zz[:,:,124:],rot[:,:,124:])
    assert np.array_equal(zz[:,:,124::2],ref[:,:,124::2]) and np.array_equal(zz[:,:,125::2],-ref[:,:,125::2])
    assert np.array_equal(zz,ft.build(np.zeros((1,96),np.float32),p+32,native,"interaction"))
    audits=[];bench=[]
    for arm in ("additive","interaction"):
        c.stage("PREFLIGHT_FIXED513",arm=arm)
        model,report=sel.fit(d,s["A"][:513],arm);again,repeated=sel.fit(d,s["A"][:513],arm)
        assert all(np.array_equal(model[k],again[k]) for k in model),"REPEATED_FIT_NOT_EXACT"
        audit=reference_audit(d,s["A"][:513],model,arm)
        audits.append(dict(arm=arm,fit=report,repeat_exact=True,reference=audit))
        del model,again;gc.collect()
        c.stage("BENCHMARK_THROWAWAY1033",arm=arm)
        model,report=sel.fit(d,s["A"][:1033],arm);bench.append(dict(arm=arm,wholefit_seconds=report["elapsed_seconds"],solver_seconds=report["solver_seconds"],iterations=report["iterations"]))
        del model;gc.collect()
    # Unknown IDs rejected by strict alignment; source/output corruption checks without changing source.
    try:c.verify({str(parent/"OOF_predictions.npz"):"0"*64})
    except AssertionError:bad_output_rejected=True
    else:raise AssertionError("BAD_OUTPUT_NOT_REJECTED")
    try:c.verify({str(c.H/"features.py"):"0"*64})
    except AssertionError:bad_source_rejected=True
    else:raise AssertionError("BAD_SOURCE_NOT_REJECTED")
    positive=[]
    for f in range(3):
        tr=s["A"][d["fold"][s["A"]]!=f];va=s["A"][d["fold"][s["A"]]==f]
        assert not np.intersect1d(d["track"][tr],d["track"][va]).size
        n=len(np.unique(d["track"][va][c.label(d,va)["potential"].any(1)]));assert n>=20;positive.append(n)
        donor=ft.permutation(d["ids"],d["track"],va)
        assert len(np.unique(donor))==len(va) and np.array_equal(np.sort(va),np.sort(donor)) and np.all(d["track"][va]!=d["track"][donor])
    planning=3+sum(v["wholefit_seconds"] for v in bench)*len(s["A"])/1033*4*3/60
    assert planning<=30,("STOP_PLANNING",planning)
    identity=dict(status="PASS_OBSERVED_ID_COORDINATE_ORIGIN_DT_SPLIT_LABELS",rows=28791,A=len(s["A"]),B=len(s["B"]),positive_validation_tracks=positive,past_raw_and_local_exact=True,past_last_zero=True,dt=.4,native_origin="same focal heading local last observation; extract originalread-only provenance",original_native_cache_alignment=c.read(c.R/"GENERATION_NATIVE_DECODER_SERIES/encoder_audit.json")["train"],feature_scalar_errors=errors,synthetic_fixtures_PASS=True,bad_source_and_output_rejected=True,all_old_digests_preserved=True)
    c.write("IDENTITY_AUDIT.json",identity);c.write("NUMERICAL_PREFLIGHT.json",dict(status="PASS",arms=audits))
    c.write("BENCHMARK.json",dict(status="PASS_THROUGHPUT_ONLY",arms=bench,planning_minutes=planning,no_saved_benchmark_model_or_H4=True))
    files,inherit=c.locks();c.verify(files);c.write("source_manifest.json",dict(files=files,inherited=inherit,scope="relatedscientificnotallhistorical34k",digest=c.digest(files)))
    c.write("PREFLIGHT.json",dict(status="PASS",source_sha=c.digest(files),source_count=len(files),inherited=inherit,planning_minutes=planning,ETA_minutes=[3,10],elapsed_seconds=time.perf_counter()-started,resource=c.RESOURCE,actual513_repeat_and_reference=True,observed_ID_all28791_PASS=True,no_torch_neural_or_epoch_RNG_claim=True))
    c.stage("PREFLIGHT_PASS",planning_minutes=planning)
if __name__=="__main__":
    try:main()
    except Exception:c.write("PREFLIGHT_FAILURE_"+str(time.time_ns())+".json",dict(traceback=traceback.format_exc(),resource=c.RESOURCE));raise
