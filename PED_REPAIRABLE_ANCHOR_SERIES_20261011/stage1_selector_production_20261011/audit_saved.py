"""Independent saved S1 completion arithmetic. No model fit, no torch."""
import common as c
import numpy as np,time,traceback
from numerical_audit import certificate
def audit():
    t=time.perf_counter();r=c.read(c.H/"RESULT.json");assert r["status"]=="PASS_S1_EXECUTION";files=c.read(c.H/"source_manifest.json")["files"];c.verify(files);c.verify(r["output_sha"])
    d,s=c.load();la=c.npz(c.H/"labels_offline.npz")
    for start in range(0,len(d["ids"]),256):
        ix=np.arange(start,min(start+256,len(d["ids"])));z=c.label(d,ix,True)
        for k in z:
            if z[k].dtype==bool:assert np.array_equal(z[k],la[k][ix])
            else:assert np.max(abs(z[k]-la[k][ix]))<=1e-12
    A=s["A"];o=c.npz(c.H/"OOF_predictions.npz");assert np.array_equal(o["sample_id"],d["ids"][A]) and np.array_equal(o["index"],A)
    fold_summaries=[]
    for f in range(3):
        tr=A[d["fold"][A]!=f];va=A[d["fold"][A]==f];m=c.npz(c.H/"models"/("fold_"+str(f)+".npz"));man=c.read(c.H/"models"/("fold_"+str(f)+"_manifest.json"))
        assert c.sha(c.H/"models"/("fold_"+str(f)+".npz"))==man["model_sha"]
        mu,sd=c.normalizer(d,tr);assert np.array_equal(mu,m["mean"]) and np.array_equal(sd,m["std"]) and np.array_equal(m["fit_ids"],d["ids"][tr])
        xx=c.x(d,tr).reshape(-1,120).astype(float);xx=(xx-mu)/sd
        cert=certificate(xx,c.label(d,tr)["potential"].reshape(-1).astype(np.int8),m["coef"],float(m["intercept"][0]));assert cert["gradient_inf"]<=1e-6;del xx
        sc=c.infer(d,va,mu,sd,m["coef"],float(m["intercept"][0]));assert np.array_equal(sc,o["scores"][d["fold"][A]==f])
        assert not np.intersect1d(d["track"][tr],d["track"][va]).size
        fs,_,_=c.summaries(d,va,c.label(d,va,True),sc);fold_summaries.append(fs)
    lab=c.label(d,A,True);summary,slots,hit=c.summaries(d,A,lab,o["scores"]);ci=c.bootstrap(d,A,lab,hit)
    # JSON representations remove numpy/container differences without tolerance inflation.
    def normalize(x):return __import__("json").loads(__import__("json").dumps(x,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()))
    assert normalize(summary)==r["OOF_summary"] and normalize(ci)==r["bootstrap"]
    assert all(normalize(v)==r["folds"][f]["summary"] for f,v in enumerate(fold_summaries))
    engineering=all(v["estimate"]>=.05 and v["CI975"][0]>0 for v in ci if v["group"]=="overall")
    changefold=all(v["change"]["policies"]["selected"]["H4"]>=v["change"]["policies"]["bottom"]["H4"] for v in fold_summaries)
    ar=c.read(c.H/"OOF_RESULT.json");assert ar["engineering_gain_gate"]==engineering and ar["change_each_fold_gate"]==changefold
    assert r["OOF_gate"]==bool(engineering and changefold)
    Bpass=False
    if r["OOF_gate"]:
        m=c.npz(c.H/"models/full_A.npz");man=c.read(c.H/"models/full_A_manifest.json")
        assert c.sha(c.H/"models/full_A.npz")==man["model_sha"]
        mu,sd=c.normalizer(d,A);assert np.array_equal(mu,m["mean"]) and np.array_equal(sd,m["std"]) and np.array_equal(m["fit_ids"],d["ids"][A])
        xx=c.x(d,A).reshape(-1,120).astype(float);xx=(xx-mu)/sd
        cert=certificate(xx,c.label(d,A)["potential"].reshape(-1).astype(np.int8),m["coef"],float(m["intercept"][0]));assert cert["gradient_inf"]<=1e-6;del xx
        B=s["B"];sc=c.infer(d,B,mu,sd,m["coef"],float(m["intercept"][0]));bs,_,_=c.summaries(d,B,c.label(d,B,True),sc)
        assert normalize(bs)==r["B_transfer"]
        bp=bs["overall"]["policies"];bc=bs["change"]["policies"]
        Bpass=bp["selected"]["H4"]>bp["bottom"]["H4"] and bp["selected"]["H4"]>bp["random"]["H4"] and bc["selected"]["H4"]>=bc["bottom"]["H4"]
    assert r["B_transfer_gate"]==bool(Bpass) and r["S2_gate"]==bool(r["OOF_gate"] and Bpass)
    for k,v in slots.items():assert np.array_equal(v,o[k+"_slots"])
    for name in r["frozen_policy_exports"]:
        z=c.npz(c.H/"frozen_policy"/(name+".npz"));m=c.npz(c.H/"models/full_A.npz");ix=s[name]
        assert np.array_equal(z["sample_id"],d["ids"][ix])
        sc=c.infer(d,ix,m["mean"],m["std"],m["coef"],float(m["intercept"][0]))
        assert np.array_equal(z["scores"],sc) and np.array_equal(z["selected_slots"],c.core.select4(sc))
        assert np.array_equal(z["index"],ix) and np.array_equal(z["bottom_slots"],c.core.bottom4(d["logs"][ix])) and np.array_equal(z["random_slots"],c.core.fixed_random4(d["ids"][ix]))
    c.verify(files);c.verify(r["output_sha"]);c.guard()
    c.write("completion_review/REVIEW.json",dict(status="PASS_ALL_SAVED_SELECTOR_LABELS_IDS_NORMALIZERS_SCORE_SLOTS_BOOTSTRAP_SHA",source_count=len(files),output_count=len(r["output_sha"]),independent_labels=True,all_OOF_normalizers_train_only=True,all_scores_and_slots_EXACT=True,bootstrap_records=len(ci),bootstrap_MAX0=True,source_output_SHA_initial_final_PASS=True,no_torch_optimizer_newfit=True,not_external_review=True,elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE))
    print("AUDIT_PASS",time.perf_counter()-t,flush=True)
if __name__=="__main__":
    try:audit()
    except Exception:c.write("AUDIT_FAILURE_"+str(time.time_ns())+".json",dict(traceback=traceback.format_exc(),resource=c.RESOURCE));raise
