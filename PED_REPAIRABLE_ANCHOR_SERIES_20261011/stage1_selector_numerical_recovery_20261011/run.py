"""S1 sole scientific run. OOF gates before complete-A fit; no selector sweep."""
import common as c
import selector as sel
import numpy as np,time,traceback,gc
def main():
    c.guard();pre=c.read(c.H/"PREFLIGHT.json");assert pre["status"]=="PASS" and pre["planning_minutes"]<=30
    files=c.read(c.H/"source_manifest.json")["files"];assert c.digest(files)==pre["source_sha"]
    c.stage("INITIAL_SOURCE_OLD_SHA",done=0,total=4);c.verify(files)
    prep=c.read(c.H/"IDENTITY_AUDIT.json");assert prep["status"]=="PASS_IDENTITY_LABEL_PREPARATION_ONLY"
    d,s=c.load();reg=c.npz(c.H/"split_registry.npz");assert np.array_equal(reg["sample_id"],d["ids"])
    for k in s:assert np.array_equal(reg[k+"_index"],s[k])
    labfile=c.npz(c.H/"labels_offline.npz");assert np.array_equal(labfile["sample_id"],d["ids"])
    A=s["A"];oof=np.full((len(A),20),np.nan);fold_records=[];done=[]
    for f in range(3):
        c.stage("FIT_SELECTOR_OOF",fold=f,done=f,total=4)
        tr=A[d["fold"][A]!=f];va=A[d["fold"][A]==f]
        model,fitrep=sel.fit(d,tr);man=sel.manifest(model,tr,d,pre["source_sha"])
        mp=c.save("models/fold_"+str(f)+".npz",**model);scores=sel.predict(d,va,model);oof[d["fold"][A]==f]=scores
        lab=c.label(d,va);summary,slots,hit=c.summaries(d,va,lab,scores)
        c.save("oof/fold_"+str(f)+".npz",sample_id=d["ids"][va],index=va,scores=scores,selected_slots=slots["selected"],bottom_slots=slots["bottom"],random_slots=slots["random"])
        c.write("models/fold_"+str(f)+"_manifest.json",dict(man,model_sha=c.sha(mp),fit_report=fitrep))
        fold_records.append(dict(fold=f,summary=summary,fit_report=fitrep));done.append(f)
        c.write("selector_progress.json",dict(folds_done=done,full_A_done=False))
    assert np.isfinite(oof).all()
    lab=c.label(d,A);summary,slots,hit=c.summaries(d,A,lab,oof);ci=c.bootstrap(d,A,lab,hit)
    c.save("OOF_predictions.npz",sample_id=d["ids"][A],index=A,scores=oof,selected_slots=slots["selected"],bottom_slots=slots["bottom"],random_slots=slots["random"])
    cs=[x for x in ci if x["group"]=="overall"]
    engineering=all(x["estimate"]>=.05 and x["CI975"][0]>0 for x in cs)
    changefold=all(fr["summary"]["change"]["policies"]["selected"]["H4"]>=fr["summary"]["change"]["policies"]["bottom"]["H4"] for fr in fold_records)
    A_pass=engineering and changefold;B_pass=False;B_report=None;exports=[]
    c.write("OOF_RESULT.json",dict(status="PASS" if A_pass else "STOP_S1_OOF_GATE_NOT_MET",summary=summary,folds=fold_records,bootstrap=ci,engineering_gain_gate=engineering,change_each_fold_gate=changefold,potential_not_trajectory_success=True))
    if A_pass:
        c.stage("FIT_SELECTOR_FULL_A",done=3,total=4);model,fitrep=sel.fit(d,A)
        man=sel.manifest(model,A,d,pre["source_sha"]);mp=c.save("models/full_A.npz",**model)
        c.write("models/full_A_manifest.json",dict(man,model_sha=c.sha(mp),fit_report=fitrep))
        B=s["B"];sc=sel.predict(d,B,model);lb=c.label(d,B);B_report,bs,bh=c.summaries(d,B,lb,sc)
        b=B_report["overall"]["policies"];bc=B_report["change"]["policies"]
        B_pass=b["selected"]["H4"]>b["bottom"]["H4"] and b["selected"]["H4"]>b["random"]["H4"] and bc["selected"]["H4"]>=bc["bottom"]["H4"]
        c.save("frozen_policy/B.npz",sample_id=d["ids"][B],index=B,scores=sc,selected_slots=bs["selected"],bottom_slots=bs["bottom"],random_slots=bs["random"])
        exports.append("B")
        if B_pass:
            # No outcome-dependent re-fitting. Only one full-A selector for all populations.
            for name in ("outer","inner","original_CAL"):
                c.stage("FROZEN_SLOT_EXPORT",split=name,done=4,total=4)
                ix=s[name];sc=sel.predict(d,ix,model);ss=c.core.select4(sc)
                c.save("frozen_policy/"+name+".npz",sample_id=d["ids"][ix],index=ix,scores=sc,selected_slots=ss,bottom_slots=c.core.bottom4(d["logs"][ix]),random_slots=c.core.fixed_random4(d["ids"][ix]))
                exports.append(name)
    c.stage("FINAL_SAVED_ARITHMETIC_AND_SOURCE",done=4 if A_pass else 3,total=4)
    out={str(p):c.sha(p) for dirname in ("models","oof","frozen_policy") for p in (c.H/dirname).glob("*") if p.is_file()}
    for name in ("split_registry.npz","labels_offline.npz","OOF_predictions.npz","OOF_RESULT.json","IDENTITY_AUDIT.json"):out[str(c.H/name)]=c.sha(c.H/name)
    c.verify(files);c.verify(out);c.guard()
    result=dict(status="PASS_S1_EXECUTION",S2_gate=bool(A_pass and B_pass),OOF_gate=bool(A_pass),B_transfer_gate=bool(B_pass),OOF_summary=summary,folds=fold_records,bootstrap=ci,B_transfer=B_report,frozen_policy_exports=exports,source_sha=pre["source_sha"],source_count=len(files),output_sha=out,resource=c.RESOURCE,model_fits=4 if A_pass else 3,no_torch_neural_forward_backward_optimizer=True,potential_not_physical_or_learned_coverage=True,formal_trajectory_study_increment=0)
    c.write("RESULT.json",result);c.write("PRESERVATION.json",dict(status="PASS",source_count=len(files),inherited=pre["inherited"],output_count=len(out),all_related_initial_final_SHA=True,not_all_historical34k=True));c.write("RESOURCE_USAGE.json",dict(status="PASS",**c.RESOURCE))
    c.stage("SUCCESS",done=result["model_fits"],S2_gate=result["S2_gate"])
if __name__=="__main__":
    try:main()
    except Exception:c.write("FAILURE_"+str(time.time_ns())+".json",dict(traceback=traceback.format_exc(),resource=c.RESOURCE));raise
