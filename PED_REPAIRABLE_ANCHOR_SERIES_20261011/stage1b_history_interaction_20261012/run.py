"""Sole paired S1b OOF run; conditional fullA/B only after registered gate."""
import common as c,selector as sel,features as ft
import numpy as np,time,traceback
def arm_run(d,s,arm,pre):
    A=s["A"];oof=np.full((len(A),20),np.nan);swapped=np.full_like(oof,np.nan);folds=[];all_donor=np.full(len(A),-1)
    for f in range(3):
        c.stage("FIT_SELECTOR_OOF",arm=arm,fold=f,done=f+(0 if arm=="additive" else 3),total=6)
        mask=d["fold"][A]==f;tr=A[~mask];va=A[mask];model,rep=sel.fit(d,tr,arm)
        mp=c.save(arm+"/models/fold_"+str(f)+".npz",**model)
        c.write(arm+"/models/fold_"+str(f)+"_manifest.json",dict(sel.manifest(model,tr,d,pre["source_sha"],arm),model_sha=c.sha(mp),fit_report=rep))
        scores=sel.predict(d,va,model,arm);donor=ft.permutation(d["ids"],d["track"],va);swap=sel.predict(d,va,model,arm,donor)
        oof[mask]=scores;swapped[mask]=swap;all_donor[mask]=donor
        lab=c.label(d,va);summary,slots,hit=c.summaries(d,va,lab,scores);swap_summary,ss,sh=c.summaries(d,va,lab,swap)
        c.save(arm+"/oof/fold_"+str(f)+".npz",sample_id=d["ids"][va],index=va,scores=scores,swapped_scores=swap,history_donor_index=donor,history_donor_ids=d["ids"][donor],selected_slots=slots["selected"],swapped_slots=ss["selected"],bottom_slots=slots["bottom"],random_slots=slots["random"])
        folds.append(dict(fold=f,fit_report=rep,summary=summary,swapped_summary=swap_summary,slot_set_changed=int(np.any(slots["selected"]!=ss["selected"],1).sum())))
        c.write("selector_progress.json",dict(arm=arm,folds_done=f+1,total_arms=2,total_OOF_fits=6))
    lab=c.label(d,A);summary,slots,hit=c.summaries(d,A,lab,oof);ss,sv,sh=c.summaries(d,A,lab,swapped);ci=c.bootstrap(d,A,lab,hit)
    c.save(arm+"/OOF_predictions.npz",sample_id=d["ids"][A],index=A,scores=oof,swapped_scores=swapped,history_donor_index=all_donor,history_donor_ids=d["ids"][all_donor],selected_slots=slots["selected"],swapped_slots=sv["selected"],bottom_slots=slots["bottom"],random_slots=slots["random"])
    engineering=all(x["estimate"]>=.05 and x["CI975"][0]>0 for x in ci if x["group"]=="overall")
    changefold=all(x["summary"]["change"]["policies"]["selected"]["H4"]>=x["summary"]["change"]["policies"]["bottom"]["H4"] for x in folds)
    record=dict(arm=arm,summary=summary,bootstrap=ci,folds=folds,swapped_summary=ss,history_swap_slot_set_changed=int(np.any(slots["selected"]!=sv["selected"],1).sum()),engineering_gain_gate=engineering,change_each_fold_gate=changefold,OOF_gate=engineering and changefold)
    c.write(arm+"/OOF_RESULT.json",record);return record,hit,sh
def main():
    pre=c.read(c.H/"PREFLIGHT.json");assert pre["status"]=="PASS" and pre["planning_minutes"]<=30
    files=c.read(c.H/"source_manifest.json")["files"];assert c.digest(files)==pre["source_sha"];c.verify(files)
    c.stage("INITIAL_SOURCE_OLD_SHA",done=0,total=6);d,s=c.load()
    records={};hits={};swaps={}
    for arm in ("additive","interaction"):records[arm],hits[arm],swaps[arm]=arm_run(d,s,arm,pre)
    A=s["A"];lab=c.label(d,A)
    pair=c.bootstrap(d,A,lab,dict(selected=hits["interaction"]["selected"],bottom=hits["additive"]["selected"],random=hits["additive"]["selected"]))
    pair=[dict(x,reference="additive",exploratory_not_extra_gate=True) for x in pair if x["reference"]=="bottom"]
    swap_ci={}
    for arm in records:
        cc=c.bootstrap(d,A,lab,dict(selected=hits[arm]["selected"],bottom=swaps[arm]["selected"],random=swaps[arm]["selected"]))
        swap_ci[arm]=[dict(x,reference="history_swapped",OOD_not_understanding=True) for x in cc if x["reference"]=="bottom"]
    treatment=records["interaction"];Apass=treatment["OOF_gate"];Bpass=False;Breport=None;exports=[]
    if Apass:
        c.stage("FIT_INTERACTION_FULL_A",done=6,total=7)
        model,rep=sel.fit(d,A,"interaction");mp=c.save("interaction/models/full_A.npz",**model)
        c.write("interaction/models/full_A_manifest.json",dict(sel.manifest(model,A,d,pre["source_sha"],"interaction"),model_sha=c.sha(mp),fit_report=rep))
        B=s["B"];score=sel.predict(d,B,model,"interaction");Breport,slots,hit=c.summaries(d,B,c.label(d,B),score)
        p=Breport["overall"]["policies"];g=Breport["change"]["policies"]
        Bpass=p["selected"]["H4"]>p["bottom"]["H4"] and p["selected"]["H4"]>p["random"]["H4"] and g["selected"]["H4"]>=g["bottom"]["H4"]
        c.save("frozen_policy/B.npz",sample_id=d["ids"][B],index=B,scores=score,selected_slots=slots["selected"],bottom_slots=slots["bottom"],random_slots=slots["random"]);exports.append("B")
        if Bpass:
            for name in ("outer","inner","original_CAL"):
                ix=s[name];score=sel.predict(d,ix,model,"interaction")
                c.save("frozen_policy/"+name+".npz",sample_id=d["ids"][ix],index=ix,scores=score,selected_slots=c.core.select4(score),bottom_slots=c.core.bottom4(d["logs"][ix]),random_slots=c.core.fixed_random4(d["ids"][ix]));exports.append(name)
    c.stage("FINAL_SOURCE_OUTPUT_SHA",done=6+int(Apass),total=6+int(Apass))
    outputs={str(p):c.sha(p) for arm in records for p in (c.H/arm).rglob("*") if p.is_file()}
    outputs.update({str(p):c.sha(p) for p in (c.H/"frozen_policy").glob("*") if p.is_file()})
    c.verify(files);c.verify(outputs)
    result=dict(status="PASS_S1B_EXECUTION",arms=records,interaction_minus_additive_CI=pair,correct_minus_swapped_CI=swap_ci,OOF_gate=bool(Apass),S2_gate=bool(Apass and Bpass),B_transfer_gate=bool(Bpass),B_transfer=Breport,frozen_policy_exports=exports,source_count=len(files),source_sha=pre["source_sha"],output_sha=outputs,model_fits=6+int(Apass),resource=c.RESOURCE,formal_increment=0,no_torch_neural_training=True,potential_not_real_rescue=True)
    c.write("RESULT.json",result);c.write("PRESERVATION.json",dict(status="PASS",source_count=len(files),inherited=pre["inherited"],output_count=len(outputs),all_relatedSHA_unchanged=True));c.write("RESOURCE_USAGE.json",c.RESOURCE)
    c.stage("SUCCESS",done=result["model_fits"],S2_gate=result["S2_gate"])
if __name__=="__main__":
    try:main()
    except Exception:c.write("FAILURE_"+str(time.time_ns())+".json",dict(traceback=traceback.format_exc(),resource=c.RESOURCE));raise
