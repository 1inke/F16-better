"""Independent saved S1b reconstruction; no fit/torch/neural forward."""
import common as c,features as ft
from numerical_audit import certificate,sigmoid
import numpy as np,time
def exact(a,b,at="root"):
    if isinstance(a,dict):
        assert set(a)==set(b),(at,"keys")
        for k in a:exact(a[k],b[k],at+"/"+k)
    elif isinstance(a,(list,tuple,np.ndarray)):
        assert len(a)==len(b),(at,"length")
        for j,(x,y) in enumerate(zip(a,b)):exact(x,y,at+"/"+str(j))
    else:assert a==b,(at,a,b)
def independent_x(d,ix,arm,history=None):
    ans=[]
    for begin in range(0,len(ix),128):
        c.guard();bb=ix[begin:begin+128];hi=bb if history is None else history[begin:begin+128]
        ans.append(ft.independent(d["p2"][hi],d["past"][hi],d["native"][bb],arm))
    return np.concatenate(ans)
def independent_predict(d,ix,model,arm,history=None):
    chunks=[]
    for begin in range(0,len(ix),128):
        c.guard();bb=ix[begin:begin+128];hi=bb if history is None else history[begin:begin+128]
        X=ft.independent(d["p2"][hi],d["past"][hi],d["native"][bb],arm).astype(float)
        chunks.append(sigmoid(((X-model["mean"])/model["std"])@model["coef"]+float(model["intercept"][0])))
    return np.concatenate(chunks)
def main():
    start=time.perf_counter();r=c.read(c.H/"RESULT.json");files=c.read(c.H/"source_manifest.json")["files"]
    assert r["status"]=="PASS_S1B_EXECUTION";assert c.read(c.H/"worker_state.json")["status"]=="SUCCESS" and c.read(c.H/"exit_status.json")["status"]=="SUCCESS"
    c.verify(files);c.verify(r["output_sha"])
    if (c.H/"completion_review/REVIEW.json").exists():
        old=c.read(c.H/"completion_review/REVIEW.json");assert old["status"]=="PASS" and old["source_sha"]==r["source_sha"] and old["scientific_outputs_sha"]==r["output_sha"];print("COMPLETE_SHA_COMPATIBLE_SKIP",flush=True);return
    d,s=c.load();A=s["A"];lab=c.label(d,np.arange(len(d["ids"])),True);cached=c.npz(c.H/"labels_offline.npz")
    for k in lab:
        if k in cached:
            assert np.array_equal(lab[k],cached[k]) if lab[k].dtype==bool else abs(lab[k]-cached[k]).max()<1e-12
    hit_byarm={};swappedhits={};certs=[]
    for arm in ("additive","interaction"):
        print("AUDIT_ARM",arm,flush=True)
        for f in range(3):
            tr=A[d["fold"][A]!=f];va=A[d["fold"][A]==f];model=c.npz(c.H/arm/("models/fold_"+str(f)+".npz"));man=c.read(c.H/arm/("models/fold_"+str(f)+"_manifest.json"))
            assert np.array_equal(model["fit_ids"],d["ids"][tr]) and man["model_sha"]==c.sha(c.H/arm/("models/fold_"+str(f)+".npz")) and man["source_manifest_sha"]==r["source_sha"]
            X=independent_x(d,tr,arm)
            assert np.array_equal(X,c.x(d,tr,arm))
            X=X.reshape(-1,X.shape[-1]).astype(float);mu=X.mean(0);sd=np.where(X.std(0)<1e-6,1.,X.std(0))
            assert np.array_equal(mu,model["mean"]) and np.array_equal(sd,model["std"])
            cert=certificate((X-mu)/sd,c.label(d,tr)["potential"].reshape(-1).astype(np.int8),model["coef"],float(model["intercept"][0]));assert cert["gradient_inf"]<=1e-6
            exact(cert,man["fit_report"]["independent_objective_certificate"]);certs.append(dict(arm=arm,fold=f,certificate=cert))
            del X
            z=c.npz(c.H/arm/("oof/fold_"+str(f)+".npz"));assert np.array_equal(z["sample_id"],d["ids"][va]) and np.array_equal(z["index"],va)
            donor=ft.permutation(d["ids"],d["track"],va);assert np.array_equal(z["history_donor_index"],donor) and np.array_equal(z["history_donor_ids"],d["ids"][donor])
            sc=independent_predict(d,va,model,arm);sw=independent_predict(d,va,model,arm,donor)
            assert np.array_equal(z["scores"],sc) and np.array_equal(z["swapped_scores"],sw)
            su,slots,hh=c.summaries(d,va,c.label(d,va),sc);ss,sv,sh=c.summaries(d,va,c.label(d,va),sw)
            exact(su,r["arms"][arm]["folds"][f]["summary"]);exact(ss,r["arms"][arm]["folds"][f]["swapped_summary"])
            for k in ("selected","bottom","random"):assert np.array_equal(z[k+"_slots"],slots[k])
            assert np.array_equal(z["swapped_slots"],sv["selected"])
        z=c.npz(c.H/arm/"OOF_predictions.npz");assert np.array_equal(z["sample_id"],d["ids"][A])
        su,slots,hh=c.summaries(d,A,c.label(d,A),z["scores"]);ss,sv,sh=c.summaries(d,A,c.label(d,A),z["swapped_scores"])
        exact(su,r["arms"][arm]["summary"]);exact(ss,r["arms"][arm]["swapped_summary"]);exact(c.bootstrap(d,A,c.label(d,A),hh),r["arms"][arm]["bootstrap"])
        engineering=all(x["estimate"]>=.05 and x["CI975"][0]>0 for x in r["arms"][arm]["bootstrap"] if x["group"]=="overall")
        change=all(x["summary"]["change"]["policies"]["selected"]["H4"]>=x["summary"]["change"]["policies"]["bottom"]["H4"] for x in r["arms"][arm]["folds"])
        assert (engineering and change)==r["arms"][arm]["OOF_gate"];hit_byarm[arm]=hh;swappedhits[arm]=sh
    pp=c.bootstrap(d,A,c.label(d,A),dict(selected=hit_byarm["interaction"]["selected"],bottom=hit_byarm["additive"]["selected"],random=hit_byarm["additive"]["selected"]))
    pp=[dict(x,reference="additive",exploratory_not_extra_gate=True) for x in pp if x["reference"]=="bottom"];exact(pp,r["interaction_minus_additive_CI"])
    for arm in hit_byarm:
        pp=c.bootstrap(d,A,c.label(d,A),dict(selected=hit_byarm[arm]["selected"],bottom=swappedhits[arm]["selected"],random=swappedhits[arm]["selected"]))
        pp=[dict(x,reference="history_swapped",OOD_not_understanding=True) for x in pp if x["reference"]=="bottom"];exact(pp,r["correct_minus_swapped_CI"][arm])
    assert r["OOF_gate"]==r["arms"]["interaction"]["OOF_gate"]
    if r["OOF_gate"]:
        model=c.npz(c.H/"interaction/models/full_A.npz");assert np.array_equal(model["fit_ids"],d["ids"][A]);X=independent_x(d,A,"interaction").reshape(-1,136).astype(float)
        assert np.array_equal(X.mean(0),model["mean"]) and np.array_equal(np.where(X.std(0)<1e-6,1.,X.std(0)),model["std"]);del X
        for name in r["frozen_policy_exports"]:
            ix=s[name];z=c.npz(c.H/"frozen_policy"/(name+".npz"));sc=independent_predict(d,ix,model,"interaction")
            assert np.array_equal(sc,z["scores"]) and np.array_equal(c.core.select4(sc),z["selected_slots"]) and np.array_equal(d["ids"][ix],z["sample_id"])
        B=s["B"];br,bs,bh=c.summaries(d,B,c.label(d,B),c.npz(c.H/"frozen_policy/B.npz")["scores"]);exact(br,r["B_transfer"])
        p=br["overall"]["policies"];g=br["change"]["policies"];bg=p["selected"]["H4"]>p["bottom"]["H4"] and p["selected"]["H4"]>p["random"]["H4"] and g["selected"]["H4"]>=g["bottom"]["H4"]
        assert bg==r["B_transfer_gate"]
    else:assert not r["B_transfer_gate"] and not r["frozen_policy_exports"]
    assert r["S2_gate"]==(r["OOF_gate"] and r["B_transfer_gate"])
    old=c.read(c.H.parent/"stage1_selector_production_20261011/RESULT.json");oldscore=c.npz(c.H.parent/"stage1_selector_production_20261011/OOF_predictions.npz")
    assert np.array_equal(oldscore["sample_id"],d["ids"][A])
    oldsummary,_,_=c.summaries(d,A,c.label(d,A),oldscore["scores"]);exact(oldsummary,old["OOF_summary"])
    c.verify(files);c.verify(r["output_sha"]);c.guard()
    c.write("completion_review/REVIEW.json",dict(status="PASS",source_sha=r["source_sha"],source_count=len(files),scientific_outputs_sha=r["output_sha"],all28791_label_ID_observation_exact=True,all_six_model_stats_score_slots_swap_gates_bootstrap_exact=True,old120_saved_reference_exact=True,old120_summary=oldsummary,max_error=0,certificates=certs,elapsed_seconds=time.perf_counter()-start,resource=c.RESOURCE,no_fit_torch_or_neural_forward=True,saved_math_not_external_review=True))
    print("AUDIT_PASS",time.perf_counter()-start,flush=True)
if __name__=="__main__":main()
