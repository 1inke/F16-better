"""R1 one bounded label-generation and matchedOOF production."""
import common as c
c.guard()
import selector as sel,numerics as n,labels as lab
from preflight import checked
import numpy as np,time,traceback
def main():
    started=time.perf_counter();pre=c.read(c.H/"PREFLIGHT.json");assert pre["status"]=="PASS" and pre["planning_minutes"]<=30
    files=c.read(c.H/"source_manifest.json")["files"];assert c.digest(files)==pre["source_sha"];c.verify(files)
    c.stage("INITIAL_SOURCE_SHA",done=0,total=1022);d=c.load();sample=c.npz(c.H/"sample_design.npz")
    assert all(np.array_equal(sample[k],v) for k,v in d.items() if isinstance(v,np.ndarray))
    recs=[];paths=[];raws=[];safe=np.zeros((384,20),bool)
    for i in range(384):
        for s in np.flatnonzero(d["envelope"][i]):
            for mode in ("shape","pointwise"):
                c.stage("ALL_POTENTIAL_ANCHOR_FIXED_SOLVE",done=len(recs),total=1022,sample_id=int(d["ids"][i]),slot=int(s),mode=mode)
                t=time.perf_counter();z,p=checked(d["native"][i],d["gt"][i],int(s),mode)
                safe[i,s]|=z["safe_witness"]
                recs.append(dict(window=i,index=int(d["index"][i]),sample_id=int(d["ids"][i]),slot=int(s),mode=mode,elapsed=time.perf_counter()-t,**{k:v for k,v in z.items() if k not in ("raw32","unit_controls")}))
                paths.append(p);raws.append(z["raw32"]);c.write("partial_records.json",recs)
    assert len(recs)==1022
    c.save("solutions.npz",paths=np.array(paths),raw32=np.array(raws),window=np.array([r["window"] for r in recs]),slot=np.array([r["slot"] for r in recs]),mode=np.array([r["mode"] for r in recs]))
    c.write("records.json",recs);y,cert=lab.build(d["native"],d["gt"],d["envelope"],safe);c.save("labels.npz",sample_id=d["ids"],labels=y,verified_safe=safe,envelope=d["envelope"])
    opportunity=safe.any(1);supports=[]
    for f in range(3):
        va=d["fold"]==f;known=va&opportunity;common=(~va)&opportunity&(y==0).any(1)
        supports.append(dict(fold=f,validation_windows=int(va.sum()),P_windows=int(known.sum()),P_tracks=len(np.unique(d["track"][known])),common_train_windows=int(common.sum()),common_train_tracks=len(np.unique(d["track"][common]))))
    support_gate=all(z["P_windows"]>=20 and z["P_tracks"]>=10 and z["common_train_windows"]>0 for z in supports)
    foldrecs=[];pred={};fitreports=[]
    if support_gate:
        scores={mode:np.full((384,20),np.nan) for mode in ("envelope","witness")}
        for f in range(3):
            va=d["fold"]==f;tr=(~va)&opportunity&(y==0).any(1);X=d["X"][tr].astype(float);mean=X.mean((0,1));std=np.where(X.std((0,1))<1e-6,1.,X.std((0,1)));weights=[]
            assert not np.intersect1d(d["track"][tr],d["track"][va]).size
            for mode in scores:
                c.stage("MATCHED_SELECTOR_OOF",fold=f,mode=mode,done=len(fitreports),total=6)
                m,rep=sel.fit(d["X"][tr],d["envelope"][tr],y[tr],d["track"][tr],mode,mean,std);scores[mode][va]=sel.scores(d["X"][va],m);weights.append(rep["total_weight"])
                path=mode+"/models/fold_"+str(f);c.save(path+".npz",**m,fit_index=np.flatnonzero(tr),fit_ids=d["ids"][tr])
                c.write(path+"_manifest.json",dict(source_sha=pre["source_sha"],split_sha=c.sha(c.H/"sample_design.npz"),labels_sha=c.sha(c.H/"labels.npz"),mode=mode,fold=f,fit_report=rep,model_sha=c.sha(c.H/(path+".npz")),supervision_policy="sameN / Epositive vs Ppositive U_mask",classes_not_probability=True))
                fitreports.append(dict(fold=f,mode=mode,**rep))
            assert abs(weights[0]-weights[1])<1e-8
        for mode,score in scores.items():
            assert np.isfinite(score).all();pred[mode]=lab.rank4(score);c.save(mode+"/OOF.npz",sample_id=d["ids"],scores=score,slots4=pred[mode])
    pred["fixed_random"]=lab.random4(d["ids"]);pred["bottom"]=lab.bottom4(d["logs"]);pred["old_S1b"]=lab.rank4(d["old_scores"])
    summaries={};hits={}
    for mode,slots in pred.items():summaries[mode],hits[mode],_,_,_=lab.metrics(slots,y,d["groups"],d["scene"])
    c.save("policy_slots.npz",sample_id=d["ids"],**pred)
    cis=[];change_gate=False;gain_gate=False
    if support_gate:
        for j,ref in enumerate(("envelope","fixed_random","bottom")):
            ci=lab.bootstrap(d["track"],opportunity,hits["witness"].astype(float)-hits[ref],20261012+j);cis.append(dict(reference=ref,**ci))
        gain_gate=all(r["estimate"]>=.05 and r["CI98333"][0]>0 for r in cis)
        change_gate=True
        for f in range(3):
            mask=(d["fold"]==f)&(d["groups"]==2)&opportunity
            h={mode:int(v[mask].sum()) for mode,v in hits.items()};den=int(mask.sum())
            local=den>0 and h["witness"]>=h["envelope"] and h["witness"]>=h["fixed_random"];change_gate&=local;foldrecs.append(dict(fold=f,change_known_windows=den,hits=h,change_gate=bool(local)))
    gate=bool(support_gate and gain_gate and change_gate)
    c.stage("FINAL_SOURCE_OUTPUT_SHA",done=1022,total=1022);out={str(p):c.sha(p) for name in ("solutions.npz","records.json","labels.npz","policy_slots.npz") for p in [c.H/name]}
    for mode in ("envelope","witness"):
        out.update({str(p):c.sha(p) for p in (c.H/mode).rglob("*") if p.is_file()})
    c.verify(files);c.verify(out);c.guard()
    result=dict(status="PASS_R1_EXECUTION" if support_gate else "PASS_LABEL_AUDIT_INSUFFICIENT_OOF_SUPPORT",support_gate=support_gate,gain_gate=gain_gate,change_gate=bool(change_gate),B_transfer_screen_gate=gate,S2_started=False,B_evaluated=False,windows=384,potential_anchors=511,solves=1022,support=supports,fit_reports=fitreports,summary=summaries,bootstrap=cis,fold_change=foldrecs,label_counts={str(v):int((y==v).sum()) for v in (-1,0,1)},safe_windows=int(opportunity.sum()),endpoint_certificate=cert,source_sha=pre["source_sha"],source_count=len(files),output_sha=out,elapsed_seconds=time.perf_counter()-started,resource=c.RESOURCE,no_neural_training=True,unknown_not_negative=True,potential_and_witness_not_deployedcoverage=True,formal_increment=0)
    c.write("RESULT.json",result);c.write("PRESERVATION.json",dict(status="PASS",source_count=len(files),output_count=len(out),related_allSHA_unchanged=True,not_allhistorical34k=True));c.stage("SUCCESS",B_transfer_screen_gate=gate)
if __name__=="__main__":
    try:main()
    except Exception:c.write("FAILURE_"+str(time.time_ns())+".json",dict(traceback=traceback.format_exc(),resource=c.RESOURCE));raise
