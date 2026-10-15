"""Full saved R1 independent labels/paths/weighted-fit/OOF arithmetic; never refit."""
import common as c,numerics as n,labels as lab
import numpy as np,time
def exact(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple,np.ndarray)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b,(a,b)
def independent_weights(e,y,track,mode):
    positive=e if mode=="envelope" else y==1;negative=y==0
    w=np.zeros(e.shape,float)
    for i in range(len(e)):
        nw=int((track==track[i]).sum())
        for j in range(20):
            if positive[i,j]:w[i,j]=.5/int(positive[i].sum())/nw
            elif negative[i,j]:w[i,j]=.5/int(negative[i].sum())/nw
    w*=e.size/w.sum();return positive.astype(np.int8),positive|negative,w
def main():
    t=time.perf_counter();r=c.read(c.H/"RESULT.json");assert c.read(c.H/"worker_state.json")["status"]=="SUCCESS" and c.read(c.H/"exit_status.json")["status"]=="SUCCESS"
    files=c.read(c.H/"source_manifest.json")["files"];c.verify(files);c.verify(r["output_sha"])
    review=c.H/"completion_review/REVIEW.json"
    if review.exists():
        prior=c.read(review);assert prior["status"]=="PASS" and prior["source_sha"]==r["source_sha"] and prior["outputs_sha"]==r["output_sha"];print("SHA_COMPLETE_SKIP");return
    d=c.load();saved=c.npz(c.H/"sample_design.npz");assert all(np.array_equal(saved[k],v) for k,v in d.items() if isinstance(v,np.ndarray))
    recs=c.read(c.H/"records.json");sol=c.npz(c.H/"solutions.npz");safe=np.zeros((384,20),bool);todo=[(i,int(s),mode) for i in range(384) for s in np.flatnonzero(d["envelope"][i]) for mode in ("shape","pointwise")]
    assert len(recs)==len(todo)==len(sol["paths"])==1022;maxerr=0.;disagreements=0
    for k,(i,s,mode) in enumerate(todo):
        if k%32==0:c.guard()
        checkpoint=c.npz(c.H/"case_checkpoints"/(str(k)+".npz"));assert np.array_equal(checkpoint["raw32"],sol["raw32"][k]) and np.array_equal(checkpoint["path"],sol["paths"][k])
        z=recs[k];assert (z["window"],z["slot"],z["mode"])==(i,s,mode) and z["sample_id"]==int(d["ids"][i])
        assert (int(sol["window"][k]),int(sol["slot"][k]),str(sol["mode"][k]))==(i,s,mode)
        audit,q,bank=n.independent_verify(d["native"][i],d["gt"][i],s,sol["raw32"][k]);prod,p,b=n.validate(d["native"][i],d["gt"][i],s,sol["raw32"][k])
        error=float(abs(q-sol["paths"][k]).max());maxerr=max(maxerr,error);assert error<=2e-12
        assert np.array_equal(bank[np.arange(20)!=s],d["native"][i].astype(float)[np.arange(20)!=s])
        valid=bool(audit["witness"] and prod["witness"]);assert valid==z["safe_witness"];safe[i,s]|=valid
        disagreements+=int(audit["witness"]!=prod["witness"])
        for field in ("ADE","FDE"):assert abs(audit[field]-z[field])<=2e-12
        assert abs(audit["motion"]-np.array(z["motion"])).max()<=2e-12
    y,cert=lab.build(d["native"],d["gt"],d["envelope"],safe);stored=c.npz(c.H/"labels.npz");assert np.array_equal(y,stored["labels"]) and np.array_equal(safe,stored["verified_safe"])
    assert not np.any((y==0)&d["envelope"]);op=safe.any(1);supports=[];fit_diagnostics=[]
    for f in range(3):
        va=d["fold"]==f;tr=(~va)&op&(y==0).any(1);known=va&op
        supports.append(dict(fold=f,validation_windows=int(va.sum()),P_windows=int(known.sum()),P_tracks=len(np.unique(d["track"][known])),common_train_windows=int(tr.sum()),common_train_tracks=len(np.unique(d["track"][tr]))))
        if not r["support_gate"]:continue
        X=d["X"][tr].astype(float);mean=X.mean((0,1));std=np.where(X.std((0,1))<1e-6,1.,X.std((0,1)))
        for mode in ("envelope","witness"):
            m=c.npz(c.H/mode/("models/fold_"+str(f)+".npz"));man=c.read(c.H/mode/("models/fold_"+str(f)+"_manifest.json"))
            assert man["source_sha"]==r["source_sha"] and man["model_sha"]==c.sha(c.H/mode/("models/fold_"+str(f)+".npz"))
            assert np.array_equal(m["mean"],mean) and np.array_equal(m["std"],std) and np.array_equal(m["fit_ids"],d["ids"][tr]) and np.array_equal(m["fit_index"],np.flatnonzero(tr))
            target,known,w=independent_weights(d["envelope"][tr],y[tr],d["track"][tr],mode);yy,kk,ww=lab.supervision(d["envelope"][tr],y[tr],d["track"][tr],mode)
            assert np.array_equal(target,yy) and np.array_equal(known,kk) and np.array_equal(w,ww) and np.array_equal(m["known"],known) and np.array_equal(m["weight"],w)
            z=((X-mean)/std)[known];target=target[known];sw=w[known];score=z@m["coef"]+float(m["intercept"][0]);from scipy.special import expit
            residual=sw*(expit(score)-target);g=np.r_[(z.T@residual+m["coef"])/sw.sum(),residual.sum()/sw.sum()];assert abs(g).max()<=1e-6
            zz=c.npz(c.H/mode/"OOF.npz");vscore=((d["X"][va].astype(float)-mean)/std)@m["coef"]+float(m["intercept"][0])
            assert np.array_equal(vscore,zz["scores"][va]);fit_diagnostics.append(dict(fold=f,mode=mode,gradient_inf=float(abs(g).max())))
    exact(supports,r["support"]);gate=all(z["P_windows"]>=20 and z["P_tracks"]>=10 and z["common_train_windows"]>0 for z in supports);assert gate==r["support_gate"]
    policies=c.npz(c.H/"policy_slots.npz");hit={};summary={}
    for mode,slots in policies.items():
        if mode=="sample_id":assert np.array_equal(slots,d["ids"]);continue
        if mode in ("envelope","witness"):
            zz=c.npz(c.H/mode/"OOF.npz");assert np.array_equal(slots,lab.rank4(zz["scores"]))
        elif mode=="fixed_random":assert np.array_equal(slots,lab.random4(d["ids"]))
        elif mode=="bottom":assert np.array_equal(slots,lab.bottom4(d["logs"]))
        elif mode=="old_S1b":assert np.array_equal(slots,lab.rank4(d["old_scores"]))
        summary[mode],hit[mode],_,_,_=lab.metrics(slots,y,d["groups"],d["scene"])
    exact(summary,r["summary"]);gain=False;change=False
    if gate:
        cis=[dict(reference=ref,**lab.bootstrap(d["track"],op,hit["witness"].astype(float)-hit[ref],20261012+j)) for j,ref in enumerate(("envelope","fixed_random","bottom"))];exact(cis,r["bootstrap"])
        gain=all(z["estimate"]>=.05 and z["CI98333"][0]>0 for z in cis);folds=[];change=True
        for f in range(3):
            mask=(d["fold"]==f)&(d["groups"]==2)&op;h={mode:int(v[mask].sum()) for mode,v in hit.items()};den=int(mask.sum());local=den>0 and h["witness"]>=h["envelope"] and h["witness"]>=h["fixed_random"];change&=local
            folds.append(dict(fold=f,change_known_windows=den,hits=h,change_gate=bool(local)))
        exact(folds,r["fold_change"])
    assert (bool(gate and gain and change))==r["B_transfer_screen_gate"]
    c.verify(files);c.verify(r["output_sha"]);c.guard()
    c.write(review,dict(status="PASS",source_sha=r["source_sha"],source_count=len(files),outputs_sha=r["output_sha"],all1022_actualpaths_literal_safety_and_labels_PASS=True,decode_max_error=maxerr,arithmetic_classification_disagreements=disagreements,all384_ids_features_discovery_track_exclusion=True,all_models_matched_ids_stats_weights_OOF_scores_CI_gates_PASS=True,fit_certificates=fit_diagnostics,elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE,no_solver_or_selector_fit_replay=True,not_external_review=True))
    print("AUDIT_PASS",time.perf_counter()-t)
if __name__=="__main__":main()
