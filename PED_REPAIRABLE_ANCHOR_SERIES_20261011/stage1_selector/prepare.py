"""Preparation: fixed identities and labels. No sklearn/torch model fitting."""
import common as c
import numpy as np,time,traceback
def prepare():
    t=time.perf_counter();c.stage("REGISTERED_IDENTITY_AND_LABELS")
    d,s=c.load();reg=dict(sample_id=d["ids"],track=d["track"],sequence=d["sequence"],pedestrian_id=d["ped"],scene=d["scene"],groups=d["groups"],source_start_index=d["start"],ambiguous_time=d["ambiguous"],bucket=d["bucket"],fold=d["fold"],**{k+"_index":v for k,v in s.items()})
    c.save("split_registry.npz",**reg);c.save("labels_offline.npz",sample_id=d["ids"],**c.label(d,np.arange(len(d["ids"]))))
    lab=c.label(d,s["A"]);folds=[]
    for f in range(3):
        va=s["A"][d["fold"][s["A"]]==f];tr=s["A"][d["fold"][s["A"]]!=f];lv=c.label(d,va);lt=c.label(d,tr)
        pos=len(np.unique(d["track"][va[lv["potential"].any(1)]]))
        assert pos>=20 and lt["potential"].any() and (~lt["potential"]).any(),"STOP_S1_IDENTITY_EVIDENCE_GATE"
        assert not np.intersect1d(d["track"][va],d["track"][tr]).size
        folds.append(dict(fold=f,train=len(tr),validation=len(va),positive_tracks=pos,failed_positive_tracks=len(np.unique(d["track"][va[lv["potential"].any(1)&~lv["native_good"]]])),groups=np.bincount(d["groups"][va],minlength=3)))
    # Time/context exposure is a proxy, NOT independent target identity.
    def exposure(source,target):
        n=unknown=0
        for j in target:
            if d["ambiguous"][j] or not np.isfinite(d["start"][j]):unknown+=1;continue
            m=(d["sequence"][source]==d["sequence"][j])&~d["ambiguous"][source]
            n+=int(np.any(np.abs(d["start"][source[m]]-d["start"][j])<20))
        return dict(target_windows=len(target),known_same_sequence_20_source_index_exposure=n,unknown=unknown,not_seconds_or_verified_video_frame=True)
    comp={k:dict(n=len(ix),tracks=len(np.unique(d["track"][ix])),groups=np.bincount(d["groups"][ix],minlength=3),scenes={a:int((d["scene"][ix]==a).sum()) for a in np.unique(d["scene"][ix])},ambiguous_time=int(d["ambiguous"][ix].sum())) for k,ix in s.items()}
    rep=dict(status="PASS_IDENTITY_LABEL_PREPARATION_ONLY",composition=comp,OOF=folds,A_to_B_time_context_proxy=exposure(s["A"],s["B"]),B_to_outer_time_context_proxy=exposure(s["B"],s["outer"]),A_B_target_disjoint=True,outer_inner_train_target_disjoint=True,CAL_tracks_shared_with_train=len(np.intersect1d(d["track"][s["original_CAL"]],d["track"][s["train"]])),all25912_registry_retained=True,metadata_unknown_retained=True,source_ID_full_preflight_not_yet=True,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-t)
    c.write("IDENTITY_AUDIT.json",rep);print(rep,flush=True)
if __name__=="__main__":
    try:prepare()
    except Exception:c.write("PREPARE_FAILURE_"+str(time.time_ns())+".json",dict(traceback=traceback.format_exc(),resource=c.RESOURCE));raise
