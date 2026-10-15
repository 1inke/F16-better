"""CPU-only exploratory saved-label/ranking contrast; no fit or new prediction."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as c
c.guard()
import numpy as np
def main():
    t=time.perf_counter();r=c.read(c.H/"RESULT.json");files=c.read(c.H/"source_manifest.json")["files"].copy();files.update(r["output_sha"]);files[str(c.H/"RESULT.json")]=c.sha(c.H/"RESULT.json");c.verify(files)
    d=c.npz(c.H/"sample_design.npz");y=c.npz(c.H/"labels.npz")["labels"];e=d["envelope"];known=(y==1).any(1)&(y==0).any(1)
    sl=c.npz(c.H/"policy_slots.npz");records={}
    for mode,slots in sl.items():
        if mode=="sample_id":continue
        pick=np.take_along_axis(y,slots,1);records[mode]=dict(selected_candidate_label_counts={str(v):int((pick==v).sum()) for v in (-1,0,1)})
    a=sl["envelope"];w=sl["witness"];same=np.all(a==w,1);safe=(y==1).any(1)
    common=int(known.sum());ep=int(e[known].sum());pp=int((y[known]==1).sum())
    train_contrast=dict(common_windows=common,all_training_envelope_positive_candidates=ep,witness_positive_candidates=pp,additional_envelope_positive_unknown_candidates=ep-pp,all_511_envelope_positives=int(e.sum()),envelope_unknown_excluded_with_noP_training_windows=int(e[~known].sum()),maskedU_not_missing_at_random=True)
    # Within-window native-path score discrimination only; U is not a true negative.
    rank_summary={}
    for mode in ("envelope","witness"):
        z=c.npz(c.H/mode/"OOF.npz")["scores"];records_bygroup={}
        for name,mask in [("overall",np.ones(384,bool)),("stable",d["groups"]==0),("transition",d["groups"]==1),("change",d["groups"]==2)]:
            pnnum=pnden=punum=puden=0.;best_rank=[]
            for i in np.flatnonzero(mask&safe):
                p=z[i,y[i]==1];n=z[i,y[i]==0];u=z[i,y[i]==-1]
                diff=p[:,None]-n[None,:];pnnum+=(diff>0).sum()+.5*(diff==0).sum();pnden+=diff.size
                diff=p[:,None]-u[None,:];punum+=(diff>0).sum()+.5*(diff==0).sum();puden+=diff.size
                order=np.argsort(-z[i],kind="stable");best_rank.append(int(np.flatnonzero(y[i,order]==1)[0])+1)
            records_bygroup[name]=dict(P_greater_N_pair_fraction=float(pnnum/pnden) if pnden else None,P_greater_U_pair_fraction=float(punum/puden) if puden else None,P_U_not_true_label_AUC=True,best_foundP_rank_mean=float(np.mean(best_rank)) if best_rank else None,best_foundP_rank_median=float(np.median(best_rank)) if best_rank else None)
        rank_summary[mode]=records_bygroup
    fold_metrics=[]
    for f in range(3):
        ix=d["fold"]==f;op=ix&safe;hits={m:int((np.take_along_axis(y,s,1)==1).any(1)[op].sum()) for m,s in sl.items() if m!="sample_id"}
        fold_metrics.append(dict(fold=f,known_opportunity=int(op.sum()),hits=hits))
    c.verify(files);c.guard();c.write("completion_review/SUPERVISION_CONTRAST.json",dict(status="PASS_EXPLORATORY_SAVED_CONTRAST",train_contrast=train_contrast,selected_labels=records,matched_selector_same_slots_all=int(same.sum()),total_windows=384,matched_selector_same_slots_knownP=int((same&safe).sum()),knownP_windows=int(safe.sum()),fold_metrics=fold_metrics,ranking=rank_summary,scientific_source_output_SHA_unchanged=True,no_fit_solver_torch_new_prediction=True,not_extra_gate_or_confirmation=True,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-t));print("SUPPLEMENT_PASS")
if __name__=="__main__":main()
