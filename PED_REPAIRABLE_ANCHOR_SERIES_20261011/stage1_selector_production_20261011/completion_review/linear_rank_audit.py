"""Posthoc linear rank interpretation, no fitting or neural forward."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common as c
import numpy as np
def audit():
    files=c.read(c.H/"source_manifest.json")["files"];r=c.read(c.H/"RESULT.json");c.verify(files);c.verify(r["output_sha"])
    d,s=c.load();A=s["A"];o=c.npz(c.H/"OOF_predictions.npz");records=[]
    for f in range(3):
        ix=A[d["fold"][A]==f];m=c.npz(c.H/"models"/("fold_"+str(f)+".npz"))
        z=d["native"][ix].reshape(-1,20,24).astype(float);sc=((z-m["mean"][96:])/m["std"][96:])@m["coef"][96:]
        full=o["selected_slots"][d["fold"][A]==f];bare=c.core.select4(sc);diff=int((full!=bare).any(1).sum())
        records.append(dict(fold=f,n=len(ix),selected4_without_observation_offset_EXACT=diff==0,mismatch_windows=diff,saved_probability_min=float(o["scores"][d["fold"][A]==f].min()),saved_probability_max=float(o["scores"][d["fold"][A]==f].max())))
    c.verify(files);c.verify(r["output_sha"])
    c.write("completion_review/LINEAR_RANK_INTERPRETATION.json",dict(status="PASS_SAVED_LINEAR_RANK_ATTRIBUTION",equation="z_ij=common_observed_offset_i+linear_native24_ij",all_OOF_selected4_EXACT_without_common_observed_offset=all(x["mismatch_windows"]==0 for x in records),records=records,no_fit_torch_or_neural_forward=True,no_change_to_selection_or_H4=True,source_output_SHA_unchanged=True,resource=c.RESOURCE))
    print(records,flush=True)
if __name__=="__main__":audit()
