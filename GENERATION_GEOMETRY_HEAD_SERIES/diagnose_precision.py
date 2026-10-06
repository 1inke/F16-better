"""Fit-only design diagnosis; dev numbers are labelled confirmatory diagnostics only."""
import run_geometry as h
import numpy as np
import torch
import json
from pathlib import Path
if __name__=='__main__':
    rows=[]
    interpolation=np.zeros((12,3),np.float32)
    for segment in range(3):
        for t in range(4):
            interpolation[4*segment+t,segment]=(t+1)/4
            if segment:interpolation[4*segment+t,segment-1]=1-(t+1)/4
    cfg=h.c.settings(1)
    for split,path in [('fit',h.p.TRAIN_CACHE),('dev',h.p.DEV_CACHE)]:
        cache=h.p.load_npz(path)
        if split=='fit':
            ids,_=h.p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
            values=h.r.prepared(cache,False,fit=ids)
        else:
            values=h.r.prepared(cache,False);ids=np.arange(len(cache['sample_id']))
        base,speed,future=[x[ids] for x in values[1:4]];groups=cache['change_group'][ids]
        family=h.p.classify_velocity(speed[:,1:],future[:,-1]/12,h.r.m.LABEL_CFG)
        candidate_family=h.p.classify_velocity(np.repeat(speed[:,1:],20,axis=0),base[:,:,-1].reshape(-1,2)/12,h.r.m.LABEL_CFG).reshape(-1,20)
        distance=np.linalg.norm(base-future[:,None],axis=-1)
        best=(distance.mean(-1)+.5*distance[:,:,-1]).argmin(-1)
        selected=distance[np.arange(len(ids)),best]
        projection=np.einsum('tp,npc->ntc',interpolation,future[:,[3,7,11]])
        representation_error=np.linalg.norm(projection-future,axis=-1)
        for group,mask in [('overall',np.ones(len(ids),bool)),('change',groups==2)]+[(name,(groups==2)&(family==i)) for i,name in enumerate(h.p.CLASS_NAMES)]:
            if not mask.any():continue
            row=dict(split=split,group=group,n=int(mask.sum()),family_covered=float((candidate_family[mask]==family[mask,None]).any(-1).mean()),
                selected_joint_ADE=float(selected[mask].mean()),selected_joint_FDE=float(selected[mask,-1].mean()),
                GT_anchor_interpolation_ADE=float(representation_error[mask].mean()),
                first4_error=float(selected[mask,:4].mean()),middle4_error=float(selected[mask,4:8].mean()),last4_error=float(selected[mask,8:].mean()))
            rows.append(row)
    h.m.write_json(h.HERE/'precision_diagnosis.json',dict(rows=rows,design_basis='FIT_ONLY; GTanchor interpolation is unattainableGTdiagnostic notmodel/inference/baseline',test_access=False))
    print(json.dumps(rows,indent=2))
