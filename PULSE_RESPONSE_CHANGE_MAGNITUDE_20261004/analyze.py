"""Posthoc saved-coordinate magnitude analysis. No inference/optimizer/old edits."""
from pathlib import Path
import sys,json,hashlib,importlib.util
sys.dont_write_bytecode=True
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
INPUT=ROOT/'PULSE_RESPONSE_VISUAL_REVIEW_20261004/data/exact_arrays.npz'
EVENT=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/common.py'
REVIEW=ROOT/'GENERATION_PULSE_RESPONSE_SERIES_20261004/completion_review/REVIEW.json'
DESIGN=ROOT/'GENERATION_PULSE_RESPONSE_SERIES_20261004/DESIGN.md'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def describe(x):
    x=np.asarray(x,dtype=float).reshape(-1)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)),p95=float(np.quantile(x,.95)),max=float(x.max())) if len(x) else dict(n=0)
def error(pred,gt):
    d=np.linalg.norm(pred-gt[:,None],axis=-1);return d.mean(-1),d[:,:,-1]
def closest(a,f):return np.maximum(a/.2,f/.4).argmin(-1)
def goodness(a,f):return ((a<=.2)&(f<=.4)).any(-1)
def main():
    assert not (HERE/'RESULT.json').exists(),'Preserve prior output'
    inputs=[INPUT,EVENT,REVIEW,DESIGN,Path(__file__)]
    hashes={str(p):sha(p) for p in inputs}
    with np.load(INPUT,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    spec=importlib.util.spec_from_file_location('magnitude_saved_event_rules',EVENT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ids=data['sample_id'];gt=data['future'];past=data['past'];clear=data['clear_event'];ar=np.arange(len(ids))
    truth=module.events(past,gt+past[:,-1,None]);assert np.array_equal(clear,(truth['y']==1)&~truth['uncertain'])
    assert len(ids)==1516 and int(clear.sum())==469
    native=data['native'];native_a,native_f=error(native,gt);native_good=goodness(native_a,native_f)
    ne=module.events(np.repeat(past,20,axis=0),(native+past[:,-1,None,None]).reshape(-1,12,2))
    event_cache={}
    for key in ['native']+[f'{a}{s}' for s in (1,2,3) for a in ('fixed','learned')]:
        p=data[key];ee=ne if key=='native' else module.events(np.repeat(past,20,axis=0),(p+past[:,-1,None,None]).reshape(-1,12,2))
        event_cache[key]={field:ee[field].reshape(len(ids),20) for field in ('y','kind','onset_step','direction')}
    rows=[];cross_rows=[];event_rows=[]
    for seed in (1,2,3):
        new=data[f'learned{seed}'];na,nf=error(new,gt);ng=goodness(na,nf)
        for ref in ('native',f'fixed{seed}'):
            old=data[ref];oa,of=error(old,gt);og=goodness(oa,of)
            delta=new-old;dis=np.linalg.norm(delta,axis=-1)
            new_step=np.diff(np.concatenate([np.zeros_like(new[:,:,:1]),new],axis=2),axis=2)/.4
            old_step=np.diff(np.concatenate([np.zeros_like(old[:,:,:1]),old],axis=2),axis=2)/.4
            heading_valid=(np.linalg.norm(new_step,axis=-1)>=.2)&(np.linalg.norm(old_step,axis=-1)>=.2)
            angle=np.abs(np.degrees(np.arctan2(new_step[...,0]*old_step[...,1]-new_step[...,1]*old_step[...,0],(new_step*old_step).sum(-1))))
            oslot,nslot=closest(oa,of),closest(na,nf)
            omargin=np.maximum(oa/.2,of/.4).min(-1);nmargin=np.maximum(na/.2,nf/.4).min(-1)
            changed=(~og&ng)|(og&~ng)
            for group,mask in [('overall',np.ones(len(ids),bool)),('clear_event',clear)]:
                oevent,nevent=event_cache[ref],event_cache[f'learned{seed}']
                kind_change=(oevent['kind']!=nevent['kind']).mean(-1)
                ychange=(oevent['y']!=nevent['y']).mean(-1)
                rows.append(dict(seed=seed,reference=ref,group=group,n=int(mask.sum()),
                    # Statistics across per-case averages; candidates/timesteps are NOT independent replicates.
                    mean_same_slot_displacement_m=describe(dis.mean((1,2))[mask]),
                    max_same_slot_displacement_m=describe(dis.max((1,2))[mask]),
                    mean_endpoint_displacement_m=describe(dis[:,:,-1].mean(-1)[mask]),
                    mean_speed_vector_change_mps=describe(np.linalg.norm(new_step-old_step,axis=-1).mean((1,2))[mask]),
                    joint_closest_slot_changed=int((mask&(oslot!=nslot)).sum()),
                    minADE_improvement_m=describe((oa.min(-1)-na.min(-1))[mask]),
                    minFDE_improvement_m=describe((of.min(-1)-nf.min(-1))[mask]),
                    mean_candidate_event_kind_flip_fraction=float(kind_change[mask].mean()),
                    mean_candidate_event_present_flip_fraction=float(ychange[mask].mean()),
                    pooled_heading_difference_degrees_descriptive=describe(angle[mask][heading_valid[mask]]),
                    near_boundary_crosses={str(b):int((mask&changed&(np.abs(omargin-1)<=b)&(np.abs(nmargin-1)<=b)).sum()) for b in (.02,.05,.10)},
                    all_crosses=int((mask&changed).sum())))
                for category,cmask in [('rescued',mask&~og&ng),('lost',mask&og&~ng)]:
                    ia=ar[cmask]
                    cross_rows.append(dict(seed=seed,reference=ref,group=group,category=category,n=int(cmask.sum()),
                        IDs=ids[cmask].tolist(),both_margin_within_5pct=int((cmask&(np.abs(omargin-1)<=.05)&(np.abs(nmargin-1)<=.05)).sum()),
                        both_margin_within_10pct=int((cmask&(np.abs(omargin-1)<=.10)&(np.abs(nmargin-1)<=.10)).sum()),
                        before_margin=describe(omargin[cmask]),after_margin=describe(nmargin[cmask]),
                        same_slot_displacement_case_mean_m=describe(dis.mean((1,2))[cmask]),
                        closest_path_ADE_delta_new_minus_old_m=describe((na[ar,nslot]-oa[ar,oslot])[cmask]),
                        closest_path_FDE_delta_new_minus_old_m=describe((nf[ar,nslot]-of[ar,oslot])[cmask]),
                        closest_path_geometry_change_m=describe(np.linalg.norm(new[ia,nslot[cmask]]-old[ia,oslot[cmask]],axis=-1).mean(-1))))
            # GT-assisted same joint oracle; event labels remain operational, not latent intention.
        joint=(na+.5*nf).argmin(-1);kind=event_cache[f'learned{seed}']['kind'][ar,joint]
        onset=event_cache[f'learned{seed}']['onset_step'][ar,joint]
        n_joint=(native_a+.5*native_f).argmin(-1)
        native_kind=event_cache['native']['kind'][ar,n_joint]
        nkindok=native_kind==truth['kind'];kindok=kind==truth['kind']
        event_rows.append(dict(seed=seed,group='clear_event',n=int(clear.sum()),
            native_oracle_kind_match=int((clear&nkindok).sum()),new_oracle_kind_match=int((clear&kindok).sum()),
            kind_corrected=int((clear&~nkindok&kindok).sum()),kind_lost=int((clear&nkindok&~kindok).sum()),
            native_oracle_kind_changed=int((clear&(native_kind!=kind)).sum()),
            native_oracle_event_detected=int((clear&(event_cache['native']['onset_step'][ar,n_joint]<99)).sum()),
            learned_oracle_event_detected=int((clear&(onset<99)).sum())))
    for p,digest in hashes.items():assert sha(Path(p))==digest,p
    result=dict(status='PASS_SAVED_COORDINATE_MAGNITUDE_ONLY',input_hashes=hashes,rows=rows,cross_cases=cross_rows,oracle_events=event_rows,
        limitations=['User visual feedback is qualitative, not blinded or population-wide perceptual validation.',
            'All case/candidate/step quantiles descriptive; repeated IDs and overlapping windows, only3seeds; no new inferential p/CI.',
            '2/5/10pct boundary bands are posthoc sensitivity, not visual perception criteria or replacement formal gates.',
            'Same-slot geometry comparison valid but closest-path oracles may change slot; explicit slot-switch counts saved.',
            'Event rule flips concern coarse kinematics, NOT psychological intention or accurate beneficial use.',
            'New paths inherit native selected slot; no new corrected-path scoring.',
            'No optimizer/model inference/training/raw/test read or original scientific writes.'])
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],clear_rows=[x for x in rows if x['group']=='clear_event'],clear_cross=[x for x in cross_rows if x['group']=='clear_event' and x['reference']=='native'],events=event_rows),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
