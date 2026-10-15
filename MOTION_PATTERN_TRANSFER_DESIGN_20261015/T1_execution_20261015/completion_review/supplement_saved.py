"""CPU-only descriptive completion supplement. No Torch/model/solver/refit."""
import sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import common as c,metrics as v
import numpy as np

def event(reference,current,mask):
 return dict(n=int(mask.sum()),reference_good=int(reference[mask].sum()),current_good=int(current[mask].sum()),rescued=int((~reference&current&mask).sum()),lost=int((reference&~current&mask).sum()),retained=int((reference&current&mask).sum()),both_bad=int((~reference&~current&mask).sum()),gain_pp=100*float((current[mask].astype(float)-reference[mask]).mean()))

def main():
 start=time.perf_counter();c.guard();lock=c.read(c.H/'source_manifest.json');result=c.read(c.H/'RESULT.json');boot=c.read(c.H/'BOOTSTRAP.json')
 c.verify(lock['files']);c.verify(result['output_sha'])
 refs={};populations=[];events=[];scene_rows=[];all_groups=[];history=[];mode_rows=[];P0rows=[];native_rows=[]
 for split in ('FIT','outer','inner','original_CAL'):
  p0=c.npz(c.H/f'P0/{split}_metrics.npz');z=c.npz(c.H/f'P0/{split}.npz');p0good=p0['vectors5'][:,0].astype(bool);nativegood=p0['native_vectors5'][:,0].astype(bool)
  refs[split]=dict(groups=p0['groups'],track=p0['track'],ids=p0['sample_id'],p0=p0good,native=nativegood,scene=z['scene'])
  for group,mask in v.masks(z['groups']):
   vec=p0['native_vectors5'][mask].mean(0);n=int(mask.sum());tracks=len(np.unique(z['track'][mask]));populations.append(dict(split=split,group=group,n=n,tracks=tracks,scenes={s:int(np.count_nonzero(mask&(z['scene']==s))) for s in np.unique(z['scene'])}))
   native_rows.append(dict(split=split,group=group,n=n,tracks=tracks,good=int(nativegood[mask].sum()),coverage=float(vec[0]),minADE=float(vec[1]),minFDE=float(vec[2]),J=float(vec[3]),endpoint_good=int((p0['native_vectors5'][mask,4]).sum()),motion=v.motion(z['native20'][mask],True)))
   P0rows.append(dict(split=split,group=group,tracks=tracks,**result['P0'][split][group]))
  del z,p0
 for record in result['records']:
  arm,seed=record['arm'],record['seed'];directory=c.H/f'runs/{arm}/seed_{seed}';hh=c.read(directory/'history.json')
  history.append(dict(arm=arm,seed=seed,first={k:hh[0][k] for k in hh[0] if k!='batches'},last={k:hh[-1][k] for k in hh[-1] if k!='batches'},epochs=len(hh),weighted_loss_not_fullFITcoverage=True))
  for split in ('FIT','outer','inner','original_CAL'):
   c.guard();st=c.npz(directory/f'{split}_metrics.npz');z=c.npz(directory/f'{split}.npz');ref=refs[split]
   assert np.array_equal(st['sample_id'],ref['ids']) and np.array_equal(st['groups'],ref['groups'])
   good=st['vectors5'][:,0].astype(bool)
   for group,mask in v.masks(st['groups']):
    metrics=record['splits'][split][group];row=dict(arm=arm,seed=seed,split=split,group=group,tracks=len(np.unique(st['track'][mask])),**metrics)
    row['motion_harm_percent']={k:100*(metrics['motion'][k]/metrics['native_motion'][k]-1) if metrics['native_motion'][k]!=0 else None for k in metrics['motion']}
    row['false_stitched_ADEFDE_good']=int(((st['ADE'].min(1)<=.2)&(st['FDE'].min(1)<=.4)&~good&mask).sum());all_groups.append(row)
    for label,r in (('own_P0',ref['p0']),('native20',ref['native'])):events.append(dict(arm=arm,seed=seed,split=split,group=group,reference=label,**event(r,good,mask)))
   # No effect-based sample or scene selection: enumerate every existing scene.
   if split=='outer':
    for scene in np.unique(ref['scene']):
     for group,gmask in v.masks(st['groups']):
      mask=gmask&(ref['scene']==scene)
      if not mask.any():continue
      values=st['vectors5'][mask].mean(0)
      scene_rows.append(dict(arm=arm,seed=seed,scene=scene,group=group,n=int(mask.sum()),tracks=len(np.unique(st['track'][mask])),good=int(good[mask].sum()),P0_good=int(ref['p0'][mask].sum()),native_good=int(ref['native'][mask].sum()),minADE=float(values[1]),minFDE=float(values[2]),J=float(values[3])))
   if split=='FIT':
    correction=z['correction'].astype(float)
    for mode in range(20):
     assigned=st['posthoc_fixed_mode']==mode
     mode_rows.append(dict(arm=arm,seed=seed,mode=mode,fixed_windows=int(assigned.sum()),Qwinner_windows=int((st['quality_winner']==mode).sum()),Jclosest_windows=int((st['J_closest']==mode).sum()),mean_path_drift_m=float(np.linalg.norm(correction[:,mode],axis=-1).mean()),mean_endpoint_drift_m=float(np.linalg.norm(correction[:,mode,-1],axis=-1).mean()),fixed_population_any_candidate_good=int(good[assigned].sum()),fixed_population_own_slot_good=int(st['good_candidate'][assigned,mode].sum()),fixed_Q_responsibility_switch_rate=float((st['quality_winner'][assigned]!=mode).mean())))
   del st,z
 for seed in (42,43,44):
  for split in ('FIT','outer','inner','original_CAL'):
   az=c.npz(c.H/f'runs/dynamic_winner/seed_{seed}/{split}_metrics.npz');bz=c.npz(c.H/f'runs/fixed_motion_responsibility/seed_{seed}/{split}_metrics.npz')
   assert np.array_equal(az['sample_id'],bz['sample_id'])
   for group,mask in v.masks(az['groups']):events.append(dict(arm='fixed_motion_responsibility',seed=seed,split=split,group=group,reference='A_dynamic_winner',**event(az['vectors5'][:,0].astype(bool),bz['vectors5'][:,0].astype(bool),mask)))
 physics={}
 for arm in ('dynamic_winner','fixed_motion_responsibility'):
  rows=[r for r in all_groups if r['arm']==arm];physics[arm]=dict(total_groups=len(rows),all_fiveMotion_pass_groups=sum(not r['motion_failed'] for r in rows),failed_by_metric={k:sum(k in r['motion_failed'] for r in rows) for k in rows[0]['motion']},harm_percent_range={k:[min(r['motion_harm_percent'][k] for r in rows if r['motion_harm_percent'][k] is not None),max(r['motion_harm_percent'][k] for r in rows if r['motion_harm_percent'][k] is not None)] for k in rows[0]['motion']})
 p0physics=dict(total_groups=len(P0rows),all_fiveMotion_pass_groups=sum(not r['motion_failed'] for r in P0rows),failed_by_metric={k:sum(k in r['motion_failed'] for r in P0rows) for k in P0rows[0]['motion']})
 c.verify(lock['files']);c.verify(result['output_sha']);c.guard()
 source={str(c.H/'completion_review/supplement_saved.py'):c.sha(c.H/'completion_review/supplement_saved.py'),str(c.H/'RESULT.json'):c.sha(c.H/'RESULT.json')}
 c.write('completion_review/SUPPLEMENT.json',dict(status='PASS',at=c.now(),populations=populations,native_groups=native_rows,P0_groups=P0rows,final_groups=all_groups,paired_events=events,outer_scene_rows=scene_rows,training_first_last=history,FIT_mode_rows=mode_rows,physics_summary=physics,P0_physics=p0physics,primary_CIs=[r for r in boot['records'] if r['primary_first4_only']],decision=result['decision'],source_sha=lock['source_sha'],source_count=len(lock['files']),output_count=len(result['output_sha']),source_and_scientific_outputs_initial_final_unchanged=True,supplement_source_sha=source,no_Torch_no_model_forward_no_optimizer_no_new_CI_draws_no_GT_selection=True,not_external_review=True,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-start))
 print(dict(status='SAVED_SUPPLEMENT_PASS',group_rows=len(all_groups),paired_events=len(events),mode_rows=len(mode_rows),physics=physics,P0_physics=p0physics,elapsed_seconds=time.perf_counter()-start,resource=c.RESOURCE),flush=True)

if __name__=='__main__':
 try:main()
 except Exception:
  c.write('completion_review/SUPPLEMENT_FAILURE.json',dict(status='STOP',at=c.now(),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
