"""Full independent CPU saved arithmetic/contracts; no NN forward or optimizer."""
import common as c,metrics as v,prepare as prep,statistics_ci as ci,decision
import numpy as np,time,gc,traceback

def final_metrics(z):
 met,aux=v.evaluate(z['paths'],z['gt'],z['groups'],z['native20'],True);met,aux=v.enrich(z,met,aux)
 centers=c.npz(c.H/'dictionary.npz')['centers'].reshape(20,24)
 x=np.empty((len(z['paths']),24))
 for t in range(12):x[:,2*t:2*t+2]=z['gt'][:,t].astype(float)-(t+1)*.4*z['velocity']
 distance=np.column_stack([np.sum((x-mu)**2,axis=1) for mu in centers]);fixed=distance.argmin(1);aux['posthoc_fixed_mode']=fixed;Q=aux['ADE']+.5*aux['FDE']
 for group,mask in v.masks(z['groups']):
  met[group]['fixed_mode_Qwinner_switch_rate']=float((fixed[mask]!=aux['quality_winner'][mask]).mean());met[group]['fixed_mode_counts_posthoc']=np.bincount(fixed[mask],minlength=20)
  met[group]['fixed_mode_minus_minQ_posthoc']=float((Q[np.arange(len(x)),fixed]-Q.min(1))[mask].mean())
 return met,aux

def main():
 import torch
 started=time.perf_counter();lock=c.read(c.H/'source_manifest.json');result=c.read(c.H/'RESULT.json')
 assert c.read(c.H/'worker_state.json')['status']==c.read(c.H/'exit_status.json')['status']=='SUCCESS'
 assert result['status']=='PASS_T1_MOTION_PATTERN_RESPONSIBILITY_PAIRED_RUN'
 c.verify(lock['files']);c.verify(result['output_sha']);data=c.load();dictionary=c.npz(c.H/'dictionary.npz');prep.attach(data,dictionary);st=c.npz(c.H/'statistics.npz');c.eq(c.safe(st),c.safe(prep.stats(data,dictionary)),1e-12)
 assert len(result['records'])==6 and result['saved_jobs']==28 and result['epochs']==150
 certificates=[];init={};history_rows=0
 for record in result['records']:
  arm=record['arm'];seed=record['seed'];directory=c.H/f'runs/{arm}/seed_{seed}';man=c.read(directory/'run_manifest.json');c.base.done_valid(c.read(directory/'done.json'),lock['source_sha'])
  assert man['source_sha']==lock['source_sha'] and man['statistics_sha']==c.sha(c.H/'statistics.npz') and man['dictionary_sha']==c.sha(c.H/'dictionary.npz') and man['split_ids_sha']==c.sha(c.H/'split_ids.npz')
  assert man['train_ids']==data['ids'][data['split']['FIT']].tolist()
  if seed in init:assert init[seed]==man['initial_parameters_sha']
  else:init[seed]=man['initial_parameters_sha']
  history=c.read(directory/'history.json');assert len(history)==25;history_rows+=len(history)
  ck=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
  assert ck['manifest']==man and ck['epoch']==25 and c.safe(ck['history'])==history
  assert sum(r['optimization_steps'] for r in history)==1850 and all(r['n']==18909 and r['optimization_steps']==74 for r in history)
  assert set(ck['rng'])=={'torch','cuda','loader','numpy','python'}
  for key in ('mean','std','centers'):assert np.array_equal(ck['model'][key].numpy(),st[key])
  assert np.array_equal(ck['model']['time_seconds'].numpy(),np.arange(1,13)*.4)
  assert sum(t.numel() for k,t in ck['model'].items() if k.startswith('head.'))==25560
  op=ck['optimizer']['param_groups'][0]
  assert op['lr']==.001 and op['weight_decay']==.0001 and op['betas']==(.9,.999) and op['eps']==1e-8 and op['foreach']==False
  assert len(ck['optimizer']['state'])==6 and all(int(x['step'].item())==1850 for x in ck['optimizer']['state'].values())
  for ep in (5,10,15,20):
   mid=torch.load(directory/f'epoch_{ep:02d}.pt',map_location='cpu',weights_only=False)
   assert mid['manifest']==man and mid['epoch']==ep and c.safe(mid['history'])==history[:ep] and set(mid['rng'])==set(ck['rng'])
   assert all(int(x['step'].item())==74*ep for x in mid['optimizer']['state'].values());del mid
  expectedfixed=np.bincount(dictionary['labels'],minlength=20)
  for row in history:
   assert sum(row['Q_winner_counts'])==sum(row['regression_responsibility_counts'])==18909
   assert len(row['batches'])==74 and sum(b['n'] for b in row['batches'])==18909
   assert sum(b['fixed_Q_switch_count'] for b in row['batches'])/18909==row['fixed_Q_switch_rate']
   if arm=='fixed_motion_responsibility':assert np.array_equal(row['regression_responsibility_counts'],expectedfixed)
   else:assert row['regression_responsibility_counts']==row['Q_winner_counts']
  assignment=c.read(directory/'assignment_stats.json');assert len(assignment['epoch_training_responsibility'])==25
  for row,h in zip(assignment['epoch_training_responsibility'],history):assert all(row[k]==h[k] for k in row)
  del ck;gc.collect()
 for arm,seed,stem,metricsref in [(r['arm'],r['seed'],f"runs/{r['arm']}/seed_{r['seed']}",r['splits']) for r in result['records']]+[('P0',None,'P0',result['P0'])]:
  for split,ix in data['split'].items():
   path=c.H/f'{stem}/{split}.npz';z=c.npz(path);stored=c.npz(c.H/f'{stem}/{split}_metrics.npz')
   assert z['paths'].shape==(len(ix),20,12,2) and np.isfinite(z['paths']).all() and len(np.unique(z['sample_id']))==len(ix)
   for key,reference in (('sample_id',data['ids'][ix]),('groups',data['groups'][ix]),('track',data['track'][ix]),('scene',data['scene'][ix]),('gt',data['gt'][ix]),('native20',data['native'][ix].astype(float)),('velocity',data['vel'][ix])):assert np.array_equal(z[key],reference),(arm,seed,split,key)
   base=np.arange(1,13)[None,None,:,None]*.4*z['velocity'][:,None,None]+st['centers'][None]
   assert z['correction'].dtype==np.float32 and np.array_equal(z['paths'],base+z['correction'].astype(float))
   if arm=='P0':assert not np.count_nonzero(z['correction'])
   met,aux=final_metrics(z);c.eq(c.safe(met),metricsref[split],2e-11);c.eq(c.safe(met),c.read(path.with_suffix('.metrics.json')),2e-11)
   for key,a in aux.items():
    if a.dtype.kind in 'bui':assert np.array_equal(a,stored[key])
    else:assert np.max(abs(a-stored[key]))<=2e-11
   assert np.array_equal(stored['sample_id'],z['sample_id'])
   c.base.done_valid(c.read(path.with_suffix('.done.json')),lock['source_sha'])
   certificates.append(dict(arm=arm,seed=seed,split=split,n=len(ix),complete_quality_SciPy_clusters_fivemotion_correction_posthoc_assignment=True))
   del z,stored;gc.collect()
 boot=ci.compute(independent=True);c.eq(c.safe(boot),c.read(c.H/'BOOTSTRAP.json'),2e-11)
 c.eq(c.safe(decision.gates(result['records'],result['P0'],boot)),result['decision'],2e-11)
 c.verify(lock['files']);c.verify(result['output_sha']);c.guard()
 c.write('completion_review/REVIEW.json',dict(status='PASS',at=c.now(),final_jobs=24,deterministic_shared_P0_jobs=4,checkpoint_contracts=6,intermediate_checkpoints=24,history_rows=history_rows,source_count=len(lock['files']),output_count=len(result['output_sha']),certificates=certificates,bootstrap_draws=20000,family_size_reserved=48,per_draw_SHA_and_independent_track_aggregation=True,CPU_checkpoint_load_only_no_NN_forward_optimizer_or_gradient_replay=True,not_external_review=True,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-started))
 print('FULL_SAVED_AUDIT_PASS',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  c.write('completion_review/FAILURE.json',dict(status='STOP',at=c.now(),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
