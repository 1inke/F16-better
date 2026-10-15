"""Production T1 only; later stages remain gated and require separate preflight."""
import common as c,model as m,metrics as v,prepare as prep,statistics_ci as ci,decision
import numpy as np,torch,time,gc
stage=c.base.stage;SEEDS=(42,43,44)
def job(data,ix,paths,correction,name,source):
 stem=c.H/name;donefile=stem.with_suffix('.done.json')
 if donefile.exists():
  done=c.read(donefile);c.base.done_valid(done,source);return c.read(stem.with_suffix('.metrics.json'))
 z=v.sample(data,ix,paths,correction=correction,velocity=data['vel'][ix]);c.save(name,**z);z=c.npz(stem)
 metrics,aux=v.evaluate(z['paths'],z['gt'],z['groups'],z['native20']);other,ind=v.evaluate(z['paths'],z['gt'],z['groups'],z['native20'],True)
 c.eq(c.safe(metrics),c.safe(other),2e-11)
 for key,a in aux.items():
  if a.dtype.kind in 'bui':assert np.array_equal(a,ind[key])
  else:assert np.max(abs(a-ind[key]))<=2e-11
 metrics,aux=v.enrich(z,metrics,aux)
 centers=c.npz(c.H/'dictionary.npz')['centers'];residual=(z['gt'].astype(float)-np.arange(1,13)[None,:,None]*.4*z['velocity'][:,None]).reshape(len(ix),24)
 ds=np.square(residual[:,None]-centers.reshape(20,24)[None]).sum(2);fixed=ds.argmin(1)
 if np.array_equal(ix,data['split']['FIT']):assert np.array_equal(fixed,data['fixed_labels'][ix])
 Q=aux['ADE']+.5*aux['FDE'];aux['posthoc_fixed_mode']=fixed
 for group,mask in v.masks(z['groups']):
  metrics[group]['fixed_mode_Qwinner_switch_rate']=float((fixed[mask]!=aux['quality_winner'][mask]).mean())
  metrics[group]['fixed_mode_counts_posthoc']=np.bincount(fixed[mask],minlength=20)
  metrics[group]['fixed_mode_minus_minQ_posthoc']=float((Q[np.arange(len(ix)),fixed]-Q.min(1))[mask].mean())
 c.save(name.replace('.npz','_metrics.npz'),sample_id=z['sample_id'],groups=z['groups'],track=z['track'],**aux)
 c.write(stem.with_suffix('.metrics.json'),metrics)
 ff={str(p):c.sha(p) for p in (stem,c.H/name.replace('.npz','_metrics.npz'),stem.with_suffix('.metrics.json'))}
 c.write(donefile,dict(status='COMPLETE',source_sha=source,files=ff))
 return c.safe(metrics)

def binding(data,st,arm,seed,lock):
 m.seed(seed);net=m.Pattern20(st)
 return dict(c.read(c.H/'protocol.json'),arm=arm,seed=seed,source_sha=lock['source_sha'],source_manifest_sha=c.sha(c.H/'source_manifest.json'),statistics_sha=c.sha(c.H/'statistics.npz'),dictionary_sha=c.sha(c.H/'dictionary.npz'),split_ids_sha=c.sha(c.H/'split_ids.npz'),input_sha={str(p):c.sha(p) for p in c.base.INPUTS},initial_parameters_sha=m.tensor_hash(net.state_dict()),parameter_count=25560,train_ids=data['ids'][data['split']['FIT']].tolist(),old_F16_decoder_scorer_not_instantiated_or_optimized=True,forward_inputs=['p2','observed_velocity'],fresh_not_warmstart=True,P0_reference='shared_P0_four_splits_not_three_independent_seeds')

def main():
 started=time.perf_counter();lock=c.read(c.H/'source_manifest.json');stage('INITIAL_SOURCE_INPUT_OLD_SHA',runs_done=0);c.verify(lock['files'])
 data=c.load();dictionary=c.npz(c.H/'dictionary.npz');prep.attach(data,dictionary);st=c.npz(c.H/'statistics.npz');c.eq(c.safe(st),c.safe(prep.stats(data,dictionary)),1e-12)
 records=[];P0={};initials={}
 for split,ix in data['split'].items():
  stage('SHARED_FULL_P0_SAVE_QUALITY',split=split,runs_done=0)
  paths=np.arange(1,13)[None,None,:,None]*.4*data['vel'][ix,None,None]+st['centers'][None]
  P0[split]=job(data,ix,paths,np.zeros(paths.shape,np.float32),f'P0/{split}.npz',lock['source_sha'])
 for seed in SEEDS:
  for arm in m.ARMS:
   directory=c.H/f'runs/{arm}/seed_{seed}';directory.mkdir(parents=True,exist_ok=True);man=binding(data,st,arm,seed,lock)
   if seed in initials:assert initials[seed]==man['initial_parameters_sha']
   else:initials[seed]=man['initial_parameters_sha']
   if (directory/'done.json').exists():
    c.base.done_valid(c.read(directory/'done.json'),lock['source_sha']);assert c.read(directory/'run_manifest.json')==man;records.append(c.read(directory/'metrics.json'));continue
   m.seed(seed);net=m.Pattern20(st).to(m.DEVICE);opt=m.optimizer(net);loader=torch.Generator().manual_seed(seed);c.write(directory/'run_manifest.json',man)
   checkpoint=directory/'checkpoint.pt';history=m.restore(checkpoint,net,opt,loader,man) if checkpoint.exists() else []
   for ep in range(len(history)+1,26):
    stage('TRAIN_T1_RESPONSIBILITY',arm=arm,seed=seed,epoch=ep,runs_done=len(records),runs_total=6)
    history.append(m.epoch(net,opt,data,data['split']['FIT'],loader,ep,arm));m.checkpoint(checkpoint,net,opt,loader,history,man);c.write(directory/'history.json',history)
    if ep in (5,10,15,20):m.checkpoint(directory/f'epoch_{ep:02d}.pt',net,opt,loader,history,man)
   assert len(history)==25 and sum(r['optimization_steps'] for r in history)==1850
   frozen={k:x.detach().cpu().clone() for k,x in net.state_dict().items()};mets={}
   for split,ix in data['split'].items():
    stage('FULL_SAVED_QUALITY_CLUSTERS_MOTION',arm=arm,seed=seed,split=split,runs_done=len(records))
    p,delta=m.generated(net,data,ix);mets[split]=job(data,ix,p,delta,f'runs/{arm}/seed_{seed}/{split}.npz',lock['source_sha']);del p,delta;gc.collect()
   assert all(torch.equal(x.detach().cpu(),frozen[k]) for k,x in net.state_dict().items())
   record=dict(arm=arm,seed=seed,epoch=25,optimization_steps=1850,parameter_count=25560,splits=mets,all_weights_buffers_frozen_eval_EXACT=True)
   c.write(directory/'metrics.json',record);c.write(directory/'assignment_stats.json',dict(epoch_training_responsibility=[{k:r[k] for k in ('epoch','regression_responsibility_counts','Q_winner_counts','fixed_Q_switch_rate','fixed_minus_minQ','per_mode_mean_output_update_norm','mean_shared_parameter_step_norm')} for r in history],posthoc={s:{g:{k:z[k] for k in ('winner_counts','fixed_mode_counts_posthoc','fixed_mode_Qwinner_switch_rate','mode_mean_correction_m')} for g,z in met.items()} for s,met in mets.items()},parameter_updates_shared_not_mode_specific=True))
   ff={str(p):c.sha(p) for p in directory.iterdir() if p.is_file()};c.write(directory/'done.json',dict(status='COMPLETE',source_sha=lock['source_sha'],files=ff));records.append(record)
   del net,opt,frozen;gc.collect();torch.cuda.empty_cache()
 stage('CROSSED_TRACK_SEED_BOOTSTRAP_20000',runs_done=6);boot=ci.compute();c.write('BOOTSTRAP.json',boot);gate=decision.gates(records,P0,boot)
 outputs={str(p):c.sha(p) for folder in ('P0','runs','bootstrap_draws') for p in (c.H/folder).rglob('*') if p.is_file()};outputs[str(c.H/'BOOTSTRAP.json')]=c.sha(c.H/'BOOTSTRAP.json')
 stage('FINAL_SOURCE_OLD_OUTPUT_SHA',runs_done=6);c.verify(lock['files']);c.verify(outputs);c.guard()
 c.write('ARITHMETIC_AUDIT.json',dict(status='PASS',final_scientific_jobs=24,deterministic_shared_P0_jobs=4,all_saved_quality_clusters_motion_independent=True,not_gradient_or_allmodel_replay=True))
 c.write('PRESERVATION.json',dict(status='PASS',source_count=len(lock['files']),old_sources_models_results_unchanged=True,not_all_historical34k=True,old_F16_not_instantiated=True))
 c.write('RESULT.json',dict(status='PASS_T1_MOTION_PATTERN_RESPONSIBILITY_PAIRED_RUN',records=records,P0=P0,decision=gate,production_trainings=6,epochs=150,optimization_steps=11100,saved_jobs=28,source_sha=lock['source_sha'],source_count=len(lock['files']),output_sha=outputs,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-started))
 stage('SUCCESS',runs_done=6,S_T1=gate['S_T1'],Q=gate['Q'],T2_permitted=gate['T2_permitted'])
if __name__=='__main__':main()
