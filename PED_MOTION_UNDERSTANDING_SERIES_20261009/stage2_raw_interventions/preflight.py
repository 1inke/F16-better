import sys,time,gc,traceback,json
sys.dont_write_bytecode=True
import stage2 as x
import numpy as np
def main():
 t=time.perf_counter();x.guard();files=x.read(x.P/'source_manifest.json')['files'].copy();files.update(x.read(x.P/'PREFLIGHT.json')['calibration_sha'])
 for p in (x.P/'data').glob('*'):files[str(p)]=x.sha(p)
 for p in (x.P/'jobs').rglob('*'):
  if p.is_file():files[str(p)]=x.sha(p)
 for p in (x.P/'completion_review').glob('*'):
  if p.is_file():files[str(p)]=x.sha(p)
 for p in (x.P/'RESULT.json',x.P/'RESULT_ANALYSIS.md',x.P/'RESULT_ANALYSIS.json'):files[str(p)]=x.sha(p)
 for p in x.H.glob('*'):
  if p.is_file() and p.suffix in ('.py','.ps1','.md') or p.name=='protocol.json':files[str(p)]=x.sha(p)
 x.verify(files);identities=[];models=[];samples=0;bench=[]
 # All selections/design frozen before interventions evaluated.
 for fold in x.s.FOLDS:
  x.stage('PREFLIGHT_IDS_TRAIN_BANK',fold=fold);d=x.selected(fold);pool=x.bank(fold);sy=x.synthetic(d,pool,fold);x.save(f'data/{fold}_selected.npz',**d);x.save(f'data/{fold}_synthetic.npz',**sy)
  identities.append(dict(fold=fold,n=len(d['ids']),tracks=len(np.unique(d['track'])),synthetic_n=len(sy['index']),moving_n=int((np.linalg.norm(d['obs'][:,0,-1]-d['obs'][:,0,-2],axis=1)>=.02).sum()),sampling_tag='20261009-stage2',all_ids_explicit=True));samples+=len(d['ids'])
  for name in x.SPEC:
   o,v,ix,params=x.intervention(d,name,sy)
   assert np.array_equal(o[:,0,-2:],d['obs'][ix,0,-2:]) if name in ('reverse_prefix','shuffle_prefix','noNbr','farNbr','replaceNbr') or name.startswith('leave') or name.startswith('L') else True
   if name.startswith('syn'):assert np.array_equal(o[:,0],d['obs'][ix,0])
  for seed in (1,2,3):
   x.stage('PREFLIGHT_FROZEN_GPU',fold=fold,seed=seed);m,tr,cfg=x.s.build(fold,seed);old=x.s.snapshot(m);o=d['obs'][:128];v=d['valid'][:128];p,l,j,jl,att=x.forward(m,tr,o,v,True);pp,ll=x.s.forward(m,tr,o,v);assert np.array_equal(p,pp) and np.array_equal(l,ll) and att.shape==(len(o),9)
   source=x.a.load(x.P/f'jobs/{fold}_seed{seed}/outputs.npz');ix=d['full_row'][:128];err=max(float(abs(p-source['model_paths'][ix]).max()),float(abs(l-source['model_logits'][ix]).max()));assert err<=2e-5
   times=[]
   for name in ('L2','shuffle_prefix','noNbr','leave1','rotate','mirror','syn_left_03','syn_right_25'):
    ob,nv,ii,par=x.intervention(d,name,sy);start=time.perf_counter();q,ql,qj,qjl,_=x.forward(m,tr,ob[:128],nv[:128]);times.append(time.perf_counter()-start);assert np.isfinite(q).all()
   x.s.exact(m,old);models.append(dict(fold=fold,seed=seed,parameters=sum(p.numel() for p in m.parameters()),all_weights_bool_statbuffers_EXACT=True,attention_hook_BITWISE_original=True,cache_alignment_max=err))
   if fold=='eth' and seed==1:
    import torch
    for _ in range(4):x.forward(m,tr,o,v)
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(8):x.forward(m,tr,o,v)
    torch.cuda.synchronize();forward=(time.perf_counter()-start)/8
    z=dict(paths=p,logits=l,GT=d['gt'][:128],base_paths=p,base_logits=l,track=d['track'][:128],mask_all=np.ones(len(p),bool));start=time.perf_counter();r,dc,ec=x.summary(z);stats=time.perf_counter()-start;x.save('preflight/fixture.npz',**z);x.write('preflight/fixture.json',r);zz=x.a.load(x.H/'preflight/fixture.npz');rr,dd,ee=x.summary(zz);assert rr==r and np.array_equal(dd,dc) and np.array_equal(ee,ec);bench=dict(forward_128=forward,fullquality_CI_cluster_128=stats)
   del m,old,p,source;gc.collect()
 for p in (x.H/'data').glob('*'):files[str(p)]=x.sha(p)
 x.verify(files);x.write('source_manifest.json',dict(files=files,scope='Inherited stage1 source/calibrations/models and saved results plus exact new interventions, not all historical files'))
 total_windows=3*(22*samples+10*sum(i['synthetic_n'] for i in identities));nominal=total_windows/128*(bench['forward_128']+2*bench['fullquality_CI_cluster_128']);planning=max(600.,nominal+120.)
 free=__import__('shutil').disk_usage(x.H).free/2**30;estimate=total_windows*20*12*2*8*4/2**30
 assert free>=estimate+3,('DISK',free,estimate)
 fixture=x.H/'preflight/completed_done.json';x.write('preflight/completed_done.json',dict(status='SUCCESS',source_sha=x.sha(x.H/'source_manifest.json'),outputs={str(x.H/'preflight/fixture.npz'):x.sha(x.H/'preflight/fixture.npz')}));x.valid_done(fixture)
 for bad in ('source','output'):
  rec=x.read(fixture)
  if bad=='source':rec['source_sha']='0'*64
  else:rec['outputs'][str(x.H/'preflight/fixture.npz')]='0'*64
  x.write(f'preflight/bad_{bad}.json',rec)
  try:x.valid_done(x.H/f'preflight/bad_{bad}.json')
  except AssertionError:pass
  else:raise AssertionError('BAD_SOURCE_OUTPUT_ACCEPTED')
 x.write('PREFLIGHT.json',dict(status='PASS',actual_models=models,identities=identities,saved_math_repeat_exact=True,source_initial_final_PASS=True,source_files=len(files),jobs=480,windows_estimate=total_windows,benchmark=bench,nominal_compute_seconds=nominal,planning_seconds=planning,ETA_range_minutes=[planning/60,planning/60*1.8],uncompressed_output_estimate_GiB=estimate,diskfree_GiB=free,resource=x.a.RES,elapsed_seconds=time.perf_counter()-t,not_training=True,not_optimizer_RNG_resume=True))
 x.write('prelaunch_review/REVIEW.json',dict(status='PASS',actual15_frozen_attention_capture_no_output_change=True,train_only_real_velocity_bank=True,synthetic_actual_closest_all_doses_verified=True,raw_ID_cache_alignment=True,saved_quality_clusters_motion_CI_fixture=True,source_locked=True,DA1='same assistant: OOD/ordering/readout and correlation limitations retained'))
 x.stage('PREFLIGHT_PASS')
if __name__=='__main__':
 try:main()
 except Exception:x.write(f'failures/preflight_{time.time_ns()}/FAILURE.json',dict(status='FAILURE',traceback=traceback.format_exc(),sources={str(p):x.sha(p) for p in x.H.glob('*.py')},resource=x.a.RES));raise

