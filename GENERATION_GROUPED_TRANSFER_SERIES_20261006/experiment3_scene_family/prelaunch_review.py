"""Read-only final source/pipeline/reuse/resource review; not another GPU training."""
import adapter as h
import evaluate as ev
import ast,numpy as np,shutil,time,traceback
def main():
    begin=time.perf_counter();pre=h.read(h.HERE/'preflight.json')
    assert pre['status']=='PASS' and pre['resume_status']=='PASS'
    h.verify(pre['source_hashes']);old=h.read(h.HERE/'old_results_manifest.json')['files'];h.verify(old)
    identity=h.read(h.HERE/'identity_audit/RESULT.json');assert identity['status']=='PASS_IDENTITY_MATCHED_SCENE_FAMILY_SPLIT'
    assert identity['grouped_exposure']['same_sequence_20index_overlap']==0 and identity['random_exposure']['same_sequence_20index_overlap']>0
    pipeline=h.read(h.HERE/'pipeline_review.json');assert pipeline['status']=='PASS'
    resume=h.read(h.HERE/'resume_check.json');assert resume['status']=='PASS'
    directory=h.HERE/'prelaunch_review';directory.mkdir(exist_ok=True)
    name='preflight_smoke_native128';saved=h.RAW_LOAD(h.HERE/f'results/saved/{name}.npz')
    replay,stamp=ev.saved_job(name,None,dict(sample_id=saved['sample_id']),[],None)
    assert all(np.array_equal(saved[k],replay[k]) for k in saved)
    rejected=[]
    for kind in ('badsource','badoutput'):
        badname='prelaunch_'+kind;done=h.HERE/f'results/saved/{badname}_done.json'
        fixture=dict(stamp)
        if kind=='badsource':fixture['source_hashes']={}
        else:fixture['outputs']={str(h.HERE/f'results/saved/{name}.npz'):'0'*64}
        h.write(done,fixture)
        try:ev.saved_job(badname,None,dict(sample_id=saved['sample_id']),[],None)
        except AssertionError:rejected.append(kind)
        else:raise AssertionError('Bad saved source/output accepted')
    # Execute exactly the production completed-seed branch using SHA-locked
    # smoke-model fixtures; abort if train() is reached. No optimizer/forward.
    tree=ast.parse((h.HERE/'runner.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in fn.body if isinstance(n,ast.For))
    model=h.HERE/'resume_smoke/random/continuous/seed_7/balanced_protected/final.pth'
    fixture_root=directory/'completed_branch_fixture';seen=[]
    for arm in ('random','grouped'):
        for seed in (1,2,3):h.write(fixture_root/f'results/models/{arm}/seed_{seed}/balanced_protected/complete.json',dict(status='COMPLETE',epochs=80,model_hashes={str(model):h.sha(model)}))
    original_here=h.HERE;original_stage=h.stage;original_train=h.train;original_models=h.m.MODELS
    def forbid(*a,**k):raise AssertionError('Completed branch reached train')
    try:
        h.HERE=fixture_root;h.stage=lambda stage,**kw:seen.append(dict(stage=stage,**kw));h.train=forbid
        exec(compile(ast.Module(body=[loop],type_ignores=[]),'actual_runner_completed_branch','exec'),dict(h=h,cache=None,values=None,splits=None,device=None))
    finally:h.HERE=original_here;h.stage=original_stage;h.train=original_train;h.m.MODELS=original_models
    assert len(seen)==6 and all(x['stage']=='SKIP_VERIFIED_COMPLETED_SEED' for x in seen)
    with np.load(h.HERE/'identity_audit/splits.npz',allow_pickle=False) as z:outer=len(z['outer_eval_index']);inner=len(z['inner_cal_index']);common=len(z['common_statistics_index'])
    rows=(outer+inner+2879);trajectory_bytes=19*rows*20*12*2*8
    # Conservative saved path+raw/base/up/metadata envelope, not all resident.
    saved_envelope=trajectory_bytes*5;onebatch_bytes=128*20*12*2*8
    checkpoint=h.HERE/'resume_smoke/random/continuous/seed_7/balanced_protected/epoch_002.pth'
    checkpoint_bytes=(checkpoint.stat().st_size if checkpoint.exists() else model.stat().st_size)*480
    free=shutil.disk_usage(h.HERE).free;assert free>saved_envelope+checkpoint_bytes+2*2**30,'DISK_BUDGET'
    h.verify(pre['source_hashes']);h.verify(old);resource=h.guard.check()
    review=dict(status='PASS',source_count=len(pre['source_hashes']),old_files=len(old),source_old_beforeafter_UNCHANGED=True,identity=identity['status'],train_per_arm=16104,outer=outer,inner=inner,common_statistics=common,parameters=91596,
        actual_GPU8_BOTH_paired_frozen_finite_nativeidentity_originaltotalP_EXACT=True,BOTH_real2epoch_FITinner_subset_allparams_boolscalers_optimizer_allRNG_history_TRACE_EXACT=True,not_full16104_twice=True,
        actual_saved128_frozen_quality_complete_link_projectionphysics=pipeline['status'],actual_saved_done_SHA_skip=True,actual_saved_badsource_badoutput_reject=rejected,production_completed_branch_AST_SHA_skip=seen,no_model_forward_or_optimizer_in_this_review=True,
        bootstrap_primary='sequence_time_block+3seeds exploratory',bootstrap_sensitivity='target_track+3seeds separately preregistered',fixed_repetitions=2000,
        uncompressed_trajectory_GiB=trajectory_bytes/2**30,conservative_saved_output_GiB=saved_envelope/2**30,estimated_480epoch_checkpoint_GiB=checkpoint_bytes/2**30,one_batch_path_MiB=onebatch_bytes/2**20,disk_free_GiB=free/2**30,resource=resource,elapsed_seconds=time.perf_counter()-begin,formal_started=False,reviewer='same assistant inline; not external independent review')
    h.write(directory/'REVIEW.json',review);print('PRELAUNCH_REVIEW_PASS',review,flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        d=h.HERE/'failures/prelaunch_review1';d.mkdir(parents=True,exist_ok=True);(d/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
