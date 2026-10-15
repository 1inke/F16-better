"""Current-source prelaunch and negative fixtures. No science file modifications."""
import ast,json,math,shutil,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controls import H,P,S,ProductionRecovery,atomic_json,digest,guard,verify_binding
from saved_quality import compute,math as cpu

def main():
    start=time.perf_counter();guard();binding=H/'source_input_binding.json';sha=verify_binding(binding)
    resume=json.loads((H/'RESUME_PREFLIGHT.json').read_text());assert resume['status']=='PASS_CURRENT_IMPLEMENTATION_SIX_ACTUAL_RECOVERY_INPUT_GRADIENT_AND_SAVED_PIPELINES' and len(resume['records'])==6
    folder=H/'prelaunch_review';badfolder=folder/'negative_fixtures'
    if badfolder.exists():raise RuntimeError('Preserve existing negative fixtures')
    badfolder.mkdir();data=json.loads(binding.read_text());bad=json.loads(json.dumps(data));bad['files'][next(iter(bad['files']))]='0'*64;atomic_json(bad,badfolder/'source.json')
    try:ProductionRecovery(H,'full',badfolder/'source.json',fixture=True)
    except RuntimeError as e:assert 'STOP_SOURCE' in str(e)
    else:raise AssertionError('Bad source accepted')
    src=H/'preflight_runs/full/h3/continuous/f16_temporal_social_cross/seed_1/val_scene_-1';m=json.loads((src/'run_manifest.json').read_text());d=json.loads((src/'done.json').read_text());fake=badfolder/'output';fake.mkdir()
    atomic_json(m,fake/'run_manifest.json');d['outputs'][next(iter(d['outputs']))]='0'*64;atomic_json(d,fake/'done.json')
    r=ProductionRecovery(H,'full',binding,fixture=True);args=SimpleNamespace(**m['arguments'],output_root=fake,force=False)
    try:r.skip_compatible(dict(run_dir=fake,args=args))
    except RuntimeError as e:assert 'STOP_COMPLETED_OUTPUT_SHA' in str(e)
    else:raise AssertionError('Bad saved output accepted')
    args.epochs+=1
    try:r.skip_compatible(dict(run_dir=fake,args=args))
    except RuntimeError as e:assert 'STOP_COMPLETED_PROTOCOL' in str(e)
    else:raise AssertionError('Bad protocol accepted')
    audits=[];bench=[]
    for branch in ('full','last2','noNbr'):
        for stage in ('h3','v2'):
            root=H/'preflight_runs'/branch/stage/'continuous/f16_temporal_social_cross/seed_1/val_scene_-1';done=cpu.read(root/'done.json')
            assert done['epochs']==2 and len(done['original_loss_gradient_replay'])==8 and done['input_path_audit']['actual_prediction_and_logits_EXACT']
            for p,s in done['outputs'].items():assert digest(p)==s
            audits.append(done['resource_aggregate'])
            if stage=='v2':
                z=cpu.load(root/'selected_predictions.npz');arr,records=compute(z['paths'],z['GT'],z['logits'],z['original_obs'],dict(all=np.ones(len(z['GT']),bool)),.2)
                for k,v in arr.items():assert np.array_equal(v,z[k])
                assert records==cpu.read(root/'saved_quality.json')['records']
            b=cpu.read(H/'benchmark'/f'{branch}_{stage}.json');assert b['status']=='PASS_ACTUAL_THROWAWAY_THROUGHPUT' and b['warm']==4 and b['measured']==32;bench.append(dict(branch=branch,stage=stage,**b))
    ids=cpu.read(H/'IDENTITY_AUDIT.json');assert ids['status']=='PASS_OFFICIAL_RAW_ID_SPLIT_AUDIT' and len(ids['records'])==5
    fits=sum(3*dict(h3=30,v2=20)[b['stage']]*sum(math.ceil(x['train']/64) for x in ids['records'])*b['train_step_seconds'] for b in bench)
    # Each branch has all 3 seeds per fold; validation every5 plus final/selected.
    val=sum(3*sum(x['validation'] for x in ids['records'])/b['validation_rows']*(dict(h3=8,v2=6)[b['stage']])*b['validation_seconds'] for b in bench)
    output_bytes=sum(x['test']*3*3*(20*12*2*8+20*4+9*8*2*4) for x in ids['records'])
    checkpoint_estimate=90*25*1024**2;free=shutil.disk_usage(H).free
    assert free>2*(output_bytes+checkpoint_estimate)+1024**3,('STOP_DISK',free,output_bytes,checkpoint_estimate)
    # Wide planning overhead includes 90 process/model initializations, final
    # checkpoints, 45 full K20 quality exports+re-reads+independent recompute.
    planning=math.ceil((1.25*(fits+val)+1800)/60);low=math.ceil((fits+val)/60);high=math.ceil((1.8*(fits+val)+3600)/60)
    assert verify_binding(binding)==sha
    gate=dict(status='PASS_ALL_ACTUAL_PREFLIGHT_AND_FINAL_SOURCE',binding_sha=sha,folds=['eth','hotel','univ','zara01','zara02'],branches=['full','last2','noNbr'],seeds=[1,2,3],pipelines=45,stage_fits=90,epoch_rows=2250,current_real_six_recovery_cases=6,actual_original_gradient_batches=48,full_fold_two_epoch=False,source_files=len(data['files']),identity=ids['records'],benchmark=bench,nominal_training_seconds=fits,nominal_validation_seconds=val,ETA_range_minutes=[low,high],planning_minutes=planning,no_saved_benchmark_model=True,disk_free_GiB=free/2**30,conservative_saved_path_GiB=output_bytes/2**30,checkpoint_temp_estimate_GiB=checkpoint_estimate/2**30,resource_samples=audits,whole_series_complete=False)
    atomic_json(gate,H/'PREFLIGHT.json');atomic_json(dict(status='PASS',binding_sha=sha,negative_source_output_protocol_refusals=True,all_6_current_done_SHA_verified=True,all_3_actual_saved_CPU_quality_clusters_motion_EXACT=True,elapsed_seconds=time.perf_counter()-start,no_new_training_or_forward=True,not_external_review=True),H/'PRELAUNCH_REVIEW.json');print(json.dumps(gate))
if __name__=='__main__':main()
