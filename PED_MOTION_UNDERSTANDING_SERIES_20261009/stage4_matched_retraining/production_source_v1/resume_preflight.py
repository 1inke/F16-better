"""Actual bounded H3/V2, three-input-branch continuous/interrupted fixtures.

This tests epoch recovery only. A PASS does not authorize production: GPU
input-path equivalence, final source/ID audit, memory/disk and throughput gates
are additional requirements. Each run remains confined to preflight_runs.
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from controls import atomic_json, digest, guard

H=Path(__file__).resolve().parent;W=H.parent.parent.parent
PROJECT=Path('C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20')
CONFIG=W/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/motif_rbf_official_4p8.yaml'
SRC={'h3':W/'HIERARCHICAL_DECODER_SERIES/hierarchical_decoder_models.py','v2':W/'ANCHOR_CORRECTION_VALIDATION_SERIES/anchor_correction_validation_models.py'}
NAME={'h3':'h3_anchor_proposal','v2':'v2_wide_endpoint_correction'}
PREF=H/'preflight_runs';BINDING=H/'source_input_binding.json'

def eq(x,y):
    if isinstance(x,torch.Tensor):assert isinstance(y,torch.Tensor) and x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y)
    elif isinstance(x,np.ndarray):assert np.array_equal(x,y)
    elif isinstance(x,dict):
        assert x.keys()==y.keys()
        for k in x:eq(x[k],y[k])
    elif isinstance(x,(tuple,list)):
        assert len(x)==len(y)
        for a,b in zip(x,y):eq(a,b)
    else:assert x==y,(x,y)

def command(branch,stage,mode,data):
    root=PREF/branch/stage/mode
    cmd=[sys.executable,'-B','-u',str(H/'runner.py'),'--input-branch',branch,'--source-binding',str(BINDING),
        '--fixture-subset-rows','513','--project-root',str(PROJECT),'--config-file',str(CONFIG),'--data-path',data['train_data_path']['path'],
        '--num-path',data['train_num_path']['path'],'--variant','f16_temporal_social_cross','--decoder-experiment',NAME[stage],
        '--decoder-module',str(SRC[stage]),'--seed','1','--val-scene-id','-1','--validation-fraction','.05','--validation-seed','20260830',
        '--validation-split-mode','scene_tail','--epochs','2','--val-freq','1','--batch-size','64','--device','cuda:0',
        '--max-agents','9','--graph-top-k','4','--graph-radius','6','--scheduler','monotonic_cosine','--warmup-epochs','1',
        '--learning-rate','.00025','--output-root',str(root),'--oracle-coverage-weight','.20','--oracle-coverage-temperature','.10',
        '--oracle-coverage-fde-weight','.25','--checkpoint-criterion','oracle_combo','--ema-decay','.995','--focal-heading-normalize',
        '--augment-reflection','--augment-speed-min','.9','--augment-speed-max','1.1','--augment-probability','.5','--num-workers','0']
    return cmd,root/'f16_temporal_social_cross/seed_1/val_scene_-1'

def execute(cmd,log,env):
    guard();log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf-8') as f:
        result=subprocess.run(cmd,cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Actual fixture failed: '+str(log))

def main():
    begun=time.perf_counter();guard()
    if PREF.exists():raise RuntimeError('Existing fixtures preserved; no automatic rerun/overwrite')
    data=json.loads((H.parent/'SOURCE_READINESS.json').read_text(encoding='utf-8'))['data']['eth']
    from controls import verify_binding
    verify_binding(BINDING)
    rows=[];initial={}
    for branch in ('full','last2','noNbr'):
        for stage in ('h3','v2'):
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
            if stage=='v2':
                # Current branch's fresh H3 fixture, not an old full model.
                env['EXP143_SOURCE_CHECKPOINT']=str(PREF/branch/'h3/continuous/f16_temporal_social_cross/seed_1/val_scene_-1/best_selected.pth')
            c,left=command(branch,stage,'continuous',data)
            execute(c,PREF/branch/stage/'continuous.log',env)
            c,right=command(branch,stage,'interrupted',data)
            execute(c+['--fixture-stop-after','1'],PREF/branch/stage/'interrupted.log',env)
            first=torch.load(right/'epoch_resume.pt',map_location='cpu',weights_only=False)
            assert first['next_epoch']==1 and len(first['history'])==1
            execute(c+['--resume-epoch'],PREF/branch/stage/'resumed.log',env)
            a=torch.load(left/'epoch_resume.pt',map_location='cpu',weights_only=False);b=torch.load(right/'epoch_resume.pt',map_location='cpu',weights_only=False)
            for key in ('model','ema','optimizer','scheduler','RNG','history','best_ade','best_checkpoint_score','next_epoch'):eq(a[key],b[key])
            assert a['next_epoch']==2 and len(a['history'])==2
            for name in ('best_selected.pth','final_model.pth','final_model_raw.pth'):
                eq(torch.load(left/name,map_location='cpu',weights_only=True),torch.load(right/name,map_location='cpu',weights_only=True))
            m=json.loads((left/'metrics.json').read_text(encoding='utf-8'));n=json.loads((right/'metrics.json').read_text(encoding='utf-8'))
            for key in ('evaluation','evaluation_best_selected','best_checkpoint_score','locked_backend'):eq(m[key],n[key])
            state=torch.load(left/'initial_state.pt',map_location='cpu',weights_only=True)
            if stage=='h3':
                if initial:eq(initial,state)
                else:initial=state
            execute(c+['--resume-epoch'],PREF/branch/stage/'completed_sha_skip.log',env)
            done=json.loads((left/'done.json').read_text());assert len(done['original_loss_gradient_replay'])==8 and done['input_path_audit']['actual_prediction_and_logits_EXACT']
            if stage=='v2':assert done['V2_posthoc_complete']
            rows.append({'branch':branch,'stage':stage,'status':'PASS_ACTUAL_SUBSET_TWO_EPOCH_EXACT','train_rows':513,'validation_rows':12,
                'all_model_bool_buffers_EMA_optimizer_scheduler_ALL_RNG_history_and_selected_parameters_EXACT':True,
                'initial_same_H3_across_branches':stage=='h3','completed_SHA_skip_actual':True,'source_H3_current_branch_fresh':stage=='v2',
                'full_fold_two_epochs':False})
            atomic_json({'status':'RUNNING_RESUME_FIXTURES','completed':rows,'not_production':True},H/'resume_preflight_progress.json')
            print(json.dumps(rows[-1]),flush=True)
    atomic_json({'status':'PASS_CURRENT_IMPLEMENTATION_SIX_ACTUAL_RECOVERY_INPUT_GRADIENT_AND_SAVED_PIPELINES','records':rows,'elapsed_seconds':time.perf_counter()-begun,
        'remaining_gates':['official_ID_full_audit','benchmark_and_peak_resources','final_source_lock'],
        'not_production_preflight_PASS':True,'whole_series_complete':False},H/'RESUME_PREFLIGHT.json')

if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json({'status':'FAILURE_PRESERVED','traceback':traceback.format_exc(),'no_production_launch':True},H/'RESUME_PREFLIGHT_FAILURE.json')
        raise


