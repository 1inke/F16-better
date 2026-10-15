import json,os,sys
from controls import H,P,W
PROJECT='C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20'
SRC=dict(h3=W/'HIERARCHICAL_DECODER_SERIES/hierarchical_decoder_models.py',v2=W/'ANCHOR_CORRECTION_VALIDATION_SERIES/anchor_correction_validation_models.py')
NAMES=dict(h3='h3_anchor_proposal',v2='v2_wide_endpoint_correction')
def command(fold,seed,branch,stage,benchmark=False):
    data=json.loads((P/'SOURCE_READINESS.json').read_text())['data'][fold]
    root=H/'benchmark'/branch/stage if benchmark else H/'runs'/fold/f'seed_{seed}'/branch/stage
    cmd=[sys.executable,'-B','-u',str(H/('benchmark.py' if benchmark else 'runner.py')),'--input-branch',branch]
    if not benchmark:cmd+=['--source-binding',str(H/'source_input_binding.json'),'--production-gate',str(H/'PREFLIGHT.json')]
    cmd+=['--project-root',PROJECT,'--config-file',str(W/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/motif_rbf_official_4p8.yaml'),'--data-path',data['train_data_path']['path'],'--num-path',data['train_num_path']['path'],'--variant','f16_temporal_social_cross','--decoder-experiment',NAMES[stage],'--decoder-module',str(SRC[stage]),'--seed',str(seed),'--val-scene-id','-1','--validation-fraction','.05','--validation-seed','20260830','--validation-split-mode','scene_tail','--epochs',str(dict(h3=30,v2=20)[stage]),'--val-freq','5','--batch-size','64','--device','cuda:0','--max-agents','9','--graph-top-k','4','--graph-radius','6','--scheduler','monotonic_cosine','--warmup-epochs','1','--learning-rate','.00025','--output-root',str(root),'--oracle-coverage-weight','.20','--oracle-coverage-temperature','.10','--oracle-coverage-fde-weight','.25','--checkpoint-criterion','oracle_combo','--ema-decay','.995','--focal-heading-normalize','--augment-reflection','--augment-speed-min','.9','--augment-speed-max','1.1','--augment-probability','.5','--num-workers','0']
    if not benchmark:cmd+=['--test-data-path',data['test_data_path']['path'],'--test-num-path',data['test_num_path']['path']]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',E3_FOLD=fold,E3_STAGE=stage,E3_BRANCH=branch)
    if stage=='v2':
        env['EXP143_SOURCE_CHECKPOINT']=str(H/'preflight_runs'/branch/'h3/continuous/f16_temporal_social_cross/seed_1/val_scene_-1/best_selected.pth' if benchmark else H/'runs'/fold/f'seed_{seed}'/branch/'h3/f16_temporal_social_cross'/f'seed_{seed}'/'val_scene_-1/best_selected.pth')
    return cmd,env,root/'f16_temporal_social_cross'/f'seed_{seed}'/'val_scene_-1'


