"""Read saved canonical manifests and data SHA; no old jobs or model forward."""
from pathlib import Path
import json
from epoch_recovery import digest, atomic_json, guard

H=Path(__file__).resolve().parent
W=H.parent.parent
FOLDS=('eth','hotel','univ','zara01','zara02')
CONFIG=W/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/motif_rbf_official_4p8.yaml'

def main():
    resources=guard()
    previous=json.loads((H.parent/'stage3_knn_memory/source_manifest.json').read_text(encoding='utf-8'))['files']
    for name,expected in previous.items():
        if digest(name)!=expected:raise RuntimeError('STOP_PRIOR_SOURCE '+name)
    files=dict(previous);rows=[];data={}
    for fold in FOLDS:
        data[fold]={}
        for seed in (1,2,3):
            root=(W/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/results' if seed==1 else W/'OFFICIAL_ETH_UCY_FINAL_3SEED_SERIES/EXP_164_FINAL_V2_OFFICIAL_3SEED/results')
            for stage,epochs,counts in (('h3',30,(295474,294897)),('v2',20,(321988,26514))):
                path=root/fold/stage/'f16_temporal_social_cross'/f'seed_{seed}'/'val_scene_-1/metrics.json'
                m=json.loads(path.read_text(encoding='utf-8'));b=m['locked_backend'];s=m['split']
                assert m['epochs']==epochs and m['scheduler']=='monotonic_cosine' and m['seed']==seed
                assert s['strategy']=='scene_tail_holdout_within_training_scenes' and s['validation_fraction']==.05 and s['validation_seed']==20260830
                assert b['num_modes']==20 and b['classification_weight']==.5 and b['ema_decay']==.995
                assert b['oracle_coverage_weight']==.2 and b['oracle_coverage_temperature']==.1 and b['oracle_coverage_fde_weight']==.25
                assert b['checkpoint_criterion']=='oracle_combo' and b['focal_heading_normalize']
                assert b['augmentation']==dict(rotation=False,reflection=True,speed_min=.9,speed_max=1.1,probability=.5,curvature_max=0.)
                assert (b['parameter_counts']['total'],b['parameter_counts']['trainable'])==counts
                files[str(path)]=digest(path)
                for key in ('data_path','num_path'):
                    recorded=Path(s[key]);targetkey='train_'+key
                    if targetkey in data[fold]:assert data[fold][targetkey]['sha']==digest(recorded)
                    else:data[fold][targetkey]={'path':str(recorded),'sha':digest(recorded)}
                    files[str(recorded)]=digest(recorded)
                rows.append({'fold':fold,'seed':seed,'stage':stage,'epochs':epochs,'manifest':str(path),'params':counts,'train_n':s['train_samples'],'val_n':s['validation_samples']})
        # Native stage1 protocol already binds the official held-out test cache.
        # Use matching raw test files, never its values for training selection.
        for key,suffix in (('test_data_path','data'),('test_num_path','num')):
            raw=Path(data[fold]['train_'+suffix+'_path']['path']).with_name(f'{fold}_{suffix}_test.npy')
            data[fold][key]={'path':str(raw),'sha':digest(raw)};files[str(raw)]=digest(raw)
    for p in (CONFIG,W/'FRONT_ENCODER_SERIES/train_front_variant.py',W/'OFFICIAL_ETH_UCY_FINAL_3SEED_SERIES/run_final_v2_seeds23.py',W/'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/run_exp156_fold_local_pipeline.py',H/'isolated_trainer.py',H/'epoch_recovery.py',H/'input_transform.py',H/'DESIGN.md',H/'source_readiness.py'):
        files[str(p)]=digest(p)
    atomic_json({'files':files,'inherited_digests_not_overwritten':len(previous),'purpose':'E3 actual source+raw-input binding; not GPU/resume PASS'},H/'source_input_binding.json')
    atomic_json({'status':'PASS_SAVED_CANONICAL_METADATA_AND_RAW_SHA_ONLY','saved_training_manifests':rows,'data':data,'locked_files':len(files),
        'resource':resources,'actualGPU_or_resume_preflight':False,'old_training_manifests_not_complete_historical_source_binding':True,
        'architecture_and_recorded_objective_consistent':True,'no_old_worker_queries':True},H/'SOURCE_READINESS.json')
    print(json.dumps({'status':'PASS_METADATA_RAW_SHA_ONLY','manifests':len(rows),'files':len(files)}))

if __name__=='__main__':main()
