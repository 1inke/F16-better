"""Independent saved-only production audit; no new model forward/optimizer."""
import json,sys,time
from pathlib import Path
import numpy as np
from controls import H,atomic_json,digest,guard,verify_binding
from saved_quality import math,compute

def audit():
    verify_binding(H/'source_input_binding.json');jobs=[];pairs=[];outputs={};resources=[]
    for fold in ('eth','hotel','univ','zara01','zara02'):
        for seed in (1,2,3):
            branches={};init=[]
            for branch in ('full','last2','noNbr'):
                for stage,epochs in (('h3',30),('v2',20)):
                    dest=H/'runs'/fold/f'seed_{seed}'/branch/stage/'f16_temporal_social_cross'/f'seed_{seed}'/'val_scene_-1'
                    done=math.read(dest/'done.json');manifest=math.read(dest/'run_manifest.json')
                    assert done['status']=='SUCCESS' and done['epochs']==epochs and not done['preflight_subset_only']
                    assert manifest['source_input_binding_sha']==digest(H/'source_input_binding.json')
                    for path,sha in done['outputs'].items():assert digest(path)==sha;outputs[path]=sha
                    resources.append(done['resource_aggregate'])
                    if stage=='h3':init.append(math.read(dest/'initial_fingerprint.json')['sha'])
                    else:
                        guard();z=math.load(dest/'selected_predictions.npz');masks={k[5:]:v for k,v in z.items() if k.startswith('mask_')}
                        delta=math.read(H.parent.parent/f'calibration/{fold}.json')['delta_nonCV_candidate']
                        arrays,records=compute(z['paths'],z['GT'],z['logits'],z['original_obs'],masks,delta)
                        for k,v in arrays.items():assert np.array_equal(v,z[k]),k
                        assert records==math.read(dest/'saved_quality.json')['records']
                        jobs.append(dict(fold=fold,seed=seed,branch=branch,records=records,rows=len(z['sample_id'])))
                        branches[branch]=z
                outputs[str(dest/'done.json')]=digest(dest/'done.json')
            assert len(set(init))==1,'STOP_PAIRED_H3_INITIALIZATION'
            full=branches['full']
            for branch in ('last2','noNbr'):
                other=branches[branch];assert np.array_equal(full['sample_id'],other['sample_id']) and np.array_equal(full['GT'],other['GT'])
                for name,m in {k[5:]:v for k,v in full.items() if k.startswith('mask_')}.items():
                    if not m.any():continue
                    rec=dict(fold=fold,seed=seed,branch=branch,subset=name,n=int(m.sum()),tracks=len(np.unique(full['track'][m])))
                    for key in ('minADE','minFDE','J','coverage'):
                        d=other['candidate_'+key]-full['candidate_'+key];rec[key+'_delta']=float(d[m].mean());rec[key+'_track_CI']=math.ci(d[m],full['track'][m])
                    a=full['candidate_coverage'].astype(bool);b=other['candidate_coverage'].astype(bool)
                    rec.update(rescue=int((m&b&~a).sum()),lost=int((m&a&~b).sum()));pairs.append(rec)
            del branches
    verify_binding(H/'source_input_binding.json')
    atomic_json(dict(status='PASS_INDEPENDENT_ALL_SAVED_MATH',jobs=45,stage_fits=90,epochs=2250,records=jobs,paired= pairs,all_source_output_SHA_unchanged=True,all_selected_saved_quality_clusters_motion_EXACT=True,no_new_forward_or_optimizer=True,not_external_review=True),H/'ARITHMETIC_AUDIT.json')
    atomic_json(dict(status='PASS',stage_resources=resources,peak_training_private_GiB=max(r['peak_private_GiB'] for r in resources),min_training_physical_GiB=min(r['min_physical_GiB'] for r in resources),min_training_commit_GiB=min(r['min_commit_GiB'] for r in resources)),H/'RESOURCE_USAGE.json')
    atomic_json(dict(status='PASS_FRESH_MATCHED_E3_RETRAINING',pipelines=45,stage_fits=90,epoch_rows=2250,jobs=jobs,paired=pairs,outputs=outputs,whole_series_complete=False,E5_rotation_and_E6_neural_sibling_remaining=True),H/'RESULT.json')
if __name__=='__main__':audit()


