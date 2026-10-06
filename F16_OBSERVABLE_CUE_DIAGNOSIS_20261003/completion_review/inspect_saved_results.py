"""Read-only post-completion checks and descriptive summaries. No model/optimizer.

Writes only new completion_review artifacts. Does not replay all reader models.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / 'completion_review'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1-1e-7)
    assert np.isfinite(p).all()
    return dict(n=len(y), prevalence=float(np.mean(y)),
        AUPRC=float(average_precision_score(y,p)) if np.any(y) else None,
        AUROC=float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None,
        Brier=float(np.mean((p-y)**2)), log_loss=float(log_loss(y,p,labels=[0,1])))

def check_metrics(actual, expected):
    for key, value in actual.items():
        target = expected[key]
        assert value == target or (value is not None and target is not None and abs(value-target)<1e-12), (key,value,target)

def summarize_masks(mask, good, selected_good, event):
    n=int(mask.sum())
    return dict(n=n, candidate_good_n=int((mask&good).sum()), selected_good_n=int((mask&selected_good).sum()),
        candidate_present_selection_miss_n=int((mask&good&~selected_good).sum()),
        no_good_candidate_n=int((mask&~good).sum()),
        event_type_time_any_n=int((mask&event).sum()))

def main():
    assert not (OUT/'saved_result_review.json').exists(), 'Do not overwrite a completed audit'
    result=read(HERE/'RESULT.json')
    worker=read(HERE/'independent_worker_state.json')
    exit_info=read(HERE/'exit_status.json')
    assert result['status']=='PASS_DIAGNOSTIC_ONLY' and worker['status']=='SUCCESS' and exit_info['status']=='SUCCESS'
    assert worker['pid']==39792 and result['completed_reader_jobs']==153
    old=read(HERE/'old_results_manifest.json')['files']
    source=read(HERE/'preflight.json')['source_hashes']
    for title, collection in [('old',old),('source',source)]:
        for path, expected in collection.items():
            assert Path(path).is_file() and digest(path)==expected, (title,path)
    artifacts=result['artifact_sha256']
    mismatch=[name for name, expected in artifacts.items() if not (HERE/name).is_file() or digest(HERE/name)!=expected]
    # Worker writes its terminal exit record after RESULT is snapshotted.
    assert set(mismatch).issubset({'exit_status.json'}), mismatch
    binary=read(HERE/'binary_probes.json')
    ids_meta=np.load(HERE/'metadata.npz',allow_pickle=False)
    family_by_id=dict(zip(ids_meta['sample_id'].astype(int),ids_meta['scene_family']))
    labels=np.load(HERE/'labels/train.npz',allow_pickle=False)
    train_id_index={int(v):i for i,v in enumerate(labels['sample_id'])}
    job_n=0
    for file in sorted((HERE/'jobs').glob('*.done.json')):
        rec=read(file)
        assert rec['status']=='SUCCESS'
        for name, expected in rec['hashes'].items():
            assert digest(HERE/name)==expected
        if 'transform_file' in rec:
            assert digest(HERE/rec['transform_file'])==rec['transform_sha']
        pred=np.load(HERE/rec['predictions'],allow_pickle=False)
        assert len(np.unique(pred['sample_id']))==len(pred['sample_id'])
        expected_y=labels['y'][[train_id_index[int(v)] for v in pred['sample_id']]]
        assert np.array_equal(pred['y'],expected_y)
        check_metrics(metrics(pred['y'],pred['p']),rec['metrics'])
        job_n+=1
    assert job_n==153
    grouped={}
    reference_ids=None
    for name, rec in binary['outputs'].items():
        pred=np.load(HERE/rec['file'],allow_pickle=False)
        check_metrics(metrics(pred['y'],pred['p']),rec['metrics'])
        if reference_ids is None:
            reference_ids=pred['sample_id'].copy()
        assert np.array_equal(pred['sample_id'],reference_ids)
        families=np.array([family_by_id[int(v)] for v in pred['sample_id']])
        grouped[name]={g:metrics(pred['y'][families==g],pred['p'][families==g]) for g in np.unique(families)}
    secondary=[]
    for row in read(HERE/'secondary_and_controls.json')['secondary']:
        d=np.load(HERE/f"secondary/{row['probe']}_{row['target']}.npz",allow_pickle=False)
        assert len(np.unique(d['sample_id']))==len(d['sample_id'])
        expected_y=labels[row['target']][[train_id_index[int(v)] for v in d['sample_id']]]
        assert np.array_equal(d['truth'],expected_y)
        accuracy=float(np.mean(d['truth']==d['prediction']))
        assert abs(accuracy-row['accuracy'])<1e-12
        family=np.array([family_by_id[int(v)] for v in d['sample_id']])
        prior_pred=np.empty_like(d['truth'])
        by_group={}
        for g in np.unique(family):
            tr=d['truth'][family!=g];classes,counts=np.unique(tr,return_counts=True)
            prior_pred[family==g]=classes[counts.argmax()]
            by_group[g]=dict(n=int((family==g).sum()), accuracy=float(np.mean(d['prediction'][family==g]==d['truth'][family==g])))
        secondary.append(dict(probe=row['probe'],target=row['target'],n=len(d['truth']),accuracy=accuracy,
            OOF_training_majority_accuracy=float(np.mean(prior_pred==d['truth'])), by_scene=by_group,
            inference='Actual positive-event subset only; no deployment accuracy or significance claim'))
    dev=np.load(HERE/'candidate_audit.npz',allow_pickle=False)
    features=np.load(HERE/'features/dev.npz',allow_pickle=False)
    devlabel=np.load(HERE/'labels/dev.npz',allow_pickle=False)
    assert np.array_equal(dev['sample_id'],features['sample_id']) and np.array_equal(dev['sample_id'],devlabel['sample_id'])
    assert features['paths'].shape==(1516,20,12,2) and np.isfinite(features['paths']).all()
    assert np.array_equal(features['logits'].argmax(1),dev['selected'])
    error=np.linalg.norm(features['paths']-devlabel['future'][:,None],axis=-1)
    np.testing.assert_allclose(error.mean(-1),dev['ade'],atol=2e-5,rtol=0)
    np.testing.assert_allclose(error[:,:,-1],dev['fde'],atol=2e-5,rtol=0)
    groups={'all':np.ones(1516,bool),'clear_event':dev['GT_event'].astype(bool),
        'no_clear_event':~dev['GT_event'].astype(bool)&~dev['GT_uncertain'],
        'boundary_uncertain':dev['GT_uncertain'],'near_static':dev['near_static']}
    candidate={}
    for suffix in ('0.2_0.4','0.3_0.5'):
        good=dev['coverage_'+suffix];selected_good=dev['selected_good_'+suffix]
        candidate[suffix]={g:summarize_masks(mask,good,selected_good,dev['event_type_time_coverage']) for g,mask in groups.items()}
    risk=np.load(HERE/'dev_risk_readouts.npz',allow_pickle=False)
    assert np.array_equal(risk['sample_id'],dev['sample_id'])
    risk_summary={}
    for key in ('final_logistic_full_R2','final_logistic_full_E2_motion','final_logistic_full_E3'):
        risk_summary[key]={g:dict(n=int(mask.sum()),mean=float(risk[key][mask].mean()),
            high_half_n=int((mask&(risk[key]>=.5)).sum()),
            high_half_no_good_candidate_n=int((mask&(risk[key]>=.5)&~dev['coverage_0.2_0.4']).sum()),
            high_half_selection_miss_n=int((mask&(risk[key]>=.5)&dev['coverage_0.2_0.4']&~dev['selected_good_0.2_0.4']).sum())) for g,mask in groups.items()}
    routing={}
    for split in ('train','dev'):
        feature=np.load(HERE/f'features/{split}.npz',allow_pickle=False)
        allowed=feature['allowed'].astype(bool)
        raw=np.load(HERE.parent/f'GENERATION_UPSTREAM_TEMPORAL_SERIES/{split}_observations.npz',allow_pickle=False)
        assert np.array_equal(feature['sample_id'],raw['sample_id'])
        count=raw['observation'][:,304].astype(int)
        valid=np.arange(9)[None]<count[:,None]
        assert allowed.shape==valid.shape and not (allowed&~valid).any()
        routing[split]=dict(n=len(count),allowed_self_n=int(allowed[:,0].sum()),
            valid_neighbor_mean=float((count-1).mean()),allowed_neighbor_mean=float(allowed[:,1:].sum(1).mean()),
            any_valid_neighbor_n=int((count>1).sum()), any_allowed_neighbor_n=int(allowed[:,1:].any(1).sum()),
            first_neighbor_valid_n=int((count>1).sum()),first_neighbor_allowed_n=int(allowed[:,1].sum()),
            invalid_allowed_n=int((allowed&~valid).sum()),
            limitation='Mask routing only; exclusion is not proof of a missed useful cue; eligibility retained from original future-completeness preprocessing.')
    fn=read(HERE/'functional_sensitivity.json')
    trainfeature=np.load(HERE/'features/train.npz',allow_pickle=False)
    trainmap={int(v):i for i,v in enumerate(trainfeature['sample_id'])}
    route=trainfeature['allowed'][[trainmap[v] for v in fn['sample_id']],1].astype(bool)
    functional=[]
    for rec in fn['records']:
        path=np.array(rec['path_max_delta']);score=np.array(rec['logit_max_delta'])
        functional.append(dict(shift_m=rec['shift_m'],n=len(path),first_neighbor_allowed_n=int(route.sum()),
            path_delta_above_1e_5_n=int((path>1e-5).sum()),max_path_delta=float(path.max()),
            max_logit_delta=float(score.max()),selection_switch_n=int(np.sum(rec['argmax_switched'])),
            responsive_allowed_n=int(((path>1e-5)&route).sum()),responsive_not_allowed_n=int(((path>1e-5)&~route).sum())))
    output=dict(status='PASS_SAVED_ARTIFACT_REVIEW',no_optimizer=True,no_new_native_forward=True,
        worker=worker,exit=exit_info,reader_jobs=job_n,old_SHA_files=len(old),source_SHA_files=len(source),
        original_artifact_hashes_checked=len(artifacts),post_worker_lifecycle_hash_changes=mismatch,
        lifecycle_note='exit_status.json was finalized after RESULT snapshot; no scientific artifact mismatch accepted.',
        native_ID_alignment_finite_K20_and_errors_recomputed=True,
        saved_predictions_independently_recomputed_not_all_models_replayed=True,
        grouped_binary=grouped,secondary_vs_majority=secondary,candidate_class_counts=candidate,
        descriptive_DEV_risk_groups_not_selection=risk_summary,original_input_routing=routing,functional_routing_summary=functional,
        blind_human_annotations=0,formal_trial=False,major_claim=False,
        uncertainty='Only three scene families and native previous training exposure; no formal group CI/Holm or causal inference.')
    OUT.mkdir(exist_ok=True)
    (OUT/'saved_result_review.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ('grouped_binary','descriptive_DEV_risk_groups_not_selection')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
