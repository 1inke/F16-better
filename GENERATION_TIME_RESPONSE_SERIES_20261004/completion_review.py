"""CPU saved-output review only. No optimizer, model forward, or old-output edits."""
from pathlib import Path
import sys, json, csv, hashlib, importlib.util
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
OUT = HERE / 'completion_review'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

def main():
    assert not (OUT / 'REVIEW.json').exists(), 'Preserve previous supplemental review'
    h = module('completed_time_response', HERE / 'run_head.py')
    common = module('completed_event_rules', HERE.parent / 'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/common.py')
    summary = read(RESULTS / 'summary.json')
    audit = read(RESULTS / 'independent_audit.json')
    recompute = read(RESULTS / 'saved_metrics_recompute.json')
    worker = read(HERE / 'independent_worker_state.json')
    assert worker['status'] == read(RESULTS / 'exit_status.json')['status'] == 'SUCCESS'
    assert audit['status'] == recompute['status'] == 'PASS'
    assert audit['old_files_unchanged'] == 18872 and audit['registered_sources_unchanged'] == 229
    sources = read(HERE / 'preflight.json')['source_hashes']
    models = read(RESULTS / 'model_lock.json')['models']
    assert len(models) == 18
    inputs = {**sources, **models}
    for path, sha in inputs.items():
        assert digest(path) == sha, path
    for path in RESULTS.rglob('*'):
        if path.is_file():
            inputs[str(path)] = digest(path)
    cache = h.p.load_npz(h.p.DEV_CACHE)
    values = h.r.prepared(cache, False, mean=np.zeros((1,940), np.float32), std=np.ones((1,940), np.float32))
    base, future = values[1], values[3]
    past, ids = cache['_past'], cache['sample_id']
    assert len(ids) == len(np.unique(ids)) == 1516
    cfg = h.c.settings(1)
    truth = common.events(past, future + past[:, -1, None])
    groups = {'overall':np.ones(len(ids),bool), 'clear_event':(truth['y']==1)&~truth['uncertain'],
              'no_clear_event':(truth['y']==0)&~truth['uncertain'], 'uncertain':truth['uncertain']}
    native_errors = np.linalg.norm(base - future[:,None], axis=-1)
    native_good = {f'{ta}_{tf}':((native_errors.mean(-1)<=ta)&(native_errors[:,:,-1]<=tf)).any(-1)
                   for ta,tf in ((.2,.4),(.3,.5))}
    rows, valid_predictions, max_error = [], 0, 0.
    all_goods = {}
    for name, pred in [('native', base)] + [(path.name, np.load(path, allow_pickle=False)) for path in sorted(RESULTS.glob('seed*.npz'))]:
        if name != 'native':
            assert np.array_equal(pred['sample_id'], ids)
            saved = pred
            pred = saved['trajectory']
            valid_predictions += 1
        assert pred.shape == (1516,20,12,2) and np.isfinite(pred).all()
        errors = np.linalg.norm(pred - future[:,None], axis=-1)
        ade, fde = errors.mean(-1), errors[:,:,-1]
        if name != 'native':
            for key, actual in [('minade20',ade.min(-1)),('minfde20',fde.min(-1))]:
                error = float(np.max(np.abs(actual-saved[key])))
                max_error = max(max_error, error)
                assert error < 1e-6, (name,key,error)
        chosen = (ade + cfg['fde_loss_weight']*fde).argmin(-1)
        ar = np.arange(len(ids))
        selected = errors[ar, chosen]
        event = common.events(np.repeat(past,20,axis=0), (pred + past[:,-1,None,None]).reshape(-1,12,2))
        kind = event['kind'].reshape(len(ids),20)[ar,chosen]
        onset = event['onset_step'].reshape(len(ids),20)[ar,chosen]
        direction = event['direction'].reshape(len(ids),20)[ar,chosen]
        predicted_turn = event['event_types'].reshape(len(ids),20,4)[ar,chosen,0].astype(bool)
        speed = event['future_speed'].reshape(len(ids),20,12)[ar,chosen]
        good = {f'{ta}_{tf}':((ade<=ta)&(fde<=tf)).any(-1) for ta,tf in ((.2,.4),(.3,.5))}
        all_goods[name] = good
        correction = pred-base
        increments = np.diff(np.concatenate([np.zeros_like(correction[:,:,:1]),correction],axis=2),axis=2)
        blocks = increments.reshape(len(ids),20,3,4,2)
        varying = np.abs(blocks-blocks.mean(3,keepdims=True)).mean((1,2,3,4))
        for label,mask in groups.items():
            row = dict(file=name,group=label,n=int(mask.sum()),same_candidate_ade=float(selected[mask].mean()),
                       same_candidate_fde=float(selected[mask,-1].mean()),early_error=float(selected[mask,:4].mean()),
                       middle_error=float(selected[mask,4:8].mean()),late_error=float(selected[mask,8:].mean()),
                       event_kind_match=float((kind[mask]==truth['kind'][mask]).mean()),
                       speed_curve_mae=float(np.abs(speed[mask]-truth['future_speed'][mask]).mean()),
                       within_block_correction_increment_deviation_mean=float(varying[mask].mean()))
            for key,yes in good.items():
                old=native_good[key]
                row['adequacy_'+key] = dict(count=int((mask&yes).sum()),rate=float(yes[mask].mean()),
                    native_count=int((mask&old).sum()),native_missing=int((mask&~old).sum()),
                    rescued=int((mask&~old&yes).sum()),lost=int((mask&old&~yes).sum()),
                    rescue_fraction_of_native_missing=float((mask&~old&yes).sum()/max(1,(mask&~old).sum())))
            if label=='clear_event':
                detected=mask&(onset<99)
                matched=detected&(kind==truth['kind'])
                turn=mask&truth['event_types'][:,0].astype(bool)
                turn_detected=turn&predicted_turn
                row.update(event_detected_n=int(detected.sum()),event_detected_rate=float(detected.sum()/mask.sum()),
                    onset_mae_steps_detected=float(np.abs(onset[detected]-truth['onset_step'][detected]).mean()),
                    kind_matched_n=int(matched.sum()),onset_mae_steps_kind_matched=float(np.abs(onset[matched]-truth['onset_step'][matched]).mean()),
                    truth_turn_n=int(turn.sum()),truth_turn_direction_accuracy=float((direction[turn]==truth['direction'][turn]).mean()),
                    truth_turn_detected_n=int(turn_detected.sum()),
                    truth_turn_direction_accuracy_given_detected=float((direction[turn_detected]==truth['direction'][turn_detected]).mean()))
            rows.append(row)
    assert valid_predictions==18
    saved_secondary=read(RESULTS/'temporal_response_analysis.json')['rows']
    for old in saved_secondary:
        actual=next(x for x in rows if x['file']==old['file'] and x['group']==old['group'])
        for key in ('same_candidate_ade','same_candidate_fde','early_error','middle_error','late_error','event_kind_match'):
            assert abs(old[key]-actual[key])<1e-6,(old['file'],key)
        for key in ('0.2_0.4','0.3_0.5'):
            assert abs(old['adequacy_'+key]-actual['adequacy_'+key]['rate'])<1e-9
            assert old['adequacy_'+key+'_rescued']==actual['adequacy_'+key]['rescued']
            assert old['adequacy_'+key+'_lost']==actual['adequacy_'+key]['lost']
    paired=[]
    for seed in (1,2,3):
        for policy in ('constrained','legacy','final'):
            control=all_goods[f'seed{seed}_zero_{policy}.npz']
            treated=all_goods[f'seed{seed}_mean_std_{policy}.npz']
            for group,mask in groups.items():
                for key in control:
                    paired.append(dict(seed=seed,policy=policy,group=group,threshold=key,n=int(mask.sum()),
                        control_count=int((mask&control[key]).sum()),treatment_count=int((mask&treated[key]).sum()),
                        rescued_vs_control=int((mask&~control[key]&treated[key]).sum()),
                        lost_vs_control=int((mask&control[key]&~treated[key]).sum())))
    with (RESULTS/'per_seed_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        perseed=list(csv.DictReader(stream))
    histories=[]
    for arm in ('zero','mean_std'):
        for seed in (1,2,3):
            path=RESULTS/'models'/arm/f'seed_{seed}'/'balanced_protected'
            with (path/'history.csv').open(encoding='utf-8-sig',newline='') as stream:
                history=list(csv.DictReader(stream))
            assert [int(x['epoch']) for x in history]==list(range(1,81))
            complete=read(path/'complete.json')
            histories.append(dict(arm=arm,seed=seed,constrained_epoch=complete['best_feasible_epoch'],legacy_epoch=complete['legacy_epoch'],
                first_epoch={k:float(history[0][k]) for k in ('train_loss','overall_minade20','overall_minfde20','change_minade20','change_minfde20')},
                final_epoch={k:float(history[-1][k]) for k in ('train_loss','overall_minade20','overall_minfde20','change_minade20','change_minfde20')}))
    # Input SHA check after computation: no saved production/scientific output edits.
    for path,sha in inputs.items():
        assert digest(path)==sha,path
    OUT.mkdir(exist_ok=True)
    result=dict(status='PASS_SAVED_OUTPUT_SUPPLEMENT_ONLY',GT_forward=False,optimizer=False,model_forward=False,
        valid_predictions=valid_predictions,explicit_IDs=1516,K=20,independent_min_metric_max_error=max_error,
        production_inputs_before_after_unchanged=len(inputs),registered_source_files_verified=229,checkpoint_SHAs_verified=18,
        old_result_manifest_count=18872,old_result_comparison='Completed worker independent_audit PASS; not repeated in supplemental review',
        formal_major=False,gates=summary['gates'],native=summary['native'],ALL_policy_group_metrics=summary['aggregate'],
        ALL_registered_CIs=summary['bootstrap'],per_seed_metrics=perseed,temporal_rows=rows,paired_adequacy=paired,histories=histories,
        limitations=['GT-assisted same-path joint oracle is not deployed selector.',
            'Formal change group and clear_event group are different labels.',
            'Conditional onset accuracy excludes missed/mistyped events; always read with coverage.',
            'Counts across seeds are repeated same IDs; do not add as unique rescued people.',
            'Only three paired seeds, repeated exploratory ETH DEV, correlated scene samples; no SOTA/generalization/causal claim.',
            'First epoch is not exact native CAL; history change is a within-run descriptor.',
            'Nominal parameter matching does not match function capacity or optimization geometry.'])
    (OUT/'REVIEW.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'input_hashes.json').write_text(json.dumps(inputs,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],predictions=18,IDs=1516,minmetric_maxerror=max_error,inputs=len(inputs),
                         native=[x for x in rows if x['file']=='native'],
                         clear_constrained=[x for x in rows if x['group']=='clear_event' and x['file'].endswith('_constrained.npz')],
                         histories=histories),ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
