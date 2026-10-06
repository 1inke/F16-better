"""Saved-data cue stratification, CPU arithmetic only; no model or optimizer."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003'
READERS = ('R2', 'E1', 'E2_motion', 'E3')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in (z.files if keys is None else keys)}


def write(name, obj):
    path = HERE / name
    assert not path.exists(), 'Preserve existing output: ' + name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def align(ids, value):
    lookup = {int(k): i for i, k in enumerate(value['sample_id'])}
    assert len(lookup) == len(value['sample_id'])
    index = np.array([lookup[int(i)] for i in ids])
    return {k: v[index] for k, v in value.items()}


def hand_cues(r2, scale=1.):
    focal = r2[:, :16].reshape(-1, 8, 2).astype(float)
    relative = r2[:, 49:177].reshape(-1, 8, 8, 2).astype(float)
    valid = r2[:, 177:185].astype(bool)
    smooth_v = (focal[:, 2:] - focal[:, :-2]) / .8
    reference = smooth_v[:, :2].mean(1)
    recent = smooth_v[:, -2:]
    speed = np.linalg.norm(recent, axis=-1)
    reference_speed = np.linalg.norm(reference, axis=-1)
    cross = reference[:, None, 0] * recent[:, :, 1] - reference[:, None, 1] * recent[:, :, 0]
    dot = (reference[:, None] * recent).sum(-1)
    angle = np.abs(np.degrees(np.arctan2(cross, dot)))
    turn = ((angle >= 15 * scale) & (speed >= .2) & (reference_speed[:, None] >= .2)).all(1)
    speed_change = (np.abs(speed - reference_speed[:, None]) >= np.maximum(.15 * scale, .25 * scale * reference_speed[:, None])).all(1)
    start = (reference_speed <= .15) & (speed >= .3 * scale).all(1)
    stop = (reference_speed >= .3) & (speed <= .15 / scale).all(1)
    pv = (relative[:, :, -1] - relative[:, :, -3]) / .8
    position = relative[:, :, -1]
    distance = np.linalg.norm(position, axis=-1)
    vv = np.square(pv).sum(-1)
    tca = np.clip(-(position * pv).sum(-1) / np.maximum(vv, 1e-6), 0, 4.8)
    dca = np.linalg.norm(position + tca[:, :, None] * pv, axis=-1)
    closing = -(position * pv).sum(-1) / np.maximum(distance, 1e-6)
    conflict = valid & (closing >= .1 * scale) & (tca > 0) & (tca <= 4.8) & (dca <= 1. / scale)
    social = conflict.any(1)
    expected = np.stack([distance, closing, tca, dca], axis=-1) * valid[:, :, None]
    delta = float(np.max(np.abs(expected - r2[:, 185:217].reshape(-1, 8, 4))))
    assert delta < 2e-3, delta
    return dict(target_turn=turn, target_speed=speed_change, target_start=start, target_stop=stop,
                neighbor_approach=social, any=turn | speed_change | start | stop | social,
                reference_speed=reference_speed, recent_heading_deg=angle.max(-1),
                recent_speed_delta=np.abs(speed - reference_speed[:, None]).max(-1),
                engineer_reconstruction_error=delta)


def score_stats(path, gt, logits=None, at=.2, ft=.4):
    d = np.sqrt(np.square(path.astype(float) - gt[:, None].astype(float)).sum(-1))
    ade, fde = d.mean(-1), d[:, :, -1]
    good = (ade <= at) & (fde <= ft)
    selected = None if logits is None else logits.argmax(1)
    return dict(minADE=ade.min(-1), minFDE=fde.min(-1),
                joint=(ade + .5 * fde).min(-1), covered=good.any(-1),
                selected_good=good[np.arange(len(d)), selected] if selected is not None else np.zeros(len(d), bool),
                selected_ADE=ade[np.arange(len(d)), selected] if selected is not None else np.zeros(len(d)),
                selected_FDE=fde[np.arange(len(d)), selected] if selected is not None else np.zeros(len(d)))


def summarize(mask, met):
    n = int(mask.sum())
    if not n:
        return dict(n=0)
    return dict(n=n, minADE=float(met['minADE'][mask].mean()), minFDE=float(met['minFDE'][mask].mean()),
                samepath_good=int(met['covered'][mask].sum()), samepath_rate=float(met['covered'][mask].mean()),
                selected_good=int(met['selected_good'][mask].sum()),
                selected_ADE=float(met['selected_ADE'][mask].mean()), selected_FDE=float(met['selected_FDE'][mask].mean()))


def countrow(mask, groups):
    return {name: dict(n=int((mask & group).sum()), denominator=int(mask.sum()),
                       percent=float((mask & group).sum() / mask.sum() * 100) if mask.any() else None)
            for name, group in groups.items()}


def dynamics(path):
    p = np.concatenate([np.zeros_like(path[:, :, :1]), path], 2).astype(float)
    v = np.diff(p, axis=2) / .4
    a = np.diff(v, axis=2) / .4
    j = np.diff(a, axis=2) / .4
    return dict(speedmean=np.linalg.norm(v, axis=-1).mean((1, 2)),
                accelmean=np.linalg.norm(a, axis=-1).mean((1, 2)),
                jerkmean=np.linalg.norm(j, axis=-1).mean((1, 2)))


def main():
    began = time.monotonic()
    assert not (HERE / 'RESULT.json').exists(), 'New analysis only'
    final = [BASE / f'jobs/final_logistic_full_{name}.predictions.npz' for name in READERS]
    final_records = [BASE / f'jobs/final_logistic_full_{name}.done.json' for name in READERS]
    oof_names = ['OOF_logistic_full_R2', 'OOF_logistic_full_E3'] + [f'OOF_mlp_full_{name}_seed{s}' for name in ('R1', 'R2', 'E3') for s in (1, 2, 3)]
    oof_files = [BASE / (name + '.npz') for name in oof_names]
    inputs = [HERE / 'DESIGN.md', Path(__file__), BASE / 'features/dev.npz', BASE / 'features/train.npz',
              BASE / 'labels/dev.npz', BASE / 'labels/train.npz', BASE / 'metadata.npz', BASE / 'dev_risk_readouts.npz',
              BASE / 'features/dev_audit.json', BASE / 'features/train_audit.json', BASE / 'binary_probes.json',
              ROOT / 'PULSE_RESPONSE_VISUAL_REVIEW_20261004/data/exact_arrays.npz'] + final + final_records + oof_files
    hashes = {str(p): sha(p) for p in inputs}
    for split in ('train', 'dev'):
        recorded = json.loads((BASE / f'features/{split}_audit.json').read_text())['sha256']
        assert sha(BASE / f'features/{split}.npz') == recorded
    for path in final_records:
        rec = json.loads(path.read_text())
        assert rec['status'] == 'SUCCESS'
        assert sha(BASE / rec['predictions']) == rec['hashes'][rec['predictions']]
    feats = load(BASE / 'features/dev.npz', ('sample_id', 'R2'))
    ids = feats['sample_id']
    lab = align(ids, load(BASE / 'labels/dev.npz'))
    meta = align(ids, load(BASE / 'metadata.npz'))
    risk = align(ids, load(BASE / 'dev_risk_readouts.npz'))
    arrays = align(ids, load(ROOT / 'PULSE_RESPONSE_VISUAL_REVIEW_20261004/data/exact_arrays.npz'))
    assert np.array_equal(risk['y'], lab['y'])
    assert np.array_equal(arrays['clear_event'], (lab['y'] == 1) & ~lab['uncertain'])
    assert np.isfinite(feats['R2']).all()
    trainlab = load(BASE / 'labels/train.npz')
    all_train_ids = trainlab['sample_id']
    thresholds, probabilities = {}, []
    calibration = []
    for name, path in zip(READERS, final):
        saved = load(path)
        labels = align(saved['sample_id'], trainlab)
        assert np.array_equal(labels['y'], saved['y'])
        assert np.all(np.isin(saved['sample_id'], all_train_ids[25912:]))
        negative = (saved['y'] == 0) & ~labels['uncertain']
        thresholds[name] = {str(q): float(np.quantile(saved['p'][negative], q)) for q in (.2, .5, .8)}
        p = risk[f'final_logistic_full_{name}']
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        probabilities.append(p)
        calibration.append(dict(reader=name, n_negative=int(negative.sum()), n_CAL=int(len(saved['y'])), threshold=thresholds[name],
                                actual_negative_exceed_q80=float((saved['p'][negative] > thresholds[name]['0.8']).mean())))
    P = np.stack(probabilities, axis=-1)
    cues = hand_cues(feats['R2'])
    low = ~cues['any'] & (P <= np.array([thresholds[k]['0.5'] for k in READERS])).all(-1)
    high_readout = (P > np.array([thresholds[k]['0.8'] for k in READERS])).any(-1)
    high = cues['any'] | high_readout
    uncertain = ~(low | high)
    assert not (low & high).any() and (low | high | uncertain).all()
    groups = dict(detected_cue=high, low_detected_cue=low, inconclusive=uncertain)
    clear = (lab['y'] == 1) & ~lab['uncertain']
    boundary = lab['uncertain']
    native = score_stats(arrays['native'], arrays['future'], arrays['logits'])
    native_loose = score_stats(arrays['native'], arrays['future'], arrays['logits'], .3, .5)
    assert len(ids) == 1516 and clear.sum() == 469 and native['covered'].sum() == 733
    denominators = dict(all=np.ones(len(ids), bool), clear_event=clear,
                        native_generation_failure=~native['covered'], clear_event_generation_failure=clear & ~native['covered'],
                        native_argmax_failure=~native['selected_good'], clear_event_argmax_failure=clear & ~native['selected_good'],
                        no_clear_event=(lab['y'] == 0) & ~boundary, boundary_uncertain=boundary)
    table = {name: countrow(mask, groups) for name, mask in denominators.items()}
    mask_exclude = clear & low
    masks = dict(full=np.ones(len(ids), bool), low_cue_clear_event=mask_exclude,
                 remaining_all=~mask_exclude, remaining_clear_event=clear & ~low,
                 detected_cue_clear_event=clear & high, inconclusive_clear_event=clear & uncertain)
    filtered = {name: summarize(mask, native) for name, mask in masks.items()}
    filtered_loose = {name: summarize(mask, native_loose) for name, mask in masks.items()}
    remaining = clear & ~low
    problem = {
        'good_candidate_and_native_selected': remaining & native['selected_good'],
        'good_candidate_but_native_missed': remaining & native['covered'] & ~native['selected_good'],
        'no_good_candidate': remaining & ~native['covered'],
    }
    assert sum(int(m.sum()) for m in problem.values()) == int(remaining.sum())
    remainder_problem = {key: dict(n=int(mask.sum()), percent_of_remaining_clear=float(mask.sum() / remaining.sum() * 100) if remaining.any() else None,
                                   IDs=ids[mask].tolist()) for key, mask in problem.items()}
    sensitivity = []
    for scale in (.75, 1., 1.25):
        cc = hand_cues(feats['R2'], scale)
        for q in (.2, .5, .8):
            lm = ~cc['any'] & (P <= np.array([thresholds[k][str(q)] for k in READERS])).all(-1)
            sensitivity.append(dict(hand_scale=scale, reader_negative_quantile=q, low_in_clear=int((lm & clear).sum()),
                                    clear_denominator=int(clear.sum()), percent_clear=float((lm & clear).mean() / clear.mean() * 100),
                                    low_in_clear_generation_fail=int((lm & clear & ~native['covered']).sum()),
                                    denominator_clear_generation_fail=int((clear & ~native['covered']).sum()),
                                    low_in_all=int(lm.sum()), hand_no_cue_in_clear=int((~cc['any'] & clear).sum())))
    cue_details = {}
    for name, mask in cues.items():
        if isinstance(mask, np.ndarray) and mask.dtype == bool:
            cue_details[name] = dict(all=int(mask.sum()), clear=int((mask & clear).sum()),
                                     clear_fail=int((mask & clear & ~native['covered']).sum()))
    scene_rows = []
    for scene in np.unique(meta['scene_family']):
        sm = meta['scene_family'] == scene
        scene_rows.append(dict(scene=str(scene), all=int(sm.sum()), clear=int((sm & clear).sum()),
                               low_cue_clear=int((sm & clear & low).sum()), clear_fail=int((sm & clear & ~native['covered']).sum()),
                               strata=countrow(sm & clear, groups)))
    pulse = []
    ndyn = dynamics(arrays['native'])
    for seed in (1, 2, 3):
        treatment = score_stats(arrays[f'learned{seed}'], arrays['future'], arrays['logits'])
        fixed = score_stats(arrays[f'fixed{seed}'], arrays['future'], arrays['logits'])
        tdyn = dynamics(arrays[f'learned{seed}'])
        for group, mask in masks.items():
            n = int(mask.sum())
            row = dict(seed=seed, group=group, native=summarize(mask, native), learned=summarize(mask, treatment), fixed=summarize(mask, fixed),
                       rescued=int((mask & ~native['covered'] & treatment['covered']).sum()), lost=int((mask & native['covered'] & ~treatment['covered']).sum()))
            row['mean_allK_dynamics_ratio_minus1'] = {key: float(tdyn[key][mask].mean() / ndyn[key][mask].mean() - 1) if n and ndyn[key][mask].mean() > 0 else None for key in ndyn}
            pulse.append(row)
    # Additional nonlinear reader check uses FIT scene-held-out saved scores only.
    fit_features = load(BASE / 'features/train.npz', ('sample_id', 'R2'))
    fit_oof = [load(path) for path in oof_files]
    fit_ids = np.sort(fit_oof[0]['sample_id'])
    flab = align(fit_ids, trainlab)
    fx = align(fit_ids, fit_features)
    assert np.isin(fit_ids, all_train_ids[:25912]).all() and not flab['uncertain'].any()
    fit_cues = hand_cues(fx['R2'])
    fscores = np.stack([align(fit_ids, value)['p'] for value in fit_oof], -1)
    fy = flab['y'] == 1
    fit_sensitivity = []
    for q in (.2, .5, .8):
        ref = np.quantile(fscores[~fy], q, axis=0)
        fm = ~fit_cues['any'] & (fscores <= ref).all(-1)
        fit_sensitivity.append(dict(reader_negative_quantile=q, n=len(fit_ids), positive=int(fy.sum()),
                                    all_low=int(fm.sum()), positive_low=int((fy & fm).sum()),
                                    percent_positive_low=float((fy & fm).sum() / fy.sum() * 100),
                                    readers=oof_names, threshold=ref.tolist(),
                                    note='FIT OOF nonlinear-capacity sensitivity, not DEV prevalence or independent generalization.'))
    write('CLASSIFICATION.json', dict(status='DESCRIPTIVE_PROXY_NOT_IRREDUCIBILITY', reader_names=list(READERS),
          cases=[dict(sample_id=int(ids[i]), scene_family=str(meta['scene_family'][i]), GT_clear_event=bool(clear[i]),
                      onset_step=int(lab['onset_step'][i]) if clear[i] else None,
                      native_good_candidate=bool(native['covered'][i]), native_selected_good=bool(native['selected_good'][i]),
                      category='low_detected_cue' if low[i] else 'detected_cue' if high[i] else 'inconclusive',
                      hand_cues=[name for name in ('target_turn', 'target_speed', 'target_start', 'target_stop', 'neighbor_approach') if cues[name][i]],
                      reader_scores={key: float(P[i, j]) for j, key in enumerate(READERS)}) for i in range(len(ids))]))
    np.savez_compressed(HERE / 'MASKS.npz', sample_id=ids, GT_clear_event=clear, low_detected_cue=low,
                        detected_cue=high, inconclusive=uncertain, research_exclude=mask_exclude,
                        native_covered=native['covered'], native_selected_good=native['selected_good'],
                        native_minADE=native['minADE'], native_minFDE=native['minFDE'],
                        reader_scores=P)
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    # Weighted means must reconcile full with excluded+retained.
    arithmetic = {}
    for metric in ('minADE', 'minFDE'):
        total = float(native[metric].sum())
        split_total = float(native[metric][mask_exclude].sum() + native[metric][~mask_exclude].sum())
        arithmetic[metric] = abs(total - split_total)
        assert arithmetic[metric] < 1e-9
    result = dict(status='PASS_SAVED_DATA_DESCRIPTIVE_STRATIFICATION', elapsed_seconds=time.monotonic() - began,
                  n=1516, clear_event=469, verified_direct_inputs=len(hashes), source_hashes=hashes,
                  calibration=calibration, counts=table, hand_cue_counts=cue_details, hand_reconstruction_max=cues['engineer_reconstruction_error'],
                  filtered_metrics=filtered, filtered_metrics_loose=filtered_loose, remaining_clear_problem=remainder_problem,
                  sensitivity=sensitivity, scenes=scene_rows, pulse_comparison=pulse, FIT_OOF_nonlinear_sensitivity=fit_sensitivity,
                  arithmetic_reconciliation_max=arithmetic,
                  scientific_exclusion=False, confirmed_unpredictable_fraction=None,
                  limitations=['Low probe/rule response is not proof of absent cues or irreducible unpredictability.',
                  'Nonlinear FIT OOF sensitivity is a different evaluation population; no final MLP DEV predictions exist, no MLP replay done.',
                  'Cue-risk detection is not correct type/onset/direction/full-path prediction; retained cases are not proven predictable.',
                  'CAL reference quantiles and hand rules are operational exploratory choices; all planned sensitivity rows reported.',
                  'Future-completeness qualification of existing neighbor cache limits claims about deployment observable inputs.',
                  'Only original train-source subsets, correlated windows, three scene families, previously seen F16 data, repeated DEV; no reliable population CI/p/SOTA.',
                  'No human blind responses or body-pose/gaze/map information collected. No case deleted, no official mean replaced.',
                  'No native/probe training or model forward, no GPU/heldtest/rawnewdata/PID query; direct input SHA only, not all-history SHA rerun.'])
    write('RESULT.json', result)
    print(json.dumps(dict(status=result['status'], counts=table, filtered=filtered, remaining=remainder_problem, sensitivity=sensitivity,
                          fit_sensitivity=fit_sensitivity), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
