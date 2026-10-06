"""Alternate case-wise cue audit and descriptive onset/track supplements."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / 'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    assert not (HERE / 'AUDIT_AND_SUPPLEMENT.json').exists()
    result = json.loads((HERE / 'RESULT.json').read_text(encoding='utf-8'))
    cases = json.loads((HERE / 'CLASSIFICATION.json').read_text(encoding='utf-8'))['cases']
    masks = npz(HERE / 'MASKS.npz')
    direct = {**result['source_hashes']}
    direct.update({str(p): sha(p) for p in [HERE / 'RESULT.json', HERE / 'CLASSIFICATION.json', HERE / 'MASKS.npz', Path(__file__)]})
    assert all(sha(p) == h for p, h in direct.items())
    features = npz(BASE / 'features/dev.npz')
    labels = npz(BASE / 'labels/dev.npz')
    meta = npz(BASE / 'metadata.npz')
    risk = npz(BASE / 'dev_risk_readouts.npz')
    visual = npz(ROOT / 'PULSE_RESPONSE_VISUAL_REVIEW_20261004/data/exact_arrays.npz')
    ids = masks['sample_id']
    for value in (features, labels, risk, visual):
        assert np.array_equal(value['sample_id'], ids)
    lookup = {int(k): i for i, k in enumerate(meta['sample_id'])}
    mi = np.array([lookup[int(i)] for i in ids])
    hand = []
    # Per-case reference implementation, independent of analyze.py functions.
    for i, c in enumerate(cases):
        assert int(ids[i]) == c['sample_id']
        r = features['R2'][i].astype(float)
        p = r[:16].reshape(8, 2)
        vs = [(p[t + 2] - p[t]) / .8 for t in range(6)]
        ref = (vs[0] + vs[1]) / 2
        refspeed = math.hypot(*ref)
        recent = vs[-2:]
        sp = [math.hypot(*v) for v in recent]
        deg = [abs(math.degrees(math.atan2(ref[0] * v[1] - ref[1] * v[0], ref[0] * v[0] + ref[1] * v[1]))) for v in recent]
        flags = dict(target_turn=refspeed >= .2 and all(s >= .2 and a >= 15 for s, a in zip(sp, deg)),
                     target_speed=all(abs(s - refspeed) >= max(.15, .25 * refspeed) for s in sp),
                     target_start=refspeed <= .15 and all(s >= .3 for s in sp),
                     target_stop=refspeed >= .3 and all(s <= .15 for s in sp))
        relative = r[49:177].reshape(8, 8, 2)
        valid = r[177:185].astype(bool)
        near = False
        for k in range(8):
            if not valid[k]:
                continue
            v = (relative[k, -1] - relative[k, -3]) / .8
            pos = relative[k, -1]
            dist = math.hypot(*pos)
            dot = float(pos @ v)
            tca = min(4.8, max(0., -dot / max(float(v @ v), 1e-6)))
            dca = math.hypot(*(pos + tca * v))
            closing = -dot / max(dist, 1e-6)
            near |= closing >= .1 and tca > 0 and dca <= 1.
        flags['neighbor_approach'] = bool(near)
        expected = [k for k, value in flags.items() if value]
        assert expected == c['hand_cues'], (c['sample_id'], expected, c['hand_cues'])
        hand.append(bool(expected))
        thresholds = {k: row['threshold'] for row in result['calibration'] for k in [row['reader']]}
        low = not expected and all(risk[f'final_logistic_full_{k}'][i] <= thresholds[k]['0.5'] for k in cases[0]['reader_scores'])
        high = bool(expected) or any(risk[f'final_logistic_full_{k}'][i] > thresholds[k]['0.8'] for k in cases[0]['reader_scores'])
        category = 'low_detected_cue' if low else 'detected_cue' if high else 'inconclusive'
        assert category == c['category']
    hand = np.asarray(hand)
    clear = masks['GT_clear_event']
    low = masks['low_detected_cue']
    high = masks['detected_cue']
    covered = masks['native_covered']
    selected = masks['native_selected_good']
    primary = dict(detected_cue=high, low_detected_cue=low, inconclusive=masks['inconclusive'])
    populations = dict(all=np.ones(len(ids), bool), clear_event=clear, native_generation_failure=~covered,
                       clear_event_generation_failure=clear & ~covered, native_argmax_failure=~selected,
                       clear_event_argmax_failure=clear & ~selected, no_clear_event=(labels['y'] == 0) & ~labels['uncertain'],
                       boundary_uncertain=labels['uncertain'])
    for name, pop in populations.items():
        assert sum(int((pop & g).sum()) for g in primary.values()) == int(pop.sum())
        for group, g in primary.items():
            record = result['counts'][name][group]
            assert record['n'] == int((pop & g).sum()) and record['denominator'] == int(pop.sum())
    paths = visual['native'].astype(float)
    errors = np.linalg.norm(paths - visual['future'][:, None].astype(float), axis=-1)
    a, f = errors.mean(-1), errors[:, :, -1]
    assert np.array_equal(((a <= .2) & (f <= .4)).any(-1), covered)
    assert np.array_equal(((a <= .2) & (f <= .4))[np.arange(len(ids)), visual['logits'].argmax(1)], selected)
    metric_checks = []
    chosen = dict(full=np.ones(len(ids), bool), low_cue_clear_event=clear & low, remaining_all=~(clear & low),
                  remaining_clear_event=clear & ~low, detected_cue_clear_event=clear & high,
                  inconclusive_clear_event=clear & masks['inconclusive'])
    for name, mask in chosen.items():
        for metric, arr in [('minADE', a.min(-1)), ('minFDE', f.min(-1))]:
            delta = abs(float(arr[mask].mean()) - result['filtered_metrics'][name][metric])
            assert delta < 1e-10
            metric_checks.append(dict(group=name, metric=metric, abs_error=delta))
    branches = {
        'explicit_rule_cue': clear & hand,
        'no_rule_cue_but_strong_reader': clear & ~hand & high,
        'no_rule_and_low_readers': clear & low,
        'no_rule_mixed_readers': clear & masks['inconclusive'],
    }
    assert sum(int(m.sum()) for m in branches.values()) == 469
    hierarchy = {}
    for name, mask in branches.items():
        onset = labels['onset_step'][mask]
        kinds = labels['kind'][mask]
        hierarchy[name] = dict(n=int(mask.sum()), percent_of_469=float(mask.sum() / 469 * 100),
                               generation_fail=int((mask & ~covered).sum()),
                               onset_counts={str(int(t)): int((onset == t).sum()) for t in np.unique(onset)},
                               onset_median_steps=float(np.median(onset)) if mask.any() else None,
                               kind_counts={str(int(k)): int((kinds == k).sum()) for k in np.unique(kinds)})
    broad = clear & hand
    broad_retained = dict(n=int(broad.sum()), no_good=int((broad & ~covered).sum()),
                         good_missed=int((broad & covered & ~selected).sum()), selected_good=int((broad & selected).sum()),
                         minADE=float(a.min(-1)[broad].mean()), minFDE=float(f.min(-1)[broad].mean()),
                         note='Stricter research focus retaining only hand-rule cue cases, not proof predictable.')
    # Focal trend + E1-only sensitivity avoids depending on neighbor mask/geometry.
    target_hand = np.array([any(k.startswith('target_') for k in c['hand_cues']) for c in cases])
    target_rows = []
    e1p = risk['final_logistic_full_E1']
    e1threshold = next(row['threshold'] for row in result['calibration'] if row['reader'] == 'E1')
    for q in (.2, .5, .8):
        no = ~target_hand & (e1p <= e1threshold[str(q)])
        target_rows.append(dict(reader_negative_quantile=q, low_clear=int((no & clear).sum()), clear_n=469,
                                low_clear_fail=int((no & clear & ~covered).sum()), clear_fail_n=377,
                                note='Focal-history/E1 readout proxy only; different definition, not corrected irreducible fraction.'))
    low_ids = ids[clear & low]
    low_meta = []
    tracks = set()
    for i in np.flatnonzero(clear & low):
        j = mi[i]
        ambiguous = bool(meta['ambiguous_track_time'][j])
        track = (str(meta['sequence'][j]), int(meta['pedestrian_id'][j]))
        if not ambiguous:
            tracks.add(track)
        low_meta.append(dict(sample_id=int(ids[i]), sequence=track[0], source_pedestrian=track[1],
                             start_frame=int(meta['start_frame'][j]), ambiguous=ambiguous,
                             onset_step=int(labels['onset_step'][i]),
                             source_onset_frame=int(meta['start_frame'][j] + 7 + labels['onset_step'][i]) if not ambiguous else None))
    assert all(sha(p) == h for p, h in direct.items())
    audit = dict(status='PASS_CASEWISE_CUE_COUNTS_AND_SAVED_ARITHMETIC', direct_input_hashes=direct,
                 direct_files_verified=len(direct), n=1516, event_n=469,
                 cue_rules_independently_casewise_recomputed=True, category_counts_exact=True,
                 saved_native_metric_max_error=max(row['abs_error'] for row in metric_checks), metric_checks=metric_checks,
                 hierarchy=hierarchy, broad_hand_cue_remainder=broad_retained,
                 focal_E1_only_sensitivity=target_rows, low_cue_clear_IDs=low_ids.tolist(),
                 low_cue_verified_source_tracks=len(tracks), low_cue_source_metadata=low_meta,
                 high_cue_rate_among_no_future_event=float((high & populations['no_clear_event']).sum() / populations['no_clear_event'].sum()),
                 confirmed_unpredictability=False,
                 limitations=['Rules/probes have false alerts and missed cues; high-score need not identify actual future event.',
                              'All negative controls quantify one realized future, not latent intention/risk ground truth.',
                              'Source tracks/windows correlated; unique track count does not make their events independent.',
                              'No independent event-label validation/human annotation/model forward/optimizer/full-history hash rerun.'])
    (HERE / 'AUDIT_AND_SUPPLEMENT.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ('status', 'direct_files_verified', 'saved_native_metric_max_error', 'hierarchy', 'broad_hand_cue_remainder', 'focal_E1_only_sensitivity', 'low_cue_verified_source_tracks', 'high_cue_rate_among_no_future_event')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
