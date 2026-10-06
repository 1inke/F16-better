"""Alternate float64 saved-coordinate arithmetic audit, no model or optimizer."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / 'PULSE_RESPONSE_VISUAL_REVIEW_20261004/data/exact_arrays.npz'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    output = HERE / 'ARITHMETIC_AUDIT.json'
    assert not output.exists(), 'Do not overwrite an existing audit'
    result_path = HERE / 'RESULT.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    files = [Path(p) for p in result['input_hashes']] + [result_path, Path(__file__)]
    before = {str(p): digest(p) for p in files}
    assert all(before[p] == h for p, h in result['input_hashes'].items())
    audited = []
    with np.load(INPUT, allow_pickle=False) as arrays:
        clear = arrays['clear_event'].astype(bool)
        ids = arrays['sample_id']
        gt = arrays['future'].astype(np.float64)
        old = arrays['native'].astype(np.float64)
        old_errors = np.sqrt(np.square(old - gt[:, None]).sum(axis=-1))
        old_ade, old_fde = old_errors.mean(axis=-1), old_errors[:, :, -1]
        old_margins = np.maximum(old_ade / .2, old_fde / .4).min(axis=-1)
        assert len(set(ids.tolist())) == 1516 and clear.sum() == 469
        for seed in (1, 2, 3):
            new = arrays[f'learned{seed}'].astype(np.float64)
            # Alternate explicit squared-coordinate implementation.
            displacement = np.sqrt(np.square(new - old).sum(axis=-1))
            case_average = displacement.sum(axis=(1, 2)) / 240
            errors = np.sqrt(np.square(new - gt[:, None]).sum(axis=-1))
            ade, fde = errors.mean(axis=-1), errors[:, :, -1]
            margins = np.maximum(ade / .2, fde / .4).min(axis=-1)
            rescued = clear & (old_margins > 1) & (margins <= 1)
            lost = clear & (old_margins <= 1) & (margins > 1)
            near = (rescued | lost) & (abs(old_margins - 1) <= .05) & (abs(margins - 1) <= .05)
            target = next(r for r in result['rows'] if r['seed'] == seed and r['reference'] == 'native' and r['group'] == 'clear_event')
            checks = {
                'displacement_median_m': (np.median(case_average[clear]), target['mean_same_slot_displacement_m']['median']),
                'displacement_p95_m': (np.quantile(case_average[clear], .95), target['mean_same_slot_displacement_m']['p95']),
                'minADE_gain_median_m': (np.median(old_ade.min(-1)[clear] - ade.min(-1)[clear]), target['minADE_improvement_m']['median']),
                'minFDE_gain_median_m': (np.median(old_fde.min(-1)[clear] - fde.min(-1)[clear]), target['minFDE_improvement_m']['median']),
            }
            differences = {k: abs(float(a) - float(b)) for k, (a, b) in checks.items()}
            assert max(differences.values()) < 1e-6, differences
            assert int(near.sum()) == target['near_boundary_crosses']['0.05']
            for category, mask in [('rescued', rescued), ('lost', lost)]:
                stored = next(r for r in result['cross_cases'] if r['seed'] == seed and r['reference'] == 'native' and r['group'] == 'clear_event' and r['category'] == category)
                assert ids[mask].tolist() == stored['IDs']
                assert int(mask.sum()) == stored['n']
            audited.append(dict(seed=seed, arithmetic_abs_error=differences, rescued=int(rescued.sum()), lost=int(lost.sum()), near_5pct=int(near.sum()), native_adequate=int((clear & (old_margins <= 1)).sum()), learned_adequate=int((clear & (margins <= 1)).sum())))
    after = {str(p): digest(p) for p in files}
    assert before == after
    report = dict(status='PASS_ALTERNATE_FLOAT64_SAVED_ARITHMETIC', rows=audited, input_hashes=before, source_files_unchanged=True, scope='Seven direct input/analysis files; not an all-history SHA rerun, model replay, event-rule validation or human perceptual test.')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
