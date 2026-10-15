"""Label-based selector evaluation, NOT generated-path coverage or physics audit."""
import numpy as np
from math import comb
from core_matching import require
from input_contract import check_slots

REFERENCES = ("linear", "envelope", "fixed_random", "bottom")


def found_hit(slots, labels):
    labels = np.asarray(labels)
    require(labels.ndim == 2 and labels.shape[1] == 20 and np.isin(labels, [-1, 0, 1]).all(), "labels")
    check_slots(slots, len(labels))
    picked = np.take_along_axis(labels, slots, axis=1)
    return (picked == 1).any(1), (picked != 0).any(1), (labels == 1).any(1)


def describe(slots, labels, score=None):
    lower, upper, opportunity = found_hit(slots, labels)
    picked = np.take_along_axis(labels, slots, axis=1)
    chance = np.array([1.0 - (comb(20 - int(k), 4) if 20 - k >= 4 else 0) / comb(20, 4)
                       for k in (labels == 1).sum(1)])
    out = dict(windows=len(labels), P_windows=int(opportunity.sum()), hits=int(lower[opportunity].sum()),
               Hit4=float(lower[opportunity].mean()) if opportunity.any() else None,
               all_window_lower=float(lower.mean()), all_window_upper=float(upper.mean()),
               selected_U_fraction=float((picked == -1).mean()),
               uniform_random4_expectation=float(chance[opportunity].mean()) if opportunity.any() else None)
    if score is not None:
        require(score.shape == labels.shape and np.isfinite(score).all(), "scores")
        ranks = np.argsort(np.argsort(-score, axis=1, kind="stable"), axis=1) + 1
        best = np.where(labels == 1, ranks, 21).min(1)[opportunity]
        wins = ties = total = 0
        for row, y in zip(score, labels):
            delta = row[y == 1, None] - row[None, y == 0]
            wins += int((delta > 0).sum())
            ties += int((delta == 0).sum())
            total += delta.size
        out.update(best_P_rank_mean=float(best.mean()) if len(best) else None,
                   best_P_rank_median=float(np.median(best)) if len(best) else None,
                   P_greater_N_pair_fraction=wins / total if total else None,
                   P_N_tie_fraction=ties / total if total else None,
                   P_N_pair_denominator=int(total), pair_aggregation="pooled_pairs_descriptive_not_PU_AUC")
    return out


def paired_track_ci(track, mask, delta, seed):
    """Window-weighted estimand; resample whole tracks, not windows or folds."""
    track, mask, delta = np.asarray(track), np.asarray(mask), np.asarray(delta, float)
    require(mask.dtype == bool and mask.shape == track.shape == delta.shape, "bootstrap arrays")
    require(mask.any() and np.isfinite(delta).all(), "bootstrap empty/nonfinite")
    keys, inv = np.unique(track[mask], return_inverse=True)
    counts = np.bincount(inv)
    sums = np.bincount(inv, weights=delta[mask])
    rng = np.random.default_rng(seed)
    draws = np.empty(2000)
    for b in range(2000):
        sampled = rng.integers(0, len(keys), len(keys))
        draws[b] = sums[sampled].sum() / counts[sampled].sum()
    alpha = 0.05 / 4
    return dict(estimate=float(delta[mask].mean()),
                CI9875=np.quantile(draws, [alpha / 2, 1 - alpha / 2]),
                draws=draws, tracks=len(keys), windows=int(mask.sum()), seed=int(seed),
                includes_training_resampling=False)


def evaluate_screen(d, slots_by_arm):
    require(set(("rbf64",) + REFERENCES).issubset(slots_by_arm), "missing prespecified comparator")
    y, track = d["labels"], d["track"]
    hit = {name: found_hit(slots, y)[0] for name, slots in slots_by_arm.items()}
    opportunity = (y == 1).any(1)
    cis = {}
    for j, ref in enumerate(REFERENCES):
        delta = hit["rbf64"].astype(float) - hit[ref].astype(float)
        cis[ref] = paired_track_ci(track, opportunity, delta, 20261012 + j)
    folds = []
    for fold in range(3):
        known = (d["fold"] == fold) & opportunity
        change = known & (d["groups"] == 2)
        counts = {name: int(values[change].sum()) for name, values in hit.items()}
        support = int(known.sum()) >= 20 and len(np.unique(track[known])) >= 10
        change_ok = bool(change.any() and all(counts["rbf64"] >= counts[ref] for ref in REFERENCES))
        folds.append(dict(fold=fold, P_windows=int(known.sum()), P_tracks=len(np.unique(track[known])),
                          change_P_windows=int(change.sum()), change_hits=counts,
                          support_gate=support, change_gate=change_ok))
    gain = all(value["estimate"] >= 0.05 and value["CI9875"][0] > 0 for value in cis.values())
    gate = bool(gain and all(f["support_gate"] and f["change_gate"] for f in folds))
    return dict(comparisons=cis, folds=folds, gain_gate=bool(gain),
                statistical_screen_only=gate, source_and_numerical_audits_still_required=True,
                B_labels_not_available_from_this_run=True, B_started=False, S2_started=False,
                not_deployed_coverage=True)
