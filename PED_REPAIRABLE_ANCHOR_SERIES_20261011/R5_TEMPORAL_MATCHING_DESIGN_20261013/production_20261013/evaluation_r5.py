"""R5 saved selector evaluation. NOT generated trajectory quality or safety."""
import numpy as np
from core_r5 import require

REFERENCES = ("control", "pointwise", "linear", "envelope", "fixed_random", "bottom")


def hit_arrays(slots, labels):
    slots, y = np.asarray(slots), np.asarray(labels)
    require(y.ndim == 2 and y.shape[1] == 20 and len(y) > 0, "labels shape")
    require(np.isin(y, [-1, 0, 1]).all(), "label domain")
    require(slots.shape == (len(y), 4) and slots.dtype.kind in "iu", "slots shape/type")
    require(np.all((slots >= 0) & (slots < 20)), "slot range")
    require(np.all(np.diff(np.sort(slots, axis=1), axis=1) > 0), "duplicate slots")
    picked = np.take_along_axis(y, slots, axis=1)
    return (picked == 1).any(1), (picked != 0).any(1), (y == 1).any(1), picked


def describe(slots, labels, scores=None):
    low, upper, opportunity, picked = hit_arrays(slots, labels)
    miss = opportunity & ~low
    out = dict(windows=len(labels), knownP_windows=int(opportunity.sum()),
               hits=int(low[opportunity].sum()),
               Hit4=float(low[opportunity].mean()) if opportunity.any() else None,
               all_window_lower=float(low.mean()), all_window_unknown_upper=float(upper.mean()),
               selected_U_fraction=float((picked == -1).mean()),
               knownP_miss=int(miss.sum()),
               miss_all_N=int((miss & (picked == 0).all(1)).sum()),
               miss_contains_U=int((miss & (picked == -1).any(1)).sum()))
    if scores is not None:
        scores, y = np.asarray(scores), np.asarray(labels)
        require(scores.shape == y.shape and np.isfinite(scores).all(), "scores")
        order = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.argsort(order, axis=1) + 1
        best = np.where(y == 1, ranks, 21).min(1)[opportunity]
        wins = ties = total = 0
        for s, row in zip(scores, y):
            delta = s[row == 1, None] - s[None, row == 0]
            wins += int((delta > 0).sum())
            ties += int((delta == 0).sum())
            total += delta.size
        out.update(best_P_rank_mean=float(best.mean()) if len(best) else None,
                   best_P_rank_median=float(np.median(best)) if len(best) else None,
                   P_greater_N_fraction=wins / total if total else None,
                   PN_tie_fraction=ties / total if total else None, PN_pairs=total,
                   pair_fraction_is_pooled_descriptive_not_PU_AUC=True)
        out["descriptive_HitK"] = {str(k): float((np.take_along_axis(y, order[:, :k], 1) == 1)
                                                .any(1)[opportunity].mean()) if opportunity.any() else None
                                    for k in (1, 2, 4, 8)}
    return out


def paired_cis(tracks, mask, delta):
    """Six comparisons, same track draws. Window-weighted; no model refitting."""
    tracks, mask, delta = np.asarray(tracks), np.asarray(mask), np.asarray(delta, float)
    require(mask.dtype == bool and tracks.shape == mask.shape, "mask/track")
    require(delta.shape == (len(tracks), 6) and np.isfinite(delta).all() and mask.any(), "deltas")
    keys, inv = np.unique(tracks[mask], return_inverse=True)
    require(len(keys) >= 2, "at least two tracks")
    counts = np.bincount(inv)
    sums = np.column_stack([np.bincount(inv, weights=delta[mask, j]) for j in range(6)])
    rng, draws = np.random.default_rng(20261013), np.empty((2000, 6))
    for b in range(2000):
        chosen = rng.integers(0, len(keys), len(keys))
        draws[b] = sums[chosen].sum(0) / counts[chosen].sum()
    return dict(estimate=delta[mask].mean(0), CI991667=np.quantile(draws, [.05/12, 1-.05/12], axis=0).T,
                draws=draws, windows=int(mask.sum()), tracks=len(keys), seed=20261013,
                includes_training_resampling=False)


def evaluate_screen(data, slots_by_arm):
    """Requires prior pinned input/source validation; returns ONLY statistics."""
    y, tracks = np.asarray(data["labels"]), np.asarray(data["track"])
    fold, groups = np.asarray(data["fold"]), np.asarray(data["groups"])
    require(y.shape == (384, 20) and tracks.shape == fold.shape == groups.shape == (384,), "R3 population")
    require(np.isin(fold, [0, 1, 2]).all() and np.isin(groups, [0, 1, 2]).all(), "fold/group")
    for track in np.unique(tracks):
        require(len(np.unique(fold[tracks == track])) == 1, "track crosses folds")
    require(set(("temporal",) + REFERENCES).issubset(slots_by_arm), "missing comparator")
    hit = {name: hit_arrays(slots_by_arm[name], y)[0] for name in ("temporal",) + REFERENCES}
    opportunity = (y == 1).any(1)
    require(opportunity.sum() == 263, "knownP denominator changed")
    delta = np.column_stack([hit["temporal"].astype(float) - hit[ref] for ref in REFERENCES])
    ci = paired_cis(tracks, opportunity, delta)
    comparisons = {}
    for j, ref in enumerate(REFERENCES):
        comparisons[ref] = dict(estimate=float(ci["estimate"][j]), CI991667=ci["CI991667"][j].tolist(),
                               gained=int((opportunity & hit["temporal"] & ~hit[ref]).sum()),
                               lost=int((opportunity & ~hit["temporal"] & hit[ref]).sum()))
    rows = []
    for f in range(3):
        m = (fold == f) & opportunity
        change = m & (groups == 2)
        counts = {name: int(value[change].sum()) for name, value in hit.items()}
        rows.append(dict(fold=f, P_windows=int(m.sum()), P_tracks=len(np.unique(tracks[m])),
                         support=bool(m.sum() >= 20 and len(np.unique(tracks[m])) >= 10),
                         change_P=int(change.sum()), change_hits=counts,
                         change_gate=bool(change.any() and all(counts["temporal"] >= counts[r] for r in REFERENCES))))
    gain = bool(np.all(ci["estimate"] >= .05) and np.all(ci["CI991667"][:, 0] > 0))
    return dict(comparisons=comparisons, folds=rows, gain_gate=gain,
                statistical_screen_only=bool(gain and all(r["support"] and r["change_gate"] for r in rows)),
                audits_still_required=True, B_started=False, S2_started=False,
                not_generated_coverage=True), ci["draws"]
