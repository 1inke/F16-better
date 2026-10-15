"""Six preregistered exploratory statistics; no fit or real-data I/O."""
import numpy as np
from core_neighbors import require, check_tracks


def six_statistics(enrichment, supported, rank, groups, tracks, *, guard):
    """One row per known-P window. Fixed-bank whole-query-track bootstrap.

    Shared 2000 draws across six statistics; 99.1667% percentile intervals.
    Rank 'boundary'=5..8 and 'deep'=9..20, NOT the geometry near/far flag.
    Returns draw counts and bootstrap values for future saved-data audit.
    """
    e = np.asarray(enrichment, float)
    ok, rank, groups = np.asarray(supported), np.asarray(rank), np.asarray(groups)
    n = len(e)
    require(e.shape == (n,) and n > 0 and ok.dtype.kind == "b", "stat shapes/support bool")
    require(all(a.shape == (n,) for a in (ok, rank, groups)), "stat arrays")
    require(np.isin(rank, np.arange(1, 21)).all() and np.isin(groups, [0, 1, 2]).all(), "ranks/groups")
    require(np.isfinite(e[ok]).all() and callable(guard), "supported values/guard")
    tracks = check_tracks(tracks, n)
    units, inv = np.unique(tracks, return_inverse=True)
    cells, cell_masks = {}, []
    for population, pop in (("overall", np.ones(n, bool)), ("change", groups == 2)):
        for label, band in (("boundary", (rank >= 5) & (rank <= 8)), ("deep", rank >= 9)):
            whole, use = pop & band, pop & band & ok
            count, supported_count = int(whole.sum()), int(use.sum())
            track_count = len(np.unique(tracks[use]))
            fraction = supported_count / count if count else None
            passes = count > 0 and fraction >= .8 and supported_count >= 20 and track_count >= 10
            cells[population + "_" + label] = dict(windows=count, supported_windows=supported_count,
                supported_tracks=track_count, supported_fraction=fraction, unsupported_windows=count-supported_count,
                estimate=float(e[use].mean()) if supported_count else None, support_pass=bool(passes))
            cell_masks.append(use)
    # Aggregate by whole target track first; bootstrap never resamples windows independently.
    sums = np.array([np.bincount(inv, weights=np.where(m, e, 0), minlength=len(units)) for m in cell_masks])
    counts = np.array([np.bincount(inv, weights=m.astype(float), minlength=len(units)) for m in cell_masks])
    rng = np.random.default_rng(20261013)
    draw_counts = np.empty((2000, len(units)), dtype=np.int64)
    values = np.full((2000, 6), np.nan)
    for b in range(2000):
        guard()
        weight = np.bincount(rng.integers(0, len(units), size=len(units)), minlength=len(units))
        draw_counts[b] = weight
        denom = counts @ weight
        means = np.divide(sums @ weight, denom, out=np.full(4, np.nan), where=denom > 0)
        values[b] = [means[0], means[1], means[0]-means[1], means[2], means[3], means[2]-means[3]]
    records = []
    for p, offset in (("overall", 0), ("change", 3)):
        a, b = cells[p + "_boundary"], cells[p + "_deep"]
        for j, label in enumerate(("boundary", "deep", "boundary_minus_deep")):
            relevant = [a] if j == 0 else [b] if j == 1 else [a, b]
            estimate = relevant[0]["estimate"] if j < 2 else (a["estimate"]-b["estimate"] if a["estimate"] is not None and b["estimate"] is not None else None)
            finite = np.isfinite(values[:, offset+j])
            status = "EXPLORATORY_INTERVAL" if all(c["support_pass"] for c in relevant) else "INCONCLUSIVE_SUPPORT"
            if status == "EXPLORATORY_INTERVAL" and finite.sum() < 1900:
                status = "INCONCLUSIVE_BOOTSTRAP"
            ci = np.quantile(values[finite, offset+j], [.05/6/2, 1-.05/6/2]).tolist() if status == "EXPLORATORY_INTERVAL" else None
            records.append(dict(statistic=p+"_"+label, estimate=estimate, interval=ci, status=status,
                                valid_draws=int(finite.sum()), invalid_draws=int((~finite).sum()),
                                confidence=1-.05/6, training_bank_resampled=False))
    return dict(cells=cells, statistics=records, track_ids=units, draw_counts=draw_counts, values=values)


def json_safe(value):
    """Strict-JSON helper; absent denominators become null, never invented zero."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
