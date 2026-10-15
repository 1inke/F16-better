"""R4 pure NumPy geometry. No project reads, model fits, or launch on import.

P=1, N=0, U=-1. Labels are deliberately absent from retrieval interfaces.
The caller must supply the production resource guard; tests use a no-op.
"""
from dataclasses import dataclass
import numpy as np

K, SLOTS, DIM = 5, 20, 200


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_tracks(tracks, n):
    a = np.asarray(tracks)
    require(a.shape == (n,) and a.dtype.kind in "iuUS", "track must be integer/string vector")
    if a.dtype.kind in "US":
        require(np.all(a != ""), "missing track")
    return a


@dataclass(frozen=True)
class Bank:
    z: np.ndarray
    ids: np.ndarray
    slots: np.ndarray
    tracks: np.ndarray
    window: np.ndarray


def make_bank(features, ids, tracks):
    z, ids = np.asarray(features, dtype=np.float64), np.asarray(ids)
    require(z.ndim == 3 and z.shape[1:] == (SLOTS, DIM), "expected windows x20x200")
    n = len(z)
    require(n > 0 and np.isfinite(z).all(), "empty/nonfinite features")
    require(ids.shape == (n,) and ids.dtype.kind in "iu" and len(np.unique(ids)) == n, "unique integer IDs")
    t = check_tracks(tracks, n)
    return Bank(z.reshape(-1, DIM).copy(), np.repeat(ids, SLOTS),
                np.tile(np.arange(SLOTS), n), np.repeat(t, SLOTS),
                np.repeat(np.arange(n), SLOTS))


def retrieve(query_features, query_tracks, bank, *, guard, chunk=256):
    """All candidates, then one representative per DISTINCT train track.

    Exact RMS distance; ties use sample ID then physical slot. Includes U.
    Returns bank indices and distances. No labels/scores/groups/GT argument.
    Temporary geometry memory is bounded by chunk x200, not all queries.
    """
    q = np.asarray(query_features, dtype=np.float64)
    require(q.ndim == 2 and q.shape[1] == DIM and np.isfinite(q).all(), "query dimensions/finite")
    qt = check_tracks(query_tracks, len(q))
    require(callable(guard) and isinstance(chunk, int) and 1 <= chunk <= 256, "guard/chunk")
    require(qt.dtype.kind in "iu" if bank.tracks.dtype.kind in "iu" else qt.dtype.kind in "US", "track type mismatch")
    indices = np.empty((len(q), K), dtype=np.int64)
    distances = np.empty((len(q), K), dtype=np.float64)
    for row, (x, track) in enumerate(zip(q, qt)):
        guard()
        require(len(np.unique(bank.tracks[bank.tracks != track])) >= K, "STOP_INPUT_SUPPORT: fewer than five other tracks")
        d2 = np.empty(len(bank.z), dtype=np.float64)
        for start in range(0, len(bank.z), chunk):
            guard()
            delta = bank.z[start:start + chunk] - x
            d2[start:start + chunk] = np.einsum("ij,ij->i", delta, delta, optimize=False) / DIM
        require(np.isfinite(d2).all(), "nonfinite squared distance")
        order = np.lexsort((bank.slots, bank.ids, d2))
        seen, chosen = set(), []
        for j in order:
            key = bank.tracks[j].item()
            if key == track or key in seen:
                continue
            seen.add(key)
            chosen.append(j)
            if len(chosen) == K:
                break
        indices[row] = chosen
        distances[row] = np.sqrt(d2[chosen])
    return indices, distances


def distance_components(query_features, bank, indices):
    """Two additive contributions to registered d^2, NOT new search metrics."""
    q, ix = np.asarray(query_features, float), np.asarray(indices)
    require(ix.shape == (len(q), K), "neighbor shape")
    require(ix.dtype.kind in "iu" and ((ix >= 0) & (ix < len(bank.z))).all(), "neighbor indices")
    out = np.empty((len(q), K, 2))
    for i in range(len(q)):
        delta = bank.z[ix[i]] - q[i]
        out[i, :, 0] = np.sum(delta[:, :136] ** 2, axis=1) / DIM
        out[i, :, 1] = np.sum(delta[:, 136:] ** 2, axis=1) / DIM
    return out


def calibration_weights(window_tracks):
    t = check_tracks(window_tracks, len(window_tracks))
    _, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    require(len(counts) > 0, "empty tracks")
    return np.repeat(1.0 / (len(counts) * counts[inverse] * SLOTS), SLOTS)


def weighted_q95(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    require(v.ndim == 1 and v.shape == w.shape and len(v) > 0, "quantile shape")
    require(np.isfinite(v).all() and (v >= 0).all() and np.isfinite(w).all() and (w > 0).all(), "quantile finite/weights")
    order = np.argsort(v, kind="stable")
    c = np.cumsum(w[order])
    return float(v[order[min(np.searchsorted(c, .95 * c[-1], side="left"), len(v) - 1)]])


def geometry_support(distances, radius):
    d = np.asarray(distances, float)
    require(d.ndim == 2 and d.shape[1] == K and np.isfinite(d).all() and (d >= 0).all(), "distance schema")
    require(np.isfinite(radius) and radius >= 0, "radius")
    return dict(d1=d[:, 0], d5=d[:, -1], near=d[:, -1] <= radius,
                radius_ratio=d[:, -1] / radius if radius > 0 else np.full(len(d), np.nan))


def label_support(indices, bank, train_labels):
    """Posthoc only. U remains selected; expected P corrects window base rates."""
    y, ix = np.asarray(train_labels), np.asarray(indices)
    require(y.shape == (len(bank.z) // SLOTS, SLOTS) and np.isin(y, [-1, 0, 1]).all(), "PNU shape")
    require(ix.ndim == 2 and ix.shape[1] == K and ix.dtype.kind in "iu", "indices shape")
    require(((ix >= 0) & (ix < len(bank.z))).all(), "index range")
    require(all(len(np.unique(bank.tracks[row])) == K for row in ix), "neighbors must have distinct tracks")
    p, n = (y == 1).sum(1), (y == 0).sum(1)
    require(((p > 0) & (n > 0)).all(), "common train must have P and N in every window")
    selected = y.reshape(-1)[ix]
    known_mask = selected >= 0
    known = known_mask.sum(1)
    nP, nN = (selected == 1).sum(1), (selected == 0).sum(1)
    prior = (p / (p + n))[bank.window[ix]]
    observed = np.divide(nP, known, out=np.full(len(ix), np.nan), where=known > 0)
    expected = np.divide((prior * known_mask).sum(1), known, out=np.full(len(ix), np.nan), where=known > 0)
    any_expected = 1 - np.prod(np.where(known_mask, 1 - prior, 1), axis=1)
    any_expected[known == 0] = np.nan
    any_observed = np.where(known > 0, (nP > 0).astype(float), np.nan)
    return dict(nP=nP, nN=nN, nU=K - known, known=known, known_sufficient=known >= 3,
                selected_labels=selected, selected_window_prior=prior,
                observed_P=observed, expected_P=expected, enrichment=observed - expected,
                observed_anyP=any_observed, expected_anyP=any_expected)


def best_positive(scores, labels):
    """Posthoc old-model best P, never the P with the nicest neighborhood."""
    s, y = np.asarray(scores, float), np.asarray(labels)
    require(s.ndim == 2 and s.shape[1] == SLOTS and s.shape == y.shape, "score/label shape")
    require(np.isfinite(s).all() and np.isin(y, [-1, 0, 1]).all(), "scores/PNU")
    order = np.argsort(-s, axis=1, kind="stable")
    isP = np.take_along_axis(y, order, axis=1) == 1
    hasP = isP.any(1)
    first = np.argmax(isP, axis=1)
    slot = np.where(hasP, order[np.arange(len(s)), first], -1)
    rank = np.where(hasP, first + 1, -1)
    bins = np.where(~hasP, -1, np.where(rank <= 4, 0, np.where(rank <= 8, 1, 2)))
    return dict(hasP=hasP, slot=slot, rank=rank, bin=bins)


def class_window_macro(values, labels):
    """Each class: mean within each window, then equally across windows.

    Missing values never count as zero; eligible/finite denominators saved.
    """
    v, y = np.asarray(values, float), np.asarray(labels)
    require(v.shape == y.shape and v.ndim == 2 and v.shape[1] == SLOTS, "macro shape")
    require(np.isin(y, [-1, 0, 1]).all(), "PNU")
    result = {}
    for code, name in ((1, "P"), (0, "N"), (-1, "U")):
        eligible = y == code
        finite = eligible & np.isfinite(v)
        counts = finite.sum(1)
        means = np.divide(np.where(finite, v, 0).sum(1), counts,
                          out=np.full(len(v), np.nan), where=counts > 0)
        result[name] = dict(eligible_candidates=int(eligible.sum()), finite_candidates=int(finite.sum()),
                            eligible_windows=int(eligible.any(1).sum()), finite_windows=int((counts > 0).sum()),
                            mean=float(means[counts > 0].mean()) if (counts > 0).any() else None)
    return result
