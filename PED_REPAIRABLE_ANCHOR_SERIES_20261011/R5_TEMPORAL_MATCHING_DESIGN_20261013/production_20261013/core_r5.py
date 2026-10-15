"""R5 dimension-generalized copy of pinned R3 numerical core. Import does not read data, fit, or launch a worker.

fit_objective is an explicit API; synthetic tests never call it.
Only the caller can provide the audited frozen R2 features and resource guard.
Scores are within-window utilities, not future probabilities.
"""
from dataclasses import dataclass
import warnings
import numpy as np

PARAMS = dict(C=1.0, penalty="l2", class_weight=None,
              solver="newton-cholesky", tol=1e-6, max_iter=500)


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def check_features(z):
    z = np.asarray(z, dtype=np.float64)
    require(z.ndim == 3 and z.shape[1] == 20 and z.shape[2] in (200,368), "expected [N,20,200or368]")
    require(0 < len(z) <= 384 and np.isfinite(z).all(), "empty/oversized/nonfinite features")
    return z


def pointwise_weights(labels, tracks):
    """EXACT R2 arithmetic order: track equal, P/N half, total20N, U0."""
    y, tracks = np.asarray(labels), np.asarray(tracks)
    require(y.ndim == 2 and y.shape[1] == 20 and tracks.shape == (len(y),), "training shapes")
    require(len(y) > 0 and np.isin(y, [-1, 0, 1]).all(), "label domain")
    pos, neg = y == 1, y == 0
    require(np.all(pos.any(1) & neg.any(1)), "each training window needs P and N")
    _, inv, count = np.unique(tracks, return_inverse=True, return_counts=True)
    weight = (0.5 * pos / pos.sum(1, keepdims=True)
              + 0.5 * neg / neg.sum(1, keepdims=True)) / count[inv, None]
    weight *= pos.size / weight.sum()
    return pos.astype(np.int8), pos | neg, weight


@dataclass(frozen=True)
class Pairs:
    delta: np.ndarray
    weight: np.ndarray
    window: np.ndarray
    positive: np.ndarray
    negative: np.ndarray


def build_pairs(z, labels, tracks, *, guard):
    """All known P x N pairs; bounded allocation, NO U, no cross-window pair."""
    require(callable(guard), "resource guard required")
    guard()
    z, labels = check_features(z), np.asarray(labels)
    _, _, sw = pointwise_weights(labels, tracks)
    require(labels.shape == z.shape[:2], "feature/label mismatch")
    sizes = (labels == 1).sum(1) * (labels == 0).sum(1)
    count = int(sizes.sum())
    require(0 < count <= len(z) * 100, "pair budget")
    delta = np.empty((count, z.shape[2]), dtype=np.float64)
    weight = np.empty(count, dtype=np.float64)
    windows = np.empty(count, dtype=np.int64)
    positive = np.empty(count, dtype=np.int64)
    negative = np.empty(count, dtype=np.int64)
    mass, offset = sw.sum(1), 0
    for i, y in enumerate(labels):
        guard()
        p, n = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        pp, nn = np.repeat(p, len(n)), np.tile(n, len(p))
        end = offset + len(pp)
        delta[offset:end] = z[i, pp] - z[i, nn]
        weight[offset:end] = mass[i] / len(pp)
        windows[offset:end], positive[offset:end], negative[offset:end] = i, pp, nn
        offset = end
    require(np.isfinite(delta).all() and np.isfinite(weight).all(), "nonfinite pair contract")
    require(np.all(weight > 0) and np.isclose(weight.sum(), 20 * len(z), rtol=1e-14), "pair normalization")
    return Pairs(delta, weight, windows, positive, negative)


def symmetric_rows(pairs):
    """Two signed encodings are one pair, NOT independent observations."""
    d, sw = pairs.delta, pairs.weight
    return (np.concatenate((d, -d)), np.r_[np.ones(len(d)), np.zeros(len(d))],
            np.r_[sw * 0.5, sw * 0.5])


def pair_objective(pairs, coef):
    """Direct positive-margin formula independent of symmetric fitting rows."""
    d, a, w = pairs.delta, pairs.weight, np.asarray(coef, dtype=np.float64)
    require(w.shape == (d.shape[1],) and np.isfinite(w).all(), "pair coefficients")
    margin = d @ w
    q = np.exp(-np.logaddexp(0.0, margin))
    total = a.sum()
    value = (a @ np.logaddexp(0.0, -margin) + 0.5 * (w @ w)) / total
    gradient = (-d.T @ (a * q) + w) / total
    hessian = (d.T @ ((a * q * (1 - q))[:, None] * d) + np.eye(len(w))) / total
    return float(value), gradient, hessian


def logistic_objective(x, y, sw, coef, intercept=0.0, fit_intercept=True):
    """Stable BCE + C1 penalty with independently calculated gradient/Hessian."""
    x, y, a, w = (np.asarray(v, dtype=np.float64) for v in (x, y, sw, coef))
    require(x.ndim == 2 and y.shape == a.shape == (len(x),), "logistic shapes")
    require(w.shape == (x.shape[1],) and np.isin(y, [0, 1]).all(), "logistic labels/coef")
    require(all(np.isfinite(v).all() for v in (x, y, a, w)) and np.isfinite(intercept), "nonfinite logistic")
    require(np.all(a >= 0) and a.sum() > 0, "logistic weights")
    require(fit_intercept or intercept == 0, "pair intercept gauge must be zero")
    score = x @ w + float(intercept)
    prob = np.exp(-np.logaddexp(0.0, -score))
    total, residual = a.sum(), a * (prob - y)
    loss = (a @ (np.logaddexp(0.0, score) - y * score) + 0.5 * (w @ w)) / total
    design = np.column_stack((x, np.ones(len(x)))) if fit_intercept else x
    penalty = np.r_[w, 0.] if fit_intercept else w
    diagonal = np.r_[np.ones(len(w)), 0.] if fit_intercept else np.ones(len(w))
    gradient = (design.T @ residual + penalty) / total
    hessian = (design.T @ ((a * prob * (1 - prob))[:, None] * design) + np.diag(diagonal)) / total
    return float(loss), gradient, hessian


@dataclass(frozen=True)
class RankModel:
    arm: str
    coef: np.ndarray
    intercept: float


def fit_objective(z, labels, tracks, arm, *, guard):
    """Future explicit fit, NO disk output or implicit data/feature-map fitting.

    Caller must pass real input/source/resource preflight BEFORE invocation.
    A passed callable is not proof of those gates. This is NOT a launcher.
    """
    require(arm in ("pointwise", "pairwise") and callable(guard), "fit contract")
    guard()
    z = check_features(z)
    target, known, sw = pointwise_weights(labels, tracks)
    require(z.shape[:2] == target.shape, "fit dimensions")
    pairs = None
    if arm == "pointwise":
        x, y, weight = z[known], target[known], sw[known]
    else:
        pairs = build_pairs(z, labels, tracks, guard=guard)
        x, y, weight = symmetric_rows(pairs)
    guard()
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = LogisticRegression(**PARAMS, fit_intercept=(arm == "pointwise"))
        fitted.fit(x, y, sample_weight=weight)
    guard()
    require(not caught, "STOP_SOLVER_WARNING: " + "; ".join(str(item.message) for item in caught))
    require(int(fitted.n_iter_[0]) < 500 and fitted.classes_.tolist() == [0, 1], "STOP_SOLVER_CONTRACT")
    intercept = float(fitted.intercept_[0]) if arm == "pointwise" else 0.0
    model = RankModel(arm, fitted.coef_[0].copy(), intercept)
    with threadpool_limits(limits=1):
        if pairs is None:
            value, grad, _ = logistic_objective(x, y, weight, model.coef, intercept)
        else:
            value, grad, _ = pair_objective(pairs, model.coef)
    require(np.isfinite(value) and np.isfinite(grad).all() and np.abs(grad).max() <= 1e-6,
            "STOP_STATIONARITY")
    return model, dict(iterations=int(fitted.n_iter_[0]), loss=value,
                      gradient_inf=float(np.abs(grad).max()), solver_rows=len(y),
                      independent_units="tracks_not_pairs_or_signed_rows",
                      train_windows=len(z), train_tracks=len(np.unique(tracks)),
                      pairs=0 if pairs is None else len(pairs.delta),
                      total_weight=float(weight.sum()), intercept_fixed=(arm == "pairwise"))


def score(z, model):
    """Prediction takes NO labels/GT/IDs/groups/scene/logits."""
    z = check_features(z)
    require(model.arm in ("pointwise", "pairwise"), "arm")
    require(np.asarray(model.coef).shape == (z.shape[2],) and np.isfinite(model.coef).all(), "coef")
    require(np.isfinite(model.intercept) and (model.arm != "pairwise" or model.intercept == 0), "intercept")
    out = z @ model.coef + model.intercept
    require(np.isfinite(out).all(), "nonfinite scores")
    return out


def rank4(scores):
    scores = np.asarray(scores, dtype=np.float64)
    require(scores.ndim == 2 and scores.shape[1] == 20 and np.isfinite(scores).all(), "score shape")
    return np.sort(np.argsort(-scores, axis=1, kind="stable")[:, :4], axis=1)
