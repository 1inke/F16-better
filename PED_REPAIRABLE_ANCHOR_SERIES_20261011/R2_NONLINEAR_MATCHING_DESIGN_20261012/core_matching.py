"""Design reference only. Import performs no I/O, fitting, or experiments.

NumPy features and convex sklearn fitting. No legacy module imports, Torch,
solver-generated labels, GT inputs to prediction, or production entry point.
fit_selector requires the caller's actual resource guard and preflight approval.
"""
from dataclasses import dataclass
import hashlib
import warnings
import numpy as np

HISTORY = np.r_[0:96, 120:124]
CANDIDATE = np.r_[96:120, 124:136]
TAG = "r2-fixed-rbf64-v1"
PARAMS = dict(C=1.0, penalty="l2", class_weight=None,
              solver="newton-cholesky", tol=1e-6, max_iter=500)


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def check_x(x):
    x = np.asarray(x)
    require(x.ndim == 3 and x.shape[1:] == (20, 136), "X must be [N,20,136]")
    require(len(x) > 0 and np.isfinite(x).all(), "empty/nonfinite X")
    return x.astype(np.float64, copy=False)


def standardizer(rows):
    rows = np.asarray(rows, dtype=np.float64)
    require(rows.ndim == 2 and len(rows) > 0 and np.isfinite(rows).all(), "bad rows")
    mean, raw_std = rows.mean(axis=0), rows.std(axis=0)
    return mean, np.where(raw_std < 1e-6, 1.0, raw_std)


def landmark_rows(ids, number=64):
    """No labels or features consulted; row permutation preserves ID/slot keys."""
    ids = np.asarray(ids)
    require(ids.ndim == 1 and ids.dtype.kind in "iu", "integer IDs required")
    require(len(np.unique(ids)) == len(ids), "duplicate IDs")
    require(number == 64 and 20 * len(ids) >= number, "fixed64 unavailable")
    records = []
    for row, sample_id in enumerate(ids):
        for slot in range(20):
            key = f"{TAG}|{int(sample_id)}|{slot}".encode("utf8")
            records.append((hashlib.sha256(key).digest(), int(sample_id), slot, row))
    records.sort(key=lambda item: item[:3])
    chosen = records[:number]
    flat = np.array([r[3] * 20 + r[2] for r in chosen], dtype=np.int64)
    keys = np.array([[r[1], r[2]] for r in chosen], dtype=np.int64)
    return flat, keys


def rbf_values(z, centers, guard=None, chunk=256):
    """Bounded temporary [chunk,64,100], not all-pairs population allocation."""
    z, centers = np.asarray(z, float), np.asarray(centers, float)
    require(z.ndim == 2 and z.shape[1] == 136, "bad standardized rows")
    require(centers.shape == (64, 136), "exactly64 centers required")
    require(np.isfinite(z).all() and np.isfinite(centers).all(), "nonfinite RBF input")
    require(0 < chunk <= 256, "chunk must be 1..256")
    out = np.empty((len(z), 64), dtype=np.float64)
    for first in range(0, len(z), chunk):
        if guard is not None:
            guard()
        batch = z[first:first + chunk]
        dh = np.mean((batch[:, None, HISTORY] - centers[None, :, HISTORY]) ** 2, axis=2)
        dq = np.mean((batch[:, None, CANDIDATE] - centers[None, :, CANDIDATE]) ** 2, axis=2)
        out[first:first + len(batch)] = np.exp(-0.5 * (dh + dq))
    require(np.isfinite(out).all(), "nonfinite RBF output")
    return out


@dataclass(frozen=True)
class FeatureMap:
    arm: str
    mean: np.ndarray
    std: np.ndarray
    centers: np.ndarray
    center_keys: np.ndarray
    rbf_mean: np.ndarray
    rbf_std: np.ndarray


def build_map(x_train, train_ids, arm, guard=None):
    """Only TRAIN arrays enter this function. U rows allowed for unsupervised stats."""
    require(arm in ("linear", "rbf64"), "unknown arm")
    x = check_x(x_train)
    require(len(train_ids) == len(x), "ID length mismatch")
    # Same reduction axes and dtype as R1; avoid a different summation layout.
    mean = x.mean(axis=(0, 1))
    raw_std = x.std(axis=(0, 1))
    std = np.where(raw_std < 1e-6, 1.0, raw_std)
    z = ((x - mean) / std).reshape(-1, 136)
    if arm == "linear":
        return FeatureMap(arm, mean, std, np.empty((0, 136)),
                          np.empty((0, 2), dtype=np.int64), np.empty(0), np.empty(0))
    indices, keys = landmark_rows(train_ids)
    centers = z[indices].copy()
    raw = rbf_values(z, centers, guard=guard)
    rmean, rstd = standardizer(raw)
    return FeatureMap(arm, mean, std, centers, keys, rmean, rstd)


def transform(x, mapping, guard=None):
    x = check_x(x)
    require(mapping.arm in ("linear", "rbf64"), "invalid map")
    z = ((x - mapping.mean) / mapping.std).reshape(-1, 136)
    if mapping.arm == "linear":
        return z.reshape(len(x), 20, 136)
    phi = rbf_values(z, mapping.centers, guard=guard)
    phi = (phi - mapping.rbf_mean) / mapping.rbf_std
    return np.concatenate((z, phi), axis=1).reshape(len(x), 20, 200)


def witness_supervision(labels, tracks):
    """Exact R1 weighting: track equal, per-window class half, sum=20*N."""
    y, tracks = np.asarray(labels), np.asarray(tracks)
    require(y.ndim == 2 and y.shape[1] == 20 and len(y) == len(tracks), "label shape")
    require(np.isin(y, [-1, 0, 1]).all(), "unknown label code")
    pos, neg = y == 1, y == 0
    require(np.all(pos.any(1) & neg.any(1)), "training needs P and N per window")
    _, inv, count = np.unique(tracks, return_inverse=True, return_counts=True)
    weight = (0.5 * pos / pos.sum(1, keepdims=True)
              + 0.5 * neg / neg.sum(1, keepdims=True)) / count[inv, None]
    weight *= pos.size / weight.sum()
    known = pos | neg
    require(np.all(weight[~known] == 0), "U must have zero weight")
    return pos.astype(np.int8), known, weight


def objective_grad(features, target, weight, coef, intercept):
    """Independent NumPy objective, C=1, unpenalized intercept. No sklearn import."""
    x, y, sw, w = (np.asarray(a, dtype=float) for a in (features, target, weight, coef))
    require(x.ndim == 2 and y.shape == sw.shape == (len(x),), "objective shape")
    require(w.shape == (x.shape[1],) and np.all(sw >= 0) and sw.sum() > 0, "objective weights")
    z = x @ w + float(intercept)
    # Stable sigmoid, no overflow and no mutation of scores.
    probability = np.exp(-np.logaddexp(0.0, -z))
    residual = sw * (probability - y)
    total = sw.sum()
    value = float(sw @ (np.logaddexp(0.0, z) - y * z) / total + (w @ w) / (2 * total))
    gradient = np.r_[(x.T @ residual + w) / total, residual.sum() / total]
    return value, gradient


@dataclass(frozen=True)
class Selector:
    mapping: FeatureMap
    coef: np.ndarray
    intercept: float


def fit_selector(x_train, labels, tracks, train_ids, arm, *, guard):
    """FUTURE explicit fitting API, never called by unit tests or on import.

    Caller must first pass real-data/source/resource preflight and own persistence.
    This core is not a production launcher or an authorization check.
    """
    require(callable(guard), "production resource guard required")
    guard()
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    mapping = build_map(x_train, train_ids, arm, guard=guard)
    features = transform(x_train, mapping, guard=guard)
    target, known, weight = witness_supervision(labels, tracks)
    require(features.shape[:2] == target.shape, "training dimensions differ")
    x, y, sw = features[known], target[known], weight[known]
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = LogisticRegression(**PARAMS).fit(x, y, sample_weight=sw)
    guard()
    require(not any(issubclass(w.category, ConvergenceWarning) for w in caught), "STOP_CONVERGENCE")
    require(int(fitted.n_iter_[0]) < 500, "STOP_ITERATION_LIMIT")
    require(fitted.classes_.tolist() == [0, 1], "wrong class order")
    model = Selector(mapping, fitted.coef_[0].copy(), float(fitted.intercept_[0]))
    loss, gradient = objective_grad(x, y, sw, model.coef, model.intercept)
    require(np.isfinite(gradient).all() and np.max(np.abs(gradient)) <= 1e-6, "STOP_STATIONARITY")
    diagnostics = dict(iterations=int(fitted.n_iter_[0]), loss=loss,
                       gradient_inf=float(np.abs(gradient).max()),
                       warnings=[str(w.message) for w in caught], total_weight=float(sw.sum()),
                       positive_weight=float(sw[y == 1].sum()), negative_weight=float(sw[y == 0].sum()),
                       known_rows=len(y), unknown_rows=int((~known).sum()),
                       train_windows=len(x_train), train_tracks=len(np.unique(tracks)),
                       coefficients_including_intercept=len(model.coef) + 1)
    return model, diagnostics


def predict(x, model, guard=None):
    """No GT, labels, IDs, groups, tracks, scenes, or logits accepted."""
    features = transform(x, model.mapping, guard=guard)
    score = features @ model.coef + model.intercept
    require(np.isfinite(score).all(), "nonfinite score")
    return score


def rank4(score):
    score = np.asarray(score, float)
    require(score.ndim == 2 and score.shape[1] == 20 and np.isfinite(score).all(), "bad score")
    return np.sort(np.argsort(-score, axis=1, kind="stable")[:, :4], axis=1)


def model_arrays(model):
    """Numeric-only payload for np.savez; caller saves manifest separately."""
    m = model.mapping
    return dict(arm=np.array(m.arm), mean=m.mean.copy(), std=m.std.copy(),
                centers=m.centers.copy(), center_keys=m.center_keys.copy(),
                rbf_mean=m.rbf_mean.copy(), rbf_std=m.rbf_std.copy(),
                coef=model.coef.copy(), intercept=np.array([model.intercept]))


def restore_model(a):
    arm = str(np.asarray(a["arm"]).item())
    require(arm in ("linear", "rbf64"), "model arm")
    m = FeatureMap(arm, *(np.array(a[k], copy=True) for k in
                         ("mean", "std", "centers", "center_keys", "rbf_mean", "rbf_std")))
    dim = 136 if arm == "linear" else 200
    require(m.mean.shape == m.std.shape == (136,) and np.all(m.std > 0), "model normalization")
    require(np.asarray(a["coef"]).shape == (dim,) and np.asarray(a["intercept"]).shape == (1,), "model coefficients")
    count = 0 if arm == "linear" else 64
    require(m.centers.shape == (count, 136) and m.center_keys.shape == (count, 2), "model centers")
    require(m.rbf_mean.shape == m.rbf_std.shape == (count,) and np.all(m.rbf_std > 0), "model RBF normalization")
    for key in ("mean", "std", "centers", "rbf_mean", "rbf_std", "coef", "intercept"):
        require(np.isfinite(a[key]).all(), f"nonfinite {key}")
    return Selector(m, np.array(a["coef"], copy=True), float(a["intercept"][0]))
