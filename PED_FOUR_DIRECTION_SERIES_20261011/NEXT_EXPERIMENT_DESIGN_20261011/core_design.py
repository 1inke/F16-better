"""Design reference only. No file I/O, model import, training, or launcher.
GT may enter envelope_targets only. Deployment-facing functions exclude labels.
Run synthetic tests with: python -B -m unittest discover -s <this_directory>.
"""
from hashlib import sha256
import numpy as np


def _finite(x):
    x = np.asarray(x)
    if not np.isfinite(x).all():
        raise ValueError("non-finite input")
    return x


def _paths(x, k=None):
    x = _finite(x)
    if x.ndim != 4 or x.shape[-2:] != (12, 2):
        raise ValueError("expected [batch, candidates, 12, 2]")
    if k is not None and x.shape[1] != k:
        raise ValueError("candidate count mismatch")
    return x


def radius():
    return .20 * (np.arange(1, 13, dtype=np.float64) / 12.) ** 2


def envelope_targets(native20, gt_future):
    """Offline only: optimistic position-ball envelope, NOT physical feasibility."""
    p = _paths(native20, 20).astype(np.float64)
    gt = _finite(gt_future).astype(np.float64)
    if gt.shape != (len(p), 12, 2):
        raise ValueError("GT shape mismatch")
    e = np.linalg.norm(p - gt[:, None], axis=-1)
    low = np.maximum(e - radius()[None, None], 0.)
    j = np.maximum(e.mean(-1) / .2, e[..., -1] / .4)
    lower_j = np.maximum(low.mean(-1) / .2, low[..., -1] / .4)
    candidate_good = j <= 1.
    # One candidate must satisfy BOTH conditions.
    potential = (j > 1.) & (lower_j <= 1.)
    return dict(candidate_good=candidate_good, native_good=candidate_good.any(1),
                potential=potential, j=j, lower_j=lower_j)


def identity_bucket(sequence, track, tag="repair-address-v1", modulus=5):
    """Split membership only; never include this value in model features."""
    if not sequence or not track or modulus < 1:
        raise ValueError("require verified nonempty sequence/track")
    # Length-delimited key prevents ambiguous concatenations.
    key = f"{tag}|{len(str(sequence))}:{sequence}|{len(str(track))}:{track}"
    return int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "big") % modulus


def selector_features(p2_observed, native20):
    p = _paths(native20, 20)
    z = _finite(p2_observed)
    if z.shape != (len(p), 96):
        raise ValueError("expected observed P2 [batch,96]")
    return np.concatenate((np.broadcast_to(z[:, None], (len(p), 20, 96)),
                           p.reshape(len(p), 20, 24)), axis=-1).astype(np.float32)


def select4(scores20):
    s = _finite(scores20)
    if s.ndim != 2 or s.shape[1] != 20:
        raise ValueError("expected [batch,20] scores")
    # Stable sort resolves exact score ties by original physical index.
    return np.sort(np.argsort(-s, axis=1, kind="stable")[:, :4], axis=1)


def bottom4(native_logits):
    s = _finite(native_logits)
    if s.ndim != 2 or s.shape[1] != 20:
        raise ValueError("expected [batch,20] logits")
    return np.sort(np.argsort(-s, axis=1, kind="stable")[:, 16:], axis=1)


def fixed_random4(sample_ids):
    # Only the registered random comparator uses IDs, never the learned selector.
    rows = []
    for sid in sample_ids:
        ranks = [(sha256(f"repair-random-v1|{sid}|{j}".encode()).digest(), j)
                 for j in range(20)]
        rows.append(sorted(j for _, j in sorted(ranks)[:4]))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 4)


def _slots(slots4, n):
    s = np.asarray(slots4)
    if s.shape != (n, 4) or not np.issubdtype(s.dtype, np.integer):
        raise ValueError("expected integer [batch,4] slot map")
    if np.any((s < 0) | (s >= 20)) or np.any(np.diff(s, axis=1) <= 0):
        raise ValueError("slots must be unique, ascending, and in [0,20)")
    return s


def gather(native20, slots4):
    p = _paths(native20, 20)
    s = _slots(slots4, len(p))
    return p[np.arange(len(p))[:, None], s].copy()


def scatter_native(native20, slots4, proposal4):
    p = _paths(native20, 20)
    s = _slots(slots4, len(p))
    q = _paths(proposal4, 4)
    if len(q) != len(p):
        raise ValueError("batch mismatch")
    out = p.astype(np.result_type(p.dtype, q.dtype), copy=True)
    out[np.arange(len(p))[:, None], s] = q
    return out


def position3_decode(anchor4, raw_controls):
    """NumPy mirror of the existing position3; no new physical safety claim."""
    a = _paths(anchor4, 4).astype(np.float64)
    raw = _finite(raw_controls).astype(np.float64)
    if raw.shape != (len(a), 4, 3, 2):
        raise ValueError("expected raw controls [batch,4,3,2]")
    u = np.tanh(raw)
    u /= np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1.)
    t = np.arange(1, 13, dtype=np.float64)
    basis = np.stack([np.interp(t, [0, 4, 8, 12], np.eye(4)[j+1])
                      for j in range(3)], axis=1)
    node_radius = .20 * (np.array([4., 8., 12.]) / 12.) ** 2
    proposal = np.einsum("ti,bkid->bktd", basis,
                         u * node_radius[None, None, :, None])
    norm = np.linalg.norm(proposal, axis=-1)
    ratio = np.ones_like(norm)
    np.divide(radius()[None, None], norm, out=ratio, where=norm > 0)
    lam = np.minimum(1., ratio.min(-1))
    residual = proposal * lam[:, :, None, None]
    return a + residual, residual, lam


def potential_hit(potential20, slots4):
    """Evaluation only. Callers must not feed this result to deployment policy."""
    p = np.asarray(potential20, dtype=bool)
    if p.ndim != 2 or p.shape[1] != 20:
        raise ValueError("expected [batch,20] labels")
    s = _slots(slots4, len(p))
    return p[np.arange(len(p))[:, None], s].any(1)


def accept_edit_subset(native_window, slots4, proposal4, selected_scores, motion):
    """S3 reference for ONE window. motion([20,12,2]) -> five nonnegative values.
    Caller must inject the locked dt=.4 definitions, including speed origin and
    percentile convention; this generic kernel deliberately does not redefine them.
    No GT, ID, groups, scene, or oracle utility accepted.
    """
    p = _paths(np.asarray(native_window)[None], 20)[0].astype(np.float64)
    s = _slots(np.asarray(slots4)[None], 1)[0]
    q = _paths(np.asarray(proposal4)[None], 4)[0].astype(np.float64)
    score = _finite(selected_scores).astype(np.float64)
    if score.shape != (4,) or np.any((score < 0) | (score > 1)):
        raise ValueError("require four frozen selector probabilities")
    base = _finite(motion(p))
    if base.shape != (5,) or (base < 0).any():
        raise ValueError("registered motion must return 5 nonnegative values")
    best, best_mask, best_key = p.copy(), 0, (0., 0, 0)
    for mask in range(1, 16):
        active = np.array([(mask >> j) & 1 for j in range(4)], dtype=bool)
        candidate = p.copy()
        candidate[s[active]] = q[active]
        m = _finite(motion(candidate))
        if m.shape != (5,) or (m < 0).any():
            raise ValueError("invalid registered motion")
        # Literal gate: no epsilon, including a zero native denominator.
        if np.all(m <= 1.01 * base):
            key = (float(score[active].sum()), -int(active.sum()), -mask)
            if key > best_key:
                best, best_mask, best_key = candidate, mask, key
    return best, best_mask


def temporal_basis():
    t = np.arange(8, dtype=np.float64) + .5
    return np.stack([np.ones(8)/8,
                     np.sqrt(2)/8*np.cos(np.pi*t/8),
                     np.sqrt(2)/8*np.cos(2*np.pi*t/8)])


def temporal_readout(projected_tokens, arm):
    """Input is AFTER the common trainable Linear(96,32), not the raw P2 mean."""
    z = _finite(projected_tokens)
    if z.ndim != 3 or z.shape[1:] != (8, 32):
        raise ValueError("expected [batch,8,32]")
    coeff = np.einsum("kt,btc->bkc", temporal_basis(), z)
    if arm == "mean":
        coeff = np.repeat(coeff[:, :1], 3, axis=1)
    elif arm != "temporal_dct":
        raise ValueError("unknown temporal arm")
    return coeff.reshape(len(z), 96)
