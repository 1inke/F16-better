"""Frozen-candidate selectors. Feature construction never reads labels/future."""
from __future__ import annotations

import hashlib
import random
import numpy as np
import torch
from torch import nn

INPUT_KEYS = ('past_relative', 'candidate_final', 'mode_feature', 'context_feature', 'native_logits')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def split_ids(ids, fraction, seed):
    """Stable explicit-ID split; exploratory, not a scene-independent split."""
    ids = np.asarray(ids)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate sample IDs')
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(f'{seed}:{int(ids[i])}'.encode()).digest())
    n = max(1, int(len(ids) * fraction))
    if len(ids) - n < 2:
        raise ValueError('Not enough fit samples')
    return np.asarray(order[n:]), np.asarray(order[:n])


def features(data):
    """All geometry in metres in the existing focal-last-observation frame."""
    past = np.asarray(data['past_relative'], dtype=np.float32)
    cand = np.asarray(data['candidate_final'], dtype=np.float32)
    logits = np.asarray(data['native_logits'], dtype=np.float32)
    n, k, t, _ = cand.shape
    vel = past[:, -1] - past[:, -2]
    cvel = np.diff(np.concatenate([np.zeros((n, k, 1, 2), np.float32), cand], axis=2), axis=2)
    ctx = np.broadcast_to(data['context_feature'][:, None], (n, k, data['context_feature'].shape[-1]))
    base = np.concatenate([
        data['mode_feature'], ctx, cand.reshape(n, k, -1), cvel.reshape(n, k, -1),
        np.broadcast_to(vel[:, None], (n, k, 2)),
        (logits - logits.mean(axis=1, keepdims=True))[..., None],
    ], axis=-1).astype(np.float32)
    times = np.arange(1 - past.shape[1], 1, dtype=np.float32)[None, :, None]
    cv = times * vel[:, None]
    return base, past, cv


def history_tokens(past):
    vel = np.diff(past, axis=1, prepend=past[:, :1])
    acc = np.diff(vel, axis=1, prepend=vel[:, :1])
    time = np.broadcast_to(np.linspace(-1, 0, past.shape[1], dtype=np.float32)[None, :, None],
                           (len(past), past.shape[1], 1))
    return np.concatenate([past, vel, acc, time], axis=-1).astype(np.float32)


def history_for(past, cv, variant, permutation=None):
    if variant in ('linear', 'cv_history'):
        return history_tokens(cv)
    if variant == 'real_history':
        return history_tokens(past)
    if variant == 'shuffled_residual':
        if permutation is None or sorted(permutation.tolist()) != list(range(len(past))):
            raise ValueError('Explicit complete permutation required')
        # Keep own final position and final observed velocity; only permute deviation from CV.
        return history_tokens(cv + (past - cv)[permutation])
    raise ValueError(variant)


class Normalizer:
    def __init__(self, base, hist):
        self.base_mean = base.mean((0, 1), keepdims=True)
        self.base_std = np.maximum(base.std((0, 1), keepdims=True), 1e-5)
        self.hist_mean = hist.mean((0, 1), keepdims=True)
        self.hist_std = np.maximum(hist.std((0, 1), keepdims=True), 1e-5)

    def apply(self, base, hist):
        return ((base - self.base_mean) / self.base_std).astype(np.float32), \
               ((hist - self.hist_mean) / self.hist_std).astype(np.float32)


class Selector(nn.Module):
    def __init__(self, base_dim, hist_dim, variant, hidden=64, history_hidden=32):
        super().__init__()
        self.variant = variant
        if variant == 'linear':
            self.linear = nn.Linear(base_dim, 1)
        else:
            self.base = nn.Sequential(nn.Linear(base_dim, hidden), nn.GELU())
            self.history = nn.GRU(hist_dim, history_hidden, batch_first=True)
            self.query = nn.Linear(hidden, history_hidden)
            self.head = nn.Sequential(nn.Linear(hidden + 2 * history_hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, base, hist):
        if self.variant == 'linear':
            out = self.linear(base).squeeze(-1)
        else:
            b = self.base(base)
            h, _ = self.history(hist)
            q = self.query(b)
            attention = torch.softmax(torch.einsum('bkd,btd->bkt', q, h) / q.shape[-1]**0.5, dim=-1)
            context = torch.einsum('bkt,btd->bkd', attention, h)
            out = self.head(torch.cat([b, context, q * context], -1)).squeeze(-1)
        return out - out.mean(dim=-1, keepdim=True)


def utility_target(ade):
    return -(ade - ade.mean(dim=-1, keepdim=True))


def calibrate(logits, utility, ade, scale, alphas):
    best = None
    for alpha in sorted(alphas):
        scores = logits + alpha * scale * utility
        indices = scores.argmax(1)
        value = float(ade[np.arange(len(ade)), indices].mean())
        if best is None or value < best['ade'] - 1e-10:
            best = dict(alpha=float(alpha), ade=value)
    return best


def metrics(ade, fde, indices, native_indices):
    r = np.arange(len(ade))
    chosen, native, oracle = ade[r, indices], ade[r, native_indices], ade.min(1)
    regret = chosen - oracle
    improvement = native - chosen
    changed = indices != native_indices
    high = native - oracle > 0.2
    gap = np.sort(ade, axis=1)[:, 1] - oracle
    result = dict(n=len(ade), selected_ade=float(chosen.mean()), selected_fde=float(fde[r, indices].mean()),
                  minade=float(oracle.mean()), minfde=float(fde.min(1).mean()), regret=float(regret.mean()),
                  near_oracle_05=float((regret <= .05).mean()), exact_oracle=float((indices == ade.argmin(1)).mean()),
                  improvement_m=float(improvement.mean()), regret_p90=float(np.quantile(regret, .9)),
                  intervention_rate=float(changed.mean()), harmful_rate=float((changed & (improvement < -.05)).mean()),
                  useful_rate=float((changed & (improvement > .05)).mean()),
                  mean_harm_m=float(np.maximum(-improvement, 0).mean()),
                  mean_benefit_m=float(np.maximum(improvement, 0).mean()))
    for name, mask in [('high_regret', high), ('small_gap_high_regret', (gap < .02) & high),
                       ('small_gap_low_regret', (gap < .02) & ~high)]:
        result[name + '_n'] = int(mask.sum())
        result[name + '_improvement_m'] = float(improvement[mask].mean()) if mask.any() else None
    return result
