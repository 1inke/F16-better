"""
Redundancy Metrics for Probability and Quality Targets.

Computes:
- Jensen-Shannon (JS) Divergence
- Kullback-Leibler (KL) Divergence
- Cosine Similarity
- Top-k Agreement & Jaccard Overlap
- Shannon Entropy & Effective Mode Count
"""
from __future__ import annotations

import numpy as np


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Computes Jensen-Shannon divergence per sample between distributions p and q.
    p, q: [N, K]
    Returns: [N]
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    # Re-normalize
    p = p / np.sum(p, axis=-1, keepdims=True)
    q = q / np.sum(q, axis=-1, keepdims=True)
    m = 0.5 * (p + q)

    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=-1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=-1)
    return 0.5 * (kl_pm + kl_qm)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Computes KL(p || q) per sample.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    q = q / np.sum(q, axis=-1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def cosine_similarity(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Cosine similarity between vectors p and q along last dimension.
    """
    dot = np.sum(p * q, axis=-1)
    norm_p = np.linalg.norm(p, axis=-1)
    norm_q = np.linalg.norm(q, axis=-1)
    denom = np.clip(norm_p * norm_q, eps, None)
    return dot / denom


def topk_jaccard_overlap(p: np.ndarray, q: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Computes Jaccard overlap between top-k indices of p and q.
    J = |A ∩ B| / |A ∪ B| = |A ∩ B| / (2k - |A ∩ B|)
    """
    N = len(p)
    top_p = np.argsort(-p, axis=-1)[:, :k]
    top_q = np.argsort(-q, axis=-1)[:, :k]

    jaccards = np.zeros(N, dtype=np.float64)
    for i in range(N):
        set_p = set(top_p[i])
        set_q = set(top_q[i])
        intersection = len(set_p & set_q)
        union = len(set_p | set_q)
        jaccards[i] = intersection / union if union > 0 else 1.0
    return jaccards


def top1_agreement(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Returns boolean array indicating if argmax matches.
    """
    return (np.argmax(p, axis=-1) == np.argmax(q, axis=-1)).astype(np.float64)


def compute_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Shannon entropy H(p) in nats.
    """
    p = np.clip(p, eps, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    return -np.sum(p * np.log(p), axis=-1)


def effective_mode_count(p: np.ndarray) -> np.ndarray:
    """
    K_eff = exp(H(p))
    """
    return np.exp(compute_entropy(p))
