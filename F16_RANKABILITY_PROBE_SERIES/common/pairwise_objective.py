"""
Decision-Relevant Pairwise Ranking Objective for Rankability Probe Series.

Enforces Butler Rule: research.goal_aligned_evaluation
Focuses supervision on decision-relevant candidate comparisons:
- Ignores near-tie candidate pairs where |ADE_i - ADE_j| < 0.02 m.
- Weights pairs by min((ADE_j - ADE_i) / 0.20, 1.0) so pairs with large regret difference (>20cm)
  receive full weight.
- Loss: L_pair = w_ij * softplus(-(s_i - s_j))
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionRelevantPairwiseLoss(nn.Module):
    def __init__(self, tie_threshold: float = 0.02, max_diff_scale: float = 0.20):
        super().__init__()
        self.tie_threshold = tie_threshold
        self.max_diff_scale = max_diff_scale

    def forward(self, scores: torch.Tensor, ades: torch.Tensor) -> torch.Tensor:
        """
        scores: [B, K] - unnormalized score for each mode (higher means model predicts better)
        ades:   [B, K] - ground-truth ADE for each mode (lower is better)

        Returns:
            scalar loss tensor
        """
        # delta_ij[b, i, j] = ades[b, j] - ades[b, i]
        # When delta_ij > 0, candidate i has strictly lower ADE than candidate j (i is better than j)
        delta_ij = ades.unsqueeze(1) - ades.unsqueeze(2)  # [B, K, K]

        # Valid pair condition: i is better than j by at least tie_threshold (0.02m)
        valid_mask = delta_ij > self.tie_threshold

        # Decision-relevant weighting: saturated at max_diff_scale (0.20m)
        weights = torch.clamp(delta_ij / self.max_diff_scale, max=1.0) * valid_mask.float()

        # Score difference: s_diff[b, i, j] = scores[b, i] - scores[b, j]
        s_diff = scores.unsqueeze(2) - scores.unsqueeze(1)

        # Softplus penalty on -(s_i - s_j): penalizes when s_i <= s_j
        pair_loss = weights * F.softplus(-s_diff)

        total_weight = weights.sum().clamp(min=1.0)
        return pair_loss.sum() / total_weight


class OracleVsRestPairwiseLoss(nn.Module):
    """
    R5 Objective: Direct Top-1 aligned pairwise ranking (§14).
    For each sample:
      k* = argmin_k ADE_k
      For j != k*:
        Delta_j = ADE_j - ADE_{k*}
        Filter Delta_j < tie_threshold (0.02m)
        Weight w_j = min(Delta_j / max_diff_scale, 1.0)
        L = sum_{j != k*} w_j * softplus(-(s_{k*} - s_j)) / sum_j w_j
    """
    def __init__(self, tie_threshold: float = 0.02, max_diff_scale: float = 0.20):
        super().__init__()
        self.tie_threshold = tie_threshold
        self.max_diff_scale = max_diff_scale

    def forward(self, scores: torch.Tensor, ades: torch.Tensor) -> torch.Tensor:
        """
        scores: [B, K]
        ades:   [B, K]
        """
        B, K = scores.shape
        oracle_idx = torch.argmin(ades, dim=-1)  # [B]

        batch_idx = torch.arange(B, device=scores.device)
        s_oracle = scores[batch_idx, oracle_idx].unsqueeze(1)  # [B, 1]
        ade_oracle = ades[batch_idx, oracle_idx].unsqueeze(1)  # [B, 1]

        # Delta_j = ADE_j - ADE_{k*} (>= 0)
        delta_j = ades - ade_oracle  # [B, K]

        # Filter near-tie competitor (< 0.02m)
        valid_mask = delta_j > self.tie_threshold  # [B, K]

        # Regret weighting capped at 0.20m
        w_j = torch.clamp(delta_j / self.max_diff_scale, max=1.0) * valid_mask.float()

        # Score diff: s_{k*} - s_j
        s_diff = s_oracle - scores  # [B, K]

        loss = w_j * F.softplus(-s_diff)
        total_w = w_j.sum().clamp(min=1.0)
        return loss.sum() / total_w


class ListwiseTop1Loss(nn.Module):
    """
    R6 Objective: Listwise Top-1 Classification (§16).
    If tau is None: Hard Oracle Cross Entropy: CE(s, k*)
    If tau is float: Soft Listwise Cross Entropy with target q_k = softmax(-(ADE_k - ADE_min)/tau)
    """
    def __init__(self, tau: float | None = None):
        super().__init__()
        self.tau = tau

    def forward(self, scores: torch.Tensor, ades: torch.Tensor) -> torch.Tensor:
        """
        scores: [B, K]
        ades:   [B, K]
        """
        oracle_idx = torch.argmin(ades, dim=-1)  # [B]
        if self.tau is None:
            return F.cross_entropy(scores, oracle_idx)

        # Soft listwise target
        ade_min = torch.min(ades, dim=-1, keepdim=True)[0]
        q = F.softmax(-(ades - ade_min) / self.tau, dim=-1)
        log_p = F.log_softmax(scores, dim=-1)
        return -(q * log_p).sum(dim=-1).mean()
