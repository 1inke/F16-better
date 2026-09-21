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
