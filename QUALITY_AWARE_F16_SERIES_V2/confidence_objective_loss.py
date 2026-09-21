"""
Confidence Objective ModeQueryLoss for F16 Confidence Objective Refinement V2.

Strictly preserves all regression, endpoint, diversity, and short-horizon losses
from locked ModeQueryLoss, while enabling clean plug-in replacement of the
classification target and hard-winner weighting without auxiliary CE stacking.
"""
from __future__ import annotations

import os
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from FRONT_ENCODER_SERIES.locked_mode_query_loss import WeightedTrajectoryLoss
from QUALITY_AWARE_F16_SERIES_V2.common.confidence_targets import (
    build_original_target,
    build_fixed_quality_target,
    build_fixed_ade_target,
)


class ConfidenceObjectiveModeQueryLoss(nn.Module):
    """
    Plug-in drop-in replacement for ModeQueryLoss with customizable confidence objectives.
    """

    SUPPORTED_MODES = (
        "original_exact",
        "original_soft_only",
        "fixed_quality",
        "fixed_ade",
        "fixed_quality_soft_only",
    )

    def __init__(
        self,
        cfg,
        objective_mode: str = "original_exact",
        fixed_tau: float = 0.10,
        hard_weight: float = 0.25,
    ):
        super().__init__()
        decoder = cfg.MODEL.DECODER
        self.winner_fde_weight = float(decoder.WINNER_FDE_WEIGHT)
        self.classification_weight = float(decoder.CLASSIFICATION_WEIGHT)
        self.endpoint_weight = float(decoder.ENDPOINT_LOSS_WEIGHT)
        self.soft_temperature = float(decoder.SOFT_TEMPERATURE)
        self.hard_classification_weight = float(decoder.HARD_CLASSIFICATION_WEIGHT)
        self.diversity_weight = float(decoder.DIVERSITY_WEIGHT)
        self.diversity_margin = float(decoder.DIVERSITY_MARGIN)
        self.short_horizon_weight = float(decoder.SHORT_HORIZON_WEIGHT)
        self.short_horizon_steps = int(decoder.SHORT_HORIZON_STEPS)

        if objective_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unknown objective_mode '{objective_mode}'. Must be one of {self.SUPPORTED_MODES}"
            )

        self.objective_mode = objective_mode
        self.fixed_tau = float(fixed_tau)

        # Force exact locked weights for original_exact and soft_only
        if self.objective_mode == "original_exact":
            self.hard_weight = self.hard_classification_weight
        elif self.objective_mode in ("original_soft_only", "fixed_quality_soft_only"):
            self.hard_weight = 0.0
        else:
            self.hard_weight = float(hard_weight)

        self.regression = WeightedTrajectoryLoss(cfg)

    @classmethod
    def from_environment(cls, cfg, model=None) -> "ConfidenceObjectiveModeQueryLoss":
        """
        Instantiates loss from environment variables with safety locks.
        """
        mode = os.environ.get("QA_OBJECTIVE_MODE", "original_exact").strip()
        tau = float(os.environ.get("QA_FIXED_TAU", "0.10"))
        hard_w = float(os.environ.get("QA_HARD_WEIGHT", "0.25"))

        # Strict lock: when mode is original_exact, ignore tau and hard_w from env
        if mode == "original_exact":
            hard_w = float(cfg.MODEL.DECODER.HARD_CLASSIFICATION_WEIGHT)

        return cls(
            cfg=cfg,
            objective_mode=mode,
            fixed_tau=tau,
            hard_weight=hard_w,
        )

    def forward(
        self,
        predictions: Tensor,
        target: Tensor,
        logits: Tensor,
        cfg=None,
        sample_weight: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        del cfg

        # 1. Winner gathering for regression & endpoint
        # Use exact distance computation
        distances = torch.linalg.vector_norm(predictions - target.unsqueeze(-3), dim=-1)
        raw_quality = distances.mean(dim=-1) + self.winner_fde_weight * distances[..., -1]
        winner = raw_quality.detach().argmin(dim=-1)

        index = winner[..., None, None, None].expand(
            *winner.shape, 1, predictions.shape[-2], predictions.shape[-1]
        )
        winner_prediction = predictions.gather(-3, index).squeeze(-3)

        # Sample weight handling
        if sample_weight is not None:
            weights = sample_weight.to(
                device=predictions.device, dtype=predictions.dtype
            ).reshape(predictions.shape[0])
            weights = weights / weights.mean().clamp_min(1.0e-8)

            def weighted_batch_mean(values: Tensor) -> Tensor:
                per_sample = values.reshape(values.shape[0], -1).mean(dim=-1)
                return (per_sample * weights).mean()
        else:
            weights = None
            weighted_batch_mean = None

        # 2. Regression loss
        if sample_weight is None:
            regression = self.regression(winner_prediction, target)
        else:
            regression_values = self.regression.criterion(
                winner_prediction, target
            ).sum(dim=-1)
            regression_values = regression_values * self.regression.weights
            regression = weighted_batch_mean(regression_values)

        # 3. Confidence Target Construction
        if self.objective_mode in ("original_exact", "original_soft_only"):
            q, target_winner, _ = build_original_target(
                predictions=predictions,
                target=target,
                winner_fde_weight=self.winner_fde_weight,
                soft_temperature=self.soft_temperature,
            )
        elif self.objective_mode in ("fixed_quality", "fixed_quality_soft_only"):
            q, target_winner, _ = build_fixed_quality_target(
                predictions=predictions,
                target=target,
                tau=self.fixed_tau,
                fde_weight=self.winner_fde_weight,
            )
        elif self.objective_mode == "fixed_ade":
            q, target_winner, _ = build_fixed_ade_target(
                predictions=predictions,
                target=target,
                tau=self.fixed_tau,
            )
        else:
            raise ValueError(f"Unsupported mode: {self.objective_mode}")

        # 4. Classification loss (Soft CE + optional Hard CE)
        soft_values = -(q * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        if self.hard_weight > 0.0:
            hard_values = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_winner.reshape(-1),
                reduction="none",
            ).reshape_as(target_winner)
        else:
            hard_values = torch.zeros_like(soft_values)

        if sample_weight is None:
            soft = soft_values.mean()
            hard = hard_values.mean()
        else:
            soft = weighted_batch_mean(soft_values)
            hard = weighted_batch_mean(hard_values)

        classification = soft + self.hard_weight * hard

        # 5. Endpoint loss
        endpoint_values = F.smooth_l1_loss(
            winner_prediction[..., -1, :], target[..., -1, :], reduction="none"
        )
        endpoint = (
            endpoint_values.mean()
            if sample_weight is None
            else weighted_batch_mean(endpoint_values)
        )

        # 6. Short-horizon loss
        short_steps = min(max(self.short_horizon_steps, 1), predictions.shape[-2])
        short_values = F.smooth_l1_loss(
            winner_prediction[..., :short_steps, :],
            target[..., :short_steps, :],
            reduction="none",
        )
        short_horizon = (
            short_values.mean()
            if sample_weight is None
            else weighted_batch_mean(short_values)
        )

        # 7. Diversity loss
        endpoints = predictions[..., -1, :]
        pairwise = torch.cdist(endpoints, endpoints)
        upper = torch.triu(torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
        if upper.any():
            diversity_values = F.relu(self.diversity_margin - pairwise)
            if sample_weight is None:
                diversity = diversity_values[upper].mean()
            else:
                pair_count = upper.reshape(upper.shape[0], -1).sum(dim=-1).clamp_min(1)
                diversity_per_sample = (
                    diversity_values * upper
                ).reshape(upper.shape[0], -1).sum(dim=-1) / pair_count
                diversity = (diversity_per_sample * weights).mean()
        else:
            diversity = predictions.new_zeros(())

        # 8. Total loss
        total = (
            regression
            + self.classification_weight * classification
            + self.endpoint_weight * endpoint
            + self.diversity_weight * diversity
            + self.short_horizon_weight * short_horizon
        )

        return total, {
            "regression": regression.detach(),
            "classification": classification.detach(),
            "soft_classification": soft.detach(),
            "hard_classification": hard.detach(),
            "endpoint": endpoint.detach(),
            "diversity": diversity.detach(),
            "short_horizon": short_horizon.detach(),
            "winner": winner.detach(),
            "q_target": q.detach(),
        }
