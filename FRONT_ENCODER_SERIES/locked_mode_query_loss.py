"""Exact Q20 ModeQueryLoss without unrelated repository dependencies."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class WeightedTrajectoryLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        pred_len = math.ceil(cfg.PREDICTION.PRED_TIME * cfg.DATA.FREQUENCY)
        start, end = float(cfg.LOSS.START_WEIGHT), float(cfg.LOSS.END_WEIGHT)
        if cfg.LOSS.WEIGHTED_PENALTY == "exponential":
            weights = torch.exp(torch.linspace(math.log(start), math.log(end), pred_len))
        elif cfg.LOSS.WEIGHTED_PENALTY == "linear":
            weights = torch.linspace(start, end, pred_len)
        elif cfg.LOSS.WEIGHTED_PENALTY == "quadratic":
            weights = torch.linspace(start, end, pred_len).square()
        else:
            weights = torch.ones(pred_len)
        self.register_buffer("weights", weights)
        self.criterion = nn.SmoothL1Loss(reduction="none")

    def forward(self, prediction, target):
        loss = self.criterion(prediction, target).sum(dim=-1)
        return (loss * self.weights).mean()


class ModeQueryLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        decoder = cfg.MODEL.DECODER
        self.winner_fde_weight = float(decoder.WINNER_FDE_WEIGHT)
        self.classification_weight = float(decoder.CLASSIFICATION_WEIGHT)
        self.endpoint_weight = float(decoder.ENDPOINT_LOSS_WEIGHT)
        self.soft_classification = bool(decoder.SOFT_CLASSIFICATION)
        self.soft_temperature = float(decoder.SOFT_TEMPERATURE)
        self.hard_classification_weight = float(decoder.HARD_CLASSIFICATION_WEIGHT)
        self.diversity_weight = float(decoder.DIVERSITY_WEIGHT)
        self.diversity_margin = float(decoder.DIVERSITY_MARGIN)
        self.short_horizon_weight = float(decoder.SHORT_HORIZON_WEIGHT)
        self.short_horizon_steps = int(decoder.SHORT_HORIZON_STEPS)
        self.regression = WeightedTrajectoryLoss(cfg)

    def forward(self, predictions, target, logits, cfg=None, sample_weight=None):
        del cfg
        distances = torch.linalg.vector_norm(predictions - target.unsqueeze(-3), dim=-1)
        quality = distances.mean(dim=-1) + self.winner_fde_weight * distances[..., -1]
        winner = quality.detach().argmin(dim=-1)
        index = winner[..., None, None, None].expand(
            *winner.shape, 1, predictions.shape[-2], predictions.shape[-1]
        )
        winner_prediction = predictions.gather(-3, index).squeeze(-3)
        if sample_weight is None:
            regression = self.regression(winner_prediction, target)
        else:
            weights = sample_weight.to(
                device=predictions.device, dtype=predictions.dtype
            ).reshape(predictions.shape[0])
            weights = weights / weights.mean().clamp_min(1.0e-8)

            def weighted_batch_mean(values):
                per_sample = values.reshape(values.shape[0], -1).mean(dim=-1)
                return (per_sample * weights).mean()

            regression_values = self.regression.criterion(
                winner_prediction, target
            ).sum(dim=-1)
            regression_values = regression_values * self.regression.weights
            regression = weighted_batch_mean(regression_values)
        if self.soft_classification:
            gap = quality.detach() - quality.detach().min(dim=-1, keepdim=True).values
            scale = gap.mean(dim=-1, keepdim=True).clamp_min(0.05)
            soft_target = torch.softmax(-gap / (self.soft_temperature * scale), dim=-1)
            soft_values = -(soft_target * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
            hard_values = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), winner.reshape(-1),
                reduction="none",
            ).reshape_as(winner)
            if sample_weight is None:
                soft = soft_values.mean()
                hard = hard_values.mean()
            else:
                soft = weighted_batch_mean(soft_values)
                hard = weighted_batch_mean(hard_values)
            classification = soft + self.hard_classification_weight * hard
        else:
            classification_values = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), winner.reshape(-1),
                reduction="none",
            ).reshape_as(winner)
            classification = (
                classification_values.mean() if sample_weight is None
                else weighted_batch_mean(classification_values)
            )
        endpoint_values = F.smooth_l1_loss(
            winner_prediction[..., -1, :], target[..., -1, :], reduction="none"
        )
        endpoint = (
            endpoint_values.mean() if sample_weight is None
            else weighted_batch_mean(endpoint_values)
        )
        short_steps = min(max(self.short_horizon_steps, 1), predictions.shape[-2])
        short_values = F.smooth_l1_loss(
            winner_prediction[..., :short_steps, :], target[..., :short_steps, :],
            reduction="none",
        )
        short_horizon = (
            short_values.mean() if sample_weight is None
            else weighted_batch_mean(short_values)
        )
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
        total = (
            regression + self.classification_weight * classification
            + self.endpoint_weight * endpoint + self.diversity_weight * diversity
            + self.short_horizon_weight * short_horizon
        )
        return total, {
            "regression": regression.detach(), "classification": classification.detach(),
            "endpoint": endpoint.detach(), "diversity": diversity.detach(),
            "short_horizon": short_horizon.detach(), "winner": winner.detach(),
        }
