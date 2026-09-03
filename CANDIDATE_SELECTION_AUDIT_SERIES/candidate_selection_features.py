"""Pre-registered 25-feature bank for EXP-186A candidate selection audit."""
from __future__ import annotations

import torch
from torch import Tensor


FEATURES = (
    "native_logit", "native_probability", "native_rank",
    "speed_mismatch", "heading_mismatch", "acceleration_mismatch",
    "turning_mismatch", "mean_jerk", "path_length_ratio", "endpoint_speed_ratio",
    "correction_t4", "correction_t8", "correction_t12", "correction_total",
    "correction_ratio", "anchor_curvature", "endpoint_centrality",
    "trajectory_centrality", "h3_centrality", "score_weighted_centrality",
    "nearest_candidate_distance", "endpoint_local_density", "trajectory_local_density",
    "mean_social_clearance", "collision_count",
)

# Every exported value is oriented so larger means a candidate is expected to be better.
HIGHER_IS_BETTER = {name: True for name in FEATURES}
SUPPLEMENTARY = {"mean_social_clearance", "collision_count"}
SCORE_FEATURES = {"native_logit", "native_probability", "native_rank"}
STORED_AS = {
    "native_logit": "native_logit",
    "native_probability": "softmax(native_logit)",
    "native_rank": "1=best_native_rank, 0=worst",
    "speed_mismatch": "-absolute_speed_mismatch",
    "heading_mismatch": "-absolute_heading_mismatch",
    "acceleration_mismatch": "-L2_acceleration_mismatch",
    "turning_mismatch": "-absolute_turning_mismatch",
    "mean_jerk": "-mean_jerk_norm",
    "path_length_ratio": "-abs(log(path_length_ratio))",
    "endpoint_speed_ratio": "-abs(log(endpoint_speed_ratio))",
    "correction_t4": "-V2_correction_norm_t4",
    "correction_t8": "-V2_correction_norm_t8",
    "correction_t12": "-V2_correction_norm_t12",
    "correction_total": "-sum_V2_correction_norm",
    "correction_ratio": "-correction_to_anchor_extent_ratio",
    "anchor_curvature": "-mean_anchor_turn_angle",
    "endpoint_centrality": "-mean_endpoint_pair_distance",
    "trajectory_centrality": "-mean_trajectory_pair_distance",
    "h3_centrality": "-weighted_H3_pair_distance",
    "score_weighted_centrality": "-native_probability_weighted_trajectory_distance",
    "nearest_candidate_distance": "-nearest_trajectory_distance",
    "endpoint_local_density": "Gaussian_endpoint_density",
    "trajectory_local_density": "Gaussian_trajectory_density",
    "mean_social_clearance": "CV_proxy_mean_clearance_supplementary",
    "collision_count": "-CV_proxy_collision_count_supplementary",
}


def _angle(a: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:
    cosine = (a * b).sum(-1) / (
        torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(b, dim=-1)
    ).clamp_min(eps)
    return torch.acos(cosine.clamp(-1.0, 1.0))


def candidate_feature_bank(
    prediction: Tensor,
    logits: Tensor,
    past: Tensor,
    num_valid: Tensor,
    source_anchors: Tensor,
    corrected_anchors: Tensor,
    sigma_endpoint: float,
    sigma_trajectory: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return directed [B,20,25] features and zero-parameter rule scores."""
    eps = 1e-6
    batch, modes, steps = prediction.shape[:3]
    focal = past[:, 0]
    last = focal[:, -1]
    observed_velocity = focal[:, -1] - focal[:, -2]
    previous_velocity = focal[:, -2] - focal[:, -3]
    observed_speed = torch.linalg.vector_norm(observed_velocity, dim=-1).clamp_min(eps)
    observed_acceleration = observed_velocity - previous_velocity
    observed_turn = _angle(previous_velocity, observed_velocity)

    full = torch.cat((last[:, None, None].expand(-1, modes, 1, -1), prediction), dim=2)
    velocity = torch.diff(full, dim=2)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    acceleration = torch.diff(velocity, dim=2)
    jerk = torch.diff(acceleration, dim=2)

    speed_mismatch = (speed[..., 0] - observed_speed[:, None]).abs()
    heading_mismatch = _angle(velocity[..., 0, :], observed_velocity[:, None])
    acceleration_mismatch = torch.linalg.vector_norm(acceleration[..., 0, :] - observed_acceleration[:, None], dim=-1)
    candidate_turn = _angle(velocity[..., 0, :], velocity[..., 1, :])
    turning_mismatch = (candidate_turn - observed_turn[:, None]).abs()
    mean_jerk = torch.linalg.vector_norm(jerk, dim=-1).mean(-1)
    length_ratio = speed.sum(-1) / (steps * observed_speed[:, None]).clamp_min(eps)
    endpoint_speed_ratio = speed[..., -1] / observed_speed[:, None]

    correction = torch.linalg.vector_norm(corrected_anchors - source_anchors, dim=-1)
    correction_total = correction.sum(-1)
    anchor_extent = torch.linalg.vector_norm(source_anchors - last[:, None, None], dim=-1).sum(-1)
    correction_ratio = correction_total / anchor_extent.clamp_min(eps)
    anchor_segments = torch.diff(
        torch.cat((last[:, None, None].expand(-1, modes, 1, -1), source_anchors), dim=2), dim=2
    )
    anchor_curvature = torch.stack(
        (_angle(anchor_segments[..., 0, :], anchor_segments[..., 1, :]),
         _angle(anchor_segments[..., 1, :], anchor_segments[..., 2, :])), dim=-1
    ).mean(-1)

    endpoint_distance = torch.cdist(prediction[..., -1, :], prediction[..., -1, :])
    trajectory_distance = torch.linalg.vector_norm(
        prediction[:, :, None] - prediction[:, None, :], dim=-1
    ).mean(-1)
    anchor_distance = torch.linalg.vector_norm(
        source_anchors[:, :, None] - source_anchors[:, None, :], dim=-1
    )
    h3_distance = (anchor_distance * torch.tensor((0.2, 0.3, 0.5), device=prediction.device)).sum(-1)
    denominator = max(1, modes - 1)
    endpoint_centrality = endpoint_distance.sum(-1) / denominator
    trajectory_centrality = trajectory_distance.sum(-1) / denominator
    h3_centrality = h3_distance.sum(-1) / denominator
    probability = torch.softmax(logits, dim=1)
    score_weighted_centrality = (trajectory_distance * probability[:, None]).sum(-1)
    h3_weighted_mbr = (h3_distance * probability[:, None]).sum(-1)
    eye = torch.eye(modes, dtype=torch.bool, device=prediction.device)[None]
    nearest = trajectory_distance.masked_fill(eye, float("inf")).min(-1).values
    endpoint_density = torch.exp(-endpoint_distance.square() / (2 * max(sigma_endpoint, eps) ** 2)).masked_fill(eye, 0).sum(-1)
    trajectory_density = torch.exp(-trajectory_distance.square() / (2 * max(sigma_trajectory, eps) ** 2)).masked_fill(eye, 0).sum(-1)

    if past.shape[1] > 1:
        other_last = past[:, 1:, -1]
        other_velocity = past[:, 1:, -1] - past[:, 1:, -2]
        timeline = torch.arange(1, steps + 1, dtype=past.dtype, device=past.device)
        # Constant velocity is only a supplementary social-clearance proxy here.
        other_future = other_last[:, :, None] + timeline[None, None, :, None] * other_velocity[:, :, None]
        distance = torch.linalg.vector_norm(prediction[:, :, None] - other_future[:, None], dim=-1)
        valid_other = torch.arange(past.shape[1] - 1, device=past.device)[None] < (num_valid - 1).clamp_min(0)[:, None]
        distance = distance.masked_fill(~valid_other[:, None, :, None], float("inf"))
        clearance_t = distance.min(dim=2).values
        no_other = ~valid_other.any(dim=1)
        mean_clearance = clearance_t.clamp_max(10.0).mean(-1)
        collision_count = ((distance < 0.20) & valid_other[:, None, :, None]).sum(dim=(2, 3)).float()
        mean_clearance[no_other] = 10.0
        collision_count[no_other] = 0.0
    else:
        mean_clearance = torch.full((batch, modes), 10.0, device=prediction.device)
        collision_count = torch.zeros_like(mean_clearance)

    order = torch.argsort(logits, dim=1, descending=True)
    rank = torch.empty_like(order, dtype=torch.float32)
    rank.scatter_(1, order, torch.arange(modes, device=logits.device, dtype=torch.float32)[None].expand(batch, -1))
    rank_score = 1.0 - rank / denominator
    closeness_length = -torch.log(length_ratio.clamp_min(eps)).abs()
    closeness_endpoint_speed = -torch.log(endpoint_speed_ratio.clamp_min(eps)).abs()
    values = (
        logits, probability, rank_score,
        -speed_mismatch, -heading_mismatch, -acceleration_mismatch, -turning_mismatch,
        -mean_jerk, closeness_length, closeness_endpoint_speed,
        -correction[..., 0], -correction[..., 1], -correction[..., 2], -correction_total,
        -correction_ratio, -anchor_curvature, -endpoint_centrality, -trajectory_centrality,
        -h3_centrality, -score_weighted_centrality, -nearest,
        endpoint_density, trajectory_density, mean_clearance, -collision_count,
    )
    feature = torch.stack(values, dim=-1)
    assert feature.shape[-1] == len(FEATURES)
    return feature, {
        "R1_medoid": -trajectory_centrality,
        "R2_weighted_MBR": -score_weighted_centrality,
        "R3_H3_MBR": -h3_weighted_mbr,
    }


def within_sample_z(values: Tensor, eps: float = 1e-6) -> Tensor:
    return (values - values.mean(dim=1, keepdim=True)) / values.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
