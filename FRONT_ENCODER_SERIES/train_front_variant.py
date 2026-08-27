"""Train one front-encoder ablation with the Q20 backend locked."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent


class TrajectoryDataset(Dataset):
    """Minimal no-scene ETH/UCY loader; avoids image-only dependencies."""

    def __init__(self, data_path: Path, num_path: Path, cfg) -> None:
        self.data = np.load(data_path, mmap_mode="r")
        self.num = np.load(num_path, mmap_mode="r")
        self.past_frames = int(cfg.DATA.FREQUENCY * cfg.PREDICTION.OBS_TIME)
        self.future_frames = int(math.ceil(cfg.DATA.FREQUENCY * cfg.PREDICTION.PRED_TIME))
        self.scale = float(cfg.PREDICTION.TRAJ_SCALE)
        self.max_agents = int(cfg.DATA.MAX_AGENTS)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        coordinates = np.array(
            self.data[index, : self.max_agents, :, -2:],
            dtype=np.float32,
            copy=True,
        ) / self.scale
        past = torch.from_numpy(coordinates[:, : self.past_frames])
        future = torch.from_numpy(coordinates[:, self.past_frames : self.past_frames + self.future_frames])
        num_valid = torch.tensor(
            min(int(self.num[index]), self.max_agents), dtype=torch.int64
        )
        empty = torch.empty(0, dtype=torch.float32)
        return past, future, num_valid, empty, empty, empty


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"))
    parser.add_argument("--config-file", default="configs/motif_rbf_univ.yaml")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--num-path", type=Path, default=None)
    parser.add_argument("--test-data-path", type=Path, default=None)
    parser.add_argument("--test-num-path", type=Path, default=None)
    parser.add_argument("--max-agents", type=int, default=None)
    parser.add_argument("--graph-top-k", type=int, default=None)
    parser.add_argument("--graph-radius", type=float, default=None)
    parser.add_argument("--variant", required=True, choices=[
        "f0_baseline", "f1_robust_state", "f2_temporal_queries",
        "f3_factorized", "f4_relation_tokens", "f5_dual_graph",
        "f6_light_factorized",
        "f7_ultralight_factorized",
        "f8_heading_aligned_relation",
        "f9_soft_risk_bias",
        "f10_multiscale_relation",
        "f11_motion_auxiliary",
        "f12_focal_conditioned",
        "f13_dynamic_routing",
        "f14_grouped_attention",
        "f15_edge_iteration",
        "f16_temporal_social_cross",
        "f17_sparsemax_attention",
        "f18_relation_qk",
        "f19_entmax15_attention",
        "f20_node_structure_encoding",
        "f21_agent_proxy_attention",
        "f22_temporal_cross_entmax15",
        "f23_dynamic_temporal_cross",
        "f24_temporal_dropkey",
        "f25_temporal_relative_bias",
        "f26_adaptive_dual_social",
        "f27_density_adaptive_social",
        "f28_temporal_landmark4",
        "f29_temporal_landmark2",
        "f30_graph_additive_attention",
        "f31_cascaded_group_attention",
        "f32_grouped_query_attention",
        "f33_multi_query_attention",
        "f34_identity_value_attention",
        "f35_temporal_half_pool",
        "f36_fixed_half_projection",
        "f37_lowdim_qk_attention",
        "f38_recent_exact_early_pool",
        "f39_motion_event_sampling",
        "f40_motion_token_squeeze",
        "f41_edge_list_attention",
        "f42_dual_scale_sparse",
        "f43_relation_type_bias",
        "f44_second_order_context",
        "f45_temporal_risk_bias",
        "f46_multiscale_temporal_residual",
        "f47_multiscale_temporal_replace",
        "f48_ttc_dca_graph",
        "f49_ttc_dca_multiscale_replace",
        "f50_enhanced_kinematic_state",
        "f51_scaled_cosine_attention",
        "f52_pair_deformable_temporal",
        "f53_kdta_recent_exact",
        "f54_kdta_trend_event_recent",
        "f55_pair_temporal_soft_prior",
        "f56_motion_adaptive_key_smoothing",
        "f57_relation_grouped_vector_attention",
        "f58_compute_neutral_differential",
    ])
    parser.add_argument("--decoder-experiment", default="q20")
    parser.add_argument("--decoder-module", type=Path, default=None)
    parser.add_argument("--encoder-experiment", default="none")
    parser.add_argument("--encoder-module", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val-scene-id", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--validation-seed", type=int, default=20260824)
    parser.add_argument(
        "--validation-split-mode",
        choices=["stratified_random", "scene_tail", "motion_challenge"],
        default="stratified_random",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("results/front_encoder_series"))
    parser.add_argument("--verifier-root", type=Path, default=None)
    parser.add_argument("--fusion-alpha", type=float, default=0.30)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--scheduler", choices=["warm_restarts", "monotonic_cosine", "constant"],
        default="warm_restarts",
    )
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=None)
    parser.add_argument("--aux-motion-weight", type=float, default=0.0)
    parser.add_argument("--oracle-coverage-weight", type=float, default=0.0)
    parser.add_argument("--oracle-coverage-temperature", type=float, default=0.10)
    parser.add_argument("--oracle-coverage-fde-weight", type=float, default=0.25)
    parser.add_argument("--oracle-coverage-warmup-epochs", type=float, default=0.0)
    parser.add_argument("--oracle-coverage-normalized", action="store_true")
    parser.add_argument(
        "--scene-sampling", choices=["none", "sqrt_scene", "uniform_scene"],
        default="none",
    )
    parser.add_argument("--motion-difficulty-sampling", action="store_true")
    parser.add_argument("--motion-difficulty-strength", type=float, default=0.8)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--augment-rotation", action="store_true")
    parser.add_argument("--augment-reflection", action="store_true")
    parser.add_argument("--augment-speed-min", type=float, default=1.0)
    parser.add_argument("--augment-speed-max", type=float, default=1.0)
    parser.add_argument("--augment-probability", type=float, default=1.0)
    parser.add_argument("--augment-curvature-max", type=float, default=0.0)
    parser.add_argument("--future-difficulty-weight", type=float, default=0.0)
    parser.add_argument("--focal-heading-normalize", action="store_true")
    parser.add_argument("--velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--acceleration-loss-weight", type=float, default=0.0)
    parser.add_argument("--masked-history-weight", type=float, default=0.0)
    parser.add_argument("--masked-history-probability", type=float, default=0.15)
    parser.add_argument("--equivariance-weight", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-criterion", choices=["original_ade", "oracle_combo"],
        default="original_ade",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


class ModelEMA:
    """Training-only exponential moving average; no inference architecture cost."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = model.state_dict()
        for name, target in self.module.state_dict().items():
            value = source[name].detach()
            if target.is_floating_point():
                target.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                target.copy_(value)


def augment_trajectories(
    past: torch.Tensor, future: torch.Tensor, args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a shared rigid transform and mild per-trajectory speed scaling."""
    batch = past.shape[0]
    uses_augmentation = (
        args.augment_rotation or args.augment_reflection
        or args.augment_speed_min != 1.0 or args.augment_speed_max != 1.0
        or args.augment_curvature_max != 0.0
    )
    if not uses_augmentation:
        return past, future
    probability = min(max(float(args.augment_probability), 0.0), 1.0)
    active = torch.rand(batch, device=past.device) < probability
    if args.augment_rotation or args.augment_reflection:
        origin = past.mean(dim=(1, 2), keepdim=True)
        if args.augment_rotation:
            angle = torch.rand(batch, device=past.device) * (2.0 * math.pi)
            angle = angle * active.to(angle.dtype)
        else:
            angle = torch.zeros(batch, device=past.device)
        cosine, sine = angle.cos(), angle.sin()
        matrix = torch.stack(
            (cosine, -sine, sine, cosine), dim=-1
        ).reshape(batch, 1, 1, 2, 2)
        if args.augment_reflection:
            reflect = (torch.rand(batch, device=past.device) < 0.5) & active
            sign = torch.where(reflect, -1.0, 1.0)
            matrix[..., 0] *= sign[:, None, None, None]
        past = torch.matmul((past - origin).unsqueeze(-2), matrix).squeeze(-2) + origin
        future = torch.matmul((future - origin).unsqueeze(-2), matrix).squeeze(-2) + origin
    if args.augment_speed_min != 1.0 or args.augment_speed_max != 1.0:
        low, high = float(args.augment_speed_min), float(args.augment_speed_max)
        sampled = torch.empty(batch, 1, 1, 1, device=past.device).uniform_(low, high)
        scale = torch.where(active[:, None, None, None], sampled, torch.ones_like(sampled))
        anchor = past[:, :, -1:].clone()
        past = anchor + scale * (past - anchor)
        future = anchor + scale * (future - anchor)
    if args.augment_curvature_max != 0.0:
        maximum = abs(float(args.augment_curvature_max))
        curvature = torch.empty(batch, device=past.device).uniform_(-maximum, maximum)
        curvature = curvature * active.to(curvature.dtype)
        anchor = past[:, :1, -1:].clone()

        def bend(coordinates: torch.Tensor, relative_time: torch.Tensor) -> torch.Tensor:
            angle = curvature[:, None, None] * relative_time[None, None]
            cosine, sine = angle.cos(), angle.sin()
            matrix = torch.stack(
                (cosine, -sine, sine, cosine), dim=-1
            ).reshape(batch, 1, coordinates.shape[2], 2, 2)
            return (
                torch.matmul((coordinates - anchor).unsqueeze(-2), matrix)
                .squeeze(-2) + anchor
            )

        past_time = torch.arange(
            1 - past.shape[2], 1, device=past.device, dtype=past.dtype
        )
        future_time = torch.arange(
            1, future.shape[2] + 1, device=future.device, dtype=future.dtype
        )
        past = bend(past, past_time)
        future = bend(future, future_time)
    return past, future


def future_motion_difficulty_weights(
    past: torch.Tensor, future: torch.Tensor, strength: float
) -> torch.Tensor:
    """Training-only GT difficulty weights; future is never a model input."""
    focal_past = past[:, 0]
    focal_future = future[:, 0]
    last_step = focal_past[:, -1] - focal_past[:, -2]
    path = torch.cat((focal_past[:, -1:], focal_future), dim=1)
    velocity = path[:, 1:] - path[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    motion_scale = torch.linalg.vector_norm(last_step, dim=-1).clamp_min(0.02)
    acceleration_ratio = torch.linalg.vector_norm(
        acceleration, dim=-1
    ).mean(dim=-1) / motion_scale
    previous = velocity[:, :-1]
    current = velocity[:, 1:]
    cross = previous[..., 0] * current[..., 1] - previous[..., 1] * current[..., 0]
    dot = (previous * current).sum(dim=-1)
    turn = torch.atan2(cross.abs(), dot).mean(dim=-1) / math.pi
    steps = focal_future.shape[1]
    cv_endpoint = focal_past[:, -1] + float(steps) * last_step
    cv_deviation = torch.linalg.vector_norm(
        focal_future[:, -1] - cv_endpoint, dim=-1
    ) / (float(steps) * motion_scale + 0.10)
    difficulty = (
        0.40 * torch.tanh(acceleration_ratio)
        + 0.30 * turn.clamp(0.0, 1.0)
        + 0.30 * torch.tanh(cv_deviation)
    ).detach()
    weights = torch.exp(float(strength) * (difficulty - 0.5))
    weights = weights.clamp(0.70, 1.50)
    return weights / weights.mean().clamp_min(1.0e-8)


def normalize_to_focal_heading(
    past: torch.Tensor, future: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the complete scene into the focal agent's smoothed heading frame."""
    recent_lag = min(3, past.shape[-2] - 1)
    displacement = past[:, 0, -1] - past[:, 0, -1 - recent_lag]
    speed = torch.linalg.vector_norm(displacement, dim=-1)
    default = torch.zeros_like(displacement)
    default[:, 0] = 1.0
    heading = torch.where(
        (speed > 0.03)[:, None], displacement / speed[:, None].clamp_min(1.0e-6), default
    )
    cosine, sine = heading[:, 0], heading[:, 1]
    # Row-vector rotation by -heading: [x*c+y*s, -x*s+y*c].
    matrix = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        past.shape[0], 1, 1, 2, 2
    )
    origin = past[:, :1, -1:]
    past_local = torch.matmul((past - origin).unsqueeze(-2), matrix).squeeze(-2)
    future_local = torch.matmul((future - origin).unsqueeze(-2), matrix).squeeze(-2)
    return past_local, future_local


def mirror_lateral(values: torch.Tensor) -> torch.Tensor:
    """Reflect heading-normalized trajectories across the forward axis."""
    sign = values.new_tensor((1.0, -1.0))
    return values * sign


def symmetric_set_consistency(
    reference: torch.Tensor, candidate: torch.Tensor
) -> torch.Tensor:
    """Permutation-free bidirectional trajectory-set matching loss."""
    pairwise = torch.linalg.vector_norm(
        reference.detach()[:, :, None] - candidate[:, None, :], dim=-1
    ).mean(dim=-1)
    reference_to_candidate = pairwise.min(dim=-1).values.mean()
    candidate_to_reference = pairwise.min(dim=-2).values.mean()
    return 0.5 * (reference_to_candidate + candidate_to_reference)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def zscore(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.maximum(std, 1.0e-6)


def selection_metrics(ade: np.ndarray, fde: np.ndarray, selection: np.ndarray) -> dict:
    rows = np.arange(len(ade))
    selected_ade = ade[rows, selection]
    selected_fde = fde[rows, selection]
    best = ade.argmin(axis=1)
    regret = selected_ade - ade.min(axis=1)
    return {
        "ade": float(selected_ade.mean()),
        "fde": float(selected_fde.mean()),
        "hit_best_ade": float((selection == best).mean()),
        "regret": float(regret.mean()),
        "regret_p95": float(np.quantile(regret, 0.95)),
        "active_modes": int(np.unique(selection).size),
    }


def decoder_fingerprint(model: torch.nn.Module) -> str:
    state = model.decoder.state_dict()
    description = [(name, tuple(tensor.shape)) for name, tensor in state.items()]
    return hashlib.sha256(repr(description).encode("utf-8")).hexdigest()[:16]


def load_verifier(path: Path, device: torch.device):
    sys.path.insert(0, str(WORKSPACE))
    from train_social_stimulus_verifier import SocialStimulusVerifier, VerifierConfig, rotate_to_heading
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SocialStimulusVerifier(VerifierConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), rotate_to_heading


def evaluate(
    model, loader, device, verifier_info=None, alpha: float = 0.30,
    focal_heading_normalize: bool = False,
) -> dict:
    model.eval()
    ade_parts, fde_parts, logits_parts, verifier_parts = [], [], [], []
    coarse_ade_parts, coarse_fde_parts, selection_recall_parts = [], [], []
    anchor_correction_parts, progress_knot_parts = [], []
    mode_groups = None
    verifier, rotate = verifier_info if verifier_info else (None, None)
    with torch.inference_mode():
        for past, future, num_valid, _, _, _ in loader:
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            if focal_heading_normalize:
                past, future = normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            prediction = model(
                past - center, None, None, mode="testing",
                num_valid=num_valid, graph_loc=past,
            )[2] + center.unsqueeze(-3)
            candidates = prediction[:, 0]
            target = future[:, 0]
            errors = torch.linalg.vector_norm(candidates - target[:, None], dim=-1)
            ade_parts.append(errors.mean(dim=-1).cpu().numpy())
            fde_parts.append(errors[..., -1].cpu().numpy())
            logits_parts.append(model.last_mode_logits[:, 0].cpu().numpy())
            coarse = getattr(model.decoder, "last_coarse_trajectories", None)
            selected = getattr(model.decoder, "last_selected_indices", None)
            source_anchors = getattr(model.decoder, "last_anchor_proposals", None)
            corrected_anchors = getattr(model.decoder, "last_corrected_anchors", None)
            current_mode_groups = getattr(model.decoder, "mode_groups", None)
            if current_mode_groups is not None:
                mode_groups = current_mode_groups.detach().cpu().numpy()
            progress_knots = getattr(model.decoder, "last_progress_knots", None)
            if source_anchors is not None and corrected_anchors is not None:
                anchor_correction = torch.linalg.vector_norm(
                    corrected_anchors[:, 0] - source_anchors[:, 0], dim=-1
                )
                anchor_correction_parts.append(anchor_correction.cpu().numpy())
            if progress_knots is not None:
                progress_knot_parts.append(progress_knots[:, 0].cpu().numpy())
            if coarse is not None:
                coarse_absolute = coarse[:, 0] + center[:, 0, None]
                coarse_errors = torch.linalg.vector_norm(
                    coarse_absolute - target[:, None], dim=-1
                )
                coarse_ade = coarse_errors.mean(dim=-1)
                coarse_fde = coarse_errors[..., -1]
                coarse_ade_parts.append(coarse_ade.cpu().numpy())
                coarse_fde_parts.append(coarse_fde.cpu().numpy())
                if selected is not None:
                    oracle_mode = coarse_ade.argmin(dim=-1)
                    focal_selected = selected[:, 0]
                    recalled = (focal_selected == oracle_mode[:, None]).any(dim=-1)
                    selection_recall_parts.append(recalled.float().cpu().numpy())
            if verifier is not None:
                distribution = verifier.distribution(past, num_valid)
                centered = candidates - distribution["origin"][:, None, None, :]
                local = rotate(centered, distribution["heading"])
                verifier_parts.append(verifier.score_candidates(distribution, local).cpu().numpy())
    ade = np.concatenate(ade_parts)
    fde = np.concatenate(fde_parts)
    logits = np.concatenate(logits_parts)
    original = selection_metrics(ade, fde, logits.argmax(axis=1))
    rows = np.arange(len(ade))
    best_ade = ade.argmin(axis=1)
    result = {
        "samples": int(len(ade)),
        "original": original,
        "oracle": {
            "min_ade_20": float(ade.min(axis=1).mean()),
            "same_path_fde": float(fde[rows, best_ade].mean()),
            "min_fde_20": float(fde.min(axis=1).mean()),
        },
        "fusion_alpha": float(alpha),
        "fused": None,
    }
    if mode_groups is not None:
        selected = logits.argmax(axis=1)
        oracle = ade.argmin(axis=1)
        selected_group = mode_groups[selected]
        oracle_group = mode_groups[oracle]
        group_best = np.empty(len(ade), dtype=np.float32)
        for row in range(len(ade)):
            members = mode_groups == selected_group[row]
            group_best[row] = ade[row, members].min()
        result["mode_group_diagnostics"] = {
            "group_hit_oracle_ade": float((selected_group == oracle_group).mean()),
            "cross_group_error": float((selected_group != oracle_group).mean()),
            "group_regret": float((group_best - ade.min(axis=1)).mean()),
            "selected_group_usage": np.bincount(
                selected_group, minlength=int(mode_groups.max()) + 1
            ).tolist(),
        }
    if verifier_parts:
        scores = np.concatenate(verifier_parts)
        fused = (1.0 - alpha) * zscore(logits) + alpha * zscore(scores)
        result["fused"] = selection_metrics(ade, fde, fused.argmax(axis=1))
    if coarse_ade_parts:
        coarse_ade = np.concatenate(coarse_ade_parts)
        coarse_fde = np.concatenate(coarse_fde_parts)
        coarse_best = coarse_ade.argmin(axis=1)
        coarse_rows = np.arange(len(coarse_ade))
        diagnostics = {
            "coarse_min_ade_20": float(coarse_ade.min(axis=1).mean()),
            "coarse_same_path_fde": float(
                coarse_fde[coarse_rows, coarse_best].mean()
            ),
            "coarse_min_fde_20": float(coarse_fde.min(axis=1).mean()),
            "refinement_min_ade_delta": float(
                ade.min(axis=1).mean() - coarse_ade.min(axis=1).mean()
            ),
            "refinement_min_fde_delta": float(
                fde.min(axis=1).mean() - coarse_fde.min(axis=1).mean()
            ),
        }
        if selection_recall_parts:
            diagnostics["top_m_coarse_oracle_ade_recall"] = float(
                np.concatenate(selection_recall_parts).mean()
            )
        if anchor_correction_parts:
            correction = np.concatenate(anchor_correction_parts, axis=0)
            diagnostics["anchor_correction_mean"] = correction.mean(axis=(0, 1)).tolist()
            diagnostics["anchor_correction_p95"] = np.percentile(
                correction, 95, axis=(0, 1)
            ).tolist()
            diagnostics["anchor_correction_max"] = correction.max(axis=(0, 1)).tolist()
        if progress_knot_parts:
            knots = np.concatenate(progress_knot_parts, axis=0)
            diagnostics["progress_knot_mean"] = knots.mean(axis=(0, 1)).tolist()
            diagnostics["progress_knot_p05"] = np.percentile(
                knots, 5, axis=(0, 1)
            ).tolist()
            diagnostics["progress_knot_p95"] = np.percentile(
                knots, 95, axis=(0, 1)
            ).tolist()
        result["proposal_diagnostics"] = diagnostics
    return result


def run() -> None:
    args = arguments()
    project = args.project_root.resolve()
    output_root = args.output_root if args.output_root.is_absolute() else project / args.output_root
    run_dir = output_root / args.variant / f"seed_{args.seed}" / f"val_scene_{args.val_scene_id}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file() and not args.force:
        print(f"[skip] {metrics_path}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(project), str(project / "scripts"), str(HERE)]
    os.chdir(project)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from locked_mode_query_loss import ModeQueryLoss
    from front_encoder_models import FrontEncoderASTRA

    seed_everything(args.seed)
    device = select_device(args.device)
    cfg_path = Path(args.config_file)
    if not cfg_path.is_absolute():
        cfg_path = project / cfg_path
    cfg = load_locked_q20(cfg_path)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.TRAIN.SEED = args.seed
    cfg.TRAIN.NUM_EPOCH = args.epochs
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.VAL.BATCH_SIZE = args.batch_size
    cfg.VAL.FREQ = args.val_freq
    if args.max_agents is not None:
        cfg.DATA.MAX_AGENTS = int(args.max_agents)
    if args.graph_top_k is not None:
        cfg.MODEL.GRAPH.TOP_K = int(args.graph_top_k)
    if args.graph_radius is not None:
        cfg.MODEL.GRAPH.RADIUS = float(args.graph_radius)
    if args.classification_weight is not None:
        cfg.MODEL.DECODER.CLASSIFICATION_WEIGHT = float(args.classification_weight)
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]

    data_path = (
        args.data_path.resolve()
        if args.data_path is not None
        else project / "datasets/eth_ucy/processed_data_diverse" / f"{cfg.SUBSET}_data_train.npy"
    )
    num_path = (
        args.num_path.resolve()
        if args.num_path is not None
        else project / "datasets/eth_ucy/processed_data_diverse" / f"{cfg.SUBSET}_num_train.npy"
    )
    if not data_path.is_file() or not num_path.is_file():
        raise FileNotFoundError(f"trajectory data missing: {data_path} / {num_path}")
    raw = np.load(data_path, mmap_mode="r")
    scene_ids = np.asarray(raw[:, 0, 0, 0], dtype=np.int64)
    unique, counts = np.unique(scene_ids, return_counts=True)
    if args.validation_fraction > 0.0:
        if not 0.0 < args.validation_fraction < 0.5:
            raise ValueError("--validation-fraction must be in (0, 0.5)")
        validation_rng = np.random.default_rng(args.validation_seed)
        challenge_score = None
        challenge_components = None
        if args.validation_split_mode == "motion_challenge":
            raw_num = np.asarray(np.load(num_path, mmap_mode="r"), dtype=np.int64)
            past_frames = int(cfg.DATA.FREQUENCY * cfg.PREDICTION.OBS_TIME)
            focal = np.asarray(
                raw[:, 0, :past_frames, -2:], dtype=np.float32
            ) / float(cfg.PREDICTION.TRAJ_SCALE)
            steps = np.diff(focal, axis=1)
            step_size = np.linalg.norm(steps, axis=-1)
            speed = np.median(step_size[:, -min(3, step_size.shape[1]):], axis=1)
            acceleration = np.linalg.norm(np.diff(steps, axis=1), axis=-1).mean(axis=1)
            density = np.minimum(raw_num, int(cfg.DATA.MAX_AGENTS)).astype(np.float32)

            def percentile_rank(values: np.ndarray) -> np.ndarray:
                order = np.argsort(values, kind="stable")
                ranks = np.empty(len(values), dtype=np.float32)
                ranks[order] = np.arange(len(values), dtype=np.float32)
                return ranks / max(len(values) - 1, 1)

            speed_rank = percentile_rank(speed)
            acceleration_rank = percentile_rank(acceleration)
            low_density = 1.0 - (density - 1.0) / max(int(cfg.DATA.MAX_AGENTS) - 1, 1)
            challenge_score = speed_rank + acceleration_rank + low_density
            challenge_components = {
                "speed_median_step": speed,
                "nonlinearity_mean_step_change": acceleration,
                "valid_agents": density,
                "challenge_score": challenge_score,
            }
        validation_parts = []
        for scene in unique:
            candidates = np.flatnonzero(scene_ids == scene)
            if len(candidates) < 2:
                raise ValueError(f"scene {int(scene)} has fewer than two training samples")
            validation_count = max(
                1, min(len(candidates) - 1, int(round(len(candidates) * args.validation_fraction)))
            )
            if args.validation_split_mode == "scene_tail":
                validation_parts.append(candidates[-validation_count:])
            elif args.validation_split_mode == "motion_challenge":
                ranked = candidates[np.argsort(challenge_score[candidates], kind="stable")]
                validation_parts.append(ranked[-validation_count:])
            else:
                shuffled = validation_rng.permutation(candidates)
                validation_parts.append(shuffled[:validation_count])
        val_indices = np.sort(np.concatenate(validation_parts))
        validation_mask = np.zeros(len(scene_ids), dtype=bool)
        validation_mask[val_indices] = True
        train_indices = np.flatnonzero(~validation_mask)
        split = {
            "strategy": f"{args.validation_split_mode}_holdout_within_training_scenes",
            "validation_fraction": float(args.validation_fraction),
            "validation_seed": int(args.validation_seed),
            "scene_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
            "train_samples": int(len(train_indices)),
            "validation_samples": int(len(val_indices)),
            "target_test_accessed": False,
            "data_path": str(data_path),
            "num_path": str(num_path),
        }
        if challenge_components is not None:
            split["challenge_definition"] = (
                "Within each source scene, hold out the highest combined percentile "
                "of recent speed, motion nonlinearity, and low agent density."
            )
            split["challenge_train_means"] = {
                name: float(np.mean(values[train_indices]))
                for name, values in challenge_components.items()
            }
            split["challenge_validation_means"] = {
                name: float(np.mean(values[val_indices]))
                for name, values in challenge_components.items()
            }
    else:
        if args.val_scene_id not in unique or len(unique) < 2:
            raise ValueError(f"val scene {args.val_scene_id} invalid; available={unique.tolist()}")
        train_indices = np.flatnonzero(scene_ids != args.val_scene_id)
        val_indices = np.flatnonzero(scene_ids == args.val_scene_id)
        split = {
            "strategy": "leave_one_complete_training_scene_out",
            "validation_scene_id": args.val_scene_id,
            "scene_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
            "train_samples": int(len(train_indices)),
            "validation_samples": int(len(val_indices)),
            "target_test_accessed": False,
            "data_path": str(data_path),
            "num_path": str(num_path),
        }
    (run_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "split_indices.npz", train=train_indices, validation=val_indices)

    base_dataset = TrajectoryDataset(data_path, num_path, cfg)
    generator = torch.Generator().manual_seed(args.seed)
    train_sampler = None
    sample_weights = torch.ones(len(train_indices), dtype=torch.double)
    uses_weighted_sampling = False
    if args.scene_sampling != "none":
        train_scene_ids = scene_ids[train_indices]
        train_unique, train_counts = np.unique(train_scene_ids, return_counts=True)
        count_map = {int(scene): int(count) for scene, count in zip(train_unique, train_counts)}
        exponent = -0.5 if args.scene_sampling == "sqrt_scene" else -1.0
        sample_weights *= torch.as_tensor(
            [count_map[int(scene)] ** exponent for scene in train_scene_ids],
            dtype=torch.double,
        )
        uses_weighted_sampling = True
    if args.motion_difficulty_sampling:
        if args.motion_difficulty_strength < 0.0:
            raise ValueError("--motion-difficulty-strength must be non-negative")
        past_frames = int(cfg.DATA.FREQUENCY * cfg.PREDICTION.OBS_TIME)
        focal = np.asarray(
            raw[:, 0, :past_frames, -2:], dtype=np.float32
        ) / float(cfg.PREDICTION.TRAJ_SCALE)
        steps = np.diff(focal, axis=1)
        step_size = np.linalg.norm(steps, axis=-1)
        speed = np.median(step_size[:, -min(3, step_size.shape[1]):], axis=1)
        nonlinearity = np.linalg.norm(np.diff(steps, axis=1), axis=-1).mean(axis=1)
        raw_num = np.asarray(np.load(num_path, mmap_mode="r"), dtype=np.int64)
        density = np.minimum(raw_num, int(cfg.DATA.MAX_AGENTS)).astype(np.float32)
        training_scene_ids = scene_ids[train_indices]
        training_difficulty = np.zeros(len(train_indices), dtype=np.float32)

        def within_group_rank(values: np.ndarray) -> np.ndarray:
            order = np.argsort(values, kind="stable")
            ranks = np.empty(len(values), dtype=np.float32)
            ranks[order] = np.arange(len(values), dtype=np.float32)
            return ranks / max(len(values) - 1, 1)

        for scene in np.unique(training_scene_ids):
            local = np.flatnonzero(training_scene_ids == scene)
            original = train_indices[local]
            speed_rank = within_group_rank(speed[original])
            nonlinear_rank = within_group_rank(nonlinearity[original])
            low_density = 1.0 - (
                density[original] - 1.0
            ) / max(int(cfg.DATA.MAX_AGENTS) - 1, 1)
            training_difficulty[local] = (
                speed_rank + nonlinear_rank + low_density
            ) / 3.0
        centered = training_difficulty - float(training_difficulty.mean())
        difficulty_weights = np.exp(
            float(args.motion_difficulty_strength) * centered
        )
        difficulty_weights = np.clip(difficulty_weights, 0.60, 1.70)
        difficulty_weights /= float(difficulty_weights.mean())
        sample_weights *= torch.as_tensor(difficulty_weights, dtype=torch.double)
        uses_weighted_sampling = True
        split["motion_difficulty_sampling"] = {
            "definition": (
                "Within-scene mean rank of recent speed, motion nonlinearity, "
                "and low density; exponential weights clipped to [0.60, 1.70]."
            ),
            "strength": float(args.motion_difficulty_strength),
            "difficulty_mean": float(training_difficulty.mean()),
            "difficulty_p90": float(np.quantile(training_difficulty, 0.90)),
            "weight_min": float(difficulty_weights.min()),
            "weight_max": float(difficulty_weights.max()),
        }
        (run_dir / "split.json").write_text(
            json.dumps(split, indent=2), encoding="utf-8"
        )
    if uses_weighted_sampling:
        sample_weights /= sample_weights.mean().clamp_min(1.0e-12)
        train_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_indices), replacement=True,
            generator=generator,
        )
    train_loader = DataLoader(
        Subset(base_dataset, train_indices.tolist()), batch_size=args.batch_size,
        shuffle=train_sampler is None, sampler=train_sampler, generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )
    val_loader = DataLoader(
        Subset(base_dataset, val_indices.tolist()), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )
    test_loader = None
    if (args.test_data_path is None) != (args.test_num_path is None):
        raise ValueError("--test-data-path and --test-num-path must be provided together")
    if args.test_data_path is not None:
        test_data_path = args.test_data_path.resolve()
        test_num_path = args.test_num_path.resolve()
        if not test_data_path.is_file() or not test_num_path.is_file():
            raise FileNotFoundError(
                f"test trajectory data missing: {test_data_path} / {test_num_path}"
            )
        test_dataset = TrajectoryDataset(test_data_path, test_num_path, cfg)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            drop_last=False,
        )
        split["test_samples"] = int(len(test_dataset))
        split["test_data_path"] = str(test_data_path)
        split["test_num_path"] = str(test_num_path)
        split["test_policy"] = "held_out_until_after_validation_checkpoint_selection"
        (run_dir / "split.json").write_text(
            json.dumps(split, indent=2), encoding="utf-8"
        )

    model = FrontEncoderASTRA(cfg, args.variant)
    if args.encoder_experiment != "none":
        if args.encoder_module is None:
            raise ValueError("--encoder-module is required for a custom encoder")
        encoder_path = args.encoder_module.resolve()
        encoder_spec = importlib.util.spec_from_file_location(
            "experiment_encoder_models", encoder_path
        )
        if encoder_spec is None or encoder_spec.loader is None:
            raise ImportError(f"Cannot load encoder module: {encoder_path}")
        encoder_module = importlib.util.module_from_spec(encoder_spec)
        encoder_spec.loader.exec_module(encoder_module)
        encoder_module.replace_encoder(model, cfg, args.encoder_experiment)
    if args.decoder_experiment != "q20":
        if args.decoder_module is None:
            raise ValueError("--decoder-module is required for a custom decoder")
        module_path = args.decoder_module.resolve()
        spec = importlib.util.spec_from_file_location("experiment_decoder_models", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load decoder module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.replace_decoder(model, cfg, args.decoder_experiment)
    model = model.to(device)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0.0 else None
    loss_fn = ModeQueryLoss(cfg).to(device)
    history_reconstruction_head = None
    trainable_parameters = list(model.parameters())
    if args.masked_history_weight > 0.0:
        history_reconstruction_head = torch.nn.Linear(model.token_dim, 2).to(device)
        trainable_parameters += list(history_reconstruction_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=float(cfg.TRAIN.LR), weight_decay=float(cfg.TRAIN.W_DECAY)
    )
    if args.scheduler == "warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, eta_min=float(cfg.TRAIN.MIN_LR)
        )
    elif args.scheduler == "monotonic_cosine":
        total_steps = max(args.epochs * len(train_loader), 1)
        warmup_steps = max(int(args.warmup_epochs * len(train_loader)), 1)

        def lr_multiplier(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = min(
                float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1)),
                1.0,
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    else:
        scheduler = None
    parameter_counts = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "decoder": sum(p.numel() for p in model.decoder.parameters()),
    }
    contract = {
        "decoder_class": type(model.decoder).__name__,
        "decoder_fingerprint": decoder_fingerprint(model),
        "num_modes": int(getattr(model.decoder, "num_modes", cfg.MODEL.DECODER.NUM_MODES)),
        "decoder_experiment": args.decoder_experiment,
        "encoder_experiment": args.encoder_experiment,
        "classification_weight": float(cfg.MODEL.DECODER.CLASSIFICATION_WEIGHT),
        "loss_class": type(loss_fn).__name__,
        "fusion_alpha": args.fusion_alpha,
        "oracle_coverage_weight": args.oracle_coverage_weight,
        "oracle_coverage_temperature": args.oracle_coverage_temperature,
        "oracle_coverage_fde_weight": args.oracle_coverage_fde_weight,
        "oracle_coverage_warmup_epochs": args.oracle_coverage_warmup_epochs,
        "oracle_coverage_normalized": args.oracle_coverage_normalized,
        "scene_sampling": args.scene_sampling,
        "motion_difficulty_sampling": args.motion_difficulty_sampling,
        "motion_difficulty_strength": args.motion_difficulty_strength,
        "ema_decay": args.ema_decay,
        "augmentation": {
            "rotation": args.augment_rotation,
            "reflection": args.augment_reflection,
            "speed_min": args.augment_speed_min,
            "speed_max": args.augment_speed_max,
            "probability": args.augment_probability,
            "curvature_max": args.augment_curvature_max,
        },
        "future_difficulty_weight": args.future_difficulty_weight,
        "focal_heading_normalize": args.focal_heading_normalize,
        "velocity_loss_weight": args.velocity_loss_weight,
        "acceleration_loss_weight": args.acceleration_loss_weight,
        "masked_history_weight": args.masked_history_weight,
        "masked_history_probability": args.masked_history_probability,
        "equivariance_weight": args.equivariance_weight,
        "training_only_auxiliary_parameters": (
            sum(p.numel() for p in history_reconstruction_head.parameters())
            if history_reconstruction_head is not None else 0
        ),
        "checkpoint_criterion": args.checkpoint_criterion,
        "hierarchical_decoder": {
            "cv_reference": False,
            "anchor_steps": list(getattr(model.decoder, "anchor_steps", ())),
            "top_m": getattr(model.decoder, "top_m", None),
            "use_refiner": getattr(model.decoder, "use_refiner", False),
        } if hasattr(model.decoder, "anchor_steps") else None,
        "parameter_counts": parameter_counts,
    }
    (run_dir / "locked_backend_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    verifier_root = args.verifier_root or (project / "results/social_verifier_crossval/scale_0p030")
    verifier_path = verifier_root / f"seed_{args.seed}" / f"val_scene_{args.val_scene_id}" / "verifier_best.pth"
    verifier_info = load_verifier(verifier_path, device) if verifier_path.is_file() else None
    history = []
    best_ade = float("inf")
    best_checkpoint_score = float("inf")
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        training_policy = getattr(model.decoder, "apply_training_policy", None)
        if callable(training_policy):
            training_policy(model)
        losses = []
        equivariance_losses = []
        for past, future, num_valid, _, _, _ in train_loader:
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = augment_trajectories(past, future, args)
            if args.focal_heading_normalize:
                past, future = normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            if hasattr(model.decoder, "set_training_target"):
                model.decoder.set_training_target(future - center)
            model_past = past
            history_mask = None
            if history_reconstruction_head is not None:
                batch, agents, steps, _ = past.shape
                time_valid = torch.zeros(steps, dtype=torch.bool, device=device)
                time_valid[1:-1] = True
                ids = torch.arange(agents, device=device)
                agent_valid = ids[None] < num_valid.long()[:, None]
                history_mask = (
                    torch.rand(batch, agents, steps, device=device)
                    < float(args.masked_history_probability)
                ) & agent_valid[:, :, None] & time_valid[None, None]
                interpolated = past.clone()
                interpolated[:, :, 1:-1] = 0.5 * (
                    past[:, :, :-2] + past[:, :, 2:]
                )
                model_past = torch.where(history_mask[..., None], interpolated, past)
            prediction = model(
                model_past - center, future - center, None, mode="train",
                num_valid=num_valid, graph_loc=model_past,
            )[2]
            logits = model.last_mode_logits
            sample_weight = None
            if args.future_difficulty_weight > 0.0:
                sample_weight = future_motion_difficulty_weights(
                    past, future, args.future_difficulty_weight
                )
            loss, _ = loss_fn(
                prediction[:, :1], (future - center)[:, :1], logits[:, :1], cfg,
                sample_weight=sample_weight,
            )
            if args.oracle_coverage_weight > 0.0:
                target = (future - center)[:, :1].unsqueeze(-3)
                distances = torch.linalg.vector_norm(
                    prediction[:, :1] - target, dim=-1
                )
                quality = (
                    distances.mean(dim=-1)
                    + float(args.oracle_coverage_fde_weight) * distances[..., -1]
                )
                temperature = max(float(args.oracle_coverage_temperature), 1.0e-4)
                # Detached responsibilities provide gradients to several
                # plausible candidates without letting the weighting itself
                # collapse. The existing endpoint diversity remains active.
                detached_quality = quality.detach()
                if args.oracle_coverage_normalized:
                    gap = detached_quality - detached_quality.min(
                        dim=-1, keepdim=True
                    ).values
                    quality_scale = gap.mean(dim=-1, keepdim=True).clamp_min(0.05)
                    responsibility_logits = -gap / (temperature * quality_scale)
                else:
                    responsibility_logits = -detached_quality / temperature
                responsibilities = torch.softmax(responsibility_logits, dim=-1)
                coverage_values = (responsibilities * quality).sum(dim=-1)
                if sample_weight is None:
                    coverage = coverage_values.mean()
                else:
                    coverage_per_sample = coverage_values.reshape(
                        coverage_values.shape[0], -1
                    ).mean(dim=-1)
                    coverage = (coverage_per_sample * sample_weight).mean()
                coverage_scale = 1.0
                if args.oracle_coverage_warmup_epochs > 0.0:
                    coverage_scale = min(
                        float(epoch + 1) / float(args.oracle_coverage_warmup_epochs), 1.0
                    )
                loss = loss + float(args.oracle_coverage_weight) * coverage_scale * coverage
            auxiliary_training_loss = getattr(
                model.decoder, "auxiliary_training_loss", None
            )
            if callable(auxiliary_training_loss):
                auxiliary = auxiliary_training_loss(
                    prediction[:, :1], (future - center)[:, :1], logits[:, :1]
                )
                if not torch.isfinite(auxiliary):
                    raise RuntimeError("Decoder auxiliary training loss is not finite")
                loss = loss + auxiliary
            if args.velocity_loss_weight > 0.0 or args.acceleration_loss_weight > 0.0:
                focal_prediction = prediction[:, :1]
                focal_target = (future - center)[:, :1]
                last_observed = (past - center)[:, :1, -1]
                predicted_velocity = torch.cat(
                    (
                        focal_prediction[..., :1, :] - last_observed[:, :, None, None, :],
                        focal_prediction[..., 1:, :] - focal_prediction[..., :-1, :],
                    ),
                    dim=-2,
                )
                target_velocity = torch.cat(
                    (
                        focal_target[..., :1, :] - last_observed[:, :, None, :],
                        focal_target[..., 1:, :] - focal_target[..., :-1, :],
                    ),
                    dim=-2,
                )
                velocity_error = torch.nn.functional.smooth_l1_loss(
                    predicted_velocity,
                    target_velocity[:, :, None],
                    reduction="none",
                ).mean(dim=(-1, -2))
                with torch.no_grad():
                    trajectory_error = torch.linalg.vector_norm(
                        focal_prediction - focal_target[:, :, None], dim=-1
                    )
                    kinetic_quality = (
                        trajectory_error.mean(dim=-1)
                        + float(args.oracle_coverage_fde_weight) * trajectory_error[..., -1]
                    )
                    kinetic_weights = torch.softmax(
                        -kinetic_quality / max(float(args.oracle_coverage_temperature), 1.0e-4),
                        dim=-1,
                    )
                if args.velocity_loss_weight > 0.0:
                    velocity_loss = (kinetic_weights * velocity_error).sum(dim=-1).mean()
                    loss = loss + float(args.velocity_loss_weight) * velocity_loss
                if args.acceleration_loss_weight > 0.0:
                    predicted_acceleration = predicted_velocity[..., 1:, :] - predicted_velocity[..., :-1, :]
                    target_acceleration = target_velocity[..., 1:, :] - target_velocity[..., :-1, :]
                    acceleration_error = torch.nn.functional.smooth_l1_loss(
                        predicted_acceleration,
                        target_acceleration[:, :, None],
                        reduction="none",
                    ).mean(dim=(-1, -2))
                    acceleration_loss = (kinetic_weights * acceleration_error).sum(dim=-1).mean()
                    loss = loss + float(args.acceleration_loss_weight) * acceleration_loss
            if args.aux_motion_weight > 0.0 and hasattr(model, "motion_aux_head"):
                features = model.agent_aware_trans_encoder.last_agent_features[:, 0]
                auxiliary_prediction = model.motion_aux_head(features)
                next_velocity = future[:, 0, 0] - past[:, 0, -1]
                mean_velocity = (
                    future[:, 0, -1] - past[:, 0, -1]
                ) / float(future.shape[-2])
                auxiliary_target = torch.cat((next_velocity, mean_velocity), dim=-1)
                auxiliary_loss = torch.nn.functional.smooth_l1_loss(
                    auxiliary_prediction, auxiliary_target
                )
                loss = loss + float(args.aux_motion_weight) * auxiliary_loss
            if history_reconstruction_head is not None and history_mask is not None:
                temporal_features = model.agent_aware_trans_encoder.last_temporal_features
                reconstruction = history_reconstruction_head(temporal_features)
                reconstruction_target = past - center
                reconstruction_error = torch.nn.functional.smooth_l1_loss(
                    reconstruction, reconstruction_target, reduction="none"
                ).mean(dim=-1)
                masked_count = history_mask.sum().clamp_min(1)
                reconstruction_loss = (
                    reconstruction_error * history_mask.to(reconstruction_error.dtype)
                ).sum() / masked_count
                loss = loss + float(args.masked_history_weight) * reconstruction_loss
            if args.equivariance_weight > 0.0:
                centered_past = model_past - center
                centered_future = future - center
                mirrored_prediction = model(
                    mirror_lateral(centered_past),
                    mirror_lateral(centered_future),
                    None,
                    mode="train",
                    num_valid=num_valid,
                    graph_loc=mirror_lateral(model_past),
                )[2]
                restored_prediction = mirror_lateral(mirrored_prediction[:, :1])
                equivariance_loss = symmetric_set_consistency(
                    prediction[:, 0], restored_prediction[:, 0]
                )
                loss = loss + float(args.equivariance_weight) * equivariance_loss
                equivariance_losses.append(float(equivariance_loss.detach().cpu()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            if args.scheduler == "warm_restarts":
                scheduler.step(epoch + len(losses) / max(len(train_loader), 1))
            elif scheduler is not None:
                scheduler.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "lr": optimizer.param_groups[0]["lr"]}
        if equivariance_losses:
            row["equivariance_loss"] = float(np.mean(equivariance_losses))
        if (epoch + 1) % args.val_freq == 0 or epoch + 1 == args.epochs:
            evaluation_model = ema.module if ema is not None else model
            val_metrics = evaluate(
                evaluation_model, val_loader, device, verifier_info, args.fusion_alpha,
                args.focal_heading_normalize,
            )
            row["val_original_ade"] = val_metrics["original"]["ade"]
            row["val_fused_ade"] = val_metrics["fused"]["ade"] if val_metrics["fused"] else None
            row["val_min_ade_20"] = val_metrics["oracle"]["min_ade_20"]
            row["val_min_fde_20"] = val_metrics["oracle"]["min_fde_20"]
            row["val_same_path_fde"] = val_metrics["oracle"]["same_path_fde"]
            if row["val_original_ade"] < best_ade:
                best_ade = row["val_original_ade"]
                torch.save(evaluation_model.state_dict(), run_dir / "best_original_ade.pth")
            if args.checkpoint_criterion == "oracle_combo":
                checkpoint_score = row["val_min_ade_20"] + 0.5 * row["val_min_fde_20"]
            else:
                checkpoint_score = row["val_original_ade"]
            row["val_checkpoint_score"] = checkpoint_score
            if checkpoint_score < best_checkpoint_score:
                best_checkpoint_score = checkpoint_score
                torch.save(evaluation_model.state_dict(), run_dir / "best_selected.pth")
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    evaluation_model = ema.module if ema is not None else model
    if ema is not None:
        torch.save(model.state_dict(), run_dir / "final_model_raw.pth")
    if history_reconstruction_head is not None:
        torch.save(history_reconstruction_head.state_dict(), run_dir / "training_only_history_head.pth")
    torch.save(evaluation_model.state_dict(), run_dir / "final_model.pth")
    final_metrics = evaluate(
        evaluation_model, val_loader, device, verifier_info, args.fusion_alpha,
        args.focal_heading_normalize,
    )
    selected_state = torch.load(run_dir / "best_selected.pth", map_location=device, weights_only=True)
    evaluation_model.load_state_dict(selected_state)
    selected_metrics = evaluate(
        evaluation_model, val_loader, device, verifier_info, args.fusion_alpha,
        args.focal_heading_normalize,
    )
    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate(
            evaluation_model, test_loader, device, None, args.fusion_alpha,
            args.focal_heading_normalize,
        )
    result = {
        "variant": args.variant, "decoder_experiment": args.decoder_experiment,
        "seed": args.seed, "validation_scene_id": args.val_scene_id,
        "epochs": args.epochs, "scheduler": args.scheduler,
        "elapsed_seconds": time.perf_counter() - started,
        "split": split, "locked_backend": contract,
        "verifier_checkpoint": str(verifier_path) if verifier_info else None,
        "evaluation": final_metrics,
        "evaluation_best_selected": selected_metrics,
        "evaluation_test_best_selected": test_metrics,
        "best_checkpoint_score": best_checkpoint_score,
    }
    metrics_path.write_text(json.dumps(json_ready(result), indent=2, ensure_ascii=False), encoding="utf-8")
    with (run_dir / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in history for k in row}))
        writer.writeheader(); writer.writerows(history)
    print(f"[done] {metrics_path}")


if __name__ == "__main__":
    run()
