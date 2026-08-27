"""Train a ground-truth-only social stimulus verifier for ASTRA/Q20.

The verifier learns p(future trajectory | focal history, neighbour history)
from real training trajectories.  It never consumes Q20 winner labels and it
never opens the ETH/UCY target test split.  Validation is performed by holding
out one complete source scene from the existing training file.

Example
-------
python train_social_stimulus_verifier.py \
  --project-root "C:\\Users\\linke\\Desktop\\DRSE\\ASTRA_ACCURACY_Q20" \
  --config-file configs/motif_rbf_univ.yaml \
  --epochs 60 --seed 1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


@dataclass(frozen=True)
class VerifierConfig:
    pred_len: int
    frequency: float
    num_components: int = 8
    self_hidden: int = 96
    social_hidden: int = 64
    decoder_hidden: int = 192
    min_scale: float = 0.03
    max_log_scale: float = 2.0
    heading_speed_eps: float = 0.05
    dropout: float = 0.10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rotate_to_heading(vectors: Tensor, heading: Tensor) -> Tensor:
    """Rotate row-vector coordinates into the focal heading frame."""
    cos = heading[:, 0]
    sin = heading[:, 1]
    while cos.ndim < vectors.ndim - 1:
        cos = cos.unsqueeze(-1)
        sin = sin.unsqueeze(-1)
    x = vectors[..., 0]
    y = vectors[..., 1]
    return torch.stack((x * cos + y * sin, -x * sin + y * cos), dim=-1)


def finite_difference(sequence: Tensor, frequency: float) -> Tensor:
    first = torch.zeros_like(sequence[..., :1, :])
    delta = sequence[..., 1:, :] - sequence[..., :-1, :]
    return torch.cat((first, delta), dim=-2) * float(frequency)


class SocialStimulusVerifier(nn.Module):
    """Trajectory-level diagonal-GMM verifier conditioned on past motion."""

    def __init__(self, config: VerifierConfig) -> None:
        super().__init__()
        self.config = config

        # Focal input: local position, velocity and acceleration (6 values).
        self.self_input = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, config.self_hidden),
            nn.GELU(),
        )
        self.self_encoder = nn.GRU(
            config.self_hidden,
            config.self_hidden,
            batch_first=True,
        )

        # Per-neighbour/time input: relative position, relative velocity,
        # distance and closing speed (6 values).  Masked max aggregation keeps
        # compute bounded by the existing MAX_AGENTS setting.
        self.social_edge = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, config.social_hidden),
            nn.GELU(),
            nn.Linear(config.social_hidden, config.social_hidden),
            nn.GELU(),
        )
        self.social_encoder = nn.GRU(
            config.social_hidden + 1,
            config.social_hidden,
            batch_first=True,
        )

        context_dim = config.self_hidden + config.social_hidden
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, config.decoder_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.mixture_head = nn.Linear(
            config.decoder_hidden, config.num_components
        )
        self.mean_head = nn.Linear(
            config.decoder_hidden,
            config.num_components * config.pred_len * 2,
        )
        self.scale_head = nn.Linear(
            config.decoder_hidden,
            config.num_components * config.pred_len * 2,
        )

        # Begin with a broad distribution around zero rather than extremely
        # small variances, which otherwise makes early GMM training unstable.
        nn.init.zeros_(self.mean_head.bias)
        nn.init.constant_(self.scale_head.bias, math.log(0.5))

    def normalize(
        self,
        past: Tensor,
        future: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor | None, Tensor]:
        """Translate and heading-align a batch of multi-agent trajectories."""
        focal = past[:, 0]
        origin = focal[:, -1]
        last_velocity = (
            focal[:, -1] - focal[:, -2]
        ) * self.config.frequency
        speed = torch.linalg.vector_norm(last_velocity, dim=-1, keepdim=True)
        default_heading = torch.zeros_like(last_velocity)
        default_heading[:, 0] = 1.0
        heading = torch.where(
            speed > self.config.heading_speed_eps,
            last_velocity / speed.clamp_min(1.0e-6),
            default_heading,
        )

        centered_past = past - origin[:, None, None, :]
        local_past = rotate_to_heading(centered_past, heading)
        local_future = None
        if future is not None:
            centered_future = future - origin[:, None, None, :]
            local_future = rotate_to_heading(centered_future, heading)
        return local_past, local_future, heading

    def encode(self, local_past: Tensor, num_valid: Tensor) -> Tensor:
        focal = local_past[:, 0]
        focal_velocity = finite_difference(focal, self.config.frequency)
        focal_acceleration = finite_difference(
            focal_velocity, self.config.frequency
        )
        focal_features = torch.cat(
            (focal, focal_velocity, focal_acceleration), dim=-1
        )
        focal_emb = self.self_input(focal_features)
        _, focal_hidden = self.self_encoder(focal_emb)
        focal_hidden = focal_hidden[-1]

        neighbours = local_past[:, 1:]
        batch, neighbour_count, history, _ = neighbours.shape
        if neighbour_count == 0:
            social_hidden = focal_hidden.new_zeros(
                batch, self.config.social_hidden
            )
        else:
            neighbour_velocity = finite_difference(
                neighbours, self.config.frequency
            )
            relative_position = neighbours - focal[:, None]
            relative_velocity = (
                neighbour_velocity - focal_velocity[:, None]
            )
            distance = torch.linalg.vector_norm(
                relative_position, dim=-1, keepdim=True
            )
            closing = -(
                relative_position * relative_velocity
            ).sum(dim=-1, keepdim=True) / distance.clamp_min(1.0e-3)
            edge_features = torch.cat(
                (
                    relative_position,
                    relative_velocity,
                    distance,
                    closing,
                ),
                dim=-1,
            )
            edge_emb = self.social_edge(edge_features)

            valid_neighbour_count = (num_valid.long() - 1).clamp(
                min=0, max=neighbour_count
            )
            neighbour_ids = torch.arange(
                neighbour_count, device=local_past.device
            )
            neighbour_mask = (
                neighbour_ids[None, :] < valid_neighbour_count[:, None]
            )
            expanded_mask = neighbour_mask[:, :, None, None]
            masked_edge = edge_emb.masked_fill(~expanded_mask, -1.0e4)
            social_sequence = masked_edge.max(dim=1).values
            has_neighbour = neighbour_mask.any(dim=1)
            social_sequence = torch.where(
                has_neighbour[:, None, None],
                social_sequence,
                torch.zeros_like(social_sequence),
            )
            count_feature = (
                valid_neighbour_count.float()
                / max(float(neighbour_count), 1.0)
            )[:, None, None].expand(-1, history, 1)
            social_input = torch.cat(
                (social_sequence, count_feature), dim=-1
            )
            _, social_hidden_gru = self.social_encoder(social_input)
            social_hidden = social_hidden_gru[-1]

        return self.context(torch.cat((focal_hidden, social_hidden), dim=-1))

    def distribution(
        self, past: Tensor, num_valid: Tensor
    ) -> Dict[str, Tensor]:
        local_past, _, heading = self.normalize(past)
        context = self.encode(local_past, num_valid)
        batch = past.shape[0]
        components = self.config.num_components
        pred_len = self.config.pred_len
        mixture_logits = self.mixture_head(context)
        means = self.mean_head(context).view(
            batch, components, pred_len, 2
        )
        raw_scale = self.scale_head(context).view(
            batch, components, pred_len, 2
        )
        scales = F.softplus(raw_scale).clamp_min(self.config.min_scale)
        scales = scales.clamp_max(math.exp(self.config.max_log_scale))
        return {
            "mixture_logits": mixture_logits,
            "means": means,
            "scales": scales,
            "heading": heading,
            "origin": past[:, 0, -1],
        }

    def trajectory_log_prob(
        self, distribution: Dict[str, Tensor], local_trajectory: Tensor
    ) -> Tensor:
        """Return log p(Y|H,S) for [B,T,2] local-frame trajectories."""
        means = distribution["means"]
        scales = distribution["scales"]
        target = local_trajectory[:, None]
        standardized = (target - means) / scales
        element_log_prob = -0.5 * standardized.square()
        element_log_prob = element_log_prob - torch.log(scales)
        element_log_prob = element_log_prob - 0.5 * math.log(2.0 * math.pi)
        component_log_prob = element_log_prob.sum(dim=(-1, -2))
        log_weights = F.log_softmax(
            distribution["mixture_logits"], dim=-1
        )
        return torch.logsumexp(log_weights + component_log_prob, dim=-1)

    def score_candidates(
        self, distribution: Dict[str, Tensor], local_candidates: Tensor
    ) -> Tensor:
        """Score [B,K,T,2] candidates without Q20 branch labels."""
        means = distribution["means"][:, None]
        scales = distribution["scales"][:, None]
        candidates = local_candidates[:, :, None]
        standardized = (candidates - means) / scales
        log_prob = -0.5 * standardized.square() - torch.log(scales)
        log_prob = log_prob - 0.5 * math.log(2.0 * math.pi)
        component_log_prob = log_prob.sum(dim=(-1, -2))
        log_weights = F.log_softmax(
            distribution["mixture_logits"], dim=-1
        )[:, None]
        return torch.logsumexp(log_weights + component_log_prob, dim=-1)

    def forward(
        self, past: Tensor, future: Tensor, num_valid: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        distribution = self.distribution(past, num_valid)
        _, local_future_all, _ = self.normalize(past, future)
        if local_future_all is None:
            raise RuntimeError("Future normalization unexpectedly returned None.")
        local_target = local_future_all[:, 0]
        nll = -self.trajectory_log_prob(distribution, local_target).mean()
        return nll, distribution, local_target


def choose_validation_scene(
    scene_ids: np.ndarray,
    requested_scene: int | None,
    target_fraction: float,
) -> int:
    unique, counts = np.unique(scene_ids.astype(np.int64), return_counts=True)
    if unique.size < 2:
        raise ValueError(
            "Scene-held-out validation needs at least two source scenes in "
            "the training file."
        )
    if requested_scene is not None:
        if requested_scene not in unique:
            raise ValueError(
                f"val-scene-id={requested_scene} not in {unique.tolist()}"
            )
        return int(requested_scene)
    fractions = counts.astype(np.float64) / counts.sum()
    return int(unique[np.argmin(np.abs(fractions - target_fraction))])


def build_scene_split(
    data_path: Path,
    val_scene_id: int | None,
    val_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    all_data = np.load(data_path, mmap_mode="r")
    if all_data.ndim != 4 or all_data.shape[-1] < 4:
        raise ValueError(f"Unexpected ETH data shape: {all_data.shape}")
    scene_ids = np.asarray(all_data[:, 0, 0, 0], dtype=np.int64)
    chosen = choose_validation_scene(
        scene_ids, val_scene_id, val_fraction
    )
    val_indices = np.flatnonzero(scene_ids == chosen).astype(np.int64)
    train_indices = np.flatnonzero(scene_ids != chosen).astype(np.int64)
    unique, counts = np.unique(scene_ids, return_counts=True)
    metadata: Dict[str, object] = {
        "strategy": "leave_one_complete_training_scene_out",
        "validation_scene_id": chosen,
        "scene_counts": {
            str(int(k)): int(v) for k, v in zip(unique, counts)
        },
        "train_samples": int(train_indices.size),
        "validation_samples": int(val_indices.size),
        "test_split_accessed": False,
    }
    return train_indices, val_indices, metadata


def focal_metrics(
    distribution: Dict[str, Tensor], target: Tensor
) -> Dict[str, float]:
    means = distribution["means"]
    errors = torch.linalg.vector_norm(
        means - target[:, None], dim=-1
    )
    component_ade = errors.mean(dim=-1)
    best = component_ade.argmin(dim=-1)
    row = torch.arange(target.shape[0], device=target.device)
    top = distribution["mixture_logits"].argmax(dim=-1)
    return {
        "oracle_ade": component_ade[row, best].sum().item(),
        "oracle_same_path_fde": errors[row, best, -1].sum().item(),
        "top1_ade": component_ade[row, top].sum().item(),
        "top1_fde": errors[row, top, -1].sum().item(),
        "mixture_entropy": (
            -F.softmax(distribution["mixture_logits"], dim=-1)
            * F.log_softmax(distribution["mixture_logits"], dim=-1)
        ).sum(dim=-1).sum().item(),
    }


def move_batch(batch: Iterable[object], device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    items = list(batch)
    past = items[0].to(device=device, dtype=torch.float32)
    future = items[1].to(device=device, dtype=torch.float32)
    num_valid = items[2].to(device=device).long()
    return past, future, num_valid


def run_epoch(
    model: SocialStimulusVerifier,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    use_amp: bool,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "nll": 0.0,
        "oracle_ade": 0.0,
        "oracle_same_path_fde": 0.0,
        "top1_ade": 0.0,
        "top1_fde": 0.0,
        "mixture_entropy": 0.0,
    }
    sample_count = 0
    amp_enabled = use_amp and device.type == "cuda"

    for batch in loader:
        past, future, num_valid = move_batch(batch, device)
        batch_size = past.shape[0]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=torch.float16,
            ):
                nll, distribution, target = model(
                    past, future, num_valid
                )
            if not torch.isfinite(nll):
                raise FloatingPointError(f"Non-finite NLL: {nll.item()}")
            if training:
                nll.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_metrics = focal_metrics(distribution, target)
        totals["nll"] += nll.item() * batch_size
        for key, value in batch_metrics.items():
            totals[key] += value
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("DataLoader produced no samples.")
    return {key: value / sample_count for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: SocialStimulusVerifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: Dict[str, float],
    split_metadata: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "model_type": "ground_truth_social_stimulus_verifier",
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "validation_metrics": validation_metrics,
            "split": split_metadata,
            "normalization": {
                "origin": "last_observed_focal_position",
                "rotation": "last_velocity_heading_aligned",
                "coordinate_unit": "ETH_dataset output after TRAJ_SCALE",
            },
            "training_arguments": vars(args),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ground-truth-only social stimulus verifier."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"),
    )
    parser.add_argument(
        "--config-file",
        default="configs/motif_rbf_univ.yaml",
        help="Path relative to project-root unless absolute.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-components", type=int, default=8)
    parser.add_argument("--self-hidden", type=int, default=96)
    parser.add_argument("--social-hidden", type=int, default=64)
    parser.add_argument("--decoder-hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--min-scale", type=float, default=0.03)
    parser.add_argument("--val-scene-id", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    config_path = Path(args.config_file)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Import the repository's config and dataset only after its root is on the
    # module path.  chdir is required because ETH_dataset resolves data paths
    # relative to the project directory.
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    cache_root = project_root / ".cache"
    matplotlib_cache = cache_root / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import munch  # type: ignore
    import yaml  # type: ignore
    from data.eth import ETH_dataset  # type: ignore

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = munch.munchify(yaml.safe_load(handle))
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.SEQ_AUG = False

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    train_data_value = getattr(cfg.DATA, "TRAIN_DATA_PATH", None)
    if train_data_value:
        train_data_path = Path(train_data_value)
        if not train_data_path.is_absolute():
            train_data_path = project_root / train_data_path
    else:
        train_data_path = (
            project_root
            / "datasets"
            / "eth_ucy"
            / "processed_data_diverse"
            / f"{cfg.SUBSET}_data_train.npy"
        )
    if not train_data_path.is_file():
        raise FileNotFoundError(f"Training data not found: {train_data_path}")

    train_indices, val_indices, split_metadata = build_scene_split(
        train_data_path,
        val_scene_id=args.val_scene_id,
        val_fraction=args.val_fraction,
    )
    dataset = ETH_dataset(cfg, mode="training", img_transforms=None)
    train_subset = Subset(dataset, train_indices.tolist())
    val_subset = Subset(dataset, val_indices.tolist())
    generator = torch.Generator().manual_seed(args.seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # The stored ETH/UCY arrays contain the authoritative future length.
    # This avoids the historical 4.6 s * 2.5 Hz ambiguity (the repository's
    # processed samples contain 12 future points).
    sample_future = dataset[0][1]
    pred_len = int(sample_future.shape[-2])
    model_config = VerifierConfig(
        pred_len=pred_len,
        frequency=float(cfg.DATA.FREQUENCY),
        num_components=args.num_components,
        self_hidden=args.self_hidden,
        social_hidden=args.social_hidden,
        decoder_hidden=args.decoder_hidden,
        min_scale=args.min_scale,
        dropout=args.dropout,
    )
    model = SocialStimulusVerifier(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1.0e-6
    )

    if args.output_dir is None:
        output_dir = (
            project_root
            / "results"
            / f"social_stimulus_verifier_{cfg.SUBSET}_seed{args.seed}"
        )
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "split_indices.npz",
        train_indices=train_indices,
        validation_indices=val_indices,
    )
    with (output_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split_metadata, handle, indent=2, ensure_ascii=False)

    print(json.dumps({
        "project_root": str(project_root),
        "config": str(config_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "model_config": asdict(model_config),
        "split": split_metadata,
    }, indent=2, ensure_ascii=False))

    best_nll = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.grad_clip,
            args.amp,
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                grad_clip=args.grad_clip,
                use_amp=args.amp,
            )
        scheduler.step(validation_metrics["nll"])
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        with (output_dir / "history.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(history, handle, indent=2, ensure_ascii=False)
        print(json.dumps(record, ensure_ascii=False))

        if validation_metrics["nll"] < best_nll - 1.0e-4:
            best_nll = validation_metrics["nll"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "verifier_best.pth",
                model,
                optimizer,
                epoch,
                validation_metrics,
                split_metadata,
                args,
            )
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            output_dir / "verifier_last.pth",
            model,
            optimizer,
            epoch,
            validation_metrics,
            split_metadata,
            args,
        )
        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; best epoch={best_epoch}, "
                f"best validation NLL={best_nll:.6f}."
            )
            break

    summary = {
        "best_epoch": best_epoch,
        "best_validation_nll": best_nll,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(output_dir / "verifier_best.pth"),
        "test_split_accessed": False,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
