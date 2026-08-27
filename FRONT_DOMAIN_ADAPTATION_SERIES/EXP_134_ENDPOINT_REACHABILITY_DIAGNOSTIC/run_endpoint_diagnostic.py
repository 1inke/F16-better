"""Diagnose endpoint reachability using the frozen EXP-133 five-fold models."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
SERIES = HERE.parent
WORKSPACE = SERIES.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
CHECKPOINT_ROOT = (
    WORKSPACE / "ETH_UCY_FULL_CROSSVAL_SERIES" /
    "EXP_133_CURRENT_MODEL_FULL_ETH_UCY" / "results"
)
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
DECODER_MODULE = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS" /
    "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
)
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def setup_model(fold: str, device: torch.device):
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA

    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.TRAIN.BATCH_SIZE = 64
    cfg.VAL.BATCH_SIZE = 64
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]

    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    spec = importlib.util.spec_from_file_location("kinematic_decoder", DECODER_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(DECODER_MODULE)
    decoder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder_module)
    decoder_module.replace_decoder(model, cfg, "kinematic_k20")
    checkpoint = (
        CHECKPOINT_ROOT / fold / "f16_temporal_social_cross" /
        "seed_1" / "val_scene_-1" / "best_selected.pth"
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model.to(device).eval(), cfg, checkpoint


def summarize(values: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    result = {"samples": int(mask.sum())}
    if not mask.any():
        return result
    for name, array in values.items():
        selected = array[mask]
        result[name] = float(selected.mean())
        if name.endswith("fde") or name.endswith("ade"):
            result[name + "_p90"] = float(np.quantile(selected, 0.90))
    return result


def diagnose_fold(
    fold: str, device: torch.device, batch_size: int, max_samples: int
) -> tuple[dict, list[dict]]:
    model, cfg, checkpoint = setup_model(fold, device)
    from train_front_variant import TrajectoryDataset, normalize_to_focal_heading
    dataset = TrajectoryDataset(
        DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy", cfg
    )
    if max_samples > 0:
        dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    collected: dict[str, list[np.ndarray]] = {}
    sample_offset = 0
    rows: list[dict] = []
    with torch.inference_mode():
        for past, future, num_valid, _, _, _ in loader:
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            prediction = model(
                past - center, None, None, mode="testing",
                num_valid=num_valid, graph_loc=past,
            )[2] + center.unsqueeze(-3)
            candidates = prediction[:, 0]
            target = future[:, 0]

            focal = past[:, 0]
            last = focal[:, -1]
            last_step = focal[:, -1] - focal[:, -2]
            speed = torch.linalg.vector_norm(last_step, dim=-1, keepdim=True)
            default = torch.zeros_like(last_step)
            default[:, 0] = 1.0
            heading = torch.where(
                speed > 0.02, last_step / speed.clamp_min(1.0e-6), default
            )
            base_relative = last_step * float(model.pred_len)
            anchors = model.decoder.anchors
            fixed_relative = base_relative[:, None] + model.decoder.rotate_local(
                anchors[None].expand(len(past), -1, -1), heading
            )
            cv_endpoint = last + base_relative
            fixed_endpoints = last[:, None] + fixed_relative
            learned_endpoints = candidates[:, :, -1]
            target_endpoint = target[:, -1]

            path_error = torch.linalg.vector_norm(
                candidates - target[:, None], dim=-1
            )
            ade = path_error.mean(dim=-1)
            fde = path_error[:, :, -1]
            best_ade_mode = ade.argmin(dim=-1)
            best_fde_mode = fde.argmin(dim=-1)
            batch_ids = torch.arange(len(past), device=device)

            pairwise = torch.cdist(learned_endpoints, learned_endpoints)
            pair_mask = ~torch.eye(pairwise.shape[-1], dtype=torch.bool, device=device)
            pair_values = pairwise[:, pair_mask]
            steps = focal[:, 1:] - focal[:, :-1]
            acceleration = steps[:, 1:] - steps[:, :-1]

            batch_values = {
                "cv_fde": torch.linalg.vector_norm(cv_endpoint - target_endpoint, dim=-1),
                "fixed_anchor_oracle_fde": torch.cdist(
                    target_endpoint[:, None], fixed_endpoints
                ).squeeze(1).min(dim=-1).values,
                "learned_oracle_fde": fde.min(dim=-1).values,
                "learned_oracle_ade": ade.min(dim=-1).values,
                "same_path_fde": fde[batch_ids, best_ade_mode],
                "mode_consistency": (best_ade_mode == best_fde_mode).float(),
                "endpoint_pair_distance": pair_values.mean(dim=-1),
                "collapsed_pair_fraction": (pair_values < 0.10).float().mean(dim=-1),
                "learned_offset_size": torch.linalg.vector_norm(
                    learned_endpoints - fixed_endpoints, dim=-1
                ).mean(dim=-1),
                "recent_speed": torch.linalg.vector_norm(steps[:, -3:], dim=-1).median(dim=-1).values,
                "nonlinearity": torch.linalg.vector_norm(acceleration, dim=-1).mean(dim=-1),
                "valid_agents": num_valid.float(),
            }
            numpy_values = {
                name: tensor.detach().cpu().numpy() for name, tensor in batch_values.items()
            }
            for name, array in numpy_values.items():
                collected.setdefault(name, []).append(array)
            for index in range(len(past)):
                row = {"fold": fold, "sample_index": sample_offset + index}
                row.update({name: float(array[index]) for name, array in numpy_values.items()})
                rows.append(row)
            sample_offset += len(past)

    values = {name: np.concatenate(parts) for name, parts in collected.items()}
    speed_q = np.quantile(values["recent_speed"], (0.25, 0.75))
    nonlinear_q = np.quantile(values["nonlinearity"], (0.25, 0.75))
    all_mask = np.ones(len(values["recent_speed"]), dtype=bool)
    groups = {
        "all": all_mask,
        "high_speed": values["recent_speed"] >= speed_q[1],
        "high_nonlinearity": values["nonlinearity"] >= nonlinear_q[1],
        "low_density": values["valid_agents"] <= 2,
        "joint_challenge": (
            (values["recent_speed"] >= speed_q[1])
            & (values["nonlinearity"] >= nonlinear_q[1])
            & (values["valid_agents"] <= 2)
        ),
    }
    metric_values = {
        name: array for name, array in values.items()
        if name not in {"recent_speed", "nonlinearity", "valid_agents"}
    }
    summary = {
        "fold": fold,
        "checkpoint": str(checkpoint),
        "samples": int(len(all_mask)),
        "thresholds": {
            "speed_q25_q75": [float(value) for value in speed_q],
            "nonlinearity_q25_q75": [float(value) for value in nonlinear_q],
        },
        "groups": {
            name: summarize(metric_values, mask) for name, mask in groups.items()
        },
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--only", choices=("all", *FOLDS), default="all")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=HERE / "results")
    args = parser.parse_args()
    device = select_device(args.device)
    folds = FOLDS if args.only == "all" else (args.only,)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_rows = []
    for fold in folds:
        print(f"[diagnose] {fold}", flush=True)
        summary, rows = diagnose_fold(fold, device, args.batch_size, args.max_samples)
        summaries.append(summary)
        all_rows.extend(rows)
        (args.output_root / f"{fold}_diagnostic.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    metric_names = [
        "cv_fde", "fixed_anchor_oracle_fde", "learned_oracle_fde",
        "learned_oracle_ade", "same_path_fde", "mode_consistency",
        "endpoint_pair_distance", "collapsed_pair_fraction", "learned_offset_size",
    ]
    macro = {}
    if summaries:
        for metric in metric_names:
            macro[metric] = float(np.mean([
                item["groups"]["all"][metric] for item in summaries
            ]))
    result = {
        "description": "Frozen EXP-133 endpoint reachability and mode-set diagnostic",
        "completed_folds": len(summaries),
        "macro_equal_fold": macro,
        "folds": summaries,
    }
    (args.output_root / "diagnostic_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if all_rows:
        with (args.output_root / "per_sample_diagnostics.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"[done] {args.output_root / 'diagnostic_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
