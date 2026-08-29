"""EXP-148: SDD video-scale counterfactual audit on the selected S0 model.

This is an analysis-only experiment.  It never retrains the model and never
uses scene/video IDs as model inputs.  Ground truth is used only to estimate
an oracle upper bound for a *future* scale-adaptation experiment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
MODEL_MODULE = ROOT / "current_scene_query_models.py"
DATA_DIR = WORKSPACE / "SDD_STANDARD_POTENTIAL_SERIES" / "processed_data"
ASSET_DIR = ROOT / "processed_scene_assets"
EXPERIMENT = ROOT / "EXP_148_SDD_VIDEO_SCALE_COUNTERFACTUAL_AUDIT"
DEFAULT_CHECKPOINT = (
    ROOT / "EXP_147B_CURRENT_DECODER_SDD_SCENE_QUERY" / "results" / "s0_current" /
    "f16_temporal_social_cross" / "seed_1" / "val_scene_1" / "best_selected.pth"
)
SCALE_GRID = np.asarray([0.55, 0.70, 0.85, 1.00, 1.20, 1.45, 1.75], np.float32)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        result = np.empty_like(order, dtype=np.float64)
        result[order] = np.arange(len(v), dtype=np.float64)
        return result
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = EXPERIMENT / "audit_summary.json"
    if output.is_file() and not args.force:
        print(f"[skip] {output}; use --force to recompute")
        return
    required = [args.checkpoint, DATA_DIR / "sdd_standard_test_data.npy", DATA_DIR / "sdd_standard_test_num.npy", ASSET_DIR / "sdd_test_scene_ids.npy", ASSET_DIR / "scene_asset_manifest.json"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.dry_run:
        print("[dry-run] checkpoint=" + str(args.checkpoint))
        print("[dry-run] test_data=" + str(DATA_DIR / "sdd_standard_test_data.npy"))
        print("[dry-run] output=" + str(output))
        return

    # The upstream ASTRA decoder resolves its PCA basis relative to PROJECT.
    os.chdir(PROJECT)
    sys.path[:0] = [str(FRONT), str(PROJECT)]
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    trainer = load_module("exp148_trainer_helpers", FRONT / "train_front_variant.py")
    decoder_module = load_module("exp148_current_decoder", MODEL_MODULE)
    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 16; cfg.MODEL.GRAPH.TOP_K = 4; cfg.MODEL.GRAPH.RADIUS = 6.0
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg.device = str(device); cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    decoder_module.replace_decoder(model, cfg, "s0_current")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    model = model.to(device).eval()
    dataset = trainer.TrajectoryDataset(DATA_DIR / "sdd_standard_test_data.npy", DATA_DIR / "sdd_standard_test_num.npy", cfg, ASSET_DIR / "sdd_test_scene_ids.npy")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    metadata = json.loads((ASSET_DIR / "scene_asset_manifest.json").read_text(encoding="utf-8"))
    id_to_video = {int(v): k for k, v in metadata["video_to_scene_id"].items()}
    image_wh = np.asarray(json.loads((ASSET_DIR / "scene_asset_manifest.json").read_text(encoding="utf-8")).get("image_wh", []))
    # image sizes are stored in the semantic bank; only metadata remains human-readable.
    bank = np.load(ASSET_DIR / "sdd_semantic_bank.npz", allow_pickle=False)
    image_wh = np.asarray(bank["image_wh"], dtype=np.float32)

    records = []
    with torch.inference_mode():
        for past, future, num_valid, scene_ids, _, _ in loader:
            absolute_past = past.numpy()
            past, future, num_valid = past.to(device), future.to(device), num_valid.to(device).int()
            local_past, local_future = trainer.normalize_to_focal_heading(past, future)
            center = local_past.mean(dim=-2, keepdim=True)
            prediction = model(local_past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=local_past)[2] + center.unsqueeze(-3)
            candidate = prediction[:, 0]
            target = local_future[:, 0]
            origin = local_past[:, 0, -1]
            relative_pred = candidate - origin[:, None, None]
            relative_target = target - origin[:, None]
            base_error = torch.linalg.vector_norm(relative_pred - relative_target[:, None], dim=-1)
            base_ade = base_error.mean(dim=-1).min(dim=-1).values
            base_fde = base_error[..., -1].min(dim=-1).values
            grid = torch.as_tensor(SCALE_GRID, device=device)[None, :, None, None, None]
            scaled = relative_pred[:, None] * grid
            scale_error = torch.linalg.vector_norm(scaled - relative_target[:, None, None], dim=-1)
            # First choose the best of 20 candidates at each scale, then choose
            # the best scale.  The latter is an oracle-only upper bound.
            grid_ade_by_scale = scale_error.mean(dim=-1).amin(dim=-1)
            grid_fde_by_scale = scale_error[..., -1].amin(dim=-1)
            grid_ade = grid_ade_by_scale.amin(dim=-1)
            grid_fde = grid_fde_by_scale.amin(dim=-1)
            selected_scale = torch.argmin(grid_fde_by_scale, dim=-1)
            past_px = absolute_past[:, 0] * 20.0
            history_span = np.linalg.norm(past_px[:, -1] - past_px[:, 0], axis=-1)
            recent_speed = np.median(np.linalg.norm(np.diff(past_px[:, -4:], axis=1), axis=-1), axis=1)
            endpoint_radius = torch.linalg.vector_norm(relative_target[:, -1], dim=-1).cpu().numpy() * 20.0
            candidate_radius = torch.linalg.vector_norm(relative_pred[..., -1, :], dim=-1).mean(dim=-1).cpu().numpy() * 20.0
            for index, scene in enumerate(scene_ids.numpy().astype(int)):
                width, height = image_wh[scene]
                records.append({
                    "scene_id": scene, "video": id_to_video[scene], "samples_in_video": None,
                    "image_width": float(width), "image_height": float(height), "image_diag": float(np.hypot(width, height)),
                    "history_span_px": float(history_span[index]), "recent_speed_px": float(recent_speed[index]),
                    "gt_endpoint_radius_px": float(endpoint_radius[index]), "candidate_endpoint_radius_px": float(candidate_radius[index]),
                    "base_minADE_px": float(base_ade[index].cpu() * 20.0), "base_minFDE_px": float(base_fde[index].cpu() * 20.0),
                    "grid_oracle_minADE_px": float(grid_ade[index].cpu() * 20.0), "grid_oracle_minFDE_px": float(grid_fde[index].cpu() * 20.0),
                    "oracle_scale_multiplier": float(SCALE_GRID[int(selected_scale[index])]),
                })

    fields = ["history_span_px", "recent_speed_px", "gt_endpoint_radius_px", "candidate_endpoint_radius_px", "base_minADE_px", "base_minFDE_px", "grid_oracle_minADE_px", "grid_oracle_minFDE_px", "oracle_scale_multiplier"]
    groups = {"all": records, "little": [r for r in records if r["video"].startswith("little/")], "non_little": [r for r in records if not r["video"].startswith("little/")]}
    grouped = {}
    for name, values in groups.items():
        grouped[name] = {key: float(np.mean([r[key] for r in values])) for key in fields}
        grouped[name]["samples"] = len(values)
        grouped[name]["grid_oracle_ADE_gain_pct"] = float((grouped[name]["grid_oracle_minADE_px"] / grouped[name]["base_minADE_px"] - 1.0) * 100.0)
        grouped[name]["grid_oracle_FDE_gain_pct"] = float((grouped[name]["grid_oracle_minFDE_px"] / grouped[name]["base_minFDE_px"] - 1.0) * 100.0)
    by_video = {}
    for video in sorted({r["video"] for r in records}):
        values = [r for r in records if r["video"] == video]
        by_video[video] = {key: float(np.mean([r[key] for r in values])) for key in fields}
        by_video[video]["samples"] = len(values)
    multiplier = np.asarray([r["oracle_scale_multiplier"] for r in records])
    correlations = {key: spearman(np.asarray([r[key] for r in records]), multiplier) for key in ("history_span_px", "recent_speed_px", "image_diag", "candidate_endpoint_radius_px", "gt_endpoint_radius_px")}
    summary = {"experiment": "EXP-148 SDD video-scale counterfactual audit", "checkpoint": str(args.checkpoint), "scale_grid": SCALE_GRID.tolist(), "groups": grouped, "spearman_with_oracle_scale": correlations, "by_video": by_video, "interpretation": "Grid improvements are oracle-only upper bounds: GT chooses the multiplier. A scale module is justified only if this upper bound is material and observed-only quantities predict the multiplier."}
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(EXPERIMENT / "sample_records.npz", records=np.asarray(records, dtype=object))
    print("[done] " + str(output))


if __name__ == "__main__":
    main()
