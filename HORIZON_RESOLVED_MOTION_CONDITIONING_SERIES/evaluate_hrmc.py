"""Evaluate HRMC variants on official outer folds and motion-condition buckets."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ASTRA_PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
if str(Path(sys.executable).resolve()).casefold() != str(ASTRA_PYTHON.resolve()).casefold():
    raise SystemExit(subprocess.run([str(ASTRA_PYTHON), *sys.argv]).returncode)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
CONFIG = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
HELPER_SCRIPT = WORKSPACE / "INDIVIDUAL_MOTION_REPRESENTATION_AUDIT" / "run_motion_representation_audit.py"
MODEL_MODULE = ROOT / "motion_conditioned_decoder_models.py"
SOURCE = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "EXP_156B_FOLD_LOCAL_CURRENT_DECODER" / "results"
BASELINE_CONDITIONS = WORKSPACE / "H3_V2_CORRECTION_ATTRIBUTION_AUDIT" / "H3_V2_CORRECTION_ATTRIBUTION_AUDIT" / "condition_summary.csv"
RESULTS = ROOT / "results"
EXPERIMENT = ROOT / "EXPERIMENT_MOTION_CONDITIONED_PROPOSAL"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
VARIANTS = {
    "p1": "p1_horizon_motion_proposal_residual",
    "c1": "c1_horizon_motion_correction_residual",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_dir(fold, stage, seed):
    return SOURCE / fold / stage / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def run_dir(variant, fold, seed):
    return RESULTS / variant / fold / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


@torch.inference_mode()
def collect(model, loader, device, normalize, helper):
    parts = {key: [] for key in ("minADE", "minFDE", "sameFDE", "J", "past")}
    for step in (4, 8, 12):
        parts[f"adapter_norm_t{step}"] = []
        parts[f"base_correction_norm_t{step}"] = []
        parts[f"final_correction_norm_t{step}"] = []
    for past, future, num, _, _, _ in loader:
        past, future, num = past.to(device), future.to(device), num.to(device).int()
        past, future = normalize(past, future)
        center = past.mean(dim=-2, keepdim=True)
        prediction = model(past - center, None, None, mode="testing", num_valid=num, graph_loc=past)[2]
        candidates, target = prediction[:, 0] + center[:, 0, 0][:, None, None], future[:, 0]
        error = torch.linalg.vector_norm(candidates - target[:, None], dim=-1)
        ade, fde = error.mean(-1), error[..., -1]
        winner = ade.argmin(-1); rows = torch.arange(len(ade), device=device)
        parts["minADE"].append(ade.min(-1).values.cpu().numpy())
        parts["minFDE"].append(fde.min(-1).values.cpu().numpy())
        parts["sameFDE"].append(fde[rows, winner].cpu().numpy())
        parts["J"].append((ade.min(-1).values + 0.5 * fde.min(-1).values).cpu().numpy())
        parts["past"].append(past.cpu().numpy())
        for index, step in enumerate((4, 8, 12)):
            for name, tensor in (
                ("adapter", model.decoder.last_adapter_delta),
                ("base_correction", model.decoder.last_base_correction),
                ("final_correction", model.decoder.last_final_correction),
            ):
                norm = torch.linalg.vector_norm(tensor[:, 0, :, index], dim=-1).mean(-1)
                parts[f"{name}_norm_t{step}"].append(norm.cpu().numpy())
    result = {key: np.concatenate(value) for key, value in parts.items()}
    result.update(helper.observation_features(result.pop("past")))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--variants", nargs="+", choices=("all", *VARIANTS), default=["all"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-root", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Evaluation smoke-test limit; 0 evaluates the complete fold.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # The evaluator later switches to the ASTRA project directory.  Resolve
    # user-supplied paths first so relative smoke-test paths remain stable.
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.output_dir = args.output_dir.resolve()
    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    variants = tuple(VARIANTS) if "all" in args.variants else tuple(args.variants)
    sys.path.insert(0, str(HELPER_SCRIPT.parent))
    import run_motion_representation_audit as helper
    required = [CONFIG, HELPER_SCRIPT, MODEL_MODULE, BASELINE_CONDITIONS]
    for fold in folds:
        required += [source_dir(fold, "h3", args.seed) / "best_selected.pth", source_dir(fold, "v2", args.seed) / "best_selected.pth"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing:\n" + "\n".join(missing))
    if args.dry_run:
        print(json.dumps({"folds": folds, "variants": variants, "seed": args.seed}, indent=2)); return

    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(WORKSPACE / "FRONT_ENCODER_SERIES")]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from train_front_variant import TrajectoryDataset, normalize_to_focal_heading
    model_module = load_module("hrmc_eval_models", MODEL_MODULE)
    baseline_rows = list(csv.DictReader(BASELINE_CONDITIONS.open("r", encoding="utf-8-sig")))
    baseline = {(row["fold"], row["condition"]): row for row in baseline_rows if row["scope"] == "outer"}
    device = torch.device(args.device)
    cfg = load_locked_q20(CONFIG); cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9; cfg.MODEL.GRAPH.TOP_K = 4; cfg.MODEL.GRAPH.RADIUS = 6.0; cfg.device = str(device); cfg.device_list = [device.index or 0]
    rows = []
    for variant in variants:
        for fold in folds:
            checkpoint = args.checkpoint_root / variant / fold / "f16_temporal_social_cross" / f"seed_{args.seed}" / "val_scene_-1" / "best_selected.pth"
            if not checkpoint.is_file():
                print(f"[skip-eval] missing {checkpoint}"); continue
            os.environ["HRMC_H3_CHECKPOINT"] = str(source_dir(fold, "h3", args.seed) / "best_selected.pth")
            os.environ["HRMC_V2_CHECKPOINT"] = str(source_dir(fold, "v2", args.seed) / "best_selected.pth")
            model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
            model_module.replace_decoder(model, cfg, VARIANTS[variant])
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            model = model.to(device).eval()
            dataset = TrajectoryDataset(DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy", cfg)
            if args.max_samples > 0:
                dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            split = np.load(source_dir(fold, "v2", args.seed) / "split_indices.npz")
            threshold = helper.source_thresholds(fold, cfg, np.asarray(split["train"], dtype=np.int64))
            data = collect(model, loader, device, normalize_to_focal_heading, helper)
            high = data["mean_acceleration"] >= threshold["mean_acceleration_q75"]
            for condition, mask in (("ALL", np.ones(len(high), dtype=bool)), ("high_acc", high), ("non_high_acc", ~high)):
                row = {"variant": variant, "fold": fold, "seed": args.seed, "condition": condition, "samples": int(mask.sum())}
                for metric in ("minADE", "minFDE", "sameFDE", "J"):
                    value = float(data[metric][mask].mean()); row[metric] = value
                    base = float(baseline[(fold, condition)][f"corrected_{'evalJ' if metric == 'J' else metric}"])
                    row[f"improvement_vs_v2_{metric}_pct"] = 100.0 * (base / value - 1.0)
                for step in (4, 8, 12):
                    for name in ("adapter_norm", "base_correction_norm", "final_correction_norm"):
                        row[f"{name}_t{step}"] = float(data[f"{name}_t{step}"][mask].mean())
                rows.append(row)
            print(f"[evaluated] {variant}/{fold}", flush=True)
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "condition_evaluation.csv", rows)
    complete = len([row for row in rows if row["condition"] == "ALL"]) == len(folds) * len(variants)
    summary = {"complete": complete, "rows": rows, "macro": {}}
    for variant in variants:
        for condition in ("ALL", "high_acc", "non_high_acc"):
            selected = [row for row in rows if row["variant"] == variant and row["condition"] == condition]
            if selected:
                summary["macro"][f"{variant}/{condition}"] = {
                    metric: float(np.mean([row[metric] for row in selected]))
                    for metric in ("minADE", "minFDE", "sameFDE", "J", "improvement_vs_v2_minADE_pct", "improvement_vs_v2_minFDE_pct", "improvement_vs_v2_J_pct")
                }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["macro"], indent=2), flush=True)


if __name__ == "__main__": main()
