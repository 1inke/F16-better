"""Zero-training audit of F16 individual-motion representation after EXP-169B/B1."""
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
    print(f"[bootstrap] relaunching with {ASTRA_PYTHON}", flush=True)
    raise SystemExit(subprocess.run([str(ASTRA_PYTHON), *sys.argv]).returncode)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
CONFIG = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
V2_MODULE = WORKSPACE / "ANCHOR_CORRECTION_VALIDATION_SERIES" / "anchor_correction_validation_models.py"
SOURCE = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "EXP_156B_FOLD_LOCAL_CURRENT_DECODER" / "results"
BASELINE = WORKSPACE / "OFFICIAL_ETH_UCY_FINAL_3SEED_SERIES" / "EXP_164_FINAL_V2_OFFICIAL_3SEED" / "summary.json"
OUTPUT = ROOT / "MOTION_REPRESENTATION_AUDIT"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
REPRESENTATIONS = ("robust_state", "temporal", "post_social", "encoder_output")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_dir(fold: str, stage: str, seed: int) -> Path:
    return SOURCE / fold / stage / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def build_model(fold, seed, cfg, model_class, device):
    h3 = run_dir(fold, "h3", seed) / "best_selected.pth"
    v2 = run_dir(fold, "v2", seed) / "best_selected.pth"
    os.environ["EXP143_SOURCE_CHECKPOINT"] = str(h3)
    model = model_class(cfg, "f16_temporal_social_cross")
    adapter = load_module(f"motion_audit_v2_{fold}", V2_MODULE)
    adapter.replace_decoder(model, cfg, "v2_wide_endpoint_correction")
    model.load_state_dict(torch.load(v2, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def observation_features(past: np.ndarray) -> dict[str, np.ndarray]:
    focal = past[:, 0]
    velocity = np.diff(focal, axis=1)
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration = np.linalg.norm(np.diff(velocity, axis=1), axis=-1)
    start, end = focal[:, 0], focal[:, -1]
    chord = end - start
    relative = focal - start[:, None]
    cross = np.abs(relative[..., 0] * chord[:, None, 1] - relative[..., 1] * chord[:, None, 0])
    denominator = np.linalg.norm(chord, axis=-1)
    return {
        "mean_step_speed": speed.mean(axis=1),
        "last_step_speed": speed[:, -1],
        "mean_acceleration": acceleration.mean(axis=1),
        "max_acceleration": acceleration.max(axis=1),
        "observation_nonlinearity": np.mean(cross / np.maximum(denominator[:, None], 1e-6), axis=1),
    }


def source_thresholds(fold, cfg, indices):
    raw = np.load(DATA / f"{fold}_data_train.npy", mmap_mode="r")
    past = np.asarray(raw[indices, : int(cfg.DATA.MAX_AGENTS), :8, -2:], dtype=np.float32)
    past /= float(cfg.PREDICTION.TRAJ_SCALE)
    obs = observation_features(past)
    result = {}
    for name in ("mean_acceleration", "max_acceleration", "observation_nonlinearity"):
        q = np.quantile(obs[name], (0.25, 0.50, 0.75))
        for i, value in enumerate(q, 1):
            result[f"{name}_q{i * 25}"] = float(value)
    return result


def sequence_statistics(sequence: torch.Tensor, prefix: str) -> dict[str, np.ndarray]:
    # sequence: [B,T,D]
    norms = torch.linalg.vector_norm(sequence, dim=-1)
    first, last = sequence[:, 0], sequence[:, -1]
    cosine = torch.nn.functional.cosine_similarity(first, last, dim=-1)
    return {
        f"{prefix}_mean_norm": norms.mean(-1).cpu().numpy(),
        f"{prefix}_last_norm": norms[:, -1].cpu().numpy(),
        f"{prefix}_delta_norm": torch.linalg.vector_norm(last - first, dim=-1).cpu().numpy(),
        f"{prefix}_first_last_cosine": cosine.cpu().numpy(),
        f"{prefix}_temporal_std": sequence.std(dim=1, unbiased=False).mean(-1).cpu().numpy(),
    }


@torch.inference_mode()
def collect(model, loader, device, normalize, sample_ids):
    cache = {}

    def robust_hook(_module, _args, output):
        cache["robust"] = output.detach()

    def encoder_hook(_module, _args, output):
        cache["encoder"] = output.detach()

    robust_handle = model.robust_state.register_forward_hook(robust_hook)
    encoder_handle = model.agent_aware_trans_encoder.register_forward_hook(encoder_hook)
    scalar_parts: dict[str, list[np.ndarray]] = {}
    vector_parts = {name: [] for name in REPRESENTATIONS}
    offset = 0
    try:
        for past, future, num, _, _, _ in loader:
            batch = len(past)
            past = past.to(device)
            future = future.to(device)
            num = num.to(device).int()
            past, future = normalize(past, future)
            center = past.mean(dim=-2, keepdim=True)
            prediction = model(past - center, None, None, mode="testing", num_valid=num, graph_loc=past)[2]
            prediction = prediction + center.unsqueeze(-3)
            candidates, target = prediction[:, 0], future[:, 0]
            error = torch.linalg.vector_norm(candidates - target[:, None], dim=-1)
            ade, fde = error.mean(-1), error[..., -1]
            min_ade, min_fde = ade.min(-1).values, fde.min(-1).values
            winner = ade.argmin(-1)
            row = torch.arange(batch, device=device)

            robust_sequence = cache["robust"][:, 0]
            temporal_sequence = model.agent_aware_trans_encoder.last_temporal_features[:, 0].detach()
            post_social = model.agent_aware_trans_encoder.last_agent_features[:, 0].detach()
            encoded = cache["encoder"].reshape(batch, model.obs_len, model.num_pedestrians, -1)[:, :, 0]
            vectors = {
                "robust_state": robust_sequence.mean(1),
                "temporal": temporal_sequence.mean(1),
                "post_social": post_social,
                "encoder_output": encoded.mean(1),
            }
            for name, value in vectors.items():
                vector_parts[name].append(value.cpu().numpy())

            scalars = {
                "sample_id": np.asarray(sample_ids[offset: offset + batch], dtype=np.int64),
                "past": past.cpu().numpy(),
                "minADE": min_ade.cpu().numpy(),
                "minFDE": min_fde.cpu().numpy(),
                "sameFDE": fde[row, winner].cpu().numpy(),
                "J": (min_ade + 0.5 * min_fde).cpu().numpy(),
                **sequence_statistics(robust_sequence, "robust"),
                **sequence_statistics(temporal_sequence, "temporal"),
                **sequence_statistics(encoded, "encoder"),
                "post_social_norm": torch.linalg.vector_norm(post_social, dim=-1).cpu().numpy(),
                "post_social_channel_std": post_social.std(dim=-1, unbiased=False).cpu().numpy(),
                "robust_saturation_fraction": (robust_sequence.abs() >= 0.90).float().mean(dim=(1, 2)).cpu().numpy(),
            }
            for key, value in scalars.items():
                scalar_parts.setdefault(key, []).append(value)
            offset += batch
    finally:
        robust_handle.remove()
        encoder_handle.remove()
    scalars = {key: np.concatenate(value) for key, value in scalar_parts.items()}
    vectors = {key: np.concatenate(value) for key, value in vector_parts.items()}
    scalars.update(observation_features(scalars.pop("past")))
    return scalars, vectors


def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    result = np.empty(len(x), dtype=np.float64)
    sorted_x = np.asarray(x)[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def ridge_probe(train_x, train_y, test_x, test_y, alpha=1e-2):
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    test_y = np.asarray(test_y, dtype=np.float64)
    mean = train_x.mean(0)
    scale = train_x.std(0)
    scale[scale < 1e-6] = 1.0
    x_train = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    y_mean = train_y.mean()
    gram = x_train.T @ x_train + alpha * np.eye(x_train.shape[1])
    weights = np.linalg.solve(gram, x_train.T @ (train_y - y_mean))
    prediction = x_test @ weights + y_mean
    denominator = np.square(test_y - test_y.mean()).sum()
    r2 = 1.0 - np.square(test_y - prediction).sum() / denominator if denominator > 1e-12 else np.nan
    return float(r2), pearson(prediction, test_y), spearman(prediction, test_y)


def effect_size(high, low):
    high, low = np.asarray(high, dtype=np.float64), np.asarray(low, dtype=np.float64)
    if len(high) < 2 or len(low) < 2:
        return None
    pooled = np.sqrt((high.var(ddof=1) + low.var(ddof=1)) / 2.0)
    return float((high.mean() - low.mean()) / pooled) if pooled > 1e-12 else 0.0


def condition_rows(fold, scope, data, thresholds):
    acc_edges = [thresholds[f"mean_acceleration_q{x}"] for x in (25, 50, 75)]
    nonlin_edges = [thresholds[f"observation_nonlinearity_q{x}"] for x in (25, 50, 75)]
    bins = {
        **{f"acc_Q{i + 1}": np.digitize(data["mean_acceleration"], acc_edges) == i for i in range(4)},
        **{f"nonlinearity_Q{i + 1}": np.digitize(data["observation_nonlinearity"], nonlin_edges) == i for i in range(4)},
    }
    rows = []
    scalar_names = (
        "minADE", "minFDE", "J", "robust_mean_norm", "robust_delta_norm",
        "robust_saturation_fraction", "temporal_mean_norm", "temporal_delta_norm",
        "post_social_norm", "encoder_mean_norm", "encoder_delta_norm",
    )
    for name, mask in bins.items():
        row = {"fold": fold, "scope": scope, "condition": name, "samples": int(mask.sum())}
        for metric in scalar_names:
            row[metric] = float(data[metric][mask].mean()) if mask.any() else None
        rows.append(row)
    return rows


def fold_summary(fold, outer, threshold):
    high = outer["mean_acceleration"] >= threshold["mean_acceleration_q75"]
    low = ~high
    row = {"fold": fold, "samples": len(high), "high_acc_samples": int(high.sum())}
    for metric in ("minADE", "minFDE", "J"):
        hv, lv = float(outer[metric][high].mean()), float(outer[metric][low].mean())
        row[f"high_acc_{metric}"] = hv
        row[f"non_high_acc_{metric}"] = lv
        row[f"high_acc_{metric}_gap_pct"] = 100.0 * (hv / lv - 1.0)
    for metric in ("robust_mean_norm", "temporal_mean_norm", "post_social_norm", "encoder_mean_norm"):
        row[f"high_acc_{metric}"] = float(outer[metric][high].mean())
        row[f"non_high_acc_{metric}"] = float(outer[metric][low].mean())
        row[f"high_vs_non_high_{metric}_effect_size"] = effect_size(outer[metric][high], outer[metric][low])
    row["robust_norm_acc_spearman"] = spearman(outer["robust_mean_norm"], outer["mean_acceleration"])
    row["temporal_norm_acc_spearman"] = spearman(outer["temporal_mean_norm"], outer["mean_acceleration"])
    row["encoder_norm_acc_spearman"] = spearman(outer["encoder_mean_norm"], outer["mean_acceleration"])
    row["robust_norm_error_spearman"] = spearman(outer["robust_mean_norm"], outer["J"])
    row["temporal_norm_error_spearman"] = spearman(outer["temporal_mean_norm"], outer["J"])
    row["encoder_norm_error_spearman"] = spearman(outer["encoder_mean_norm"], outer["J"])
    return row


def verify_reproduction(fold, outer, seed):
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    expected = next(x for x in payload["rows"] if x["fold"] == fold and int(x["seed"]) == seed)
    mapping = {"minADE": "minADE", "minFDE": "minFDE", "sameFDE": "same_path_FDE"}
    checks = {}
    for observed_name, expected_name in mapping.items():
        observed = float(outer[observed_name].mean())
        target = float(expected[expected_name])
        difference = observed - target
        checks[observed_name] = {"expected": target, "observed": observed, "absolute_difference": difference}
        if abs(difference) > 2e-5:
            raise RuntimeError(f"Official V2 reproduction failed: {fold}/{observed_name}: {difference}")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="Smoke-test cap per outer/inner split; 0 means full.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    folds = FOLDS if "all" in args.folds else tuple(args.folds)

    required = [CONFIG, V2_MODULE, BASELINE]
    for fold in folds:
        required += [
            run_dir(fold, "h3", args.seed) / "best_selected.pth",
            run_dir(fold, "v2", args.seed) / "best_selected.pth",
            run_dir(fold, "v2", args.seed) / "split_indices.npz",
            DATA / f"{fold}_data_train.npy", DATA / f"{fold}_num_train.npy",
            DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy",
        ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    if args.dry_run:
        print(json.dumps({"audit": "individual_motion_representation", "training": False, "folds": folds, "seed": args.seed, "output": str(args.output)}, indent=2))
        return

    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(WORKSPACE / "FRONT_ENCODER_SERIES")]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from train_front_variant import TrajectoryDataset, normalize_to_focal_heading

    device = torch.device(args.device)
    cfg = load_locked_q20(CONFIG)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0]

    per_sample, bins, fold_rows, probes, threshold_rows, reproduction = [], [], [], [], [], {}
    for fold in folds:
        print(f"[motion-audit] {fold}", flush=True)
        split = np.load(run_dir(fold, "v2", args.seed) / "split_indices.npz")
        train_indices = np.asarray(split["train"], dtype=np.int64)
        inner_indices = np.asarray(split["validation"], dtype=np.int64)
        threshold = source_thresholds(fold, cfg, train_indices)
        threshold_rows.append({"fold": fold, **threshold})
        model = build_model(fold, args.seed, cfg, FrontEncoderASTRA, device)

        datasets = {
            "inner": (TrajectoryDataset(DATA / f"{fold}_data_train.npy", DATA / f"{fold}_num_train.npy", cfg), inner_indices),
            "outer": (TrajectoryDataset(DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy", cfg), None),
        }
        collected = {}
        for scope in ("inner", "outer"):
            dataset, indices = datasets[scope]
            if indices is None:
                indices = np.arange(len(dataset), dtype=np.int64)
            if args.max_samples:
                indices = indices[: args.max_samples]
            subset = Subset(dataset, indices.tolist())
            loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            print(f"[motion-audit] {fold}/{scope} n={len(subset)}", flush=True)
            scalars, vectors = collect(model, loader, device, normalize_to_focal_heading, indices)
            collected[scope] = (scalars, vectors)
            bins += condition_rows(fold, scope, scalars, threshold)
            for i in range(len(indices)):
                row = {"fold": fold, "scope": scope}
                for name, values in scalars.items():
                    row[name] = int(values[i]) if name == "sample_id" else float(values[i])
                per_sample.append(row)

        inner, inner_vectors = collected["inner"]
        outer, outer_vectors = collected["outer"]
        fold_rows.append(fold_summary(fold, outer, threshold))
        if not args.max_samples:
            reproduction[fold] = verify_reproduction(fold, outer, args.seed)
        targets = {
            "mean_acceleration": (np.log1p(inner["mean_acceleration"]), np.log1p(outer["mean_acceleration"])),
            "observation_nonlinearity": (np.log1p(inner["observation_nonlinearity"]), np.log1p(outer["observation_nonlinearity"])),
            "future_J": (np.log1p(inner["J"]), np.log1p(outer["J"])),
        }
        for representation in REPRESENTATIONS:
            for target, (train_y, test_y) in targets.items():
                r2, corr, rank_corr = ridge_probe(inner_vectors[representation], train_y, outer_vectors[representation], test_y)
                probes.append({
                    "fold": fold, "representation": representation, "target": target,
                    "train_samples": len(train_y), "test_samples": len(test_y),
                    "outer_r2": r2, "outer_pearson": corr, "outer_spearman": rank_corr,
                })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    complete = set(folds) == set(FOLDS) and args.max_samples == 0
    suffix = "" if complete else "_partial"
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / f"per_sample_representation{suffix}.csv", per_sample)
    write_csv(args.output / f"per_fold_summary{suffix}.csv", fold_rows)
    write_csv(args.output / f"condition_bins{suffix}.csv", bins)
    write_csv(args.output / f"linear_probe_summary{suffix}.csv", probes)
    write_csv(args.output / f"source_thresholds{suffix}.csv", threshold_rows)
    macro = {}
    if fold_rows:
        for key in fold_rows[0]:
            if key != "fold" and all(isinstance(row.get(key), (int, float)) and row.get(key) is not None for row in fold_rows):
                macro[key] = float(np.mean([row[key] for row in fold_rows]))
    payload = {
        "audit": "individual_motion_representation",
        "complete": complete,
        "training": False,
        "model": "EXP-164 official F16+H3+V2 seed1",
        "full_model_reproduction": reproduction,
        "macro_fold_summary": macro,
        "per_fold": fold_rows,
        "linear_probes": probes,
        "interpretation_guardrail": "Norm correlations alone do not prove information absence; use them together with cross-domain linear decodability and error stratification.",
        "next_step": "manual mechanism decision after audit; do not train a new encoder from norm-only evidence",
    }
    (args.output / f"summary{suffix}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"complete": complete, "macro": macro, "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
