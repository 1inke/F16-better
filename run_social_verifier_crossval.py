"""Run scene-held-out cross-validation for the social stimulus verifier.

This script performs the next development step without touching the target
ETH/UCY test split:

1. train one verifier while holding out each OOF-scorable source scene;
2. score the matching out-of-fold Q20 candidates;
3. search one global Q20/verifier fusion weight using scene-equal metrics;
4. compare verifier minimum scales and random seeds;
5. write machine-readable results and a Markdown report.

The existing ``train_social_stimulus_verifier.py`` file must be available.
By default this script tests min_scale={0.03, 0.05, 0.08} with seed 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch


@dataclass
class SceneCandidates:
    scene_id: int
    global_index: np.ndarray
    candidates: np.ndarray
    target: np.ndarray
    original_logits: np.ndarray


def parse_number_list(values: Sequence[str], cast):
    output = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                output.append(cast(token))
    return output


def safe_scale_tag(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def load_oof_candidates(oof_dir: Path) -> Dict[int, SceneCandidates]:
    parts: Dict[int, Dict[str, List[np.ndarray]]] = {}
    files = sorted(oof_dir.glob("fold*_candidates.npz"))
    if not files:
        raise FileNotFoundError(f"No fold*_candidates.npz in {oof_dir}")
    keys = ("global_index", "candidates", "target", "original_logits")
    for path in files:
        with np.load(path) as archive:
            scene_ids = archive["scene_id"].astype(np.int64)
            for scene_id in np.unique(scene_ids):
                mask = scene_ids == scene_id
                scene_parts = parts.setdefault(
                    int(scene_id), {key: [] for key in keys}
                )
                for key in keys:
                    scene_parts[key].append(np.asarray(archive[key][mask]))

    result: Dict[int, SceneCandidates] = {}
    for scene_id, scene_parts in parts.items():
        arrays = {
            key: np.concatenate(value, axis=0)
            for key, value in scene_parts.items()
        }
        order = np.argsort(arrays["global_index"], kind="stable")
        arrays = {key: value[order] for key, value in arrays.items()}
        if np.unique(arrays["global_index"]).size != len(order):
            raise ValueError(f"Duplicate OOF indices in scene {scene_id}")
        result[scene_id] = SceneCandidates(
            scene_id=scene_id,
            global_index=arrays["global_index"].astype(np.int64),
            candidates=arrays["candidates"].astype(np.float32),
            target=arrays["target"].astype(np.float32),
            original_logits=arrays["original_logits"].astype(np.float32),
        )
    return result


def standardize_per_sample(scores: np.ndarray) -> np.ndarray:
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return (scores - mean) / np.maximum(std, 1.0e-6)


def selection_metrics(
    ade: np.ndarray,
    fde: np.ndarray,
    selection: np.ndarray,
) -> Dict[str, float]:
    rows = np.arange(ade.shape[0])
    selected_ade = ade[rows, selection]
    selected_fde = fde[rows, selection]
    best = ade.argmin(axis=1)
    regret = selected_ade - ade.min(axis=1)
    return {
        "samples": int(ade.shape[0]),
        "ade": float(selected_ade.mean()),
        "fde": float(selected_fde.mean()),
        "hit_best_ade": float((selection == best).mean()),
        "regret": float(regret.mean()),
        "median_ade": float(np.median(selected_ade)),
        "ade_p95": float(np.quantile(selected_ade, 0.95)),
        "regret_p95": float(np.quantile(regret, 0.95)),
        "catastrophic_regret_gt_0_5": float((regret > 0.5).mean()),
        "active_modes": int(np.unique(selection).size),
    }


def load_verifier(checkpoint_path: Path, module_path: Path, device: torch.device):
    sys.path.insert(0, str(module_path.parent))
    from train_social_stimulus_verifier import (  # type: ignore
        SocialStimulusVerifier,
        VerifierConfig,
        rotate_to_heading,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = SocialStimulusVerifier(
        VerifierConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, rotate_to_heading, checkpoint


def score_scene(
    model,
    rotate_to_heading,
    scene: SceneCandidates,
    all_data: np.ndarray,
    all_num: np.ndarray,
    traj_scale: float,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    scores = []
    all_scales = []
    indices = scene.global_index
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            stop = start + batch_size
            batch_indices = indices[start:stop]
            past_np = np.asarray(
                all_data[batch_indices, :, :8, -2:], dtype=np.float32
            ) / float(traj_scale)
            num_np = np.asarray(all_num[batch_indices], dtype=np.int64)
            candidates_np = scene.candidates[start:stop]
            past = torch.from_numpy(past_np).to(device)
            num_valid = torch.from_numpy(num_np).to(device)
            candidates = torch.from_numpy(candidates_np).to(device)
            distribution = model.distribution(past, num_valid)
            centered = (
                candidates
                - distribution["origin"][:, None, None, :]
            )
            local_candidates = rotate_to_heading(
                centered, distribution["heading"]
            )
            batch_scores = model.score_candidates(
                distribution, local_candidates
            )
            scores.append(batch_scores.cpu().numpy())
            all_scales.append(distribution["scales"].cpu().numpy())

    scores_np = np.concatenate(scores, axis=0)
    scales_np = np.concatenate(all_scales, axis=0)
    diagnostics: Dict[str, object] = {
        "score_finite": bool(np.isfinite(scores_np).all()),
        "scale_at_min_fraction": float(
            (scales_np <= model.config.min_scale + 1.0e-6).mean()
        ),
        "scale_quantiles": np.quantile(
            scales_np, [0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0]
        ).tolist(),
    }
    if not diagnostics["score_finite"]:
        raise FloatingPointError(
            f"Non-finite verifier scores in scene {scene.scene_id}"
        )
    return scores_np, diagnostics


def run_training(
    python: Path,
    trainer: Path,
    project_root: Path,
    config_file: str,
    output_dir: Path,
    scene_id: int,
    seed: int,
    min_scale: float,
    epochs: int,
    patience: int,
    batch_size: int,
    device: str,
    force: bool,
) -> Path:
    checkpoint = output_dir / "verifier_best.pth"
    if checkpoint.is_file() and not force:
        print(f"[skip] existing checkpoint: {checkpoint}")
        return checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(trainer),
        "--project-root",
        str(project_root),
        "--config-file",
        config_file,
        "--output-dir",
        str(output_dir),
        "--val-scene-id",
        str(scene_id),
        "--seed",
        str(seed),
        "--min-scale",
        str(min_scale),
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--no-amp",
    ]
    log_path = output_dir / "train_console.log"
    print(
        f"[train] scale={min_scale:.3f} seed={seed} "
        f"val_scene={scene_id}"
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=project_root,
            check=False,
            env=os.environ.copy(),
        )
    if completed.returncode != 0 or not checkpoint.is_file():
        raise RuntimeError(
            f"Training failed for scale={min_scale}, seed={seed}, "
            f"scene={scene_id}. See {log_path}"
        )
    return checkpoint


def evaluate_run(
    verifier_scores: np.ndarray,
    scene: SceneCandidates,
    alpha_grid: np.ndarray,
) -> Dict[str, object]:
    errors = np.linalg.norm(
        scene.candidates - scene.target[:, None], axis=-1
    )
    ade = errors.mean(axis=-1)
    fde = errors[:, :, -1]
    rows = np.arange(len(ade))
    original_selection = scene.original_logits.argmax(axis=1)
    verifier_selection = verifier_scores.argmax(axis=1)
    z_original = standardize_per_sample(scene.original_logits)
    z_verifier = standardize_per_sample(verifier_scores)
    alpha_metrics: Dict[str, Dict[str, float]] = {}
    for alpha in alpha_grid:
        fused = (1.0 - alpha) * z_original + alpha * z_verifier
        selection = fused.argmax(axis=1)
        alpha_metrics[f"{alpha:.4f}"] = selection_metrics(
            ade, fde, selection
        )
    best = ade.argmin(axis=1)
    return {
        "oracle": {
            "ade": float(ade.min(axis=1).mean()),
            "same_path_fde": float(fde[rows, best].mean()),
        },
        "original": selection_metrics(ade, fde, original_selection),
        "verifier_only": selection_metrics(ade, fde, verifier_selection),
        "alpha_metrics": alpha_metrics,
    }


def aggregate_alpha_grid(
    fold_results: Mapping[str, Mapping[str, object]],
    alpha_grid: np.ndarray,
) -> List[Dict[str, float]]:
    aggregates = []
    for alpha in alpha_grid:
        key = f"{alpha:.4f}"
        rows = [
            result["evaluation"]["alpha_metrics"][key]
            for result in fold_results.values()
        ]
        original_rows = [
            result["evaluation"]["original"]
            for result in fold_results.values()
        ]
        mean_ade = float(np.mean([row["ade"] for row in rows]))
        mean_fde = float(np.mean([row["fde"] for row in rows]))
        original_mean_ade = float(
            np.mean([row["ade"] for row in original_rows])
        )
        original_mean_fde = float(
            np.mean([row["fde"] for row in original_rows])
        )
        aggregates.append(
            {
                "alpha": float(alpha),
                "scene_seed_mean_ade": mean_ade,
                "scene_seed_mean_fde": mean_fde,
                "ade_delta_vs_original": mean_ade - original_mean_ade,
                "fde_delta_vs_original": mean_fde - original_mean_fde,
                "ade_better_folds": int(
                    sum(
                        row["ade"] < original["ade"]
                        for row, original in zip(rows, original_rows)
                    )
                ),
                "fde_better_folds": int(
                    sum(
                        row["fde"] < original["fde"]
                        for row, original in zip(rows, original_rows)
                    )
                ),
                "folds": len(rows),
                "mean_regret": float(
                    np.mean([row["regret"] for row in rows])
                ),
                "mean_ade_p95": float(
                    np.mean([row["ade_p95"] for row in rows])
                ),
                "mean_catastrophic_rate": float(
                    np.mean(
                        [
                            row["catastrophic_regret_gt_0_5"]
                            for row in rows
                        ]
                    )
                ),
            }
        )
    return aggregates


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    selected: Mapping[str, object],
    scale_summaries: Sequence[Mapping[str, object]],
    fold_results: Mapping[str, Mapping[str, object]],
    test_accessed: bool,
) -> str:
    lines = [
        "# Social stimulus verifier cross-validation",
        "",
        f"- Target test split accessed: **{str(test_accessed).lower()}**",
        "- Fusion weights were selected using scene-equal OOF metrics.",
        "- Negative deltas indicate improvement over original Q20 logits.",
        "",
        "## Selected configuration",
        "",
        f"- min_scale: `{selected['min_scale']}`",
        f"- alpha: `{selected['alpha']:.4f}`",
        f"- scene/seed mean ADE: `{selected['scene_seed_mean_ade']:.4f}`",
        f"- scene/seed mean FDE: `{selected['scene_seed_mean_fde']:.4f}`",
        f"- ADE delta: `{selected['ade_delta_vs_original']:+.4f}`",
        f"- FDE delta: `{selected['fde_delta_vs_original']:+.4f}`",
        f"- ADE better folds: `{selected['ade_better_folds']}/{selected['folds']}`",
        f"- FDE better folds: `{selected['fde_better_folds']}/{selected['folds']}`",
        "",
        "## Scale comparison",
        "",
        "| min_scale | alpha | ADE | FDE | ADE delta | FDE delta | ADE wins | FDE wins |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scale_summaries:
        lines.append(
            f"| {row['min_scale']:.3f} | {row['alpha']:.2f} | "
            f"{row['scene_seed_mean_ade']:.4f} | "
            f"{row['scene_seed_mean_fde']:.4f} | "
            f"{row['ade_delta_vs_original']:+.4f} | "
            f"{row['fde_delta_vs_original']:+.4f} | "
            f"{row['ade_better_folds']}/{row['folds']} | "
            f"{row['fde_better_folds']}/{row['folds']} |"
        )

    lines.extend(
        [
            "",
            "## Selected configuration by fold",
            "",
            "| scale | seed | scene | N | original ADE/FDE | fused ADE/FDE | hit | regret | clamp rate |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    selected_scale = float(selected["min_scale"])
    alpha_key = f"{float(selected['alpha']):.4f}"
    for key, result in sorted(fold_results.items()):
        if not math.isclose(float(result["min_scale"]), selected_scale):
            continue
        evaluation = result["evaluation"]
        original = evaluation["original"]
        fused = evaluation["alpha_metrics"][alpha_key]
        diagnostics = result["diagnostics"]
        lines.append(
            f"| {result['min_scale']:.3f} | {result['seed']} | "
            f"{result['scene_id']} | {original['samples']} | "
            f"{original['ade']:.4f}/{original['fde']:.4f} | "
            f"{fused['ade']:.4f}/{fused['fde']:.4f} | "
            f"{fused['hit_best_ade']:.3f} | {fused['regret']:.4f} | "
            f"{diagnostics['scale_at_min_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "Proceed to final all-training-scenes verifier only if the selected "
            "configuration improves ADE and FDE in most held-out scenes and "
            "does not materially increase P95 ADE or catastrophic regret.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validate and calibrate the social verifier."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"),
    )
    parser.add_argument(
        "--trainer",
        type=Path,
        default=Path(__file__).with_name(
            "train_social_stimulus_verifier.py"
        ),
    )
    parser.add_argument(
        "--config-file", default="configs/motif_rbf_univ.yaml"
    )
    parser.add_argument(
        "--oof-dir", default="results/oof_generator_gate_3fold"
    )
    parser.add_argument(
        "--output-dir", default="results/social_verifier_crossval"
    )
    parser.add_argument(
        "--scenes", nargs="+", default=None,
        help="Scene ids, e.g. --scenes 0 1 2 3 4. Default: all OOF scenes.",
    )
    parser.add_argument(
        "--seeds", nargs="+", default=["1"],
        help="Verifier seeds; accepts spaces or commas.",
    )
    parser.add_argument(
        "--min-scales", nargs="+", default=["0.03", "0.05", "0.08"],
        help="Minimum GMM scales; accepts spaces or commas.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    trainer = args.trainer.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(project_root)
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    oof_dir = Path(args.oof_dir)
    if not oof_dir.is_absolute():
        oof_dir = project_root / oof_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_root = project_root / ".cache"
    matplotlib_cache = cache_root / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    seeds = parse_number_list(args.seeds, int)
    min_scales = parse_number_list(args.min_scales, float)
    scene_data = load_oof_candidates(oof_dir)
    if args.scenes is None:
        scenes = sorted(scene_data)
    else:
        scenes = parse_number_list(args.scenes, int)
    missing_scenes = sorted(set(scenes) - set(scene_data))
    if missing_scenes:
        raise ValueError(f"Scenes absent from OOF candidates: {missing_scenes}")

    config_path = Path(args.config_file)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    import yaml  # type: ignore

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    subset = raw_config["SUBSET"]
    traj_scale = float(raw_config["PREDICTION"]["TRAJ_SCALE"])
    data_root = project_root / "datasets/eth_ucy/processed_data_diverse"
    all_data = np.load(
        data_root / f"{subset}_data_train.npy", mmap_mode="r"
    )
    all_num = np.load(
        data_root / f"{subset}_num_train.npy", mmap_mode="r"
    )
    alpha_count = int(round(1.0 / args.alpha_step))
    alpha_grid = np.linspace(0.0, 1.0, alpha_count + 1)

    fold_results: Dict[str, Dict[str, object]] = {}
    for min_scale in min_scales:
        for seed in seeds:
            for scene_id in scenes:
                run_dir = (
                    output_dir
                    / f"scale_{safe_scale_tag(min_scale)}"
                    / f"seed_{seed}"
                    / f"val_scene_{scene_id}"
                )
                checkpoint_path = run_training(
                    python=Path(sys.executable),
                    trainer=trainer,
                    project_root=project_root,
                    config_file=str(config_path),
                    output_dir=run_dir,
                    scene_id=scene_id,
                    seed=seed,
                    min_scale=min_scale,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.train_batch_size,
                    device=args.device,
                    force=args.force,
                )
                device = torch.device(args.device)
                model, rotate_to_heading, checkpoint = load_verifier(
                    checkpoint_path, trainer, device
                )
                scores, diagnostics = score_scene(
                    model,
                    rotate_to_heading,
                    scene_data[scene_id],
                    all_data,
                    all_num,
                    traj_scale,
                    device,
                    args.score_batch_size,
                )
                evaluation = evaluate_run(
                    scores, scene_data[scene_id], alpha_grid
                )
                key = (
                    f"scale={min_scale:.3f}|seed={seed}|scene={scene_id}"
                )
                fold_results[key] = {
                    "min_scale": float(min_scale),
                    "seed": int(seed),
                    "scene_id": int(scene_id),
                    "best_epoch": int(checkpoint["epoch"]),
                    "validation_nll": float(
                        checkpoint["validation_metrics"]["nll"]
                    ),
                    "checkpoint": str(checkpoint_path),
                    "diagnostics": diagnostics,
                    "evaluation": evaluation,
                }
                with (run_dir / "oof_evaluation.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(
                        fold_results[key], handle, indent=2,
                        ensure_ascii=False
                    )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    scale_summaries = []
    alpha_tables: Dict[str, List[Dict[str, float]]] = {}
    for min_scale in min_scales:
        selected_folds = {
            key: value
            for key, value in fold_results.items()
            if math.isclose(float(value["min_scale"]), min_scale)
        }
        alpha_rows = aggregate_alpha_grid(selected_folds, alpha_grid)
        alpha_tables[f"{min_scale:.3f}"] = alpha_rows
        # Select by scene-equal mean ADE; FDE is the deterministic tie-break.
        best = min(
            alpha_rows,
            key=lambda row: (
                row["scene_seed_mean_ade"],
                row["scene_seed_mean_fde"],
            ),
        )
        scale_summaries.append({"min_scale": min_scale, **best})

    selected = min(
        scale_summaries,
        key=lambda row: (
            row["scene_seed_mean_ade"],
            row["scene_seed_mean_fde"],
        ),
    )
    payload = {
        "protocol": {
            "target_test_split_accessed": False,
            "selection_unit": "equal_weight_per_scene_and_seed",
            "scenes": scenes,
            "seeds": seeds,
            "min_scales": min_scales,
            "alpha_grid": alpha_grid.tolist(),
        },
        "selected": selected,
        "scale_summaries": scale_summaries,
        "alpha_tables": alpha_tables,
        "fold_results": fold_results,
    }
    with (output_dir / "crossval_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_csv(output_dir / "scale_summary.csv", scale_summaries)
    for scale, rows in alpha_tables.items():
        write_csv(output_dir / f"alpha_grid_scale_{scale}.csv", rows)
    report = build_report(
        selected,
        scale_summaries,
        fold_results,
        test_accessed=False,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    main()
