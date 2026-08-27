"""Train final full-data verifiers and run the locked UNIV test once.

Locked development choices
--------------------------
- verifier min_scale: 0.03
- full-training epochs: 24
- fusion: 0.70 * z(Q20 logits) + 0.30 * z(verifier scores)
- paired Q20/verifier seeds: 1, 2, 3

No validation or test metric is used for model/epoch/hyperparameter selection.
The test split is accessed only after each full-data verifier has finished.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


LOCKED_MIN_SCALE = 0.03
LOCKED_EPOCHS = 24
LOCKED_ALPHA = 0.30
LOCKED_SEEDS = (1, 2, 3)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def standardize(scores: Tensor) -> Tensor:
    mean = scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, keepdim=True, unbiased=False)
    return (scores - mean) / std.clamp_min(1.0e-6)


def gather(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return values[np.arange(values.shape[0]), indices]


def selection_metrics(
    errors: np.ndarray,
    selection: np.ndarray,
) -> Dict[str, float]:
    mode_ade = errors.mean(axis=-1)
    mode_fde = errors[:, :, -1]
    selected_ade = gather(mode_ade, selection)
    selected_fde = gather(mode_fde, selection)
    best = mode_ade.argmin(axis=-1)
    regret = selected_ade - mode_ade.min(axis=-1)
    return {
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


def mean_sd(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available.")
    return device


def save_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def train_full_verifier(
    seed: int,
    output_dir: Path,
    dataset,
    model_config,
    verifier_class,
    run_epoch,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> Path:
    checkpoint_path = output_dir / "verifier_full_epoch24.pth"
    metadata_path = output_dir / "training_summary.json"
    if checkpoint_path.is_file() and metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        locked = metadata.get("locked_configuration", {})
        if (
            int(locked.get("epochs", -1)) == LOCKED_EPOCHS
            and math.isclose(
                float(locked.get("min_scale", -1.0)),
                LOCKED_MIN_SCALE,
            )
            and int(locked.get("seed", -1)) == seed
        ):
            print(f"[skip] full verifier seed {seed}: {checkpoint_path}")
            return checkpoint_path
        raise RuntimeError(
            f"Existing checkpoint metadata does not match locked config: "
            f"{metadata_path}. Use --force to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )
    model = verifier_class(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3.0e-4, weight_decay=1.0e-4
    )
    history = []
    started = time.perf_counter()
    for epoch in range(1, LOCKED_EPOCHS + 1):
        metrics = run_epoch(
            model=model,
            loader=loader,
            device=device,
            optimizer=optimizer,
            grad_clip=5.0,
            use_amp=False,
        )
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": metrics,
        }
        history.append(record)
        print(
            f"[verifier seed={seed}] epoch={epoch:02d} "
            f"NLL={metrics['nll']:.4f} "
            f"topADE={metrics['top1_ade']:.4f} "
            f"oracleADE={metrics['oracle_ade']:.4f}"
        )
        save_json(output_dir / "training_history.json", history)

    checkpoint = {
        "format_version": 1,
        "model_type": "ground_truth_social_stimulus_verifier_full_data",
        "model_config": asdict(model_config),
        "model_state": model.state_dict(),
        "epoch": LOCKED_EPOCHS,
        "seed": seed,
        "train_samples": len(dataset),
        "target_test_split_accessed_during_training": False,
        "locked_fusion_alpha": LOCKED_ALPHA,
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
        "final_training_metrics": history[-1]["train"],
        "locked_configuration": {
            "seed": seed,
            "epochs": LOCKED_EPOCHS,
            "min_scale": LOCKED_MIN_SCALE,
            "alpha": LOCKED_ALPHA,
        },
        "train_samples": len(dataset),
        "validation_used": False,
        "target_test_split_accessed": False,
    }
    save_json(metadata_path, summary)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_path


def load_full_verifier(
    checkpoint_path: Path,
    verifier_class,
    verifier_config_class,
    device: torch.device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint["epoch"]) != LOCKED_EPOCHS:
        raise ValueError("Full verifier checkpoint epoch is not locked epoch 24.")
    config = verifier_config_class(**checkpoint["model_config"])
    if not math.isclose(config.min_scale, LOCKED_MIN_SCALE):
        raise ValueError("Full verifier checkpoint min_scale is not 0.03.")
    model = verifier_class(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()


def evaluate_pair(
    seed: int,
    q20_checkpoint: Path,
    verifier_checkpoint: Path,
    base_cfg,
    configure_q20,
    prepare_runtime_cfg,
    build_dataset,
    astra_class,
    verifier_class,
    verifier_config_class,
    rotate_to_heading,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    stored_result_dir: Path,
    output_dir: Path,
) -> Dict[str, object]:
    cfg = prepare_runtime_cfg(
        configure_q20(copy.deepcopy(base_cfg)),
        device,
        batch_size,
    )
    dataset, _ = build_dataset(cfg, None)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    q20 = astra_class(cfg).to(device).eval()
    q20.load_state_dict(
        torch.load(q20_checkpoint, map_location=device, weights_only=True),
        strict=True,
    )
    verifier = load_full_verifier(
        verifier_checkpoint,
        verifier_class,
        verifier_config_class,
        device,
    )

    all_errors: List[np.ndarray] = []
    original_logits: List[np.ndarray] = []
    verifier_scores: List[np.ndarray] = []
    original_indices: List[np.ndarray] = []
    fused_indices: List[np.ndarray] = []
    scale_at_floor = 0
    scale_total = 0

    with torch.inference_mode():
        for batch in loader:
            past, future, num_valid, _, _, _ = batch
            past = past.to(device=device, dtype=torch.float32)
            future = future.to(device=device, dtype=torch.float32)
            num_valid = num_valid.to(device=device).int()

            center = past.mean(dim=-2, keepdim=True)
            prediction = q20(
                past - center,
                None,
                None,
                mode="testing",
                num_valid=num_valid,
                graph_loc=past,
            )[2]
            prediction = prediction + center.unsqueeze(-3)
            candidates = prediction[:, 0]
            target = future[:, 0]
            logits = q20.last_mode_logits[:, 0]

            distribution = verifier.distribution(past, num_valid)
            centered_candidates = (
                candidates
                - distribution["origin"][:, None, None, :]
            )
            local_candidates = rotate_to_heading(
                centered_candidates, distribution["heading"]
            )
            scores = verifier.score_candidates(
                distribution, local_candidates
            )
            fused = (
                (1.0 - LOCKED_ALPHA) * standardize(logits)
                + LOCKED_ALPHA * standardize(scores)
            )
            errors = torch.linalg.vector_norm(
                candidates - target[:, None], dim=-1
            )

            all_errors.append(errors.cpu().numpy())
            original_logits.append(logits.cpu().numpy())
            verifier_scores.append(scores.cpu().numpy())
            original_indices.append(logits.argmax(dim=-1).cpu().numpy())
            fused_indices.append(fused.argmax(dim=-1).cpu().numpy())
            scales = distribution["scales"]
            scale_at_floor += int(
                (scales <= LOCKED_MIN_SCALE + 1.0e-6).sum().item()
            )
            scale_total += scales.numel()

    errors_np = np.concatenate(all_errors)
    logits_np = np.concatenate(original_logits)
    verifier_np = np.concatenate(verifier_scores)
    original_np = np.concatenate(original_indices)
    fused_np = np.concatenate(fused_indices)
    mode_ade = errors_np.mean(axis=-1)
    best_ade = mode_ade.argmin(axis=-1)
    best_fde = errors_np[:, :, -1].argmin(axis=-1)

    original_metrics = selection_metrics(errors_np, original_np)
    fused_metrics = selection_metrics(errors_np, fused_np)
    oracle = {
        "minade20": float(mode_ade.min(axis=-1).mean()),
        "same_path_fde": float(
            gather(errors_np[:, :, -1], best_ade).mean()
        ),
        "minfde20": float(
            gather(errors_np[:, :, -1], best_fde).mean()
        ),
    }

    stored_path = stored_result_dir / f"q20_accuracy_seed{seed}" / "per_sample_q20.npz"
    e24_metrics = None
    reproduction = None
    if stored_path.is_file():
        with np.load(stored_path) as stored:
            stored_errors = np.asarray(stored["per_mode_errors"])
            e24_errors = np.asarray(stored["e24_errors"])
        if stored_errors.shape != errors_np.shape:
            raise ValueError(
                f"Stored Q20 shape mismatch for seed {seed}: "
                f"{stored_errors.shape} vs {errors_np.shape}"
            )
        reproduction = {
            "max_abs_error_difference": float(
                np.max(np.abs(stored_errors - errors_np))
            )
        }
        e24_metrics = {
            "ade": float(e24_errors.mean()),
            "fde": float(e24_errors[:, -1].mean()),
        }

    horizon_original = gather(errors_np, original_np).mean(axis=0)
    horizon_fused = gather(errors_np, fused_np).mean(axis=0)
    seed_output = output_dir / f"seed_{seed}"
    seed_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        seed_output / "per_sample_final.npz",
        per_mode_errors=errors_np,
        original_logits=logits_np,
        verifier_scores=verifier_np,
        original_indices=original_np,
        fused_indices=fused_np,
        best_ade_indices=best_ade,
        horizon_original=horizon_original,
        horizon_fused=horizon_fused,
    )
    result: Dict[str, object] = {
        "seed": seed,
        "samples": int(errors_np.shape[0]),
        "q20_checkpoint": str(q20_checkpoint),
        "verifier_checkpoint": str(verifier_checkpoint),
        "original": original_metrics,
        "fused": fused_metrics,
        "delta": {
            "ade": fused_metrics["ade"] - original_metrics["ade"],
            "fde": fused_metrics["fde"] - original_metrics["fde"],
            "hit_best_ade": (
                fused_metrics["hit_best_ade"]
                - original_metrics["hit_best_ade"]
            ),
            "regret": fused_metrics["regret"] - original_metrics["regret"],
        },
        "oracle": oracle,
        "e24": e24_metrics,
        "q20_reproduction": reproduction,
        "scale_at_min_fraction": scale_at_floor / max(scale_total, 1),
        "horizon_original": horizon_original.tolist(),
        "horizon_fused": horizon_fused.tolist(),
        "locked_configuration": {
            "min_scale": LOCKED_MIN_SCALE,
            "epochs": LOCKED_EPOCHS,
            "alpha": LOCKED_ALPHA,
        },
    }
    save_json(seed_output / "metrics.json", result)
    del q20, verifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def aggregate_results(results: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    metric_names = (
        "ade",
        "fde",
        "hit_best_ade",
        "regret",
        "ade_p95",
        "regret_p95",
        "catastrophic_regret_gt_0_5",
    )
    aggregate: Dict[str, object] = {}
    for method in ("original", "fused"):
        aggregate[method] = {
            metric: mean_sd(
                float(result[method][metric]) for result in results
            )
            for metric in metric_names
        }
    aggregate["delta"] = {
        metric: mean_sd(float(result["delta"][metric]) for result in results)
        for metric in ("ade", "fde", "hit_best_ade", "regret")
    }
    aggregate["better_seeds"] = {
        "ade": int(sum(result["delta"]["ade"] < 0 for result in results)),
        "fde": int(sum(result["delta"]["fde"] < 0 for result in results)),
        "total": len(results),
    }
    return aggregate


def format_pm(stats: Mapping[str, float], digits: int = 4) -> str:
    return (
        f"{stats['mean']:.{digits}f} +/- "
        f"{stats['sample_sd']:.{digits}f}"
    )


def build_report(
    results: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, object],
) -> str:
    lines = [
        "# Locked social-verifier final UNIV evaluation",
        "",
        "- This run accessed the target test split.",
        f"- Locked min_scale: `{LOCKED_MIN_SCALE}`",
        f"- Locked full-training epochs: `{LOCKED_EPOCHS}`",
        f"- Locked fusion: `0.70 * z(Q20) + {LOCKED_ALPHA:.2f} * z(verifier)`",
        f"- Seeds: `{list(LOCKED_SEEDS)}`",
        "- No test-driven checkpoint, scale or alpha selection was performed.",
        "",
        "## Per-seed results",
        "",
        "| Seed | Original ADE/FDE | Fused ADE/FDE | ADE delta | FDE delta | Hit delta | Regret delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['seed']} | "
            f"{result['original']['ade']:.4f}/{result['original']['fde']:.4f} | "
            f"{result['fused']['ade']:.4f}/{result['fused']['fde']:.4f} | "
            f"{result['delta']['ade']:+.4f} | "
            f"{result['delta']['fde']:+.4f} | "
            f"{result['delta']['hit_best_ade']:+.4f} | "
            f"{result['delta']['regret']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Three-seed aggregate",
            "",
            "| Method | ADE | FDE | Hit best-ADE | Regret | ADE P95 | Catastrophic regret |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, label in (("original", "Original Q20"), ("fused", "Fused")):
        stats = aggregate[method]
        lines.append(
            f"| {label} | {format_pm(stats['ade'])} | "
            f"{format_pm(stats['fde'])} | "
            f"{format_pm(stats['hit_best_ade'])} | "
            f"{format_pm(stats['regret'])} | "
            f"{format_pm(stats['ade_p95'])} | "
            f"{format_pm(stats['catastrophic_regret_gt_0_5'])} |"
        )
    better = aggregate["better_seeds"]
    lines.extend(
        [
            "",
            "## Locked decision",
            "",
            f"- ADE better seeds: `{better['ade']}/{better['total']}`",
            f"- FDE better seeds: `{better['fde']}/{better['total']}`",
            f"- Mean ADE delta: `{aggregate['delta']['ade']['mean']:+.4f}`",
            f"- Mean FDE delta: `{aggregate['delta']['fde']['mean']:+.4f}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train locked full-data verifiers and run final UNIV test."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"),
    )
    parser.add_argument(
        "--verifier-module",
        type=Path,
        default=Path(__file__).with_name(
            "train_social_stimulus_verifier.py"
        ),
    )
    parser.add_argument(
        "--config-file", default="configs/motif_rbf_univ.yaml"
    )
    parser.add_argument(
        "--output-dir", default="results/social_verifier_final_locked"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    verifier_module = args.verifier_module.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(project_root)
    if not verifier_module.is_file():
        raise FileNotFoundError(verifier_module)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_root = project_root / ".cache"
    matplotlib_cache = cache_root / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))
    sys.path.insert(0, str(verifier_module.parent))

    import munch  # type: ignore
    import yaml  # type: ignore
    from data.eth import ETH_dataset  # type: ignore
    from models.astra_model import ASTRA_model  # type: ignore
    from evaluate_q20_accuracy import configure_q20  # type: ignore
    from evaluate_risk_strata import (  # type: ignore
        build_dataset,
        load_config,
        prepare_runtime_cfg,
    )
    from train_social_stimulus_verifier import (  # type: ignore
        SocialStimulusVerifier,
        VerifierConfig,
        rotate_to_heading,
        run_epoch,
    )

    config_path = Path(args.config_file)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        train_cfg = munch.munchify(yaml.safe_load(handle))
    train_cfg.MODEL.USE_PRETRAINED_UNET = False
    train_cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    train_cfg.DATA.SEQ_AUG = False
    full_train_dataset = ETH_dataset(
        train_cfg, mode="training", img_transforms=None
    )
    sample_future = full_train_dataset[0][1]
    model_config = VerifierConfig(
        pred_len=int(sample_future.shape[-2]),
        frequency=float(train_cfg.DATA.FREQUENCY),
        num_components=8,
        self_hidden=96,
        social_hidden=64,
        decoder_hidden=192,
        min_scale=LOCKED_MIN_SCALE,
        dropout=0.10,
    )
    device = resolve_device(args.device)

    verifier_checkpoints: Dict[int, Path] = {}
    for seed in LOCKED_SEEDS:
        verifier_checkpoints[seed] = train_full_verifier(
            seed=seed,
            output_dir=output_dir / f"verifier_seed_{seed}",
            dataset=full_train_dataset,
            model_config=model_config,
            verifier_class=SocialStimulusVerifier,
            run_epoch=run_epoch,
            device=device,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            force=args.force_train,
        )

    # The target test split is first constructed here, after every verifier
    # has completed full-data training with locked hyperparameters.
    base_cfg = load_config(config_path)
    q20_prefix = (
        "univ_adaptive_kinematic_k4_sym0_seed{seed}_"
        "d1_state_no_motif_matched_decquery_endpoint_pca_"
        "m20_k22_qd96_msa1_relative_lossfixed_v2_last_model.pth"
    )
    results = []
    for seed in LOCKED_SEEDS:
        q20_checkpoint = project_root / "checkpoints" / q20_prefix.format(
            seed=seed
        )
        if not q20_checkpoint.is_file():
            raise FileNotFoundError(q20_checkpoint)
        print(f"[final test] paired seed {seed}")
        result = evaluate_pair(
            seed=seed,
            q20_checkpoint=q20_checkpoint,
            verifier_checkpoint=verifier_checkpoints[seed],
            base_cfg=base_cfg,
            configure_q20=configure_q20,
            prepare_runtime_cfg=prepare_runtime_cfg,
            build_dataset=build_dataset,
            astra_class=ASTRA_model,
            verifier_class=SocialStimulusVerifier,
            verifier_config_class=VerifierConfig,
            rotate_to_heading=rotate_to_heading,
            device=device,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            stored_result_dir=project_root / "results",
            output_dir=output_dir,
        )
        results.append(result)

    aggregate = aggregate_results(results)
    payload = {
        "protocol": {
            "target_test_split_accessed": True,
            "hyperparameters_locked_before_test": True,
            "min_scale": LOCKED_MIN_SCALE,
            "epochs": LOCKED_EPOCHS,
            "alpha": LOCKED_ALPHA,
            "seeds": list(LOCKED_SEEDS),
            "train_samples": len(full_train_dataset),
        },
        "per_seed": results,
        "aggregate": aggregate,
    }
    save_json(output_dir / "final_results.json", payload)
    report = build_report(results, aggregate)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nFinal results: {output_dir}")


if __name__ == "__main__":
    main()
