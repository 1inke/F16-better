"""Complete official ETH/UCY seeds 2/3 for frozen F16+H3+V2."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
OFFICIAL = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES"
CONFIG = OFFICIAL / "motif_rbf_official_4p8.yaml"
AUDIT = OFFICIAL / "EXP_156A_PROTOCOL_AUDIT" / "protocol_audit.json"
H3_MODULE = WORKSPACE / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py"
V2_MODULE = WORKSPACE / "ANCHOR_CORRECTION_VALIDATION_SERIES" / "anchor_correction_validation_models.py"
SEED1_SUMMARY = WORKSPACE / "SCORE_SYNCHRONIZATION_SERIES" / "EXP_161_FROZEN_CANDIDATE_LISTWISE_SCORER" / "summary.json"
SEED1_RESULTS = OFFICIAL / "EXP_156B_FOLD_LOCAL_CURRENT_DECODER" / "results"
DEFAULT_OUT = HERE / "EXP_164_FINAL_V2_OFFICIAL_3SEED"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
STAGES = ("h3", "v2")


def run_dir(output: Path, fold: str, stage: str, seed: int) -> Path:
    return output / "results" / fold / stage / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def seed1_metrics_path(fold: str) -> Path:
    return SEED1_RESULTS / fold / "v2" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "metrics.json"


def command(args, output: Path, fold: str, stage: str, seed: int) -> list[str]:
    module, experiment, epochs = (
        (H3_MODULE, "h3_anchor_proposal", args.h3_epochs)
        if stage == "h3"
        else (V2_MODULE, "v2_wide_endpoint_correction", args.v2_epochs)
    )
    cmd = [
        str(PYTHON), "-u", str(TRAINER),
        "--project-root", str(PROJECT),
        "--config-file", str(CONFIG),
        "--data-path", str(DATA / f"{fold}_data_train.npy"),
        "--num-path", str(DATA / f"{fold}_num_train.npy"),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", experiment,
        "--decoder-module", str(module),
        "--seed", str(seed),
        "--val-scene-id", "-1",
        "--validation-fraction", "0.05",
        "--validation-seed", "20260830",
        "--validation-split-mode", "scene_tail",
        "--epochs", str(epochs),
        "--val-freq", str(args.val_freq),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--max-agents", "9",
        "--graph-top-k", "4",
        "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine",
        "--warmup-epochs", "1",
        "--learning-rate", "0.00025",
        "--output-root", str(output / "results" / fold / stage),
        "--oracle-coverage-weight", "0.20",
        "--oracle-coverage-temperature", "0.10",
        "--oracle-coverage-fde-weight", "0.25",
        "--checkpoint-criterion", "oracle_combo",
        "--ema-decay", "0.995",
        "--focal-heading-normalize",
        "--augment-reflection",
        "--augment-speed-min", "0.9",
        "--augment-speed-max", "1.1",
        "--augment-probability", "0.5",
    ]
    if stage == "v2":
        cmd += [
            "--test-data-path", str(DATA / f"{fold}_data_test.npy"),
            "--test-num-path", str(DATA / f"{fold}_num_test.npy"),
        ]
    return cmd


def seed1_rows() -> list[dict]:
    raw = json.loads(SEED1_SUMMARY.read_text(encoding="utf-8"))["per_fold"]
    chosen = {x["fold"]: x for x in raw if x["family"] == "v2" and x["seed"] == 1}
    rows = []
    for fold in FOLDS:
        item = chosen[fold]
        old_metrics = json.loads(seed1_metrics_path(fold).read_text(encoding="utf-8"))
        rows.append({
            "fold": fold, "seed": 1, "samples": item["samples"],
            "minADE": item["oracle"]["minADE"], "minFDE": item["oracle"]["minFDE"],
            "same_path_FDE": item["oracle"]["same_path_FDE"],
            "top1_ADE": item["raw_score"]["top1_ADE"], "top1_FDE": item["raw_score"]["top1_FDE"],
            "parameters": old_metrics["locked_backend"]["parameter_counts"]["total"],
            "decoder_parameters": old_metrics["locked_backend"]["parameter_counts"]["decoder"],
            "source": "existing_official_seed1",
        })
    return rows


def collect_new_rows(output: Path, seeds: list[int]) -> list[dict]:
    rows = []
    for fold in FOLDS:
        for seed in seeds:
            path = run_dir(output, fold, "v2", seed) / "metrics.json"
            if not path.is_file():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            evaluation = metrics.get("evaluation_test_best_selected")
            if not evaluation:
                continue
            oracle = evaluation["oracle"]
            rows.append({
                "fold": fold, "seed": seed, "samples": evaluation["samples"],
                "minADE": oracle["min_ade_20"], "minFDE": oracle["min_fde_20"],
                "same_path_FDE": oracle["same_path_fde"],
                "top1_ADE": evaluation["original"]["ade"], "top1_FDE": evaluation["original"]["fde"],
                "parameters": metrics["locked_backend"]["parameter_counts"]["total"],
                "decoder_parameters": metrics["locked_backend"]["parameter_counts"]["decoder"],
                "source": "new_official_run",
            })
    return rows


def describe(group: list[dict]) -> dict:
    names = ("minADE", "minFDE", "same_path_FDE", "top1_ADE", "top1_FDE")
    return {
        key: {"mean": float(np.mean([x[key] for x in group])), "std": float(np.std([x[key] for x in group], ddof=1)) if len(group) > 1 else None}
        for key in names
    }


def summarize(output: Path, requested_seeds: list[int]) -> None:
    rows = seed1_rows() + collect_new_rows(output, requested_seeds)
    rows.sort(key=lambda x: (FOLDS.index(x["fold"]), x["seed"]))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "all_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fold_stats = []
    for fold in FOLDS:
        group = [x for x in rows if x["fold"] == fold]
        fold_stats.append({"fold": fold, "seeds": [x["seed"] for x in group], "count": len(group), "metrics": describe(group)})
    seed_macro = []
    for seed in (1, *requested_seeds):
        group = [x for x in rows if x["seed"] == seed]
        if len(group) == len(FOLDS):
            seed_macro.append({"seed": seed, **{key: float(np.mean([x[key] for x in group])) for key in ("minADE", "minFDE", "same_path_FDE", "top1_ADE", "top1_FDE")}})
    complete = len(rows) == 15 and all(x["count"] == 3 for x in fold_stats)
    payload = {
        "experiment": "EXP-164 Frozen F16+H3+V2 Official Three-Seed Validation",
        "method_frozen": "F16 + H3 + V2 + original score",
        "complete": complete,
        "expected_runs": 15,
        "available_runs": len(rows),
        "fold_statistics": fold_stats,
        "seed_macro": seed_macro,
        "macro_across_seed_means": describe(seed_macro) if len(seed_macro) == 3 else None,
        "rows": rows,
        "protocol": {"folds": list(FOLDS), "seeds": [1, 2, 3], "SDD": False, "M2": False, "new_scorer": False},
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[summary] available={len(rows)}/15 complete={complete}", flush=True)


def run_stage(args, output: Path, fold: str, stage: str, seed: int) -> None:
    destination = run_dir(output, fold, stage, seed)
    metrics = destination / "metrics.json"
    if metrics.is_file() and not args.force:
        print(f"[skip] {fold}/{stage}/seed{seed}", flush=True)
        return
    env = os.environ.copy()
    source = None
    if stage == "v2":
        source = run_dir(output, fold, "h3", seed) / "best_selected.pth"
        env["EXP143_SOURCE_CHECKPOINT"] = str(source)
    cmd = command(args, output, fold, stage, seed)
    if args.force:
        cmd.append("--force")
    if args.dry_run:
        print(f"[dry-run] {fold}/{stage}/seed{seed} " + json.dumps(cmd), flush=True)
        return
    if source is not None and not source.is_file():
        raise FileNotFoundError(f"V2 source H3 checkpoint is missing: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[run] {fold}/{stage}/seed{seed}", flush=True)
    with (destination / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{fold}/{stage}/s{seed}] {line}", end="", flush=True)
            log.write(line); log.flush()
    if process.wait() != 0:
        raise RuntimeError(f"Failed: {fold}/{stage}/seed{seed}; see {destination / 'console.log'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--h3-epochs", type=int, default=30)
    parser.add_argument("--v2-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate seeds are not allowed")
    required = (PYTHON, TRAINER, CONFIG, AUDIT, H3_MODULE, V2_MODULE, SEED1_SUMMARY)
    missing = [str(path) for path in required if not path.is_file()]
    missing += [str(seed1_metrics_path(fold)) for fold in FOLDS if not seed1_metrics_path(fold).is_file()]
    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    missing += [str(DATA / f"{fold}_{kind}.npy") for fold in folds for kind in ("data_train", "num_train", "data_test", "num_test") if not (DATA / f"{fold}_{kind}.npy").is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    if not json.loads(AUDIT.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("Official fold protocol audit has not passed")
    for fold in folds:
        for seed in args.seeds:
            run_stage(args, args.output_dir, fold, "h3", seed)
            run_stage(args, args.output_dir, fold, "v2", seed)
            if not args.dry_run:
                summarize(args.output_dir, args.seeds)
    if args.dry_run:
        print("[done] dry-run passed", flush=True)
    else:
        summarize(args.output_dir, args.seeds)
        print(f"[done] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
