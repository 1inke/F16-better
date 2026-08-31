"""Run EXP-168A official five-fold seed-1 joint hybrid K20 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WS = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
CONFIG = WS / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
TRAIN = WS / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = ROOT / "hybrid_generator_models.py"
EXPERIMENT = ROOT / "EXP_168A_HYBRID_K20_10_10"
OUT = EXPERIMENT / "results"
BASELINE = WS / "OFFICIAL_ETH_UCY_FINAL_3SEED_SERIES" / "EXP_164_FINAL_V2_OFFICIAL_3SEED" / "summary.json"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")


def run_dir(fold: str, seed: int) -> Path:
    return OUT / fold / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def baseline_seed1() -> dict:
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))
    return {row["fold"]: row for row in payload["rows"] if int(row["seed"]) == 1}


def summarize(seeds: list[int]) -> None:
    reference = baseline_seed1()
    rows = []
    responsibility_rows = []
    for fold in FOLDS:
        for seed in seeds:
            destination = run_dir(fold, seed)
            metrics_path = destination / "metrics.json"
            if not metrics_path.is_file():
                continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            evaluation = payload.get("evaluation_test_best_selected")
            if not evaluation:
                continue
            oracle = evaluation["oracle"]
            original = evaluation["original"]
            ade = float(oracle["min_ade_20"])
            fde = float(oracle["min_fde_20"])
            same = float(oracle["same_path_fde"])
            j_score = ade + 0.5 * fde
            base = reference[fold]
            base_j = float(base["minADE"]) + 0.5 * float(base["minFDE"])
            contract = payload["locked_backend"]
            row = {
                "experiment": "EXP-168A hybrid K20 10/10",
                "fold": fold,
                "seed": seed,
                "samples": evaluation["samples"],
                "minADE20": ade,
                "minFDE20": fde,
                "same_path_FDE": same,
                "J": j_score,
                "top1_ADE": original["ade"],
                "top1_FDE": original["fde"],
                "parameters_total": contract["parameter_counts"]["total"],
                "parameters_trainable": contract["parameter_counts"]["trainable"],
                "decoder_parameters": contract["parameter_counts"]["decoder"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "delta_minADE_pct": 100.0 * (ade - float(base["minADE"])) / float(base["minADE"]),
                "delta_minFDE_pct": 100.0 * (fde - float(base["minFDE"])) / float(base["minFDE"]),
                "delta_J_pct": 100.0 * (j_score - base_j) / base_j,
                "delta_samepath_pct": 100.0 * (same - float(base["same_path_FDE"])) / float(base["same_path_FDE"]),
            }
            rows.append(row)
            diagnostic_path = destination / "family_diagnostics.json"
            if diagnostic_path.is_file():
                diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
                responsibility_rows.append({
                    "fold": fold,
                    "seed": seed,
                    "winner_anchor": diagnostic["winner_family_fraction"]["anchor"],
                    "winner_free": diagnostic["winner_family_fraction"]["free"],
                    "soft_anchor": diagnostic["soft_responsibility"]["anchor"],
                    "soft_free": diagnostic["soft_responsibility"]["free"],
                    "tail_soft_anchor": diagnostic["tail_100_soft_responsibility"]["anchor"],
                    "tail_soft_free": diagnostic["tail_100_soft_responsibility"]["free"],
                    "grad_anchor": diagnostic["gradient_norm_proxy"]["anchor"],
                    "grad_free": diagnostic["gradient_norm_proxy"]["free"],
                    "active_anchor_modes": diagnostic["active_modes_by_family"]["anchor"],
                    "active_free_modes": diagnostic["active_modes_by_family"]["free"],
                    "anchor_diversity": diagnostic["mean_diversity"]["within_anchor_diversity"],
                    "free_diversity": diagnostic["mean_diversity"]["within_free_diversity"],
                    "cross_family_distance": diagnostic["mean_diversity"]["cross_family_distance"],
                    "family_collapse": diagnostic["family_collapse_tail_lt_10pct"],
                })
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    if rows:
        with (EXPERIMENT / "per_fold_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    if responsibility_rows:
        with (EXPERIMENT / "family_responsibility.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(responsibility_rows[0]))
            writer.writeheader(); writer.writerows(responsibility_rows)
    seed1 = [row for row in rows if row["seed"] == 1]
    macro = None
    gate = None
    if len(seed1) == 5:
        macro = {key: sum(row[key] for row in seed1) / 5.0 for key in ("minADE20", "minFDE20", "same_path_FDE", "J")}
        baseline_macro_j = sum(float(reference[f]["minADE"]) + 0.5 * float(reference[f]["minFDE"]) for f in FOLDS) / 5.0
        macro_improvement = 100.0 * (baseline_macro_j - macro["J"]) / baseline_macro_j
        per_fold_improvement = {row["fold"]: -row["delta_J_pct"] for row in seed1}
        responsibility_ok = (
            len(responsibility_rows) >= 5
            and all(min(x["tail_soft_anchor"], x["tail_soft_free"]) >= 0.10 for x in responsibility_rows if x["seed"] == 1)
        )
        gate = {
            "macro_J_improvement_pct": macro_improvement,
            "macro_J_improvement_ge_2pct": macro_improvement >= 2.0,
            "all_fold_regression_le_2pct": all(value >= -2.0 for value in per_fold_improvement.values()),
            "eth_or_univ_improvement_ge_4pct": max(per_fold_improvement["eth"], per_fold_improvement["univ"]) >= 4.0,
            "both_family_tail_responsibility_ge_10pct": responsibility_ok,
        }
        gate["pass"] = all(value for key, value in gate.items() if key.endswith(("2pct", "4pct", "10pct")))
    (EXPERIMENT / "summary.json").write_text(json.dumps({
        "experiment": "EXP-168A Joint Hybrid Generator K20 10/10",
        "rows": rows,
        "family_responsibility": responsibility_rows,
        "seed1_macro": macro,
        "gate": gate,
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for path in (PYTHON, TRAIN, DECODER, CONFIG, BASELINE):
        if not path.is_file():
            raise FileNotFoundError(path)
    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    for fold in folds:
        for seed in args.seeds:
            destination = run_dir(fold, seed)
            if (destination / "metrics.json").is_file() and not args.force:
                print(f"[skip] {fold}/seed{seed}")
                continue
            command = [
                str(PYTHON), "-u", str(TRAIN),
                "--project-root", str(PROJECT), "--config-file", str(CONFIG),
                "--data-path", str(DATA / f"{fold}_data_train.npy"),
                "--num-path", str(DATA / f"{fold}_num_train.npy"),
                "--test-data-path", str(DATA / f"{fold}_data_test.npy"),
                "--test-num-path", str(DATA / f"{fold}_num_test.npy"),
                "--variant", "f16_temporal_social_cross",
                "--decoder-experiment", "hybrid_k20_10_10", "--decoder-module", str(DECODER),
                "--seed", str(seed), "--val-scene-id", "-1",
                "--validation-fraction", "0.05", "--validation-seed", "20260830",
                "--validation-split-mode", "scene_tail", "--epochs", str(args.epochs),
                "--val-freq", str(args.val_freq), "--batch-size", str(args.batch_size),
                "--device", args.device, "--max-agents", "9", "--graph-top-k", "4",
                "--graph-radius", "6.0", "--scheduler", "monotonic_cosine",
                "--warmup-epochs", "1", "--learning-rate", "0.00025",
                "--output-root", str(OUT / fold), "--oracle-coverage-weight", "0.20",
                "--oracle-coverage-temperature", "0.10", "--oracle-coverage-fde-weight", "0.25",
                "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
                "--focal-heading-normalize", "--augment-reflection", "--augment-speed-min", "0.9",
                "--augment-speed-max", "1.1", "--augment-probability", "0.5",
            ]
            if args.force:
                command.append("--force")
            if args.dry_run:
                print("[dry-run] " + json.dumps(command, ensure_ascii=False))
                continue
            destination.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["EXP168_DIAGNOSTICS_PATH"] = str(destination / "family_diagnostics.json")
            print(f"[run] {fold}/seed{seed}", flush=True)
            with (destination / "console.log").open("w", encoding="utf-8") as log:
                job = subprocess.Popen(
                    command, cwd=PROJECT, env=environment, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                )
                assert job.stdout is not None
                for line in job.stdout:
                    print(f"[{fold}] {line}", end="", flush=True)
                    log.write(line); log.flush()
            if job.wait():
                raise RuntimeError(f"failed: {destination / 'console.log'}")
            summarize(args.seeds)
    if args.dry_run:
        print("[done] dry-run passed")
    else:
        summarize(args.seeds)
        print("[done] EXP-168A completed")


if __name__ == "__main__":
    main()

