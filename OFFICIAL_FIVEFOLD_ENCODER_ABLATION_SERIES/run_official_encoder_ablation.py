"""Official ETH/UCY five-fold encoder ablation with a fixed H3+V2 decoder."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
ADAPTER = ROOT / "official_v2_decoder_adapter.py"
CONFIG = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
SEED1_SOURCE = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "EXP_156B_FOLD_LOCAL_CURRENT_DECODER" / "results"
SEEDS23_SOURCE = WORKSPACE / "OFFICIAL_ETH_UCY_FINAL_3SEED_SERIES" / "EXP_164_FINAL_V2_OFFICIAL_3SEED" / "results"
EXPERIMENT = ROOT / "EXP_172_OFFICIAL_FIVEFOLD_ENCODER_ABLATION"
DEFAULT_RESULTS = EXPERIMENT / "results"

FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
VARIANTS = {
    "e0_trajectory_only": "f0_baseline",
    "e1_sparse_graph": "f3_factorized",
    "e2_relation_prior": "f4_relation_tokens",
    "e3_f16_temporal_social": "f16_temporal_social_cross",
}


def source_checkpoint(fold: str, seed: int) -> Path:
    root = SEED1_SOURCE if seed == 1 else SEEDS23_SOURCE
    return root / fold / "v2" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"


def run_directory(results: Path, fold: str, label: str, variant: str, seed: int) -> Path:
    return results / fold / label / variant / f"seed_{seed}" / "val_scene_-1"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: Path, report: Path, seeds: tuple[int, ...]) -> None:
    rows = []
    for fold in FOLDS:
        for label, variant in VARIANTS.items():
            for seed in seeds:
                metrics_path = run_directory(results, fold, label, variant, seed) / "metrics.json"
                if not metrics_path.is_file():
                    continue
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                evaluation = payload.get("evaluation_test_best_selected")
                if not evaluation:
                    continue
                oracle, original = evaluation["oracle"], evaluation["original"]
                counts = payload["locked_backend"]["parameter_counts"]
                rows.append({
                    "experiment": "EXP-172 official encoder ablation",
                    "encoder_label": label,
                    "encoder_variant": variant,
                    "fold": fold,
                    "seed": seed,
                    "decoder_source_checkpoint": str(source_checkpoint(fold, seed)),
                    "samples": evaluation["samples"],
                    "minADE20": oracle["min_ade_20"],
                    "minFDE20": oracle["min_fde_20"],
                    "same_path_FDE": oracle["same_path_fde"],
                    "J": oracle["min_ade_20"] + 0.5 * oracle["min_fde_20"],
                    "top1_ADE": original["ade"],
                    "top1_FDE": original["fde"],
                    "parameters_total": counts["total"],
                    "parameters_trainable": counts["trainable"],
                    "decoder_parameters": counts["decoder"],
                    "elapsed_seconds": payload["elapsed_seconds"],
                })
    write_csv(report / "all_runs.csv", rows)
    per_fold = []
    for label, variant in VARIANTS.items():
        for fold in FOLDS:
            group = [row for row in rows if row["encoder_label"] == label and row["fold"] == fold]
            if group:
                per_fold.append({
                    "encoder_label": label,
                    "encoder_variant": variant,
                    "fold": fold,
                    "seeds": len(group),
                    **{key: sum(float(row[key]) for row in group) / len(group) for key in ("minADE20", "minFDE20", "same_path_FDE", "J", "top1_ADE", "top1_FDE")},
                    "parameters_total": group[0]["parameters_total"],
                    "parameters_trainable": group[0]["parameters_trainable"],
                    "decoder_parameters": group[0]["decoder_parameters"],
                    "elapsed_seconds_mean": sum(float(row["elapsed_seconds"]) for row in group) / len(group),
                })
    write_csv(report / "per_fold_summary.csv", per_fold)
    macro = {}
    previous = None
    for label, variant in VARIANTS.items():
        group = [row for row in per_fold if row["encoder_label"] == label]
        if not group:
            continue
        values = {key: sum(float(row[key]) for row in group) / len(group) for key in ("minADE20", "minFDE20", "same_path_FDE", "J", "top1_ADE", "top1_FDE")}
        values.update({"encoder_variant": variant, "folds": len(group), "parameters_total": group[0]["parameters_total"], "parameters_trainable": group[0]["parameters_trainable"], "decoder_parameters": group[0]["decoder_parameters"]})
        if previous is not None:
            for key in ("minADE20", "minFDE20", "same_path_FDE", "J"):
                values[f"improvement_vs_previous_{key}_pct"] = 100.0 * (previous[key] - values[key]) / previous[key]
        macro[label] = values
        previous = values
    baseline = macro.get("e0_trajectory_only")
    if baseline:
        for values in macro.values():
            for key in ("minADE20", "minFDE20", "same_path_FDE", "J"):
                values[f"improvement_vs_e0_{key}_pct"] = 100.0 * (baseline[key] - values[key]) / baseline[key]
    complete = all(any(row["encoder_label"] == label and row["fold"] == fold for row in per_fold) for label in VARIANTS for fold in FOLDS)
    report.mkdir(parents=True, exist_ok=True)
    (report / "summary.json").write_text(json.dumps({"experiment": "EXP-172 Official Five-Fold Encoder Ablation", "complete": complete, "protocol": {"decoder": "frozen exact fold-local official V2", "decoder_source_seed1": str(SEED1_SOURCE), "decoder_source_seeds2_3": str(SEEDS23_SOURCE), "variants": VARIANTS, "seeds": seeds}, "macro": macro, "per_fold": per_fold}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--variants", nargs="+", choices=("all", *VARIANTS), default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.00025)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report-root", type=Path, default=EXPERIMENT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_root, args.report_root = args.output_root.resolve(), args.report_root.resolve()
    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    labels = tuple(VARIANTS) if "all" in args.variants else tuple(args.variants)
    seeds = tuple(args.seeds)
    for path in (PYTHON, TRAINER, ADAPTER, CONFIG):
        if not path.is_file():
            raise FileNotFoundError(path)
    for fold in folds:
        for seed in seeds:
            if not source_checkpoint(fold, seed).is_file():
                raise FileNotFoundError(source_checkpoint(fold, seed))

    commands = []
    for label in labels:
        variant = VARIANTS[label]
        for fold in folds:
            for seed in seeds:
                run = run_directory(args.output_root, fold, label, variant, seed)
                if (run / "metrics.json").is_file() and not args.force:
                    print(f"[skip] {label}/{fold}/seed{seed}", flush=True)
                    continue
                command = [
                    str(PYTHON), "-u", str(TRAINER),
                    "--project-root", str(PROJECT),
                    "--config-file", str(CONFIG),
                    "--data-path", str(DATA / f"{fold}_data_train.npy"),
                    "--num-path", str(DATA / f"{fold}_num_train.npy"),
                    "--test-data-path", str(DATA / f"{fold}_data_test.npy"),
                    "--test-num-path", str(DATA / f"{fold}_num_test.npy"),
                    "--variant", variant,
                    "--decoder-experiment", "fixed_official_v2",
                    "--decoder-module", str(ADAPTER),
                    "--seed", str(seed),
                    "--val-scene-id", "-1",
                    "--validation-fraction", "0.05",
                    "--validation-seed", "20260830",
                    "--validation-split-mode", "scene_tail",
                    "--epochs", str(args.epochs),
                    "--val-freq", str(args.val_freq),
                    "--batch-size", str(args.batch_size),
                    "--device", args.device,
                    "--max-agents", "9",
                    "--graph-top-k", "4",
                    "--graph-radius", "6.0",
                    "--scheduler", "monotonic_cosine",
                    "--warmup-epochs", "1",
                    "--learning-rate", str(args.learning_rate),
                    "--output-root", str(args.output_root / fold / label),
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
                if args.force:
                    command.append("--force")
                commands.append((label, fold, seed, run, command))
    if args.dry_run:
        for label, fold, seed, _, command in commands:
            print(f"[dry-run] {label}/{fold}/seed{seed} " + json.dumps(command))
        print(f"[done] {len(commands)} commands validated")
        return

    for label, fold, seed, run, command in commands:
        run.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["OFFICIAL_V2_DECODER_SOURCE"] = str(source_checkpoint(fold, seed))
        print(f"[run] {label}/{fold}/seed{seed}", flush=True)
        with (run / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(f"[{label}/{fold}] {line}", end="", flush=True)
                log.write(line); log.flush()
        if process.wait() != 0:
            raise RuntimeError(f"Failed: {run / 'console.log'}")
        summarize(args.output_root, args.report_root, seeds)
    summarize(args.output_root, args.report_root, seeds)
    print(f"[done] report: {args.report_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
