from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
ENCODER_MODULE = ROOT / "encoder_precision_models.py"
DECODER_MODULE = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS" /
    "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
)

EXPERIMENTS = {
    "baseline": {
        "folder": "EXP_123_F16_SAME_BATCH_BASELINE",
    },
    "eq_w002": {
        "folder": "EXP_124_EQ_MIRROR_W002",
        "equivariance_weight": 0.02,
    },
    "eq_w005": {
        "folder": "EXP_125_EQ_MIRROR_W005",
        "equivariance_weight": 0.05,
    },
    "state_gate": {
        "folder": "EXP_126_STATE_CHANNEL_GATE",
        "encoder_experiment": "state_channel_gate",
    },
    "relation_gate": {
        "folder": "EXP_127_RELATION_CHANNEL_GATE",
        "encoder_experiment": "relation_channel_gate",
    },
    "tcn_residual": {
        "folder": "EXP_128_TCN_RESIDUAL",
        "encoder_experiment": "tcn_residual",
    },
    "temporal_rope": {
        "folder": "EXP_129_TEMPORAL_ROPE",
        "encoder_experiment": "temporal_rope",
    },
}


def integer_list(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def selected_labels(text: str) -> list[str]:
    if text == "all":
        return list(EXPERIMENTS)
    labels = [value.strip() for value in text.split(",") if value.strip()]
    unknown = sorted(set(labels) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")
    return labels


def collect_summary(series_root: Path) -> None:
    rows = []
    for label, spec in EXPERIMENTS.items():
        result_root = series_root / spec["folder"] / "results"
        for metrics_path in result_root.glob("f16_temporal_social_cross/seed_*/val_scene_*/metrics.json"):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            selected = metrics["evaluation_best_selected"]
            rows.append({
                "experiment": label,
                "folder": spec["folder"],
                "seed": metrics["seed"],
                "scene": metrics["validation_scene_id"],
                "samples": selected["samples"],
                **selected["oracle"],
                "top1_ade": selected["original"]["ade"],
                "top1_fde": selected["original"]["fde"],
                "parameters": metrics["locked_backend"]["parameter_counts"],
                "elapsed_seconds": metrics["elapsed_seconds"],
            })
    rows.sort(key=lambda row: (row["experiment"], row["seed"], row["scene"]))
    (series_root / "batch_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--val-scenes", default="1,3")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--only", default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = integer_list(args.seeds)
    scenes = integer_list(args.val_scenes)
    labels = selected_labels(args.only)
    if not seeds or not scenes:
        raise ValueError("--seeds and --val-scenes must not be empty")

    for label in labels:
        spec = EXPERIMENTS[label]
        experiment_root = ROOT / spec["folder"]
        output_root = experiment_root / "results"
        for seed in seeds:
            for scene in scenes:
                run_dir = (
                    output_root / "f16_temporal_social_cross" /
                    f"seed_{seed}" / f"val_scene_{scene}"
                )
                metrics_path = run_dir / "metrics.json"
                if metrics_path.is_file() and not args.force:
                    print(f"[skip] {label} seed={seed} scene={scene}", flush=True)
                    continue
                command = [
                    str(PYTHON), "-u", str(TRAINER),
                    "--project-root", str(PROJECT),
                    "--variant", "f16_temporal_social_cross",
                    "--decoder-experiment", "kinematic_k20",
                    "--decoder-module", str(DECODER_MODULE),
                    "--seed", str(seed), "--val-scene-id", str(scene),
                    "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
                    "--device", args.device, "--scheduler", "warm_restarts",
                    "--output-root", str(output_root),
                    "--oracle-coverage-weight", "0.20",
                    "--oracle-coverage-temperature", "0.10",
                    "--checkpoint-criterion", "oracle_combo",
                    "--ema-decay", "0.995", "--focal-heading-normalize",
                    "--augment-reflection", "--augment-speed-min", "0.9",
                    "--augment-speed-max", "1.1", "--augment-probability", "0.5",
                ]
                encoder_name = spec.get("encoder_experiment")
                if encoder_name:
                    command += [
                        "--encoder-experiment", str(encoder_name),
                        "--encoder-module", str(ENCODER_MODULE),
                    ]
                weight = float(spec.get("equivariance_weight", 0.0))
                if weight > 0.0:
                    command += ["--equivariance-weight", str(weight)]
                if args.force:
                    command.append("--force")
                if args.dry_run:
                    print(
                        "[dry-run] " + json.dumps(command, ensure_ascii=False),
                        flush=True,
                    )
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                print(f"[run] {label} seed={seed} scene={scene}", flush=True)
                with (run_dir / "console.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT
                    )
                if result.returncode:
                    raise RuntimeError(
                        f"Failed: {label} seed={seed} scene={scene}; "
                        f"see {run_dir / 'console.log'}"
                    )
                collect_summary(ROOT)
    if args.dry_run:
        print("[done] dry-run validation passed", flush=True)
    else:
        collect_summary(ROOT)
        print(f"[done] summary={ROOT / 'batch_summary.json'}", flush=True)


if __name__ == "__main__":
    run()
