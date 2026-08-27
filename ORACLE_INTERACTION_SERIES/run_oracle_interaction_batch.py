"""Run the ETH/UCY oracle-focused interaction experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
TRAIN = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
DEFAULT_PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")

EXPERIMENTS = {
    "f42_dual_scale_sparse": HERE / "EXP_062_DUAL_SCALE_SPARSE",
    "f43_relation_type_bias": HERE / "EXP_063_RELATION_TYPE_BIAS",
    "f44_second_order_context": HERE / "EXP_064_SECOND_ORDER_CONTEXT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--variants", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--val-scenes", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python-executable", type=Path, default=Path(r"C:\coding_toobox\envs\astra2\python.exe"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def flatten(payload: dict) -> dict:
    oracle = payload["evaluation"]["oracle"]
    original = payload["evaluation"]["original"]
    return {
        "variant": payload["variant"],
        "seed": payload["seed"],
        "val_scene": payload["validation_scene_id"],
        "parameters": payload["locked_backend"]["parameter_counts"]["total"],
        "min_ade_20": oracle["min_ade_20"],
        "same_path_fde": oracle["same_path_fde"],
        "min_fde_20": oracle["min_fde_20"],
        "original_ade": original["ade"],
        "original_fde": original["fde"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }


def main() -> None:
    args = parse_args()
    python = args.python_executable.resolve()
    if not python.is_file():
        python = Path(sys.executable)
    all_rows = []
    for variant in args.variants:
        experiment = EXPERIMENTS[variant]
        output = experiment / "results"
        for scene in args.val_scenes:
            for seed in args.seeds:
                run_dir = output / variant / f"seed_{seed}" / f"val_scene_{scene}"
                run_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    str(python), str(TRAIN),
                    "--project-root", str(args.project_root.resolve()),
                    "--variant", variant,
                    "--decoder-experiment", "kinematic_k20",
                    "--decoder-module", str(DECODER),
                    "--seed", str(seed),
                    "--val-scene-id", str(scene),
                    "--epochs", str(args.epochs),
                    "--val-freq", str(args.val_freq),
                    "--batch-size", str(args.batch_size),
                    "--device", args.device,
                    "--output-root", str(output),
                    "--scheduler", "monotonic_cosine",
                ]
                if args.force:
                    command.append("--force")
                print(f"[run] {variant} seed={seed} val_scene={scene}", flush=True)
                with (run_dir / "console.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=args.project_root, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Failed: {variant}; see {run_dir / 'console.log'}")
                metrics = run_dir / "metrics.json"
                all_rows.append(flatten(json.loads(metrics.read_text(encoding="utf-8"))))
        (experiment / "summary.json").write_text(
            json.dumps([row for row in all_rows if row["variant"] == variant], indent=2),
            encoding="utf-8",
        )
    (HERE / "batch_summary.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"[done] {HERE / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
