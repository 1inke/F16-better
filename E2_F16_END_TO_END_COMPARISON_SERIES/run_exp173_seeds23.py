"""Resume EXP-173 with seeds 2/3 and rebuild the three-seed result tables.

This wrapper deliberately reuses the locked EXP-173 runner instead of
duplicating any training options. Existing metrics are skipped by the base
runner, so the command is safe to resume after an interruption.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
RUNNER = ROOT / "run_e2_f16_joint_comparison.py"
EVALUATOR = ROOT / "evaluate_e2_f16_joint_comparison.py"
DEFAULT_EXPERIMENT = ROOT / "EXP_173_E2_F16_END_TO_END"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
LABELS = ("e2", "e3")
STAGES = ("h3", "v2")


def metrics_path(experiment: Path, fold: str, stage: str, label: str, seed: int) -> Path:
    variant = "f4_relation_tokens" if label == "e2" else "f16_temporal_social_cross"
    return (
        experiment / "results" / fold / stage / label / variant
        / f"seed_{seed}" / "val_scene_-1" / "metrics.json"
    )


def require_complete_seed1(experiment: Path) -> None:
    missing = [
        str(metrics_path(experiment, fold, stage, label, 1))
        for label in LABELS
        for fold in FOLDS
        for stage in STAGES
        if not metrics_path(experiment, fold, stage, label, 1).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "EXP-173 seed 1 is incomplete; refusing to mix protocols. Missing:\n"
            + "\n".join(missing)
        )
    print("[preflight] seed 1 complete: 20/20 stage results", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run only EXP-173 seeds 2/3, then aggregate seeds 1/2/3."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--variants", nargs="+", choices=("all", *LABELS), default=["e2", "e3"])
    parser.add_argument("--h3-epochs", type=int, default=30)
    parser.add_argument("--v2-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiment = args.experiment_dir.resolve()
    required = (PYTHON, RUNNER, EVALUATOR)
    missing_inputs = [str(path) for path in required if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError("Missing EXP-173 entry points:\n" + "\n".join(missing_inputs))
    require_complete_seed1(experiment)

    command = [
        str(PYTHON), "-u", str(RUNNER),
        "--device", args.device,
        "--folds", *args.folds,
        "--variants", *args.variants,
        "--seeds", "2", "3",
        "--stages", "all",
        "--h3-epochs", str(args.h3_epochs),
        "--v2-epochs", str(args.v2_epochs),
        "--batch-size", str(args.batch_size),
        "--val-freq", str(args.val_freq),
        "--experiment-dir", str(experiment),
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.force:
        command.append("--force")

    print("[launch] " + json.dumps(command, ensure_ascii=False), flush=True)
    if args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)
        print("[done] seeds 2/3 dry-run passed; no training was started", flush=True)
        return

    aggregate = [
        str(PYTHON), "-u", str(EVALUATOR),
        "--experiment-dir", str(experiment),
        "--seeds", "1", "2", "3",
    ]
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    finally:
        # Keep the combined progress tables useful even if a later run fails.
        print("[aggregate] rebuilding available tables for seeds 1/2/3", flush=True)
        subprocess.run(aggregate, cwd=ROOT, check=True)
    print("[done] EXP-173 seeds 1/2/3 tables are ready", flush=True)


if __name__ == "__main__":
    main()
