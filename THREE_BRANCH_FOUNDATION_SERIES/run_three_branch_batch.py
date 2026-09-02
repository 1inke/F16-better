from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def execute(name: str, command: list[str], dry_run: bool):
    print(f"\n===== {name} =====", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return
    proc = subprocess.run(command, cwd=str(HERE.parent))
    if proc.returncode:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Sequential single-GPU runner for EXP174A, EXP175S0 and EXP176A")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--efficiency-warmup", type=int, default=100)
    parser.add_argument("--efficiency-iterations", type=int, default=1000)
    parser.add_argument("--h3-epochs", type=int, default=30)
    parser.add_argument("--v2-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-efficiency", action="store_true")
    parser.add_argument("--skip-sdd", action="store_true")
    parser.add_argument("--skip-score-audit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    py = sys.executable
    seed_args = [str(s) for s in args.seeds]
    if not args.skip_efficiency:
        execute("EXP-174A final efficiency", [py, str(HERE / "run_exp174a_final_efficiency.py"), "--device", args.device, "--warmup", str(args.efficiency_warmup), "--iterations", str(args.efficiency_iterations)], args.dry_run)
    if not args.skip_sdd:
        execute("EXP-175S0 final SDD baseline", [py, str(HERE / "run_exp175s0_final_sdd_baseline.py"), "--device", args.device, "--seeds", *seed_args, "--h3-epochs", str(args.h3_epochs), "--v2-epochs", str(args.v2_epochs), "--batch-size", str(args.batch_size)], args.dry_run)
    if not args.skip_score_audit:
        execute("EXP-176A final score audit", [py, str(HERE / "run_exp176a_final_score_audit.py"), "--device", args.device, "--folds", "all", "--seeds", *seed_args, "--batch-size", str(args.batch_size)], args.dry_run)
    print("\n[complete] requested three-branch sequence finished", flush=True)


if __name__ == "__main__":
    main()
