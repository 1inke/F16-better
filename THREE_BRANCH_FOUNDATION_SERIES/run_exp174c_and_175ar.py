from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERIES = Path(__file__).resolve().parent
EXP174C = SERIES / "EXP_174C_PROFILER_GUIDED_KERNEL_AUDIT" / "run_exp174c_profiler_audit.py"
EXP175AR = SERIES / "EXP_175AR_SOFT_SEMANTIC_REPAIR" / "run_exp175ar_soft_semantic_repair.py"


def run(command: list[str]):
    print("[run] " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def main():
    parser = argparse.ArgumentParser(description="Run EXP-174C then EXP-175A-R")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force-assets", action="store_true")
    args = parser.parse_args()
    for path in (EXP174C, EXP175AR):
        if not path.exists():
            raise FileNotFoundError(path)
    first = [sys.executable, "-u", str(EXP174C), "--device", args.device, "--seed", str(args.seed)]
    second = [sys.executable, "-u", str(EXP175AR), "--device", args.device, "--seed", str(args.seed), "--batch-size", str(args.batch_size)]
    if args.preflight_only:
        first.append("--smoke"); second.append("--preflight-only")
    if args.force_assets:
        second.append("--force-assets")
    run(first)
    run(second)
    print("[complete] EXP-174C and EXP-175A-R", flush=True)


if __name__ == "__main__":
    main()
