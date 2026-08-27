"""Repair EXP-140A General protocol, then run EXP-140B full-context refinement."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--skip-general-repair", action="store_true")
    args = parser.parse_args()
    common = ["--device", args.device, "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--seeds", *map(str, args.seeds)]
    if not args.skip_general_repair:
        repair = [str(PYTHON), str(ROOT / "run_frozen_all20_batch.py"), *common, "--splits", "general", "--validation-seed", "20260825", "--force"]
        print("[stage 1/2] repair EXP-140A General with the matched validation seed", flush=True)
        subprocess.run(repair, check=True)
    experiment = [str(PYTHON), str(ROOT / "run_full_context_all20_batch.py"), *common, "--validation-seed", "20260825"]
    print("[stage 2/2] run EXP-140B full-context all-20 refiner", flush=True)
    subprocess.run(experiment, check=True)


if __name__ == "__main__": main()

