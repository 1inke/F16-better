"""Validated one-command driver for the zero-training CSNA experiment."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0"); p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--splits", nargs="+", choices=("all", "general", "challenge"), default=["all"])
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()
    subprocess.run([str(PYTHON), str(ROOT / "smoke_check.py")], cwd=ROOT, check=True)
    command = [str(PYTHON), "-u", str(ROOT / "run_csna_audit.py"), "--device", args.device,
               "--batch-size", str(args.batch_size), "--seed", str(args.seed), "--bootstrap", str(args.bootstrap),
               "--splits", *args.splits]
    if args.max_samples is not None: command += ["--max-samples", str(args.max_samples)]
    log_path = ROOT / "console.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
        if process.wait(): raise RuntimeError(f"CSNA failed; see {log_path}")


if __name__ == "__main__": main()

