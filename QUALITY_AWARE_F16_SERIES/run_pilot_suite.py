"""
Full Pilot Automation Suite for Quality-Aware F16 Training (Seed 1, ETH Fold).

Coordinates:
- Q1: QA0 Baseline Reproduction Audit
- Q2: Lambda Sweep (0.05, 0.10, 0.20) at tau=0.10
- Q3: Tau Sweep (0.05, 0.20) at best lambda
- Q4: Temperature Calibration on best model (QA2)
- Aggregation of all required CSVs and PILOT_ANALYSIS.md
- Strict stop at STOP POINT 2.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
PYTHON = sys.executable


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n========================================================")
    print(f"[RUNNING] {desc}")
    print(f"Command: {' '.join(cmd)}")
    print(f"========================================================", flush=True)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(REPO))
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(f"Step '{desc}' failed with exit code {res.returncode}")
    print(f"[COMPLETED] {desc} in {elapsed:.1f}s\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run Full Pilot Suite")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    results_root = HERE / "results"
    base_repro_dir = results_root / "baseline_reproduction"
    qa0_dir = base_repro_dir / "QA0_STANDARD_FINETUNE"
    pilot_dir = results_root / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check QA0 reproduction
    if not (qa0_dir / "metrics.json").is_file():
        raise FileNotFoundError(f"QA0 metrics missing at {qa0_dir / 'metrics.json'}. Finish QA0 first!")

    print("[STAGE 1] Running Reproduction Audit...")
    run_cmd([PYTHON, "-u", str(HERE / "build_reproduction_audit.py")], "Audit QA0 Reproduction")

    audit_md = base_repro_dir / "REPRODUCTION_AUDIT.md"
    if not audit_md.is_file() or "[STATUS: PASS]" not in audit_md.read_text(encoding="utf-8"):
        raise RuntimeError("[STOP POINT 1 HIT] QA0 reproduction did not PASS! Halting per protocol.")
    print("[STAGE 1 PASSED] QA0 reproduction verified! Proceeding to Q2 Lambda Sweep.")

    # 2. Q2: Lambda Sweep at tau = 0.10
    lambdas = [0.05, 0.10, 0.20]
    tau_default = 0.10
    lambda_runs = []

    # Include QA0 as lambda=0.0
    lambda_runs.append({
        "run_id": "QA0_STANDARD_FINETUNE",
        "lambda_q": 0.0,
        "quality_tau": 0.10,
        "dir": qa0_dir,
    })

    for l_val in lambdas:
        run_name = f"QA1_L{int(l_val*100):03d}_T{int(tau_default*100):03d}"
        run_dir = pilot_dir / run_name
        if not (run_dir / "metrics.json").is_file():
            cmd = [
                PYTHON, "-u", str(HERE / "train_quality_aware_f16.py"),
                "--fold", "eth", "--seed", "1", "--source-type", "v2",
                "--epochs", str(args.epochs), "--batch-size", "64", "--val-freq", "5",
                "--lr", "0.00025", "--lambda-q", str(l_val), "--quality-tau", str(tau_default),
                "--output-dir", str(run_dir), "--device", args.device,
            ]
            run_cmd(cmd, f"Train {run_name} (lambda={l_val}, tau={tau_default})")
        else:
            print(f"[SKIP] {run_name} already completed.")

        lambda_runs.append({
            "run_id": run_name,
            "lambda_q": l_val,
            "quality_tau": tau_default,
            "dir": run_dir,
        })

    # Find best lambda on Validation Selected ADE among positive lambda QA1 runs
    def get_val_ade(r):
        m = json.loads((r["dir"] / "metrics.json").read_text(encoding="utf-8"))
        return m["val"]["selection"]["selected_ade"]

    positive_lambda_runs = [r for r in lambda_runs if r["lambda_q"] > 0]
    best_lambda_run = min(positive_lambda_runs, key=get_val_ade)
    best_lambda = best_lambda_run["lambda_q"]
    print(f"\n>>> [Q2 RESULT] Best lambda_q on validation: {best_lambda} (from {best_lambda_run['run_id']} with Val ADE: {get_val_ade(best_lambda_run):.4f}m)")

    # 3. Q3: Tau Sweep at best lambda (tau in [0.05, 0.20])
    all_runs = list(lambda_runs)
    tau_vals = [0.05, 0.20] # 0.10 is already completed above

    for t_val in tau_vals:
        run_name = f"QA1_L{int(best_lambda*100):03d}_T{int(t_val*100):03d}"
        run_dir = pilot_dir / run_name
        if not (run_dir / "metrics.json").is_file():
            cmd = [
                PYTHON, "-u", str(HERE / "train_quality_aware_f16.py"),
                "--fold", "eth", "--seed", "1", "--source-type", "v2",
                "--epochs", str(args.epochs), "--batch-size", "64", "--val-freq", "5",
                "--lr", "0.00025", "--lambda-q", str(best_lambda), "--quality-tau", str(t_val),
                "--output-dir", str(run_dir), "--device", args.device,
            ]
            run_cmd(cmd, f"Train {run_name} (lambda={best_lambda}, tau={t_val})")
        else:
            print(f"[SKIP] {run_name} already completed.")

        all_runs.append({
            "run_id": run_name,
            "lambda_q": best_lambda,
            "quality_tau": t_val,
            "dir": run_dir,
        })

    # Overall best pilot run
    overall_best_run = min(all_runs, key=get_val_ade)
    print(f"\n>>> [PILOT BEST] Overall best model on validation: {overall_best_run['run_id']} (Val ADE: {get_val_ade(overall_best_run):.4f}m)")

    # 4. Q4: Temperature Calibration on overall best run
    print("\n[STAGE 4] Performing Temperature Calibration on best model (QA2)...")
    cmd_calib = [
        PYTHON, "-u", str(HERE / "calibrate_temperature.py"),
        "--model-dir", str(overall_best_run["dir"]),
        "--fold", "eth", "--seed", "1", "--device", args.device,
    ]
    run_cmd(cmd_calib, f"Calibrate temperature for {overall_best_run['run_id']}")

    calib_json = overall_best_run["dir"] / "temperature_calibration_metrics.json"
    calib_metrics = json.loads(calib_json.read_text(encoding="utf-8"))

    # 5. Aggregate metrics tables and synthesize PILOT_ANALYSIS.md
    from QUALITY_AWARE_F16_SERIES.run_quality_aware_series import (
        aggregate_pilot_metrics,
        synthesize_pilot_analysis,
    )
    aggregate_pilot_metrics(pilot_dir, all_runs)
    synthesize_pilot_analysis(pilot_dir, qa0_dir, all_runs, calib_metrics)

    print("\n========================================================")
    print("[SUCCESS] Seed 1 Pilot Phase Completed!")
    print(f"Report: {pilot_dir / 'PILOT_ANALYSIS.md'}")
    print("[STOP POINT 2 REACHED] Pausing per technical guide protocol.")
    print("========================================================\n")


if __name__ == "__main__":
    main()
