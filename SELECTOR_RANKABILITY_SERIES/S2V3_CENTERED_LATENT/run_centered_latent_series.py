#!/usr/bin/env python
"""
Master Orchestrator for Centered Latent Audit & Tiny L5 Selector Series.

Coordinates stages:
- c1: Centered Latent Construction (build_centered_latents.py)
- c2: Centered Latent Statistical Audit (run_c2_centered_variance_audit.py)
- c3: Global Centered Latent Probe (run_c3_global_centered_probe.py)
- c4: Local Top-K Centered Latent Probe (run_c4_local_centered_probe.py)
- t1: Tiny L5 Selector 3-Seed Validation (run_t1_tiny_selector_multiseed.py)
- t2: Efficiency Benchmark (run_t2_efficiency_benchmark.py)
- t3: Fold Stability & Paired Bootstrap (run_t3_fold_stability.py)
- all: Executes all stages sequentially with resumability.

Maintains state.json for incremental resumability.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable

STAGES = [
    ("c1", "build_centered_latents.py", ["--seed", "1"]),
    ("c2", "run_c2_centered_variance_audit.py", ["--seed", "1"]),
    ("c3", "run_c3_global_centered_probe.py", ["--seed", "1"]),
    ("c4", "run_c4_local_centered_probe.py", ["--seed", "1"]),
    ("t1", "run_t1_tiny_selector_multiseed.py", []),
    ("t2", "run_t2_efficiency_benchmark.py", []),
    ("t3", "run_t3_fold_stability.py", []),
]


def load_state(state_file: Path) -> Dict[str, Any]:
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            pass
    return {"completed_stages": {}, "current_stage": None, "last_updated": None}


def save_state(state_file: Path, state: Dict[str, Any]):
    state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(state_file, "w", encoding="utf-8") as fp:
        json.dump(state, fp, indent=2)


def run_stage(stage_id: str, script_name: str, extra_args: List[str], force: bool = False, state_file: Path | None = None) -> bool:
    state = load_state(state_file) if state_file else {}
    if not force and state.get("completed_stages", {}).get(stage_id, {}).get("status") == "SUCCESS":
        print(f"[ORCHESTRATOR] Skipping already completed stage: {stage_id} ({script_name})")
        return True

    print(f"\n=======================================================")
    print(f"   LAUNCHING STAGE: {stage_id.upper()} -> {script_name}")
    print(f"=======================================================")

    script_path = HERE / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    cmd = [PYTHON, "-u", str(script_path)] + extra_args
    t0 = time.time()

    proc = subprocess.run(cmd)
    elapsed = time.time() - t0

    success = (proc.returncode == 0)
    if state_file:
        if "completed_stages" not in state:
            state["completed_stages"] = {}
        state["completed_stages"][stage_id] = {
            "status": "SUCCESS" if success else "FAILED",
            "script": script_name,
            "elapsed_s": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        state["current_stage"] = stage_id if not success else None
        save_state(state_file, state)

    if not success:
        print(f"[ORCHESTRATOR ERROR] Stage {stage_id} failed with return code {proc.returncode}")
        return False

    print(f"[ORCHESTRATOR SUCCESS] Stage {stage_id} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Master Orchestrator for Centered Latent Series")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["c1", "c2", "c3", "c4", "t1", "t2", "t3", "all"],
        help="Stage to run (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Force rerun even if marked SUCCESS in state.json")
    args = parser.parse_args()

    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / "state.json"

    stages_to_run = STAGES if args.stage == "all" else [s for s in STAGES if s[0] == args.stage]

    for s_id, s_script, s_args in stages_to_run:
        ok = run_stage(s_id, s_script, s_args, force=args.force, state_file=state_file)
        if not ok:
            print(f"[ORCHESTRATOR HALT] Pipeline stopped at stage {s_id}")
            sys.exit(1)

    print("\n[ORCHESTRATOR COMPLETE] All requested stages finished successfully.")


if __name__ == "__main__":
    main()
