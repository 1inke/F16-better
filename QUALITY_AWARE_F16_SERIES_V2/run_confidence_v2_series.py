"""
Orchestrator for F16 Confidence Objective Refinement V2.

Supports discrete stage execution:
- --stage protocol: P0-A strict EXP-173 protocol reproduction & verification
- --stage eval-audit: P0-B inference & evaluation equivalence audit
- --stage loss-audit: P0-C loss & gradient mathematical equivalence audit
- --stage redundancy: P1 target redundancy & gradient similarity audit
- --stage pilot: P2 confidence objective replacement pilot (requires authorization)
- --stage final: P3 3-seed x 5-fold final validation (requires authorization)

Note: --stage all is intentionally forbidden to enforce research stop points.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PYTHON = sys.executable


def load_state() -> dict:
    state_file = HERE / "results" / "state.json"
    if not state_file.is_file():
        return {}
    return json.loads(state_file.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    state_file = HERE / "results" / "state.json"
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n>>> [RUNNING] {desc}")
    print(f"Command: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(REPO))
    if res.returncode != 0:
        raise RuntimeError(f"Step '{desc}' failed with exit code {res.returncode}")
    print(f">>> [SUCCESS] {desc}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Confidence Objective Refinement V2 Orchestrator")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["protocol", "eval-audit", "loss-audit", "redundancy", "pilot", "final"],
        help="Discrete execution stage (forbidden: all)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", default="eth")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    state = load_state()

    if args.stage == "protocol":
        run_cmd(
            [PYTHON, "-u", str(HERE / "run_protocol_reproduction.py"), args.device] + (["--force"] if args.force else []),
            "P0-A Protocol Reproduction",
        )
    elif args.stage == "eval-audit":
        run_cmd(
            [PYTHON, "-u", str(HERE / "audit_eval_equivalence.py"), "--fold", args.fold, "--seed", str(args.seed), "--device", args.device],
            "P0-B Evaluation Equivalence Audit",
        )
    elif args.stage == "loss-audit":
        run_cmd(
            [PYTHON, "-u", str(HERE / "audit_loss_equivalence.py"), args.device],
            "P0-C Loss Equivalence Audit",
        )
    elif args.stage == "redundancy":
        # Check P0 prerequisites
        state = load_state()
        p0_ok = (
            state.get("p0_protocol") == "PASS"
            and state.get("p0_eval") == "PASS"
            and state.get("p0_loss") == "PASS"
        )
        if not p0_ok:
            print(f"[BLOCKED] Cannot execute redundancy audit: P0 prerequisites not passed yet: {state}")
            sys.exit(1)
        run_cmd(
            [PYTHON, "-u", str(HERE / "audit_target_redundancy.py"), "--fold", args.fold, "--seed", str(args.seed), "--device", args.device],
            "P1 Target Redundancy Audit",
        )
    elif args.stage == "pilot":
        state = load_state()
        if state.get("p1_redundancy") == "HIGH REDUNDANCY":
            print("[STOP POINT] High redundancy detected in P1! Additive loss is not recommended.")
            print("Per protocol, P2 Pilot requires explicit human decision. Stopping.")
            sys.exit(1)
        print("[PILOT] P2 pilot runner will be implemented and executed upon human decision.")
    elif args.stage == "final":
        print("[FINAL] P3 multi-seed final will be executed upon human decision.")


if __name__ == "__main__":
    main()
