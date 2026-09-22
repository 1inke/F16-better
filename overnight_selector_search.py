#!/usr/bin/env python
"""
Unattended selector experiment orchestrator.

Orchestrates the F16 Offline Selector Zoo pipeline:
1. Candidate Cache Generation (Frozen F16 OOF inference)
2. Feature Bank Construction (G0..G4)
3. Stage 1 Broad Search (Seed 1, 28+ methods, 5-fold LOO)
4. Screen Aggregation & Pareto Finalist Selection (Top-6)
5. Stage 2 Multi-Seed Confirmation (Seeds 2 & 3)
6. Stage 3 Final Academic Reporting

Properties:
- Resumable state machine (overnight_state.json)
- Per-job logs in work_root/logs
- Failure isolation (non-fatal jobs allow pipeline to proceed)
- Windows-friendly: shell=False, UTF-8 logging
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class Runner:
    def __init__(self, cfg: dict[str, Any], dry_run: bool = False, force: bool = False):
        self.cfg = cfg
        self.repo = Path(cfg["repo_root"]).resolve()
        self.work = (self.repo / cfg["work_root"]).resolve()
        self.logs = self.work / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work / "overnight_state.json"
        self.state = load_json(self.state_path) if self.state_path.exists() else {"jobs": {}}
        self.dry_run = dry_run
        self.force = force
        self.failures: list[str] = []

        self.python = cfg.get("python_executable") or sys.executable
        self.env = os.environ.copy()
        self.env["PYTHONUNBUFFERED"] = "1"
        for k, v in cfg.get("env", {}).items():
            self.env[str(k)] = str(v)

    def save_state(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def script(self, key: str) -> str:
        val = self.cfg["scripts"][key]
        p = Path(val)
        if not p.is_absolute():
            p = (self.repo / p).resolve()
        return str(p)

    def run_job(
        self,
        name: str,
        args: list[str],
        *,
        required: bool = False,
        cwd: Path | None = None,
    ) -> bool:
        jobs = self.state.setdefault("jobs", {})
        old = jobs.get(name, {})

        if old.get("status") == "success" and not self.force:
            print(f"[SKIP success] {name}")
            return True

        command = [self.python, *args]
        print("\n" + "=" * 90)
        print(f"[JOB] {name}")
        print("[CMD]", subprocess.list2cmdline(command))

        if self.dry_run:
            return True

        log_path = self.logs / f"{name.replace('/', '__').replace(':', '_')}.log"
        jobs[name] = {
            "status": "running",
            "started_at": now_iso(),
            "command": command,
            "log": str(log_path),
        }
        self.save_state()

        try:
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write(f"\n\n===== START {now_iso()} =====\n")
                log.write(subprocess.list2cmdline(command) + "\n\n")
                proc = subprocess.Popen(
                    command,
                    cwd=str(cwd or self.repo),
                    env=self.env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=False,
                )
                rc = proc.wait()

            jobs[name].update({
                "status": "success" if rc == 0 else "failed",
                "finished_at": now_iso(),
                "returncode": rc,
            })
            self.save_state()

            if rc != 0:
                self.failures.append(name)
                print(f"[FAILED rc={rc}] {name} -> {log_path}")
                if required:
                    raise RuntimeError(f"Required job failed: {name}")
                return False

            print(f"[OK] {name}")
            return True

        except KeyboardInterrupt:
            jobs[name].update({
                "status": "interrupted",
                "finished_at": now_iso(),
            })
            self.save_state()
            raise
        except Exception as exc:
            jobs[name].update({
                "status": "failed",
                "finished_at": now_iso(),
                "exception": repr(exc),
            })
            self.save_state()
            self.failures.append(name)
            if required:
                raise
            print(f"[EXCEPTION] {name}: {exc}")
            return False

    def build_seed_cache(self, seed: int) -> None:
        for fold in self.cfg["folds"]:
            self.run_job(
                f"cache/seed{seed}/{fold}",
                [
                    self.script("build_candidate_cache"),
                    "--model", "f16",
                    "--seed", str(seed),
                    "--fold", fold,
                    "--split", "test",
                    "--shard-size", str(self.cfg.get("cache_shard_size", 2048)),
                    "--save-neighbor-distribution",
                    "--output-root", self.cfg["cache_root"],
                ],
                required=False,
            )

        self.run_job(
            f"features/seed{seed}",
            [
                self.script("build_feature_bank"),
                "--seed", str(seed),
                "--cache-root", self.cfg["cache_root"],
                "--feature-root", self.cfg["feature_root"],
                "--feature-groups", ",".join(self.cfg["feature_groups"]),
            ],
            required=True,
        )

    def run_method(self, seed: int, method: str, phase: str) -> None:
        self.run_job(
            f"{phase}/seed{seed}/{method}",
            [
                self.script("run_selector_benchmark"),
                "--seed", str(seed),
                "--method", method,
                "--protocol", "leave_one_fold_out",
                "--cache-root", self.cfg["cache_root"],
                "--feature-root", self.cfg["feature_root"],
                "--output-root", f'{self.cfg["result_root"]}/{phase}/seed_{seed}',
            ],
            required=False,
        )

    def aggregate(self, phase: str, seeds: list[int]) -> None:
        seed_args = [str(s) for s in seeds]
        self.run_job(
            f"aggregate/{phase}",
            [
                self.script("aggregate_results"),
                "--input-root", f'{self.cfg["result_root"]}/{phase}',
                "--seeds", *seed_args,
                "--output-root", f'{self.cfg["result_root"]}/{phase}_summary',
            ],
            required=True,
        )

    def select_finalists(self) -> list[str]:
        finalists_path = Path(self.cfg["finalists_file"])
        out_json = (self.repo / finalists_path).resolve() if not finalists_path.is_absolute() else finalists_path
        self.run_job(
            "select_finalists",
            [
                self.script("select_finalists"),
                "--leaderboard",
                f'{self.cfg["result_root"]}/screen_summary/LEADERBOARD.csv',
                "--top-k", str(self.cfg["finalist_top_k"]),
                "--max-ade-degradation-pct",
                str(self.cfg["selection_constraints"]["max_ade_degradation_pct"]),
                "--max-fde-degradation-pct",
                str(self.cfg["selection_constraints"]["max_fde_degradation_pct"]),
                "--max-collision-increase-pp",
                str(self.cfg["selection_constraints"]["max_collision_increase_pp"]),
                "--output", self.cfg["finalists_file"],
            ],
            required=True,
        )

        if self.dry_run:
            mock_finalists = ["native", "mpn_shield_k5", "mlp_pairwise_g0124", "linear_residual_g012", "mbr_native_weighted"]
            print("[DRY-RUN FINALISTS]", ", ".join(mock_finalists))
            return mock_finalists

        data = load_json(out_json)
        methods = data.get("methods", [])
        if not methods:
            raise RuntimeError(f"No finalists found in {out_json}")
        print("[FINALISTS]", ", ".join(methods))
        return methods

    def write_failure_summary(self) -> None:
        failures = []
        for name, meta in self.state.get("jobs", {}).items():
            if meta.get("status") not in ("success",):
                failures.append({"job": name, **meta})

        path = self.work / "failure_summary.json"
        atomic_write_json(path, {"generated_at": now_iso(), "failures": failures})
        print(f"[SUMMARY] failure summary -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Overnight Selector Search Orchestrator")
    ap.add_argument("--config", default="selector_overnight_config.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--phase",
        choices=["all", "screen", "confirm"],
        default="all",
        help="screen=seed1 broad search; confirm=read finalists and run confirm seeds",
    )
    args = ap.parse_args()

    cfg = load_json(Path(args.config))
    r = Runner(cfg, dry_run=args.dry_run, force=args.force)

    try:
        # ------------------------------------------------------------------
        # Stage 1: one seed, very wide selector search
        # ------------------------------------------------------------------
        if args.phase in ("all", "screen"):
            screen_seed = int(cfg["screen_seed"])
            r.build_seed_cache(screen_seed)

            for method in cfg["screen_methods"]:
                r.run_method(screen_seed, method, "screen")

            r.aggregate("screen", [screen_seed])
            finalists = r.select_finalists()
        else:
            finalists_path = Path(cfg["finalists_file"])
            finalists_file = (r.repo / finalists_path).resolve() if not finalists_path.is_absolute() else finalists_path
            finalists = load_json(finalists_file)["methods"]

        # ------------------------------------------------------------------
        # Stage 2: only finalists receive additional seed budget
        # ------------------------------------------------------------------
        if args.phase in ("all", "confirm"):
            confirm_seeds = [int(x) for x in cfg["confirm_seeds"]]

            for seed in confirm_seeds:
                r.build_seed_cache(seed)
                for method in finalists:
                    r.run_method(seed, method, "confirm")

            final_seeds = [int(cfg["screen_seed"]), *confirm_seeds]
            r.aggregate("confirm", confirm_seeds)

            r.run_job(
                "final_report",
                [
                    r.script("final_report"),
                    "--screen-root", f'{cfg["result_root"]}/screen',
                    "--confirm-root", f'{cfg["result_root"]}/confirm',
                    "--finalists", cfg["finalists_file"],
                    "--all-seeds", *[str(s) for s in final_seeds],
                    "--output-root", f'{cfg["result_root"]}/final',
                ],
                required=False,
            )

    finally:
        r.write_failure_summary()


if __name__ == "__main__":
    main()
