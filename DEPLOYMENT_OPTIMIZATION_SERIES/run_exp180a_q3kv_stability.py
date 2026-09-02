"""Drive five sequential, independent EXP-180A benchmark processes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SERIES = Path(__file__).resolve().parent
EXP_DIR = SERIES / "EXP_180A_Q3KV_STABILITY"
WORKER = SERIES / "benchmark_exp180a_worker.py"
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")


def stream(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(SERIES), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
    if process.wait(): raise RuntimeError(f"Worker failed; see {log_path}")


def aggregate(process_files, batch_sizes, formal):
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in process_files]
    rows = [row for payload in payloads for row in payload["rows"]]
    metrics = ("latency_mean_ms", "latency_median_ms", "latency_p90_ms", "latency_p95_ms",
               "latency_p99_ms", "latency_std_ms", "throughput_samples_s", "peak_vram_mib")
    aggregate_rows, gates = [], []
    for batch_size in batch_sizes:
        by_variant = {}
        for variant in ("F16_baseline", "Q3KV"):
            selected = [row for row in rows if row["batch_size"] == batch_size and row["variant"] == variant]
            by_variant[variant] = selected
            result = {"variant": variant, "batch_size": batch_size, "processes": len(selected)}
            for metric in metrics:
                values = np.asarray([row[metric] for row in selected], np.float64)
                result[f"{metric}_cross_process_mean"] = float(values.mean())
                result[f"{metric}_cross_process_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            aggregate_rows.append(result)
        baseline, q3kv = by_variant["F16_baseline"], by_variant["Q3KV"]
        base_map = {row["process_index"]: row for row in baseline}; q_map = {row["process_index"]: row for row in q3kv}
        improvements = np.asarray([
            (base_map[index]["latency_p95_ms"] - q_map[index]["latency_p95_ms"]) / base_map[index]["latency_p95_ms"]
            for index in sorted(set(base_map) & set(q_map))
        ])
        gates.append({
            "batch_size": batch_size, "p95_improvement_cross_process_mean": float(improvements.mean()),
            "p95_improvement_cross_process_std": float(improvements.std(ddof=1)) if len(improvements) > 1 else 0.0,
            "positive_processes": int((improvements > 0).sum()),
            "p95_at_least_15_percent": bool(improvements.mean() >= 0.15),
            "at_least_4_of_5_positive": bool((improvements > 0).sum() >= 4),
            "passed": bool(formal and improvements.mean() >= 0.15 and (improvements > 0).sum() >= 4),
            "per_process_p95_improvement": improvements.tolist(),
        })
    accepted = bool(formal and all(gate["passed"] for gate in gates))
    return rows, aggregate_rows, gates, accepted, payloads[0]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)


def write_analysis(path, payload):
    lines = ["# EXP-180A result analysis", "", "## Decision", "",
             f"**{payload['decision']}** (accepted: `{str(payload['accepted']).lower()}`).", "", "## P95 stability", "",
             "| Batch | Mean improvement | Cross-process std | Positive processes | Pass |", "|---:|---:|---:|---:|---|"]
    for gate in payload["gates"]:
        lines.append(f"| {gate['batch_size']} | {100*gate['p95_improvement_cross_process_mean']:.2f}% | {100*gate['p95_improvement_cross_process_std']:.2f}% | {gate['positive_processes']}/5 | {gate['passed']} |")
    lines.extend(["", "Q3KV is retained as a deployment variant only when both registered batch sizes pass."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="EXP-180A Q3KV runtime stability audit")
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=100); parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64]); parser.add_argument("--output-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); args.output_dir = args.output_dir.resolve()
    formal = not args.smoke
    if formal and (args.processes != 5 or args.warmup != 100 or args.iterations != 2000 or args.batch_sizes != [1, 64]):
        raise ValueError("Formal EXP-180A is locked to 5 processes, warmup100, iterations2000, batches1/64")
    if not PYTHON.is_file() or not WORKER.is_file(): raise FileNotFoundError(PYTHON if not PYTHON.is_file() else WORKER)
    process_dir = args.output_dir / ("smoke_processes" if args.smoke else "processes")
    process_files = []
    for index in range(args.processes):
        result = process_dir / f"process_{index}.json"; process_files.append(result)
        if result.is_file() and not args.force:
            print(f"[skip] process={index}: {result}", flush=True); continue
        command = [str(PYTHON), "-u", str(WORKER), "--device", args.device,
                   "--process-index", str(index), "--warmup", str(args.warmup),
                   "--iterations", str(args.iterations), "--batch-sizes", *map(str, args.batch_sizes),
                   "--output", str(result)]
        print("[launch] " + subprocess.list2cmdline(command), flush=True)
        stream(command, process_dir / f"process_{index}.log")
    missing = [str(path) for path in process_files if not path.is_file()]
    if missing: raise FileNotFoundError("Missing worker results:\n" + "\n".join(missing))
    rows, aggregate_rows, gates, accepted, metadata = aggregate(process_files, args.batch_sizes, formal)
    decision = "RETAIN_Q3KV_DEPLOYMENT_VARIANT" if accepted else ("SMOKE_COMPLETE_NO_DECISION" if args.smoke else "DROP_Q3KV")
    payload = {
        "experiment": "EXP-180A", "formal_protocol": formal, "training": False,
        "processes": args.processes, "warmup": args.warmup, "iterations": args.iterations,
        "batch_sizes": args.batch_sizes, "precision": "FP32 eager", "device": metadata["device"],
        "checkpoint_protocol": metadata["checkpoint_protocol"],
        "gate_rule": "each B1 and B64: mean P95 improvement >=15% and >=4/5 processes positive",
        "gates": gates, "accepted": accepted, "decision": decision, "aggregate": aggregate_rows,
    }
    output = args.output_dir / ("smoke" if args.smoke else "results"); output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output / "per_process.csv", rows); write_csv(output / "aggregate.csv", aggregate_rows)
    write_analysis(output / "RESULT_ANALYSIS.md", payload)
    print(json.dumps({"status": "smoke_ok" if args.smoke else "complete", "decision": decision, "gates": gates}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
