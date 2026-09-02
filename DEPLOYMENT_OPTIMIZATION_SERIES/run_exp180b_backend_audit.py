"""EXP-180B: fixed-shape deployment backend audit for final F16 Joint-V2."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from benchmark_onnx import benchmark_ort, benchmark_pytorch, make_session, metric_bundle, run_numpy
from export_onnx import build_profile, export_profile


HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE / "EXP_180B_BACKEND_AUDIT"


def gpu_used_mb() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return float(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


def dataset_sample(dataset, index: int, device: torch.device):
    item = dataset[index]
    past = torch.as_tensor(item[0], dtype=torch.float32, device=device).unsqueeze(0)
    future = torch.as_tensor(item[1], dtype=torch.float32, device=device).unsqueeze(0)
    valid = torch.as_tensor(item[2], dtype=torch.long, device=device).reshape(1)
    return past, future[:, 0], valid


def correctness(wrapper, dataset, session, device, count: int):
    pt_candidates, pt_logits, ort_candidates, ort_logits, futures = [], [], [], [], []
    with torch.inference_mode():
        for index in range(min(count, len(dataset))):
            past, future, valid = dataset_sample(dataset, index, device)
            pt_c, pt_l = wrapper(past, valid)
            out_c, out_l = run_numpy(session, past.cpu().numpy(), valid.cpu().numpy())
            pt_candidates.append(pt_c.cpu().numpy()); pt_logits.append(pt_l.cpu().numpy())
            ort_candidates.append(out_c); ort_logits.append(out_l); futures.append(future.cpu().numpy())
    pt_c = np.concatenate(pt_candidates); pt_l = np.concatenate(pt_logits)
    ort_c = np.concatenate(ort_candidates); ort_l = np.concatenate(ort_logits)
    truth = np.concatenate(futures)
    pt_metrics = metric_bundle(pt_c, truth, pt_l)
    ort_metrics = metric_bundle(ort_c, truth, ort_l)
    result = {
        "samples": int(len(pt_c)),
        "candidate_max_abs": float(np.max(np.abs(pt_c - ort_c))),
        "candidate_mean_abs": float(np.mean(np.abs(pt_c - ort_c))),
        "logit_max_abs": float(np.max(np.abs(pt_l - ort_l))),
        "logit_mean_abs": float(np.mean(np.abs(pt_l - ort_l))),
    }
    for name, value in pt_metrics.items():
        result[f"pytorch_{name}"] = value
        result[f"backend_{name}"] = ort_metrics[name]
        result[f"relative_{name}_pct"] = float(abs(ort_metrics[name] - value) / max(abs(value), 1e-12) * 100.0)
    result["accuracy_pass"] = bool(
        result["relative_minADE_pct"] < 0.2 and result["relative_minFDE_pct"] < 0.2
    )
    return result


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def relative_gain(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / baseline * 100.0)


def write_report(output_dir: Path, summary: dict):
    runtime_rows = summary["runtime"]
    correctness_rows = summary["correctness"]
    lines = [
        "# EXP-180B Deployment Backend Audit", "",
        "固定模型：EXP-173 final F16 + H3 + Joint-V2 + original score + K20。", "",
        "## 判定标准", "",
        "- ONNX 相对 PyTorch 的 minADE/minFDE 变化均小于 0.2%。",
        "- 中位延迟降低至少 20%，或吞吐率提高至少 25%。", "",
        "## Runtime", "",
        "| Profile | Backend | Median ms | P95 ms | Throughput/s | Median gain | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in runtime_rows:
        gain = row.get("median_improvement_pct")
        lines.append(
            f"| {row['profile']} | {row['backend']} | {row.get('latency_median_ms', float('nan')):.4f} | "
            f"{row.get('latency_p95_ms', float('nan')):.4f} | {row.get('throughput_samples_s', float('nan')):.2f} | "
            f"{('-' if gain is None else f'{gain:.2f}%')} | {row.get('verdict', 'REFERENCE')} |"
        )
    lines += ["", "## Correctness", "",
              "| Profile | Backend | ΔminADE | ΔminFDE | Candidate max abs | Verdict |",
              "|---|---:|---:|---:|---:|---|"]
    for row in correctness_rows:
        lines.append(
            f"| {row['profile']} | {row['backend']} | {row['relative_minADE_pct']:.4f}% | "
            f"{row['relative_minFDE_pct']:.4f}% | {row['candidate_max_abs']:.6f} | "
            f"{'PASS' if row['accuracy_pass'] else 'FAIL'} |"
        )
    failures = summary.get("failures", [])
    lines += ["", "## 自动结论", "", summary["conclusion"]]
    if failures:
        lines += ["", "## 未完成后端", ""] + [f"- {x['profile']} / {x['backend']}: {x['error']}" for x in failures]
    lines += ["", "说明：ONNX 导出对固定尺寸 `EyeLike` 与旧导出器的 `atan2(0,0)` 展开做了等价兼容处理。"]
    (output_dir / "RESULT_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profiles", default="all", help="all, eth_n9, sdd_n16, or comma-separated")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--accuracy-samples", type=int, default=64)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-180B requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.profiles == "all":
        profiles = ["eth_n9", "sdd_n16"]
    else:
        profiles = [x.strip() for x in args.profiles.split(",") if x.strip()]
    invalid = set(profiles) - {"eth_n9", "sdd_n16"}
    if invalid:
        raise ValueError(f"Unknown profiles: {sorted(invalid)}")
    warmup = min(args.warmup, 2) if args.smoke else args.warmup
    iterations = min(args.iterations, 5) if args.smoke else args.iterations
    accuracy_samples = min(args.accuracy_samples, 2) if args.smoke else args.accuracy_samples
    output_dir = RESULT_ROOT / "smoke" if args.smoke else RESULT_ROOT
    artifact_dir = output_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_rows, correctness_rows, failures = [], [], []

    for profile in profiles:
        print(f"[profile] {profile}", flush=True)
        exported = export_profile(profile, artifact_dir, device, force=args.force_export)
        wrapper, dataset, _ = build_profile(profile, device)
        past, _, valid = dataset_sample(dataset, 0, device)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        baseline_runtime = benchmark_pytorch(wrapper, past, valid, device, 1, warmup, iterations)
        baseline_runtime.update({
            "profile": profile, "backend": "B0_pytorch_fp32", "verdict": "REFERENCE",
            "median_improvement_pct": None,
            "vram_peak_mb": float(torch.cuda.max_memory_allocated(device) / 1048576.0),
        })
        runtime_rows.append(baseline_runtime)

        for backend, model_key in (("B1_ort_cuda_fp32", "fp32"), ("B2_ort_cuda_fp16", "fp16")):
            print(f"[backend] {profile}/{backend}", flush=True)
            try:
                memory_before = gpu_used_mb(); start = time.perf_counter()
                session = make_session(exported[model_key], "CUDAExecutionProvider")
                initialization_s = time.perf_counter() - start
                accuracy = correctness(wrapper, dataset, session, device, accuracy_samples)
                runtime = benchmark_ort(
                    session, past.cpu().numpy(), valid.cpu().numpy(), 1, warmup, iterations
                )
                memory_after = gpu_used_mb()
                median_gain = relative_gain(baseline_runtime["latency_median_ms"], runtime["latency_median_ms"])
                throughput_gain = float(
                    (runtime["throughput_samples_s"] / baseline_runtime["throughput_samples_s"] - 1.0) * 100.0
                )
                passed = accuracy["accuracy_pass"] and (median_gain >= 20.0 or throughput_gain >= 25.0)
                runtime.update({
                    "profile": profile, "backend": backend, "initialization_s": initialization_s,
                    "median_improvement_pct": median_gain, "throughput_improvement_pct": throughput_gain,
                    "vram_delta_mb": None if memory_before is None or memory_after is None else memory_after - memory_before,
                    "verdict": "PASS" if passed else "FAIL",
                })
                accuracy.update({"profile": profile, "backend": backend})
                runtime_rows.append(runtime); correctness_rows.append(accuracy)
            except Exception as error:
                failures.append({"profile": profile, "backend": backend, "error": repr(error)})
                print(f"[warning] {profile}/{backend}: {error!r}", flush=True)
            finally:
                if "session" in locals():
                    del session
                gc.collect(); torch.cuda.empty_cache()

    tested = [row for row in runtime_rows if row["backend"] != "B0_pytorch_fp32"]
    passing = [row for row in tested if row.get("verdict") == "PASS"]
    if passing:
        names = ", ".join(f"{x['profile']}/{x['backend']}" for x in passing)
        conclusion = f"存在通过 EXP-180B 门槛的部署组合：{names}。正式保留前需结合两个 profile 综合判断。"
    elif tested:
        conclusion = "当前已完成的 ONNX Runtime 后端均未同时通过精度与速度门槛；按指导应保留 PyTorch FP32，并转 EXP-180C。"
    else:
        conclusion = "没有后端完成测试，需先检查运行环境；不能据此判断部署路线。"
    summary = {
        "experiment": "EXP-180B", "model": "EXP-173 final F16 Joint-V2", "profiles": profiles,
        "warmup": warmup, "iterations": iterations, "accuracy_samples": accuracy_samples,
        "smoke": args.smoke, "runtime": runtime_rows, "correctness": correctness_rows,
        "failures": failures, "conclusion": conclusion,
        "environment": {"torch": torch.__version__, "onnxruntime": ort.__version__, "gpu": torch.cuda.get_device_name(device)},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "runtime.csv", runtime_rows)
    write_csv(output_dir / "correctness.csv", correctness_rows)
    write_report(output_dir, summary)
    print(f"[done] {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
