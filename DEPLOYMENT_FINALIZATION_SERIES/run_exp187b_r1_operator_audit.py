"""EXP-187B-R1: isolate ONNX graph vs CUDA-provider divergence on outlier profiles."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "run_exp187b_full_ort_equivalence.py"
BASE_OUT = ROOT / "EXP_187B_FULL_ORT_EQUIVALENCE"
OUT = BASE_OUT / "EXP_187B_R1_OPERATOR_AUDIT"
DEFAULT_PROFILES = ("hotel", "univ", "zara02", "sdd")


def load_base():
    spec = importlib.util.spec_from_file_location("exp187b_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_session(path: Path, provider: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if provider not in available:
        raise RuntimeError(f"{provider} unavailable: {available}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    providers = [provider]
    if provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(path), sess_options=options, providers=providers)


def comparison(left_c, left_l, right_c, right_l, prefix: str) -> dict:
    dc, dl = np.abs(left_c - right_c), np.abs(left_l - right_l)
    return {
        f"{prefix}_candidate_mean": float(dc.mean()),
        f"{prefix}_candidate_max": float(dc.max()),
        f"{prefix}_logit_mean": float(dl.mean()),
        f"{prefix}_logit_max": float(dl.max()),
        f"{prefix}_top1_changed": int(left_l.argmax() != right_l.argmax()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def summarize_values(rows: list[dict], key: str) -> dict:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "p99_9": float(np.percentile(values, 99.9)), "max": float(values.max()),
        "count_gt_0_1": int((values > 0.1).sum()), "count_gt_0_5": int((values > 0.5).sum()),
    }


def classify(rows: list[dict]) -> str:
    pt_cpu = max(float(row["pt_cpu_candidate_max"]) for row in rows)
    pt_cuda = max(float(row["pt_cuda_candidate_max"]) for row in rows)
    cpu_cuda = max(float(row["cpu_cuda_candidate_max"]) for row in rows)
    if pt_cpu <= 0.1 and pt_cuda > 0.5 and cpu_cuda > 0.5:
        return "CUDA_PROVIDER_DIVERGENCE"
    if pt_cpu > 0.5 and pt_cuda > 0.5 and cpu_cuda <= 0.1:
        return "EXPORTED_GRAPH_OR_TOPK_DIVERGENCE"
    if pt_cpu > 0.5 and pt_cuda > 0.5 and cpu_cuda > 0.5:
        return "MIXED_GRAPH_AND_PROVIDER_DIVERGENCE"
    if max(pt_cpu, pt_cuda, cpu_cuda) <= 0.1:
        return "NUMERICALLY_STABLE"
    return "BOUNDARY_DIFFERENCE_REQUIRES_SAMPLE_TRACE"


def audit(profile: str, args, base, device: torch.device, output_dir: Path) -> dict:
    cache = output_dir / "profiles" / f"{profile}.json"
    rows_path = output_dir / "profiles" / f"{profile}_sample_diffs.csv"
    if cache.is_file() and rows_path.is_file() and not args.force:
        print(f"[reuse] {profile}", flush=True)
        return json.loads(cache.read_text(encoding="utf-8"))
    model_path = BASE_OUT / "artifacts" / f"{profile}_fp32.onnx"
    if not model_path.is_file():
        raise FileNotFoundError(f"Run EXP-187B export first: {model_path}")
    wrapper, dataset, _ = base.build(profile, device)
    cpu_session = make_session(model_path, "CPUExecutionProvider")
    cuda_session = make_session(model_path, "CUDAExecutionProvider")
    count = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)
    rows, started = [], time.perf_counter()
    with torch.inference_mode():
        for index in range(count):
            past, _, valid = base.sample(dataset, index, device)
            pt_c, pt_l = wrapper(past, valid)
            feed = {"past": past.cpu().numpy(), "num_valid": valid.cpu().numpy()}
            cpu_c, cpu_l = cpu_session.run(["candidates", "logits"], feed)
            cuda_c, cuda_l = cuda_session.run(["candidates", "logits"], feed)
            pt_c, pt_l = pt_c.cpu().numpy(), pt_l.cpu().numpy()
            row = {"profile": profile, "sample_index": index, "num_valid": int(valid.item())}
            row.update(comparison(pt_c, pt_l, cpu_c, cpu_l, "pt_cpu"))
            row.update(comparison(pt_c, pt_l, cuda_c, cuda_l, "pt_cuda"))
            row.update(comparison(cpu_c, cpu_l, cuda_c, cuda_l, "cpu_cuda"))
            row.update({"pt_top1": int(pt_l.argmax()), "cpu_top1": int(cpu_l.argmax()), "cuda_top1": int(cuda_l.argmax())})
            rows.append(row)
            if (index + 1) % args.progress_every == 0 or index + 1 == count:
                print(f"[{profile}] {index + 1}/{count}", flush=True)
    keys = [
        f"{pair}_{kind}_{stat}"
        for pair in ("pt_cpu", "pt_cuda", "cpu_cuda")
        for kind in ("candidate", "logit")
        for stat in ("mean", "max")
    ]
    result = {
        "profile": profile, "samples": count, "elapsed_seconds": time.perf_counter() - started,
        "classification": classify(rows),
        "statistics": {key: summarize_values(rows, key) for key in keys},
        "top1_change_rates": {pair: float(np.mean([row[f"{pair}_top1_changed"] for row in rows])) for pair in ("pt_cpu", "pt_cuda", "cpu_cuda")},
    }
    write_csv(rows_path, rows)
    top = sorted(rows, key=lambda row: max(row["pt_cpu_candidate_max"], row["pt_cuda_candidate_max"], row["cpu_cuda_candidate_max"]), reverse=True)[: args.top_n]
    write_csv(output_dir / "profiles" / f"{profile}_top_outliers.csv", top)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def aggregate(results: list[dict], output_dir: Path, smoke: bool) -> dict:
    classes = {row["profile"]: row["classification"] for row in results}
    if smoke:
        decision = "PREFLIGHT_OK"
    elif all(value == "NUMERICALLY_STABLE" for value in classes.values()):
        decision = "NO_REPRODUCED_OPERATOR_ANOMALY"
    elif all(value == "CUDA_PROVIDER_DIVERGENCE" for value in classes.values()):
        decision = "PATCH_OR_AVOID_CUDA_SCATTER_PATH"
    elif all(value == "EXPORTED_GRAPH_OR_TOPK_DIVERGENCE" for value in classes.values()):
        decision = "REWRITE_EXPORT_GRAPH_TOPK_SCATTER"
    else:
        decision = "TRACE_RECORDED_OUTLIERS_AND_REWRITE_DISCRETE_ROUTING"
    payload = {"experiment": "EXP-187B-R1", "same_onnx_cpu_cuda": True, "smoke": smoke, "profiles": results, "classification": classes, "decision": decision}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "profile_summary.csv", [{"profile": row["profile"], "samples": row["samples"], "classification": row["classification"], **{f"top1_{key}": value for key, value in row["top1_change_rates"].items()}} for row in results])
    lines = ["# EXP-187B-R1 Operator Audit", "", f"Decision: **{decision}**", "", "| Profile | Samples | Classification | PT-CPU Top1 | PT-CUDA Top1 | CPU-CUDA Top1 |", "|---|---:|---|---:|---:|---:|"]
    for row in results:
        rate = row["top1_change_rates"]
        lines.append(f"| {row['profile']} | {row['samples']} | {row['classification']} | {100*rate['pt_cpu']:.3f}% | {100*rate['pt_cuda']:.3f}% | {100*rate['cpu_cuda']:.3f}% |")
    (output_dir / "RESULT_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profiles", nargs="+", choices=(*DEFAULT_PROFILES, "eth", "zara01"), default=list(DEFAULT_PROFILES))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.smoke:
        args.max_samples = min(args.max_samples or 10, 100)
        output_dir = OUT / "smoke"
    else:
        output_dir = OUT
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    base = load_base()
    results = [audit(profile, args, base, device, output_dir) for profile in args.profiles]
    payload = aggregate(results, output_dir, args.smoke)
    print(json.dumps({"classification": payload["classification"], "decision": payload["decision"]}, ensure_ascii=False), flush=True)
    print(f"[done] {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
