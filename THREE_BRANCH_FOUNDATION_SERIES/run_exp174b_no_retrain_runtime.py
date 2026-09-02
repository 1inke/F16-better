"""EXP-174B: no-retrain runtime optimization for final EXP-173 F16 Joint-V2."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_exp174a_final_efficiency as efficiency  # noqa: E402


DEFAULT_OUT = HERE / "EXP_174B_NO_RETRAIN_RUNTIME_OPTIMIZATION"


def precision_context(dtype: torch.dtype | None):
    return torch.autocast(device_type="cuda", dtype=dtype) if dtype is not None else nullcontext()


def call(model, past: torch.Tensor, num_valid: torch.Tensor):
    center = past.mean(dim=-2, keepdim=True)
    output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
    return output[2] if isinstance(output, (tuple, list)) else output


def dense_batch(cfg, batch_size: int, device: torch.device):
    data = efficiency.ASTRA_ROOT / "datasets" / "eth_ucy" / "processed_data_diverse"
    raw = np.load(data / "eth_data_train.npy", mmap_mode="r")
    counts = np.asarray(np.load(data / "eth_num_train.npy", mmap_mode="r"), dtype=np.int64)
    order = np.argsort(-counts)[:batch_size]
    obs_len = int(cfg.DATA.FREQUENCY * cfg.PREDICTION.OBS_TIME)
    scale = float(cfg.PREDICTION.TRAJ_SCALE)
    past = np.asarray(raw[order, :, :obs_len, -2:], dtype=np.float32) / scale
    return torch.from_numpy(past.copy()).to(device), torch.from_numpy(counts[order].copy()).int().to(device)


def forward_peak(model, dtype, past, valid, device):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    with torch.inference_mode(), precision_context(dtype):
        call(model, past, valid); torch.cuda.synchronize(device)
    return int(max(0, torch.cuda.max_memory_allocated(device) - baseline))


def latency(model, dtype, past, valid, device, warmup: int, iterations: int):
    values = []
    with torch.inference_mode(), precision_context(dtype):
        for _ in range(warmup):
            call(model, past, valid)
        torch.cuda.synchronize(device)
        for _ in range(iterations):
            torch.cuda.synchronize(device); start = time.perf_counter()
            call(model, past, valid); torch.cuda.synchronize(device)
            values.append((time.perf_counter() - start) * 1000.0)
    return values


def collect(model, base_model, dtype, cfg, helper, device, batch_size: int, max_samples: int):
    dataset = helper.TrajectoryDataset(
        efficiency.ASTRA_ROOT / "datasets" / "eth_ucy" / "processed_data_diverse" / "eth_data_test.npy",
        efficiency.ASTRA_ROOT / "datasets" / "eth_ucy" / "processed_data_diverse" / "eth_num_test.npy", cfg,
    )
    if max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions, futures, logits_parts = [], [], []
    with torch.inference_mode(), precision_context(dtype):
        for past, future, num_valid, *_ in loader:
            past, future, num_valid = past.to(device), future.to(device), num_valid.to(device).int()
            past, future = helper.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            raw = output[2] if isinstance(output, (tuple, list)) else output
            predictions.append((raw + center.unsqueeze(-3))[:, 0].float().cpu().numpy())
            futures.append(future[:, 0].float().cpu().numpy())
            logits_parts.append(base_model.last_mode_logits[:, 0].float().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(futures), np.concatenate(logits_parts)


def metrics(prediction: np.ndarray, future: np.ndarray, logits: np.ndarray):
    distance = np.linalg.norm(prediction - future[:, None], axis=-1)
    ade, fde = distance.mean(axis=-1), distance[..., -1]
    rows = np.arange(len(ade)); best_ade = ade.argmin(axis=1); top1 = logits.argmax(axis=1)
    return {
        "samples": int(len(ade)), "minADE": float(ade.min(axis=1).mean()), "minFDE": float(fde.min(axis=1).mean()),
        "same_path_FDE": float(fde[rows, best_ade].mean()), "top1_ADE": float(ade[rows, top1].mean()),
        "top1_FDE": float(fde[rows, top1].mean()),
    }


def relative_change(value: float, reference: float) -> float:
    return float(value / max(abs(reference), 1e-12) - 1.0)


def main():
    parser = argparse.ArgumentParser(description="EXP-174B no-retrain precision/compile audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--accuracy-batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); args.output_dir = args.output_dir.resolve()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    model, cfg, helper = efficiency.build_model("F16", 1, device, 9)
    variants = {
        "B0_FP32_EAGER": {"model": model, "dtype": None, "status": "available"},
        "B1_FP16_AUTOCAST": {"model": model, "dtype": torch.float16, "status": "available"},
        "B1_BF16_AUTOCAST": {"model": model, "dtype": torch.bfloat16, "status": "available"},
    }
    failures = []
    probe_past, probe_valid = dense_batch(cfg, 1, device)
    if not args.skip_compile:
        try:
            compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            with torch.inference_mode():
                call(compiled, probe_past, probe_valid); torch.cuda.synchronize(device)
            variants["B2_TORCH_COMPILE_FP32"] = {"model": compiled, "dtype": None, "status": "available"}
        except Exception as error:
            variants["B2_TORCH_COMPILE_FP32"] = {"model": None, "dtype": None, "status": "failed", "error": repr(error)}
            failures.append({"variant": "B2_TORCH_COMPILE_FP32", "phase": "compile_probe", "error": repr(error)})
            try:
                torch._dynamo.reset()
            except Exception:
                pass
    else:
        variants["B2_TORCH_COMPILE_FP32"] = {"model": None, "dtype": None, "status": "skipped"}

    for name, variant in variants.items():
        if variant["status"] != "available":
            continue
        try:
            with torch.inference_mode(), precision_context(variant["dtype"]):
                result = call(variant["model"], probe_past, probe_valid); torch.cuda.synchronize(device)
            if not torch.isfinite(result).all():
                raise RuntimeError("non-finite prediction")
        except Exception as error:
            variant["status"] = "failed"; variant["error"] = repr(error); failures.append({"variant": name, "error": repr(error)})
    available = {name: value for name, value in variants.items() if value["status"] == "available"}
    if "B0_FP32_EAGER" not in available:
        raise RuntimeError("FP32 baseline failed")

    max_samples = 16 if args.smoke else 0
    accuracy = {}; reference_prediction = None; reference_metrics = None
    for name, variant in list(available.items()):
        print(f"[accuracy] {name}", flush=True)
        try:
            prediction, future, logits = collect(variant["model"], model, variant["dtype"], cfg, helper, device,
                                                  min(args.accuracy_batch_size, 16) if args.smoke else args.accuracy_batch_size,
                                                  max_samples)
            row = metrics(prediction, future, logits)
            if name == "B0_FP32_EAGER":
                reference_prediction, reference_metrics = prediction, row
                row.update({"prediction_mean_abs_error": 0.0, "prediction_max_abs_error": 0.0})
            else:
                row.update({"prediction_mean_abs_error": float(np.abs(prediction - reference_prediction).mean()),
                            "prediction_max_abs_error": float(np.abs(prediction - reference_prediction).max())})
            accuracy[name] = row
        except Exception as error:
            variant["status"] = "failed"; variant["error"] = repr(error); failures.append({"variant": name, "phase": "accuracy", "error": repr(error)})
            available.pop(name)

    rows = []
    batch_sizes = (1,) if args.smoke else (1, 64)
    warmup = 2 if args.smoke else args.warmup
    iterations = 5 if args.smoke else args.iterations
    for batch_size in batch_sizes:
        past, valid = dense_batch(cfg, batch_size, device)
        for name, variant in list(available.items()):
            print(f"[benchmark] {name} batch={batch_size}", flush=True)
            try:
                values = latency(variant["model"], variant["dtype"], past, valid, device, warmup, iterations)
                mean = statistics.fmean(values)
                row = {"variant": name, "batch_size": batch_size, "precision": "FP32" if variant["dtype"] is None else str(variant["dtype"]).replace("torch.", "").upper(),
                       "latency_mean_ms": mean, "latency_median_ms": statistics.median(values),
                       "latency_p95_ms": float(np.percentile(values, 95)), "throughput_samples_s": 1000.0 * batch_size / mean,
                       "forward_peak_mib": forward_peak(variant["model"], variant["dtype"], past, valid, device) / 1024**2,
                       "warmup": warmup, "iterations": iterations, "mean_valid_agents": float(valid.float().mean().item())}
                row.update({f"metric_{key}": value for key, value in accuracy[name].items() if key != "samples"})
                for metric_name in ("minADE", "minFDE", "same_path_FDE", "top1_ADE", "top1_FDE"):
                    row[f"relative_change_{metric_name}"] = relative_change(accuracy[name][metric_name], reference_metrics[metric_name])
                rows.append(row)
            except Exception as error:
                variant["status"] = "failed"; variant["error"] = repr(error); failures.append({"variant": name, "phase": f"benchmark_B{batch_size}", "error": repr(error)})

    baseline_by_batch = {row["batch_size"]: row for row in rows if row["variant"] == "B0_FP32_EAGER"}
    decisions = []
    for name, variant in variants.items():
        variant_rows = [row for row in rows if row["variant"] == name]
        if not variant_rows:
            decisions.append({"variant": name, "status": variant["status"], "accepted": False, "reason": variant.get("error", "not benchmarked")}); continue
        b1 = next(row for row in variant_rows if row["batch_size"] == 1)
        b64 = next((row for row in variant_rows if row["batch_size"] == 64), None)
        latency_gain = 1.0 - b1["latency_mean_ms"] / baseline_by_batch[1]["latency_mean_ms"]
        throughput_gain = 0.0 if b64 is None else b64["throughput_samples_s"] / baseline_by_batch[64]["throughput_samples_s"] - 1.0
        accuracy_safe = abs(b1["relative_change_minADE"]) < 0.002 and abs(b1["relative_change_minFDE"]) < 0.002
        speed_gate = latency_gain >= 0.15 or throughput_gain >= 0.20
        accepted = name != "B0_FP32_EAGER" and accuracy_safe and speed_gate
        decisions.append({"variant": name, "status": variant["status"], "batch1_latency_improvement": latency_gain,
                          "batch64_throughput_improvement": throughput_gain, "accuracy_within_0p2pct": accuracy_safe,
                          "speed_gate_pass": speed_gate, "accepted": accepted})
    accepted = [row for row in decisions if row.get("accepted")]
    recommendation = max(accepted, key=lambda row: (row["batch1_latency_improvement"], row["batch64_throughput_improvement"]))["variant"] if accepted else None
    structural_audit = {
        "B3_SDPA": {"implemented_as_new_variant": False,
                     "reason": "F16 temporal-social attention adds pair-specific relation bias and relation value messages and requires explicit flattened neighbor-time weights; fused SDPA cannot reproduce this as a safe drop-in. Any nn.MultiheadAttention submodules already use PyTorch's inference SDPA path when eligible."},
        "B4_VECTORIZATION": {"implemented_as_new_variant": False,
                             "reason": "Current graph construction uses batched cdist/topk/masks; H3 interpolation uses einsum; V2 correction is batched. No per-agent or per-candidate Python loop remains in these requested paths, so a no-retrain vectorization patch would be a no-op."},
    }
    payload = {"experiment": "EXP-174B", "training": False, "checkpoint": "EXP173 F16 Joint-V2 seed1 eth",
               "device": torch.cuda.get_device_name(device), "torch": torch.__version__, "protocol": {"warmup": warmup, "iterations": iterations,
               "synchronize_each_iteration": True, "accuracy_fold": "eth", "accuracy_tolerance": "relative minADE/minFDE < 0.2%"},
               "variant_status": {name: {key: value for key, value in variant.items() if key not in {"model", "dtype"}} for name, variant in variants.items()},
               "accuracy": accuracy, "runtime_rows": rows, "decisions": decisions, "recommendation": recommendation,
               "structural_audit": structural_audit, "failures": failures}
    if args.smoke:
        print(json.dumps({"status": "smoke_ok", "available": list(available),
                          "variant_status": {name: {key: value for key, value in variant.items() if key not in {"model", "dtype"}} for name, variant in variants.items()},
                          "failures": failures, "rows": rows}, ensure_ascii=False, indent=2)); return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "runtime_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "structural_runtime_audit.json").write_text(json.dumps(structural_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"recommendation": recommendation, "decisions": decisions, "failures": failures}, ensure_ascii=False, indent=2))
    print(f"[complete] {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
