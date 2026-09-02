from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "run_exp174a_final_efficiency.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def module_targets(model):
    return {
        "module::input_projection": model.lin_proj_xy,
        "module::drse_motion_encoder": model.drse_encoder,
        "module::robust_state": model.robust_state,
        "module::self_state_encoder": model.self_state_encoder,
        "module::f16_encoder_total": model.agent_aware_trans_encoder,
        "module::temporal_transformer": model.agent_aware_trans_encoder.temporal_encoder,
        "module::temporal_queries": model.agent_aware_trans_encoder.temporal_queries,
        "module::joint_v2_decoder": model.decoder,
    }


def timed_hooks(model):
    targets = module_targets(model)
    handles, active = [], {}

    def pre(name):
        def hook(_module, _inputs):
            ctx = torch.profiler.record_function(name)
            ctx.__enter__()
            active.setdefault(name, []).append(ctx)
        return hook

    def post(name):
        def hook(_module, _inputs, _output):
            active[name].pop().__exit__(None, None, None)
        return hook

    for name, module in targets.items():
        handles.append(module.register_forward_pre_hook(pre(name)))
        handles.append(module.register_forward_hook(post(name)))
    return handles


def value(event, modern: str, legacy: str) -> float:
    return float(getattr(event, modern, getattr(event, legacy, 0.0)) or 0.0)


def event_row(event, batch_size: int) -> dict:
    return {
        "batch_size": batch_size,
        "name": event.key,
        "calls": int(event.count),
        "self_cpu_us": value(event, "self_cpu_time_total", "self_cpu_time_total"),
        "cpu_total_us": value(event, "cpu_time_total", "cpu_time_total"),
        "self_device_us": value(event, "self_device_time_total", "self_cuda_time_total"),
        "device_total_us": value(event, "device_time_total", "cuda_time_total"),
        "self_cpu_memory_bytes": int(getattr(event, "self_cpu_memory_usage", 0) or 0),
        "self_device_memory_bytes": int(getattr(event, "self_device_memory_usage", getattr(event, "self_cuda_memory_usage", 0)) or 0),
        "flops": int(getattr(event, "flops", 0) or 0),
    }


def category(name: str) -> str:
    low = name.lower()
    if name.startswith("module::"):
        return "module_range"
    if any(x in low for x in ("cdist", "topk", "sort", "masked_fill", "nonzero")):
        return "graph_and_mask"
    if any(x in low for x in ("softmax", "scaled_dot_product", "attention")):
        return "attention_normalization"
    if any(x in low for x in ("linear", "addmm", "gemm", "mm", "matmul", "bmm")):
        return "matrix_and_mlp"
    if any(x in low for x in ("layer_norm", "native_layer_norm", "gelu", "silu", "relu")):
        return "normalization_activation"
    if any(x in low for x in ("einsum", "interpolate", "upsample")):
        return "decoder_interpolation"
    if any(x in low for x in ("copy", "to", "contiguous", "reshape", "view", "permute", "transpose")):
        return "layout_and_copy"
    return "other"


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        keys = list(rows[0]) + sorted({key for row in rows for key in row} - set(rows[0]))
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def cuda_event_fallback(base, model, past, valid, steps: int, device):
    """Inclusive CUDA module timing when CUPTI operator activities are unavailable."""
    pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
        name: [] for name in module_targets(model)
    }
    stacks: dict[str, list[torch.cuda.Event]] = {name: [] for name in pairs}
    handles = []

    def pre(name):
        def hook(_module, _inputs):
            start = torch.cuda.Event(enable_timing=True)
            start.record(); stacks[name].append(start)
        return hook

    def post(name):
        def hook(_module, _inputs, _output):
            end = torch.cuda.Event(enable_timing=True)
            end.record(); pairs[name].append((stacks[name].pop(), end))
        return hook

    for name, module in module_targets(model).items():
        handles.append(module.register_forward_pre_hook(pre(name)))
        handles.append(module.register_forward_hook(post(name)))
    total_pairs = []
    try:
        with torch.inference_mode():
            for _ in range(steps):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); base.forward(model, past, valid); end.record()
                total_pairs.append((start, end))
        torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    totals = {name: sum(a.elapsed_time(b) for a, b in values) * 1000.0 for name, values in pairs.items()}
    totals["step::model_forward"] = sum(a.elapsed_time(b) for a, b in total_pairs) * 1000.0
    return totals


def profile_case(base, model, past, valid, batch_size: int, device, warmup: int, active_steps: int, output_dir: Path):
    handles = timed_hooks(model)
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                base.forward(model, past, valid)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            schedule = torch.profiler.schedule(wait=1, warmup=2, active=active_steps, repeat=1)
            with torch.profiler.profile(
                activities=activities, schedule=schedule, record_shapes=True,
                profile_memory=True, with_flops=True,
            ) as prof:
                for _ in range(active_steps + 3):
                    with torch.profiler.record_function("step::model_forward"):
                        base.forward(model, past, valid)
                    prof.step()
            torch.cuda.synchronize(device)
            peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
    finally:
        for handle in handles:
            handle.remove()

    events = [event_row(e, batch_size) for e in prof.key_averages()]
    top = sorted(events, key=lambda x: x["self_device_us"], reverse=True)[:80]
    modules = [x for x in events if x["name"].startswith("module::") or x["name"].startswith("step::")]
    op_events = [x for x in events if not x["name"].startswith(("module::", "step::"))]
    total_self_device = sum(x["self_device_us"] for x in op_events)
    total_self_cpu = sum(x["self_cpu_us"] for x in op_events)
    categories = []
    for label in sorted({category(x["name"]) for x in op_events}):
        selected = [x for x in op_events if category(x["name"]) == label]
        cuda = sum(x["self_device_us"] for x in selected)
        cpu = sum(x["self_cpu_us"] for x in selected)
        categories.append({
            "batch_size": batch_size, "category": label,
            "self_device_us": cuda, "device_fraction": cuda / max(total_self_device, 1e-9),
            "self_cpu_us": cpu, "cpu_fraction": cpu / max(total_self_cpu, 1e-9),
            "calls": sum(x["calls"] for x in selected),
        })
    backend = "cupti_operator_activities"
    step_device = next((r["device_total_us"] for r in modules if r["name"] == "step::model_forward"), 0.0)
    if total_self_device <= 0.0 or step_device <= 0.0:
        backend = "cuda_events_module_fallback"
        event_totals = cuda_event_fallback(base, model, past, valid, active_steps, device)
        by_name = {row["name"]: row for row in modules}
        for name, duration in event_totals.items():
            if name not in by_name:
                row = event_row(type("Event", (), {"key": name, "count": active_steps})(), batch_size)
                modules.append(row); by_name[name] = row
            by_name[name]["device_total_us"] = duration
        step_device = event_totals["step::model_forward"]
    for row in modules:
        row["device_fraction_of_step_total"] = row["device_total_us"] / max(step_device, 1e-9)
    prof.export_chrome_trace(str(output_dir / f"trace_batch_{batch_size}.json"))
    return {
        "batch_size": batch_size, "active_steps": active_steps, "peak_vram_mib": peak_mib,
        "operator_self_device_total_us": total_self_device,
        "operator_self_cpu_total_us": total_self_cpu, "cuda_timing_backend": backend,
        "top": top, "modules": modules, "categories": categories,
    }


def main():
    parser = argparse.ArgumentParser(description="EXP-174C profiler-guided kernel audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--active-steps", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "results")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("EXP-174C requires CUDA profiling")
    if args.smoke:
        args.batch_sizes, args.warmup, args.active_steps = [1], 2, 2
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    base = load_module("exp174c_base", BASE_SCRIPT)
    # Batch size and agent count are independent.  The official ETH/UCY
    # efficiency protocol uses real batches with MAX_AGENTS=9; using the
    # largest batch size here incorrectly reconfigured the model to 64 agents.
    model, cfg, trainer = base.build_model("F16", args.seed, device, 9)
    cases = []
    for batch_size in args.batch_sizes:
        print(f"[profile] batch={batch_size}", flush=True)
        past, valid = base.real_batch(trainer, cfg, batch_size, device)
        cases.append(profile_case(base, model, past, valid, batch_size, device, args.warmup, args.active_steps, output_dir))

    top_rows = [r for c in cases for r in c["top"]]
    module_rows = [r for c in cases for r in c["modules"]]
    category_rows = [r for c in cases for r in c["categories"]]
    write_csv(output_dir / "top_operators.csv", top_rows)
    write_csv(output_dir / "module_ranges.csv", module_rows)
    write_csv(output_dir / "operator_categories.csv", category_rows)
    candidate_ranges = [r for r in module_rows if r["name"] in ("module::f16_encoder_total", "module::joint_v2_decoder")]
    bottlenecks = [r for r in candidate_ranges if r["device_fraction_of_step_total"] >= 0.20]
    payload = {
        "experiment": "EXP-174C", "training": False, "model": "EXP173 F16 + H3 + Joint-V2",
        "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "cases": [{k: v for k, v in c.items() if k not in ("top", "modules", "categories")} for c in cases],
        "module_ranges": module_rows, "operator_categories": category_rows,
        "decision_rule": "Only a module with inclusive device time >=20% justifies targeted structural optimization",
        "candidate_bottlenecks": bottlenecks,
        "next": "Consider EXP-174D only if a stable >=20% structural bottleneck is present in both relevant profiles",
        "smoke": args.smoke,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "smoke_ok" if args.smoke else "complete", "cases": len(cases), "bottlenecks": [x["name"] for x in bottlenecks]}, ensure_ascii=False, indent=2))
    print(f"[result] {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
