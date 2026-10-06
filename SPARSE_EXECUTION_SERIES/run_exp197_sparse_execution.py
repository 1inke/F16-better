"""EXP-197: zero-training audit of real sparse execution for formal F16."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
FORMAL = WORKSPACE / "FORMAL_ENCODER_ABLATION_SERIES"
CHECKPOINT = FORMAL / "EXP_154_FORMAL_ENCODER_ABLATION" / "results" / "general" / "d_f16_full_temporal" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"
CONTRACT = CHECKPOINT.parent / "locked_backend_contract.json"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
OUT = ROOT / "EXP_197_RESULTS"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup(device: torch.device):
    for path in (PROJECT, CHECKPOINT, CONTRACT, DATA / "eth_data_train.npy"):
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT), str(FORMAL), str(ROOT)]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    import formal_decoder_adapter
    from sparse_exec_adapter import enable_edge_first

    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    os.environ["FORMAL_DECODER_SOURCE"] = str(CHECKPOINT)

    def build(edge: bool):
        model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
        formal_decoder_adapter.replace_decoder(model, cfg, "fixed_current_mode_anchor")
        model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
        if edge:
            enable_edge_first(model)
        return model.to(device).eval()

    dense, edge = build(False), build(True)
    dense_count = sum(p.numel() for p in dense.parameters())
    edge_count = sum(p.numel() for p in edge.parameters())
    expected = json.loads(CONTRACT.read_text(encoding="utf-8"))["parameter_counts"]["total"]
    if dense_count != edge_count or dense_count != expected:
        raise RuntimeError(f"parameter contract mismatch: {dense_count}, {edge_count}, expected {expected}")
    return cfg, dense, edge


def data_arrays(fold: str = "eth"):
    raw = np.load(DATA / f"{fold}_data_train.npy", mmap_mode="r")
    counts = np.asarray(np.load(DATA / f"{fold}_num_train.npy", mmap_mode="r"), dtype=np.int64)
    return raw, counts


def make_batch(cfg, indices: np.ndarray, fold: str = "eth"):
    raw, counts = data_arrays(fold)
    obs_len = int(cfg.DATA.FREQUENCY * cfg.PREDICTION.OBS_TIME)
    scale = float(cfg.PREDICTION.TRAJ_SCALE)
    absolute = np.asarray(raw[indices, :, :obs_len, -2:], dtype=np.float32) / scale
    return torch.from_numpy(absolute.copy()), torch.from_numpy(counts[indices].copy()).int()


def call(model, absolute, counts):
    center = absolute.mean(dim=-2, keepdim=True)
    prediction = model(
        absolute - center, None, None, mode="testing", num_valid=counts, graph_loc=absolute
    )[2]
    return prediction, model.last_mode_logits


def equivalence(cfg, dense, edge, device, batches: int, batch_size: int):
    raw, _ = data_arrays("eth")
    rng = np.random.default_rng(197)
    rows = []
    for batch_id in range(batches):
        idx = rng.choice(len(raw), size=min(batch_size, len(raw)), replace=False)
        past, counts = make_batch(cfg, idx)
        past, counts = past.to(device), counts.to(device)
        with torch.inference_mode():
            dense_prediction, dense_logits = call(dense, past, counts)
            edge_prediction, edge_logits = call(edge, past, counts)
            base = dense.agent_aware_trans_encoder
            sparse = edge.agent_aware_trans_encoder
            dense_features, _, _ = base.relation_features(base._past)
            dense_encoded = base.relation_encoder(dense_features)
            selected_features = dense_features[
                sparse.last_edge_index[:, 0], sparse.last_edge_index[:, 1], sparse.last_edge_index[:, 2]
            ]
            selected_encoded = dense_encoded[
                sparse.last_edge_index[:, 0], sparse.last_edge_index[:, 1], sparse.last_edge_index[:, 2]
            ]
            rows.append({
                "batch": batch_id,
                "graph_mismatch": int((base.last_allowed != sparse.last_allowed).sum().item()),
                "relation_feature_max_error": float((selected_features - sparse.last_edge_relation_features).abs().max().item()),
                "relation_encoded_max_error": float((selected_encoded - sparse.last_edge_relation_encoded).abs().max().item()),
                "encoder_max_error": float((base.last_agent_features - sparse.last_agent_features).abs().max().item()),
                "logit_max_error": float((dense_logits - edge_logits).abs().max().item()),
                "trajectory_max_error": float((dense_prediction - edge_prediction).abs().max().item()),
                "edges": int(sparse.last_edge_index.shape[0]),
            })
    gate = {
        "graph_exact": all(r["graph_mismatch"] == 0 for r in rows),
        "relation_pass": max(r["relation_encoded_max_error"] for r in rows) < 1e-5,
        "encoder_pass": max(r["encoder_max_error"] for r in rows) < 1e-5,
        "logit_pass": max(r["logit_max_error"] for r in rows) < 1e-5,
        "trajectory_pass": max(r["trajectory_max_error"] for r in rows) < 1e-5,
    }
    gate["passed"] = all(gate.values())
    payload = {"gate": gate, "rows": rows}
    save_json("equivalence.json", payload)
    return payload


def density(cfg, dense, device, limit: int, batch_size: int):
    rows = []
    for fold in ("eth", "hotel", "univ", "zara01", "zara02"):
        raw, counts_all = data_arrays(fold)
        take = min(limit, len(raw)) if limit else len(raw)
        degrees, ns, es, ratios = [], [], [], []
        for start in range(0, take, batch_size):
            idx = np.arange(start, min(start + batch_size, take))
            past, counts = make_batch(cfg, idx, fold)
            with torch.inference_mode():
                call(dense, past.to(device), counts.to(device))
            allowed = dense.agent_aware_trans_encoder.last_allowed.cpu()
            for a, n in zip(allowed, counts):
                n = int(n)
                sub = a[:n, :n]
                nonself = int(sub.sum().item()) - n
                denom = n * max(n - 1, 1)
                degree = nonself / max(n, 1)
                ns.append(n); es.append(nonself); degrees.append(degree); ratios.append(nonself / denom if n > 1 else 0.0)
        rows.append({
            "dataset": fold, "samples": len(ns), "mean_n": float(np.mean(ns)),
            "mean_nonself_edges": float(np.mean(es)), "mean_degree": float(np.mean(degrees)),
            "density": float(np.mean(ratios)), "p90_degree": float(np.quantile(degrees, .9)),
        })
    save_csv("graph_density_summary.csv", rows)
    save_json("graph_density_summary.json", rows)
    return rows


def latency_samples(model, past, counts, iterations: int):
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call(model, past, counts)
    end.record(); torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iterations


def memory(model, past, counts):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    before_a = torch.cuda.memory_allocated(); before_r = torch.cuda.memory_reserved()
    with torch.inference_mode():
        call(model, past, counts); torch.cuda.synchronize()
    return {
        "peak_allocated_bytes": int(max(0, torch.cuda.max_memory_allocated() - before_a)),
        "peak_reserved_bytes": int(max(0, torch.cuda.max_memory_reserved() - before_r)),
    }


def benchmark(cfg, models, device, warmup: int, iterations: int, cycles: int):
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for latency and memory benchmark")
    rows = []
    for batch_size in (1, 16, 64):
        raw, counts_all = data_arrays("eth")
        order = np.argsort(-counts_all)[:batch_size]
        past, counts = make_batch(cfg, order)
        past, counts = past.to(device), counts.to(device)
        with torch.inference_mode():
            for model in models.values():
                for _ in range(warmup):
                    call(model, past, counts)
            torch.cuda.synchronize()
            samples = {name: [] for name in models}
            names = list(models)
            for cycle in range(cycles):
                order_names = names if cycle % 2 == 0 else list(reversed(names))
                for name in order_names:
                    samples[name].append(latency_samples(models[name], past, counts, iterations))
        for name, model in models.items():
            values = samples[name]
            row = {
                "model": name, "batch_size": batch_size,
                "mean_ms": statistics.fmean(values), "median_ms": statistics.median(values),
                "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
                "p90_ms": float(np.quantile(values, .90)), "p95_ms": float(np.quantile(values, .95)),
                "throughput_samples_s": 1000.0 * batch_size / statistics.fmean(values),
                **memory(model, past, counts),
            }
            rows.append(row)
    save_csv("real_efficiency.csv", rows); save_json("real_efficiency.json", rows)
    return rows


def scaling(device, warmup: int, iterations: int, cycles: int):
    from front_encoder_models import FactorizedStateRelationEncoder
    from sparse_exec_adapter import enable_edge_first
    rows = []
    for agents in (4, 8, 16, 32, 64, 96, 128):
        kwargs = dict(obs_len=8, agents=agents, dim=96, heads=4, ffn_dim=192, dropout=0.0,
                      use_relation_tokens=True, use_dual_graph=False, top_k=4, frequency=2.5,
                      use_multiscale_relation=True, use_temporal_social_cross=True)
        dense_full = FactorizedStateRelationEncoder(**kwargs).to(device).eval()
        masked = FactorizedStateRelationEncoder(**kwargs).to(device).eval()
        edge = FactorizedStateRelationEncoder(**kwargs).to(device).eval()
        masked.load_state_dict(dense_full.state_dict(), strict=True)
        edge.load_state_dict(dense_full.state_dict(), strict=True)
        holder = nn.Module(); holder.front_variant = "f16_temporal_social_cross"; holder.agent_aware_trans_encoder = edge
        enable_edge_first(holder)
        src = torch.randn(1, 8 * agents, 96, device=device)
        past = torch.randn(1, agents, 8, 2, device=device)
        counts = torch.tensor([agents], device=device)
        allowed = torch.eye(agents, dtype=torch.bool, device=device)
        for i in range(agents):
            for offset in range(1, min(4, agents - 1) + 1):
                allowed[i, (i + offset) % agents] = True
        token_mask = torch.full((8 * agents, 8 * agents), -torch.inf, device=device)
        last = 7 * agents + torch.arange(agents, device=device)
        token_mask[last[:, None], last[None, :]] = torch.where(allowed, 0.0, -torch.inf)
        samples = {"dense_full_graph": [], "masked_dense_exec": [], "edge_exec": []}
        for model in (dense_full, masked, edge): model.set_context(past, counts)
        with torch.inference_mode():
            for _ in range(warmup):
                dense_full(src, None); masked(src, token_mask); edge(src, token_mask)
            if device.type == "cuda": torch.cuda.synchronize()
            for cycle in range(cycles):
                ordered = (("dense_full_graph", dense_full, None), ("masked_dense_exec", masked, token_mask), ("edge_exec", edge, token_mask))
                if cycle % 2: ordered = tuple(reversed(ordered))
                for name, model, active_mask in ordered:
                    t0 = time.perf_counter()
                    for _ in range(iterations): model(src, active_mask)
                    if device.type == "cuda": torch.cuda.synchronize()
                    samples[name].append(1000.0 * (time.perf_counter() - t0) / iterations)
        edges = int(allowed.sum().item())
        for name, model, active_mask in (("dense_full_graph", dense_full, None), ("masked_dense_exec", masked, token_mask), ("edge_exec", edge, token_mask)):
            values = samples[name]
            mem = {}
            if device.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                before_a = torch.cuda.memory_allocated(); before_r = torch.cuda.memory_reserved()
                with torch.inference_mode(): model(src, active_mask); torch.cuda.synchronize()
                mem = {"peak_allocated_bytes": int(max(0, torch.cuda.max_memory_allocated() - before_a)),
                       "peak_reserved_bytes": int(max(0, torch.cuda.max_memory_reserved() - before_r))}
            rows.append({"n": agents, "model": name, "edges": edges, "dense_pairs": agents * agents,
                         "pair_saving": 1.0 - edges / (agents * agents),
                         "mean_ms": statistics.fmean(values), "p95_ms": float(np.quantile(values, .95)), **mem})
    save_csv("scaling.csv", rows); save_json("scaling.json", rows)
    return rows


def save_json(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "equivalence", "density", "benchmark", "scaling", "all"), default="smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--density-limit", type=int, default=2000, help="0 means all samples")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cycles", type=int, default=10)
    args = parser.parse_args()
    device = torch.device(args.device)
    cfg, dense, edge = setup(device)
    with torch.inference_mode():
        past, counts = make_batch(cfg, np.arange(2))
        d, dl = call(dense, past.to(device), counts.to(device))
        e, el = call(edge, past.to(device), counts.to(device))
    smoke = {"trajectory_max_error": float((d-e).abs().max()), "logit_max_error": float((dl-el).abs().max()),
             "parameters": sum(p.numel() for p in dense.parameters())}
    save_json("smoke.json", smoke); print(json.dumps(smoke, indent=2), flush=True)
    if args.stage == "smoke": return
    if args.stage in ("equivalence", "all"):
        result = equivalence(cfg, dense, edge, device, args.batches, args.batch_size)
        print(f"[equivalence] passed={result['gate']['passed']}", flush=True)
        if not result["gate"]["passed"]: raise RuntimeError("equivalence gate failed")
    if args.stage in ("density", "all"): density(cfg, dense, device, args.density_limit, args.batch_size)
    if args.stage in ("benchmark", "all"): benchmark(cfg, {"F16-Dense": dense, "F16-EdgeExec": edge}, device, args.warmup, args.iterations, args.cycles)
    if args.stage in ("scaling", "all"): scaling(device, args.warmup, args.iterations, args.cycles)


if __name__ == "__main__":
    main()
