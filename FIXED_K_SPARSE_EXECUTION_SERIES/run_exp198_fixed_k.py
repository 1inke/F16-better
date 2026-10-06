"""EXP-198: compare masked-dense, dynamic EdgeExec and regular Fixed-K F16."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
EXP197 = WORKSPACE / "SPARSE_EXECUTION_SERIES"
OUT = ROOT / "EXP_198_RESULTS"
sys.path[:0] = [str(EXP197), str(ROOT)]

import run_exp197_sparse_execution as previous
from fixed_k_adapter import enable_fixed_k, slots_to_allowed


def setup(device: torch.device):
    cfg, dense, edge = previous.setup(device)
    fixed = copy.deepcopy(dense)
    enable_fixed_k(fixed)
    counts = {name: sum(p.numel() for p in model.parameters()) for name, model in
              (("dense", dense), ("edge", edge), ("fixed", fixed))}
    if len(set(counts.values())) != 1 or counts["dense"] != 355048:
        raise RuntimeError(f"parameter mismatch: {counts}")
    return cfg, {"F16-MaskedDenseExec": dense, "F16-EdgeExec": edge, "F16-FixedK": fixed}


def save_json(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader(); writer.writerows(rows)


def call(model, past, counts):
    return previous.call(model, past, counts)


def equivalence(cfg, models, device, batches: int, batch_size: int):
    dense, edge, fixed = models.values()
    raw, _ = previous.data_arrays("eth")
    rng = np.random.default_rng(198)
    rows, slots = [], []
    for batch_id in range(batches):
        indices = rng.choice(len(raw), size=min(batch_size, len(raw)), replace=False)
        past, counts = previous.make_batch(cfg, indices)
        past, counts = past.to(device), counts.to(device)
        with torch.inference_mode():
            dense_prediction, dense_logits = call(dense, past, counts)
            edge_prediction, edge_logits = call(edge, past, counts)
            fixed_prediction, fixed_logits = call(fixed, past, counts)
            db = dense.agent_aware_trans_encoder
            fb = fixed.agent_aware_trans_encoder
            reconstructed = slots_to_allowed(
                fb.last_neighbor_index, fb.last_neighbor_valid, fb.agents
            )
            dense_features, _, _ = db.relation_features(db._past)
            b = torch.arange(past.shape[0], device=device).view(-1, 1, 1)
            target = torch.arange(fb.agents, device=device).view(1, -1, 1)
            selected = dense_features[b, target, fb.last_neighbor_index]
            valid = fb.last_neighbor_valid[..., None, None].expand_as(selected)
            encoded_dense = db.relation_encoder(dense_features)
            selected_encoded = encoded_dense[b, target, fb.last_neighbor_index]
            encoded_valid = fb.last_neighbor_valid[..., None, None].expand_as(selected_encoded)
            rows.append({
                "batch": batch_id,
                "graph_mismatch": int((db.last_allowed != reconstructed).sum().item()),
                "raw_relation_max_error": float((selected[valid] - fb.last_fixed_relation_features[valid]).abs().max().item()),
                "encoded_relation_max_error": float((selected_encoded[encoded_valid] - fb.last_fixed_relation_encoded[encoded_valid]).abs().max().item()),
                "dense_fixed_encoder_max_error": float((db.last_agent_features - fb.last_agent_features).abs().max().item()),
                "dense_fixed_logit_max_error": float((dense_logits - fixed_logits).abs().max().item()),
                "dense_fixed_trajectory_max_error": float((dense_prediction - fixed_prediction).abs().max().item()),
                "edge_fixed_trajectory_max_error": float((edge_prediction - fixed_prediction).abs().max().item()),
            })
            used = int(fb.last_neighbor_valid.sum().item())
            capacity = int(fb.last_neighbor_valid.numel())
            slots.append({"batch": batch_id, "used_slots": used, "capacity": capacity,
                          "slot_utilization": used / capacity, "padding_waste": 1.0 - used / capacity,
                          "fallback_pairs": int(fb.last_fixedk_fallback_pairs)})
    keys = ("raw_relation_max_error", "encoded_relation_max_error", "dense_fixed_encoder_max_error",
            "dense_fixed_logit_max_error", "dense_fixed_trajectory_max_error")
    gate = {"graph_exact": all(r["graph_mismatch"] == 0 for r in rows),
            **{key + "_pass": max(r[key] for r in rows) < 1e-5 for key in keys},
            "parameters_equal": True}
    gate["passed"] = all(gate.values())
    payload = {"gate": gate, "rows": rows, "slot_summary": {
        "utilization": sum(x["used_slots"] for x in slots) / sum(x["capacity"] for x in slots),
        "padding_waste": 1.0 - sum(x["used_slots"] for x in slots) / sum(x["capacity"] for x in slots),
        "configured_top_k": 4, "adaptive_k_max": 6,
        "social_slot_bound": int(fixed.agent_aware_trans_encoder.fixed_k_social_slots),
        "total_width_including_self": int(fixed.agent_aware_trans_encoder.fixed_k_total_slots)}}
    save_json("equivalence.json", payload); save_csv("fixed_k_slot_utilization.csv", slots)
    return payload


def latency(model, past, counts, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations): call(model, past, counts)
    end.record(); torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iterations


def memory(model, past, counts):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    before_a, before_r = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    with torch.inference_mode(): call(model, past, counts); torch.cuda.synchronize()
    return {"peak_allocated_bytes": int(max(0, torch.cuda.max_memory_allocated() - before_a)),
            "peak_reserved_bytes": int(max(0, torch.cuda.max_memory_reserved() - before_r))}


def accounted_flops(model, past, counts):
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
    ) as profile:
        call(model, past, counts); torch.cuda.synchronize()
    flops = int(sum(event.flops for event in profile.key_averages() if event.flops))
    return flops


def benchmark(cfg, models, device, warmup, iterations, cycles):
    if device.type != "cuda": raise RuntimeError("CUDA benchmark required")
    _, counts_all = previous.data_arrays("eth")
    rows = []
    names = list(models)
    for batch_size in (1, 16, 64):
        indices = np.argsort(-counts_all)[:batch_size]
        past, counts = previous.make_batch(cfg, indices)
        past, counts = past.to(device), counts.to(device)
        with torch.inference_mode():
            for model in models.values():
                for _ in range(warmup): call(model, past, counts)
            torch.cuda.synchronize()
            samples = {name: [] for name in names}
            for cycle in range(cycles):
                shift = cycle % len(names); order = names[shift:] + names[:shift]
                for name in order: samples[name].append(latency(models[name], past, counts, iterations))
        for name in names:
            values = samples[name]
            row = {"model": name, "batch_size": batch_size,
                   "mean_ms": statistics.fmean(values), "median_ms": statistics.median(values),
                   "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
                   "min_ms": min(values), "max_ms": max(values),
                   "p90_ms": float(np.quantile(values, .90)), "p95_ms": float(np.quantile(values, .95)),
                   "throughput_samples_s": 1000.0 * batch_size / statistics.fmean(values),
                   **memory(models[name], past, counts)}
            if batch_size in (1, 64):
                row["profiler_accounted_flops"] = accounted_flops(models[name], past, counts)
                row["profiler_accounted_macs"] = row["profiler_accounted_flops"] / 2
            rows.append(row)
    save_csv("real_efficiency.csv", rows); save_json("real_efficiency.json", rows)
    return rows


def make_mask(agents, social_k, device):
    allowed = torch.eye(agents, dtype=torch.bool, device=device)
    for offset in range(1, min(social_k, agents - 1) + 1):
        allowed |= torch.roll(torch.eye(agents, dtype=torch.bool, device=device), shifts=offset, dims=1)
    token_mask = torch.full((8 * agents, 8 * agents), -torch.inf, device=device)
    last = 7 * agents + torch.arange(agents, device=device)
    token_mask[last[:, None], last[None, :]] = torch.where(allowed, 0.0, -torch.inf)
    return allowed, token_mask


def scaling(device, warmup, iterations, cycles, include_256=False):
    from front_encoder_models import FactorizedStateRelationEncoder
    from sparse_exec_adapter import enable_edge_first
    sizes = [8, 16, 32, 64, 96, 128] + ([256] if include_256 else [])
    rows = []
    for agents in sizes:
        kwargs = dict(obs_len=8, agents=agents, dim=96, heads=4, ffn_dim=192, dropout=0.0,
                      use_relation_tokens=True, use_dual_graph=False, top_k=4, frequency=2.5,
                      use_multiscale_relation=True, use_temporal_social_cross=True)
        dense = FactorizedStateRelationEncoder(**kwargs).to(device).eval()
        edge = copy.deepcopy(dense); fixed = copy.deepcopy(dense)
        edge_holder = nn.Module(); edge_holder.front_variant="f16_temporal_social_cross"; edge_holder.agent_aware_trans_encoder=edge
        fixed_holder = nn.Module(); fixed_holder.front_variant="f16_temporal_social_cross"; fixed_holder.agent_aware_trans_encoder=fixed
        enable_edge_first(edge_holder); enable_fixed_k(fixed_holder, social_slots=4)
        src=torch.randn(1,8*agents,96,device=device); past=torch.randn(1,agents,8,2,device=device)
        counts=torch.tensor([agents],device=device); allowed,mask=make_mask(agents,4,device)
        variants=(("masked_dense_exec",dense),("edge_exec",edge),("fixed_k",fixed))
        for _,model in variants: model.set_context(past,counts)
        samples={name:[] for name,_ in variants}
        with torch.inference_mode():
            for _ in range(warmup):
                for _,model in variants: model(src,mask)
            torch.cuda.synchronize()
            for cycle in range(cycles):
                order=list(variants); shift=cycle%3; order=order[shift:]+order[:shift]
                for name,model in order:
                    start=time.perf_counter()
                    for _ in range(iterations): model(src,mask)
                    torch.cuda.synchronize(); samples[name].append(1000*(time.perf_counter()-start)/iterations)
        for name,model in variants:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
            with torch.inference_mode(): model(src,mask);torch.cuda.synchronize()
            values=samples[name]
            rows.append({"n":agents,"model":name,"edges":int(allowed.sum()),"dense_pairs":agents*agents,
                         "pair_saving":1-int(allowed.sum())/(agents*agents),"mean_ms":statistics.fmean(values),
                         "p95_ms":float(np.quantile(values,.95)),"throughput_agents_s":1000*agents/statistics.fmean(values),
                         "peak_allocated_bytes":int(max(0,torch.cuda.max_memory_allocated()-before))})
    save_csv("scaling.csv",rows);save_json("scaling.json",rows);return rows


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("smoke","equivalence","benchmark","scaling","all"),default="smoke")
    p.add_argument("--device",default="cuda:0");p.add_argument("--batches",type=int,default=20);p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--warmup",type=int,default=100);p.add_argument("--iterations",type=int,default=500);p.add_argument("--cycles",type=int,default=15)
    p.add_argument("--include-256",action="store_true");args=p.parse_args();device=torch.device(args.device)
    cfg,models=setup(device);dense,edge,fixed=models.values();past,counts=previous.make_batch(cfg,np.arange(2));past,counts=past.to(device),counts.to(device)
    with torch.inference_mode(): d,dl=call(dense,past,counts);e,el=call(edge,past,counts);f,fl=call(fixed,past,counts)
    smoke={"parameters":sum(x.numel() for x in dense.parameters()),"dense_edge_traj_error":float((d-e).abs().max()),
           "dense_fixed_traj_error":float((d-f).abs().max()),"dense_fixed_logit_error":float((dl-fl).abs().max())}
    save_json("smoke.json",smoke);print(json.dumps(smoke,indent=2),flush=True)
    if args.stage=="smoke":return
    if args.stage in ("equivalence","all"):
        result=equivalence(cfg,models,device,args.batches,args.batch_size);print(f"[equivalence] {result['gate']['passed']}",flush=True)
        if not result["gate"]["passed"]:raise RuntimeError("Fixed-K equivalence gate failed")
    if args.stage in ("benchmark","all"):benchmark(cfg,models,device,args.warmup,args.iterations,args.cycles)
    if args.stage in ("scaling","all"):scaling(device,args.warmup,args.iterations,args.cycles,args.include_256)


if __name__=="__main__":main()
