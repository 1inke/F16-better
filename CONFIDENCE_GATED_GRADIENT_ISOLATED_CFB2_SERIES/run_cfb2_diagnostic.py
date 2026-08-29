"""Paired B0/CFB1/CFB2 diagnostics, including tail and isolation audits."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
W = ROOT.parent
P = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
CSRS = W / "COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"
IR1 = W / "INTERACTION_RESIDUAL_SUPERVISION_SERIES"
CSO = W / "CANDIDATE_SET_OPTIMIZATION_SERIES"
CFB1 = W / "CONDITIONAL_FUTURE_BANK_SERIES"
OUT = ROOT / "diagnostic"
KS = (1, 2, 5, 10, 20)
SPLITS = ("general", "challenge")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def source(split, seed):
    return IR1 / "results" / split / "baseline" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"


def run(split, name, seed):
    if name == "reported_baseline":
        return CSRS / "results" / split / "baseline" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"
    if name == "source_baseline":
        return IR1 / "results" / split / "baseline" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"
    if name == "cfb1":
        return CFB1 / "results" / split / "cfb1_cross_scene_three_future" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"
    return ROOT / "results" / split / "cfb2_confidence_gradient_isolated" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def build(diag, split, name, seed, cfg, cls, device):
    if name == "reported_baseline":
        return diag.build(split, "baseline", seed, cfg, cls, device)
    if name == "source_baseline":
        # Build the exact IR1 source with a zero-initialized adapter. This is bitwise
        # equivalent to the source checkpoint at initialization.
        model = cls(cfg, "f16_temporal_social_cross")
        os.environ["CFB2_SOURCE_CHECKPOINT"] = str(source(split, seed))
        os.environ["CFB2_CACHE"] = str(ROOT / "cache" / f"{split}_seed_{seed}.npz")
        module = load(f"cfb2_source_model_{split}", ROOT / "cfb2_models.py")
        module.replace_decoder(model, cfg, "cfb2_confidence_gradient_isolated")
        return model.to(device).eval()
    if name == "cfb1":
        old = load(f"cfb2_old_diag_{split}", CFB1 / "run_cfb1_diagnostic.py")
        return old.build(diag, split, "cfb1", seed, cfg, cls, device)
    model = cls(cfg, "f16_temporal_social_cross")
    os.environ["CFB2_SOURCE_CHECKPOINT"] = str(source(split, seed))
    os.environ["CFB2_CACHE"] = str(ROOT / "cache" / f"{split}_seed_{seed}.npz")
    module = load(f"cfb2_diag_model_{split}", ROOT / "cfb2_models.py")
    module.replace_decoder(model, cfg, "cfb2_confidence_gradient_isolated")
    model.load_state_dict(torch.load(run(split, name, seed) / "best_selected.pth", map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def subset(values, mask):
    return {k: float(values[k][mask].mean()) for k in ("min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde")}


def curve(values, mask):
    error = np.linalg.norm(values["prediction"] - values["target"][:, None], axis=-1)
    ade, fde = error.mean(-1), error[..., -1]
    order = np.argsort(-values["score"], axis=1)
    rows = np.arange(len(order))[:, None]
    result = {}
    for k in KS:
        chosen = order[:, :k]
        result[str(k)] = {"ade": float(ade[rows, chosen][mask].min(1).mean()), "fde": float(fde[rows, chosen][mask].min(1).mean())}
    return result


def diversity(values):
    i, j = np.triu_indices(20, 1)
    prediction, anchor = values["prediction"], values["anchor"]
    return {
        "endpoint": float(np.linalg.norm(prediction[:, i, -1] - prediction[:, j, -1], axis=-1).mean()),
        "trajectory": float(np.linalg.norm(prediction[:, i] - prediction[:, j], axis=-1).mean()),
        "anchor": float(np.linalg.norm(anchor[:, i] - anchor[:, j], axis=-1).mean()),
    }


def tail_diagnostic(values):
    error = np.linalg.norm(values["prediction"] - values["target"][:, None], axis=-1).mean(-1)
    order = np.argsort(-values["score"], axis=1)
    rows = np.arange(len(order))[:, None]
    top = error[rows, order[:, :10]].min(1)
    tail_values = error[rows, order[:, 10:20]]
    tail_local = tail_values.argmin(1)
    tail = tail_values[np.arange(len(tail_values)), tail_local]
    tail_mode = order[np.arange(len(order)), 10 + tail_local]
    wins = tail < top
    counts = np.bincount(tail_mode[wins], minlength=20)
    return {
        "tail_11_20_becomes_oracle_winner_fraction": float(wins.mean()),
        "mean_ade_gain_when_tail_wins": float((top[wins] - tail[wins]).mean()) if wins.any() else 0.0,
        "tail_winner_mode_counts": counts.tolist(),
    }


def protection(base, model):
    prediction_shift = np.linalg.norm(model["prediction"] - base["prediction"], axis=-1)
    top = base["score"].argmax(1)
    return {
        "base_modes_0_17_max_abs_shift": float(np.abs(model["prediction"][:, :18] - base["prediction"][:, :18]).max()),
        "adapted_modes_18_19_mean_trajectory_shift": float(prediction_shift[:, 18:20].mean()),
        "baseline_top1_mean_trajectory_shift": float(prediction_shift[np.arange(len(top)), top].mean()),
        "logits_max_abs_shift": float(np.abs(model["score"] - base["score"]).max()),
    }


def usage(state):
    counts = state["decoder.pseudo_usage_counts"].cpu().numpy()
    total = max(int(counts.sum()), 1)
    confidence_count = int(state["decoder.pseudo_confidence_count"])
    delta_count = int(state["decoder.adapter_delta_norm_count"])
    return {
        "query_usage_fraction_0_1_2": (counts / total).tolist(),
        "pseudo_mode_counts": state["decoder.pseudo_mode_counts"].cpu().numpy().tolist(),
        "mean_used_confidence": float(state["decoder.pseudo_confidence_sum"] / max(confidence_count, 1)),
        "mean_adapter_delta_norm_during_training": float(state["decoder.adapter_delta_norm_sum"] / max(delta_count, 1)),
    }


def confidence_gate_subgroups(base, model, pseudo_count):
    result = {}
    for count in range(3):
        mask = pseudo_count == count
        row = {"samples": int(mask.sum())}
        for metric in ("min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde"):
            before = float(base[metric][mask].mean())
            after = float(model[metric][mask].mean())
            row[metric] = {"source_baseline": before, "cfb2": after, "percent_change": 100 * (after / before - 1)}
        result[str(count)] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    device = torch.device(args.device)
    diag = load("cfb2_cso_diag", CSO / "run_cso1_diagnostic.py")
    cfg, cls = diag.prepare(device)
    geom = load("cfb2_geometry", W / "INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES" / "run_geometry_diagnostic.py")
    results = []
    OUT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        base_run = run(split, "source_baseline", args.seed)
        indices = np.asarray(np.load(base_run / "split_indices.npz")["validation"], np.int64)
        info = json.loads((base_run / "split.json").read_text(encoding="utf-8"))
        dataset = geom.IndexedDataset(Path(info["data_path"]), Path(info["num_path"]), indices, cfg)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        names = ["reported_baseline", "source_baseline"]
        if (run(split, "cfb1", args.seed) / "metrics.json").is_file():
            names.append("cfb1")
        names.append("cfb2")
        values, cfb2_state = {}, None
        for name in names:
            print(f"[infer] {split}/{name} samples={len(dataset)}", flush=True)
            model = build(diag, split, name, args.seed, cfg, cls, device)
            values[name] = diag.infer(model, loader, device)
            if name == "cfb2":
                cfb2_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        labels = np.asarray(np.load(CSRS / "teacher_cache" / f"{split}_seed_{args.seed}.npz", allow_pickle=False)["need_label"])[indices].astype(bool)
        cache = np.load(ROOT / "cache" / f"{split}_seed_{args.seed}.npz", allow_pickle=False)
        validation_pseudo_count = np.asarray(cache["valid_mask"])[indices].sum(axis=-1)
        all_mask = np.ones(len(indices), bool)
        models = {}
        for name, value in values.items():
            models[name] = {
                "overall": subset(value, all_mask), "social_needed": subset(value, labels),
                "coverage_at_k": curve(value, all_mask), "social_needed_at_k": curve(value, labels),
                "diversity": diversity(value), "tail": tail_diagnostic(value),
            }
        result = {
            "split": split, "samples": len(indices), "models": models,
            "gt_main_branch_protection": protection(values["source_baseline"], values["cfb2"]),
            "pseudo_adapter_usage": usage(cfb2_state),
            "validation_by_confidence_gate": confidence_gate_subgroups(values["source_baseline"], values["cfb2"], validation_pseudo_count),
        }
        out = OUT / split; out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append(result)
    (OUT / "combined_summary.json").write_text(json.dumps({"strict_pairing": True, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
