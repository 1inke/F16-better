"""Strict CDA1 diagnostic: accuracy, coverage, diversity, responsibility and calibration."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
W = ROOT.parent
P = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
CSRS = W / "COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"
CSO = W / "CANDIDATE_SET_OPTIMIZATION_SERIES"
OUT = ROOT / "diagnostic"
SPLITS = ("general", "challenge")
MODELS = ("source_b0", "cda0_matched_continuation", "cda1_fixed_scale_anchor_mixture")
KS = (1, 2, 5, 10, 20)
METRICS = ("min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def source_run(split, seed):
    return CSRS / "results" / split / "baseline" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def run_dir(split, name, seed):
    if name == "source_b0": return source_run(split, seed)
    return ROOT / "results" / split / name / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def build(split, name, seed, cfg, cls, device):
    model = cls(cfg, "f16_temporal_social_cross")
    os.environ["CDA_SOURCE_CHECKPOINT"] = str(source_run(split, seed) / "best_selected.pth")
    os.environ["CDA_SCALE_JSON"] = str(ROOT / "scale" / f"{split}_seed_{seed}.json")
    module = load(f"cda_diag_model_{split}_{name}", ROOT / "cda_models.py")
    experiment = "cda0_matched_continuation" if name == "source_b0" else name
    module.replace_decoder(model, cfg, experiment)
    if name != "source_b0":
        model.load_state_dict(torch.load(run_dir(split, name, seed) / "best_selected.pth", map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def subset(values, mask): return {metric: float(values[metric][mask].mean()) for metric in METRICS}


def curve(values, mask):
    error = np.linalg.norm(values["prediction"] - values["target"][:, None], axis=-1)
    ade, fde = error.mean(-1), error[..., -1]
    order = np.argsort(-values["score"], axis=1); rows = np.arange(len(order))[:, None]
    result = {}
    for k in KS:
        chosen = order[:, :k]
        result[str(k)] = {"ade": float(ade[rows, chosen][mask].min(1).mean()), "fde": float(fde[rows, chosen][mask].min(1).mean())}
    return result


def diversity(values):
    i, j = np.triu_indices(20, 1); prediction, anchor = values["prediction"], values["anchor"]
    return {"endpoint": float(np.linalg.norm(prediction[:, i, -1] - prediction[:, j, -1], axis=-1).mean()), "trajectory": float(np.linalg.norm(prediction[:, i] - prediction[:, j], axis=-1).mean()), "anchor": float(np.linalg.norm(anchor[:, i] - anchor[:, j], axis=-1).mean())}


def winner_usage(values, mask):
    error = np.linalg.norm(values["prediction"] - values["target"][:, None], axis=-1)
    winner = (error.mean(-1) + 0.25 * error[..., -1]).argmin(-1)[mask]
    counts = np.bincount(winner, minlength=20); probability = counts / max(counts.sum(), 1)
    entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum()
    return {"normalized_entropy": float(entropy / np.log(20)), "never_win_modes": int((counts == 0).sum()), "modes_over_1pct": int((probability > 0.01).sum()), "modes_over_5pct": int((probability > 0.05).sum()), "frequency": probability.tolist()}


def anchor_coverage(values, mask):
    target = values["target"][:, [3, 7, 11]]
    error = np.linalg.norm(values["anchor"] - target[:, None], axis=-1)
    return {f"step_{step}": float(error[mask, :, index].min(1).mean()) for index, step in enumerate((4, 8, 12))}


def distribution(values, sigma, mask):
    anchor, target, score = values["anchor"][mask], values["target"][mask], values["score"][mask]
    diff = (anchor - target[:, None, [3, 7, 11]]) / sigma[None, None, :, None]
    log_component = -0.5 * (diff ** 2 + 2.0 * np.log(sigma)[None, None, :, None] + math.log(2.0 * math.pi)).sum(axis=(-1, -2))
    shifted = score - score.max(axis=-1, keepdims=True); log_prior = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    joint = log_prior + log_component; maximum = joint.max(axis=-1, keepdims=True)
    log_normalizer = maximum[:, 0] + np.log(np.exp(joint - maximum).sum(axis=-1))
    responsibility = np.exp(joint - log_normalizer[:, None])
    ordered = np.sort(responsibility, axis=-1)
    entropy = -(responsibility * np.log(np.maximum(responsibility, 1e-12))).sum(axis=-1)
    prior = np.exp(log_prior); predicted = prior.argmax(-1); posterior = responsibility.argmax(-1)
    anchor_winner = (diff ** 2).sum(axis=(-1, -2)).argmin(-1)
    confidence = prior[np.arange(len(prior)), predicted]; correct = predicted == anchor_winner
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        in_bin = (confidence >= lower) & (confidence < lower + 0.1)
        if in_bin.any(): ece += in_bin.mean() * abs(confidence[in_bin].mean() - correct[in_bin].mean())
    return {
        "nll_per_dimension": float((-log_normalizer / 6.0).mean()),
        "posterior_top1_mass": float(ordered[:, -1].mean()), "posterior_top3_mass": float(ordered[:, -3:].sum(-1).mean()),
        "posterior_entropy": float(entropy.mean()), "effective_modes_mean": float(np.exp(entropy).mean()),
        "effective_modes_median": float(np.median(np.exp(entropy))), "effective_modes_p90": float(np.quantile(np.exp(entropy), 0.90)),
        "score_top1_posterior_winner_agreement": float((predicted == posterior).mean()),
        "score_top1_anchor_winner_agreement": float(correct.mean()), "score_ece_against_anchor_winner": float(ece),
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--seed", type=int, default=1); args = parser.parse_args()
    device = torch.device(args.device); diag = load("cda_cso_diag", CSO / "run_cso1_diagnostic.py"); cfg, cls = diag.prepare(device)
    geom = load("cda_geometry", W / "INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES" / "run_geometry_diagnostic.py")
    OUT.mkdir(parents=True, exist_ok=True); results = []
    for split in SPLITS:
        source = source_run(split, args.seed); indices = np.asarray(np.load(source / "split_indices.npz")["validation"], np.int64)
        info = json.loads((source / "split.json").read_text(encoding="utf-8")); dataset = geom.IndexedDataset(Path(info["data_path"]), Path(info["num_path"]), indices, cfg)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        sigma = np.asarray(json.loads((ROOT / "scale" / f"{split}_seed_{args.seed}.json").read_text(encoding="utf-8"))["selected_sigma"], np.float32)
        values = {}
        for name in MODELS:
            print(f"[infer] {split}/{name} samples={len(dataset)}", flush=True); model = build(split, name, args.seed, cfg, cls, device); values[name] = diag.infer(model, loader, device); del model
            if device.type == "cuda": torch.cuda.empty_cache()
        labels = np.asarray(np.load(CSRS / "teacher_cache" / f"{split}_seed_{args.seed}.npz", allow_pickle=False)["need_label"])[indices].astype(bool)
        all_mask = np.ones(len(indices), bool); models = {}
        for name, value in values.items():
            models[name] = {
                "overall": subset(value, all_mask), "social_needed": subset(value, labels),
                "coverage_at_k": curve(value, all_mask), "social_needed_at_k": curve(value, labels),
                "diversity": diversity(value), "winner_usage": winner_usage(value, all_mask),
                "anchor_coverage": anchor_coverage(value, all_mask),
                "distribution_overall": distribution(value, sigma, all_mask),
                "distribution_social_needed": distribution(value, sigma, labels),
            }
        control, cda = models["cda0_matched_continuation"], models["cda1_fixed_scale_anchor_mixture"]
        result = {"split": split, "samples": len(indices), "fixed_sigma": sigma.tolist(), "models": models, "cda1_vs_matched_control_percent_change": {m: 100 * (cda["overall"][m] / control["overall"][m] - 1) for m in METRICS}}
        out = OUT / split; out.mkdir(parents=True, exist_ok=True); (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"); results.append(result)
    (OUT / "combined_summary.json").write_text(json.dumps({"strict_pairing": True, "primary_control": "cda0_matched_continuation", "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {OUT}")


if __name__ == "__main__": main()
