"""Train-fold-only robust anchor scale calibration for CDA1."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
W = ROOT.parent
P = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FRONT = W / "FRONT_ENCODER_SERIES"
CSRS = W / "COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"
CSO = W / "CANDIDATE_SET_OPTIMIZATION_SERIES"
SPLITS = ("general", "challenge")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def source_run(split, seed):
    return CSRS / "results" / split / "baseline" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def build_exact_source(split, seed, cfg, cls, device):
    model = cls(cfg, "f16_temporal_social_cross")
    checkpoint = source_run(split, seed) / "best_selected.pth"
    os.environ["CSRS_SOURCE_CHECKPOINT"] = str(checkpoint)
    module = load(f"cda_scale_adapter_{split}", CSRS / "current_baseline_adapter.py")
    module.replace_decoder(model, cfg, "current_mode_anchor_template")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def posterior_stats(anchor, target, score, sigma):
    diff = (anchor - target[:, None, [3, 7, 11]]) / sigma[None, None, :, None]
    log_component = -0.5 * (diff ** 2).sum(axis=(-1, -2))
    shifted = score - score.max(axis=-1, keepdims=True)
    log_prior = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    joint = log_prior + log_component
    joint -= joint.max(axis=-1, keepdims=True)
    responsibility = np.exp(joint); responsibility /= responsibility.sum(axis=-1, keepdims=True)
    entropy = -(responsibility * np.log(np.maximum(responsibility, 1e-12))).sum(axis=-1)
    return float(np.exp(entropy).mean()), float(np.sort(responsibility, axis=-1)[:, -1].mean())


def calibrate(split, seed, batch_size, device, diag, cfg, cls, geom):
    run = source_run(split, seed)
    arrays = np.load(run / "split_indices.npz")
    train = np.asarray(arrays["train"], np.int64)
    info = json.loads((run / "split.json").read_text(encoding="utf-8"))
    dataset = geom.IndexedDataset(Path(info["data_path"]), Path(info["num_path"]), train, cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    print(f"[scale] infer {split} train_samples={len(dataset)}", flush=True)
    model = build_exact_source(split, seed, cfg, cls, device)
    values = diag.infer(model, loader, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    error = np.linalg.norm(values["prediction"] - values["target"][:, None], axis=-1)
    quality = error.mean(-1) + 0.25 * error[..., -1]
    winner = quality.argmin(-1)
    winner_anchor = values["anchor"][np.arange(len(winner)), winner]
    residual = winner_anchor - values["target"][:, [3, 7, 11]]
    raw_sigma = np.median(np.abs(residual), axis=(0, 2)) / 0.67448975
    raw_sigma = np.clip(raw_sigma, 0.05, 0.80)
    candidates = []
    for multiplier in (0.75, 1.0, 1.25, 1.5, 2.0):
        sigma = raw_sigma * multiplier
        neff, top1 = posterior_stats(values["anchor"], values["target"], values["score"], sigma)
        candidates.append({"multiplier": multiplier, "sigma": sigma.tolist(), "mean_effective_modes": neff, "mean_posterior_top1": top1})
    # Deterministic train-only calibration: choose closest to N_eff=2 within the desired 1-3 band.
    selected = min(candidates, key=lambda row: abs(row["mean_effective_modes"] - 2.0))
    result = {
        "split": split, "seed": seed, "train_fold_only": True,
        "source_checkpoint": str(run / "best_selected.pth"),
        "raw_robust_sigma": raw_sigma.tolist(), "selected_sigma": selected["sigma"],
        "selected_multiplier": selected["multiplier"], "selected_train_diagnostics": selected,
        "calibration_candidates": candidates,
    }
    out = ROOT / "scale"; out.mkdir(parents=True, exist_ok=True)
    (out / f"{split}_seed_{seed}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[scale] {split} sigma={result['selected_sigma']} N_eff={selected['mean_effective_modes']:.3f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    device = torch.device(args.device)
    sys.path[:0] = [str(P), str(P / "scripts"), str(FRONT)]
    os.chdir(P)
    import sitecustomize  # noqa
    diag = load("cda_scale_diag", CSO / "run_cso1_diagnostic.py")
    cfg, cls = diag.prepare(device)
    geom = load("cda_scale_geometry", W / "INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES" / "run_geometry_diagnostic.py")
    results = [calibrate(split, args.seed, args.batch_size, device, diag, cfg, cls, geom) for split in SPLITS]
    (ROOT / "scale" / "combined_summary.json").write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
