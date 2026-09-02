from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
ASTRA_ROOT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
EXP173 = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END" / "results"
CFG_PATH = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
TRAINER = ROOT / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
MODEL_FILE = ROOT / "FRONT_ENCODER_SERIES" / "front_encoder_models.py"
ADAPTER = ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
DEFAULT_OUT = Path(__file__).resolve().parent / "EXP_176A_FINAL_SCORE_AUDIT"
DATA = ASTRA_ROOT / "datasets" / "eth_ucy" / "processed_data_diverse"
FOLDS = ["eth", "hotel", "univ", "zara01", "zara02"]
VARIANT = "f16_temporal_social_cross"
LABEL = "e3"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure(device: torch.device):
    os.chdir(ASTRA_ROOT)
    sys.path[:0] = [str(ASTRA_ROOT), str(ASTRA_ROOT / "scripts"), str(ROOT / "FRONT_ENCODER_SERIES")]
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    cfg = load_locked_q20(CFG_PATH)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    return cfg


def ckpt(fold: str, seed: int, stage: str) -> Path:
    return EXP173 / fold / stage / LABEL / VARIANT / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"


def build(fold: str, seed: int, device: torch.device, trainer, models, adapter):
    cfg = configure(device)
    h3, final = ckpt(fold, seed, "h3"), ckpt(fold, seed, "v2")
    if not h3.exists() or not final.exists():
        raise FileNotFoundError(h3 if not h3.exists() else final)
    os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(h3)
    os.environ["EXP173_DECODER_INIT_SEED"] = str(seed)
    model = models.FrontEncoderASTRA(cfg, VARIANT)
    adapter.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")
    model.load_state_dict(torch.load(final, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval(), cfg


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, axis=1, kind="mergesort")
    out = np.empty_like(order, dtype=np.float64)
    out[np.arange(x.shape[0])[:, None], order] = np.arange(x.shape[1], dtype=np.float64)[None]
    return out


def spearman(score: np.ndarray, quality: np.ndarray) -> np.ndarray:
    a, b = ranks(score), ranks(-quality)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return (a * b).sum(axis=1) / np.maximum(den, 1e-12)


def collect(model, loader, trainer, device: torch.device):
    logits_all, ade_all, fde_all = [], [], []
    with torch.inference_mode():
        for past, future, num_valid, *_ in loader:
            past, future, num_valid = past.to(device), future.to(device), num_valid.to(device).int()
            past, future = trainer.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            prediction = output[2] if isinstance(output, (tuple, list)) else output
            if prediction.ndim != 5:
                raise RuntimeError(f"Expected [B,A,K,T,2], got {tuple(prediction.shape)}")
            prediction = (prediction + center.unsqueeze(-3))[:, 0]
            target = future[:, 0, None]
            distance = torch.linalg.vector_norm(prediction - target, dim=-1)
            ade_all.append(distance.mean(dim=-1).cpu().numpy())
            fde_all.append(distance[..., -1].cpu().numpy())
            logits = getattr(model, "last_mode_logits", None)
            if logits is None:
                raise RuntimeError("Final decoder did not expose last_mode_logits")
            logits_all.append(logits[:, 0].detach().cpu().numpy())
    return np.concatenate(logits_all), np.concatenate(ade_all), np.concatenate(fde_all)


def audit(logits: np.ndarray, ade: np.ndarray, fde: np.ndarray):
    quality = ade + 0.25 * fde
    selected = logits.argmax(axis=1)
    rows = np.arange(logits.shape[0])
    best_ade, best_fde, best_q = ade.argmin(1), fde.argmin(1), quality.argmin(1)
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    entropy = -(prob * np.log(np.maximum(prob, 1e-12))).sum(1)
    order = np.argsort(-logits, axis=1)
    result = {
        "samples": int(logits.shape[0]),
        "modes": int(logits.shape[1]),
        "spearman_score_neg_quality": float(spearman(logits, quality).mean()),
        "top1_best_ade_hit": float((selected == best_ade).mean()),
        "top1_best_fde_hit": float((selected == best_fde).mean()),
        "top1_best_combined_hit": float((selected == best_q).mean()),
        "selected_regret_mean": float((quality[rows, selected] - quality[rows, best_q]).mean()),
        "selected_regret_p90": float(np.percentile(quality[rows, selected] - quality[rows, best_q], 90)),
        "selected_regret_p95": float(np.percentile(quality[rows, selected] - quality[rows, best_q], 95)),
        "score_entropy_mean": float(entropy.mean()),
        "score_entropy_normalized": float((entropy / np.log(logits.shape[1])).mean()),
        "selected_ade": float(ade[rows, selected].mean()),
        "selected_fde": float(fde[rows, selected].mean()),
        "oracle_min_ade": float(ade.min(1).mean()),
        "oracle_min_fde": float(fde.min(1).mean()),
        "oracle_min_combined": float(quality.min(1).mean()),
    }
    for k in (1, 2, 3, 5, 10):
        result[f"top{k}_combined_recovery"] = float((order[:, :k] == best_q[:, None]).any(1).mean())
    usage = np.bincount(selected, minlength=logits.shape[1])
    result["mode_usage_counts"] = usage.tolist()
    result["active_modes"] = int((usage > 0).sum())
    return result


def main():
    parser = argparse.ArgumentParser(description="EXP-176A zero-training final-score audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", default=FOLDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    folds = FOLDS if args.folds == ["all"] else args.folds
    unknown = sorted(set(folds) - set(FOLDS))
    if unknown:
        raise ValueError(f"Unknown folds: {unknown}")
    required = [(fold, seed, ckpt(fold, seed, "v2")) for fold in folds for seed in args.seeds]
    missing = [str(path) for _, _, path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    if args.dry_run:
        print(json.dumps({"status": "preflight_ok", "runs": len(required), "checkpoints": [str(x[2]) for x in required]}, ensure_ascii=False, indent=2))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sys.path.insert(0, str(ASTRA_ROOT))
    sys.path.insert(0, str(ROOT / "FRONT_ENCODER_SERIES"))
    trainer = load_module("exp176_trainer", TRAINER)
    models = load_module("exp176_models", MODEL_FILE)
    adapter = load_module("exp176_adapter", ADAPTER)
    results = []
    for fold in folds:
        for seed in args.seeds:
            print(f"[audit] fold={fold} seed={seed}", flush=True)
            model, cfg = build(fold, seed, device, trainer, models, adapter)
            dataset = trainer.TrajectoryDataset(DATA / f"{fold}_data_test.npy", DATA / f"{fold}_num_test.npy", cfg)
            if args.max_samples > 0:
                dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            logits, ade, fde = collect(model, loader, trainer, device)
            row = {"fold": fold, "seed": seed, **audit(logits, ade, fde)}
            results.append(row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    scalar_keys = [k for k, v in results[0].items() if isinstance(v, (int, float)) and k not in {"seed", "samples", "modes", "active_modes"}]
    macro = {f"{key}_mean": float(np.mean([row[key] for row in results])) for key in scalar_keys}
    macro.update({f"{key}_std": float(np.std([row[key] for row in results], ddof=1)) if len(results) > 1 else 0.0 for key in scalar_keys})
    macro["top5_gate_pass"] = bool(macro["top5_combined_recovery_mean"] >= 0.60)
    macro["top1_hard_case_pass"] = bool(macro["top1_best_combined_hit_mean"] <= 0.35)
    macro["recommendation"] = "EXP-176B learned reranker is justified" if macro["top5_gate_pass"] and macro["top1_hard_case_pass"] else "Do not train EXP-176B under the guide's gate"
    payload = {
        "experiment": "EXP-176A",
        "quality_definition": "ADE + 0.25 * FDE",
        "checkpoint": "EXP173 F16 Joint-V2 final candidate set and native score",
        "runs": results,
        "macro_equal_fold_seed": macro,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "score_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_rows = []
    for row in results:
        flat = {k: v for k, v in row.items() if k != "mode_usage_counts"}
        flat["mode_usage_counts"] = json.dumps(row["mode_usage_counts"])
        csv_rows.append(flat)
    with (args.output_dir / "per_fold_seed_score_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    print(json.dumps(macro, ensure_ascii=False, indent=2))
    print(f"[complete] {args.output_dir / 'score_audit.json'}")


if __name__ == "__main__":
    main()
