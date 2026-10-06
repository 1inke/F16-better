"""Validation-only PIF1 versus locked C00 analysis with paired bootstrap."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO), str(REPO / "SELECTOR_BENCHMARK_SERIES"), str(REPO / "FRONT_ENCODER_SERIES")]

if sys.platform == "win32":
    torch_lib = Path(sys.executable).parent / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from build_candidate_cache import configure_astra_env, load_module


ASTRA_DEFAULT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
C00_RUN = (
    REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
    / "results" / "eth" / "v2" / "e3" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1"
)
C00_CACHE = (
    REPO / "DIVERSITY_STAGE_ATTRIBUTION_SERIES" / "results"
    / "stage_attribution_analysis" / "validation_candidate_cache" / "c00.npz"
)
DEFAULT_RUN = HERE / "results" / "pif1_flow" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"
DEFAULT_STATS = HERE / "calibration" / "intent_stats.json"
PAIR_I, PAIR_J = np.triu_indices(20, k=1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_pif_model(cfg, checkpoint: Path, stats: Path, device: torch.device):
    os.environ["PIF_SOURCE_CHECKPOINT"] = str(C00_RUN / "best_selected.pth")
    os.environ["PIF_STATS_JSON"] = str(stats)
    models = load_module("pif_front_models", REPO / "FRONT_ENCODER_SERIES" / "front_encoder_models.py")
    pif = load_module("pif_decoder_analysis", HERE / "physical_intent_flow_models.py")
    model = models.FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    pif.replace_decoder(model, cfg, pif.EXPERIMENT)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def evaluate(model, loader: DataLoader, sample_ids: np.ndarray, trainer, device: torch.device) -> dict[str, np.ndarray]:
    candidate, ade, fde, logits = [], [], [], []
    with torch.inference_mode():
        for past, future, num_valid, *_ in loader:
            past, future = past.to(device), future.to(device)
            num_valid = num_valid.to(device).int()
            past, future = trainer.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)
            output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
            prediction = output[2] + center.unsqueeze(-3)
            focal, target = prediction[:, 0], future[:, 0]
            error = torch.linalg.vector_norm(focal - target[:, None], dim=-1)
            candidate.append(focal.cpu().numpy())
            ade.append(error.mean(dim=-1).cpu().numpy())
            fde.append(error[..., -1].cpu().numpy())
            logits.append(model.last_mode_logits[:, 0].cpu().numpy())
    result = {
        "sample_id": sample_ids.copy(),
        "candidate": np.concatenate(candidate).astype(np.float32),
        "ade": np.concatenate(ade).astype(np.float32),
        "fde": np.concatenate(fde).astype(np.float32),
        "logits": np.concatenate(logits).astype(np.float32),
    }
    if result["ade"].shape != (len(sample_ids), 20):
        raise RuntimeError(result["ade"].shape)
    return result


def derive(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    candidate = raw["candidate"].astype(np.float64)
    ade, fde, logits = (raw[key].astype(np.float64) for key in ("ade", "fde", "logits"))
    n = len(ade); row = np.arange(n)
    selected = logits.argmax(axis=1); oracle = ade.argmin(axis=1); minfde = fde.argmin(axis=1)
    order = np.argsort(-logits, axis=1)
    rank = np.empty_like(order); rank[row[:, None], order] = np.arange(1, 21)[None]
    oracle_rank = rank[row, oracle]
    endpoints = candidate[:, :, -1]
    pair = np.linalg.norm(endpoints[:, PAIR_I] - endpoints[:, PAIR_J], axis=-1)
    selected_ade = ade[row, selected]
    return {
        "selected_ade": selected_ade,
        "selected_fde": fde[row, selected],
        "minade": ade[row, oracle],
        "minfde": fde[row, minfde],
        "samepath_fde": fde[row, oracle],
        "regret": selected_ade - ade[row, oracle],
        "J": ade[row, oracle] + 0.5 * fde[row, minfde],
        "endpoint_spread": pair.mean(axis=1),
        "support_1p0": (fde < 1.0).sum(axis=1).astype(np.float64),
        "oracle_rank": oracle_rank.astype(np.float64),
        "recall_1": (oracle_rank <= 1).astype(np.float64),
        "recall_3": (oracle_rank <= 3).astype(np.float64),
        "duplicate_sample": (pair < 0.1).any(axis=1).astype(np.float64),
    }


def summarize(label: str, values: dict[str, np.ndarray]) -> dict:
    return {"condition": label, "N": len(values["minade"]), **{key: float(value.mean()) for key, value in values.items()}}


def bootstrap(delta: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, repetitions, 500):
        count = min(500, repetitions - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means.append(delta[indices].mean(axis=1))
    all_means = np.concatenate(means)
    low, high = np.quantile(all_means, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--astra-root", type=Path, default=ASTRA_DEFAULT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    args = parser.parse_args()
    checkpoints = {"pif1_best": args.run_dir / "best_selected.pth", "pif1_final": args.run_dir / "final_model.pth"}
    required = [C00_CACHE, C00_RUN / "best_selected.pth", C00_RUN / "split_indices.npz", args.run_dir / "split_indices.npz", args.stats, *checkpoints.values()]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    c00 = {key: np.load(C00_CACHE)[key] for key in np.load(C00_CACHE).files}
    sample_ids = np.load(args.run_dir / "split_indices.npz")["validation"].astype(np.int64)
    c00_ids = np.load(C00_RUN / "split_indices.npz")["validation"].astype(np.int64)
    if not np.array_equal(sample_ids, c00_ids) or not np.array_equal(sample_ids, c00["sample_id"]):
        raise RuntimeError("PIF1 validation IDs do not exactly match locked C00")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = configure_astra_env(REPO, args.astra_root.resolve(), device)
    trainer = load_module("pif_analysis_trainer", REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py")
    data = args.astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    dataset = trainer.TrajectoryDataset(data / "eth_data_train.npy", data / "eth_num_train.npy", cfg)
    loader = DataLoader(Subset(dataset, sample_ids.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    output = HERE / "results" / "analysis"
    cache_dir = output / "validation_candidate_cache"
    output.mkdir(parents=True, exist_ok=True); cache_dir.mkdir(parents=True, exist_ok=True)
    values = {"c00": derive(c00)}
    summaries = [summarize("c00", values["c00"])]
    provenance = {
        "protocol": "pif1_eth_seed1_validation_only_scene_tail_v1",
        "sample_count": len(sample_ids), "test_accessed": False,
        "sample_id_sha256": hashlib.sha256(sample_ids.tobytes()).hexdigest(),
        "c00_checkpoint_sha256": sha256(C00_RUN / "best_selected.pth"),
        "stats_sha256": sha256(args.stats), "conditions": {},
    }
    for label, checkpoint in checkpoints.items():
        print(f"[eval] {label}: {checkpoint}", flush=True)
        model = load_pif_model(cfg, checkpoint, args.stats, device)
        raw = evaluate(model, loader, sample_ids, trainer, device)
        np.savez_compressed(cache_dir / f"{label}.npz", **raw)
        values[label] = derive(raw); summaries.append(summarize(label, values[label]))
        provenance["conditions"][label] = {"checkpoint": str(checkpoint), "sha256": sha256(checkpoint)}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metric_names = list(values["c00"])
    paired_rows = []
    for view_index, label in enumerate(checkpoints):
        for metric_index, metric in enumerate(metric_names):
            delta = values[label][metric] - values["c00"][metric]
            low, high = bootstrap(delta, args.bootstrap_repetitions, 20260923 + 100 * view_index + metric_index)
            paired_rows.append({
                "condition": label, "metric": metric,
                "c00_mean": float(values["c00"][metric].mean()),
                "pif1_mean": float(values[label][metric].mean()),
                "delta_pif1_minus_c00": float(delta.mean()),
                "ci95_low": low, "ci95_high": high,
            })

    base = summaries[0]
    gates = {}
    for label in checkpoints:
        row = next(item for item in summaries if item["condition"] == label)
        generation_gain = 100.0 * (1.0 - row["J"] / base["J"])
        selected_gain = 100.0 * (1.0 - row["selected_ade"] / base["selected_ade"])
        recall_gain = row["recall_1"] - base["recall_1"]
        gates[label] = {
            "generation_J_gain_percent": generation_gain,
            "selected_ADE_gain_percent": selected_gain,
            "recall1_gain_probability": recall_gain,
            "generation_pass": generation_gain >= 2.0,
            "selection_safe": selected_gain >= -0.5,
            "ranking_signal": selected_gain >= 2.0 or recall_gain >= 0.03,
            "duplicate_safe": row["duplicate_sample"] <= 0.10,
        }
    both_generation = all(item["generation_pass"] for item in gates.values())
    both_safe = all(item["selection_safe"] and item["duplicate_safe"] for item in gates.values())
    both_ranking = all(item["ranking_signal"] for item in gates.values())
    if both_generation and both_safe and both_ranking:
        verdict = "ADVANCE_PIF1_TO_GENERATOR_SEEDS"
    elif both_generation and both_safe:
        verdict = "GENERATION_ONLY_REVIEW_SCORE_ALIGNMENT"
    else:
        verdict = "STOP_PIF1_AFTER_SEED1"
    verdict_payload = {"verdict": verdict, "gates": gates, "test_accessed": False}
    write_csv(output / "condition_summary.csv", summaries)
    write_csv(output / "paired_bootstrap.csv", paired_rows)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "verdict.json").write_text(json.dumps(verdict_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# PIF1 validation report", "", f"Decision: **{verdict}**", "", "| Condition | minADE | minFDE | Selected ADE | Recall@1 | Mean oracle rank | Duplicate sample rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['condition']} | {row['minade']:.6f} | {row['minfde']:.6f} | {row['selected_ade']:.6f} | {100*row['recall_1']:.2f}% | {row['oracle_rank']:.3f} | {100*row['duplicate_sample']:.2f}% |")
    lines += ["", "The pilot is validation-only. Best and final checkpoints must agree before any seed expansion; no test path is used."]
    (output / "PIF1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(verdict_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
