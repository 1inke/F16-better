"""EXP-161: frozen-candidate listwise score recalibration for V2 and M2.

Only the small scorer is trained. Encoder, decoder and all K=20 candidate
coordinates are frozen. Every fold trains on its own training split and is
evaluated once on the held-out fold.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
BASE_SCRIPT = WORKSPACE / "CANDIDATE_COMPLEMENTARITY_SERIES" / "run_exp158_exp160_joint_audit.py"
EXP159C_CACHE = WORKSPACE / "LEARNABLE_CANDIDATE_SELECTOR_SERIES" / "EXP_159C_LEARNABLE_K40_TO_K20_SELECTOR" / "cache"
DEFAULT_OUT = HERE / "EXP_161_FROZEN_CANDIDATE_LISTWISE_SCORER"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
FAMILIES = ("v2", "m2")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def motion_features(past: torch.Tensor, num_valid: torch.Tensor) -> torch.Tensor:
    focal = past[:, 0]
    velocity = focal[:, 1:] - focal[:, :-1]
    recent = velocity[:, -1]
    long = velocity[:, -min(3, velocity.shape[1]):].mean(1)
    acceleration = recent - long
    speed = torch.linalg.vector_norm(recent, dim=-1, keepdim=True)
    history = focal[:, -1] - focal[:, 0]
    distance = torch.linalg.vector_norm(past[:, :, -1] - past[:, :1, -1], dim=-1)
    valid = torch.arange(past.shape[1], device=past.device)[None] < num_valid[:, None]
    density = (((distance <= 2.0) & valid).sum(1).float() - 1.0)[:, None]
    neighbours = (num_valid.float() - 1.0).clamp_min(0.0)[:, None]
    return torch.cat((recent, long, acceleration, speed, history, neighbours, density), dim=-1)


def original_mode_embedding(model) -> np.ndarray:
    embedding = getattr(model.decoder, "mode_embedding", None)
    if embedding is None:
        raise AttributeError("Decoder has no mode_embedding; EXP-161 contract cannot be satisfied")
    value = embedding.detach().float().cpu().numpy()
    if value.shape[0] != 20:
        raise ValueError(f"Expected 20 original mode embeddings, got {value.shape}")
    return value


def collect_family(base, helper, model_class, cfg, fold: str, family: str, split: str,
                   seed: int, device: torch.device, batch_size: int, max_samples: int):
    dataset = helper.TrajectoryDataset(base.DATA / f"{fold}_data_{split}.npy",
                                       base.DATA / f"{fold}_num_{split}.npy", cfg)
    if max_samples:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = base.build_model(fold, family, seed, cfg, model_class, device)
    mode_embedding = original_mode_embedding(model)
    captured = {}

    def hook(_module, args):
        captured["tokens"] = args[0]

    handle = model.decoder.register_forward_pre_hook(hook)
    parts = {k: [] for k in ("context", "motion", "last_position", "future", "candidates", "original_logits")}
    try:
        with torch.inference_mode():
            for past, future, num_valid, _, _, _ in loader:
                past = past.to(device); future = future.to(device); num_valid = num_valid.to(device).int()
                past, future = helper.normalize_to_focal_heading(past, future)
                center = past.mean(dim=-2, keepdim=True)
                prediction = model(past - center, None, None, mode="testing",
                                   num_valid=num_valid, graph_loc=past)[2] + center.unsqueeze(-3)
                tokens = captured.pop("tokens")[:, 0]
                parts["context"].append(torch.cat((tokens[:, -1], tokens.mean(1)), dim=-1).float().cpu().numpy())
                parts["motion"].append(motion_features(past, num_valid).float().cpu().numpy())
                parts["last_position"].append(past[:, 0, -1].float().cpu().numpy())
                parts["future"].append(future[:, 0].float().cpu().numpy())
                parts["candidates"].append(prediction[:, 0].float().cpu().numpy())
                parts["original_logits"].append(model.last_mode_logits[:, 0].float().cpu().numpy())
    finally:
        handle.remove()
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    return arrays, mode_embedding


def existing_v2_cache(fold: str, split: str, seed: int, max_samples: int):
    if max_samples:
        return None
    path = EXP159C_CACHE / f"{fold}_{split}_seed_{seed}.npz"
    if not path.is_file(): return None
    with np.load(path) as raw:
        required = {"context", "motion", "last_position", "future", "candidates", "original_logits"}
        if not required.issubset(raw.files) or raw["candidates"].shape[1] != 40: return None
        return {
            "context": raw["context"], "motion": raw["motion"],
            "last_position": raw["last_position"], "future": raw["future"],
            "candidates": raw["candidates"][:, 20:],
            "original_logits": raw["original_logits"][:, 20:],
        }


def load_cache(path: Path):
    with np.load(path) as raw:
        arrays = {k: raw[k] for k in ("context", "motion", "last_position", "future", "candidates", "original_logits")}
        embedding = raw["mode_embedding"]
    return arrays, embedding


def get_cache(base, helper, model_class, cfg, fold, family, split, seed, device,
              batch_size, max_samples, cache_dir: Path, force: bool):
    path = cache_dir / f"{fold}_{family}_{split}_seed_{seed}.npz"
    expected_total = int(np.load(base.DATA / f"{fold}_data_{split}.npy", mmap_mode="r").shape[0])
    expected = min(max_samples, expected_total) if max_samples else expected_total
    if path.is_file() and not force:
        arrays, embedding = load_cache(path)
        if len(arrays["future"]) == expected:
            print(f"[cache] reuse {path}", flush=True); return arrays, embedding
    reused = existing_v2_cache(fold, split, seed, max_samples) if family == "v2" and not force else None
    if reused is not None:
        print(f"[cache] import existing frozen V2 cache fold={fold} split={split}", flush=True)
        model = base.build_model(fold, family, seed, cfg, model_class, device)
        embedding = original_mode_embedding(model); del model
        if device.type == "cuda": torch.cuda.empty_cache()
        arrays = reused
    else:
        print(f"[cache] collect family={family} fold={fold} split={split}", flush=True)
        arrays, embedding = collect_family(base, helper, model_class, cfg, fold, family, split,
                                           seed, device, batch_size, max_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, mode_embedding=embedding)
    return arrays, embedding


class FrozenCandidateScorer(nn.Module):
    def __init__(self, context_dim: int, motion_dim: int, mode_embedding: np.ndarray,
                 pred_len: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.register_buffer("original_mode_embedding", torch.from_numpy(mode_embedding).float())
        self.context = nn.Sequential(nn.LayerNorm(context_dim + motion_dim),
                                     nn.Linear(context_dim + motion_dim, hidden_dim), nn.GELU())
        geometry_dim = pred_len * 2 + pred_len + 2 + 2
        self.geometry = nn.Sequential(nn.LayerNorm(geometry_dim), nn.Linear(geometry_dim, hidden_dim), nn.GELU())
        self.mode = nn.Sequential(nn.LayerNorm(mode_embedding.shape[1]), nn.Linear(mode_embedding.shape[1], hidden_dim))
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
                                    nn.GELU(), nn.Linear(hidden_dim, 1))
        # Start as the frozen original scorer, then learn only a bounded residual
        # correction. This avoids destroying a useful ranking at initialization.
        nn.init.zeros_(self.output[-1].weight); nn.init.zeros_(self.output[-1].bias)

    def forward(self, context, motion, last, candidates, original_logits):
        relative = candidates - last[:, None, None]
        first = candidates[:, :, :1] - last[:, None, None]
        displacement = torch.cat((first, candidates[:, :, 1:] - candidates[:, :, :-1]), dim=2)
        speed = torch.linalg.vector_norm(displacement, dim=-1)
        score_z = (original_logits - original_logits.mean(1, keepdim=True)) / original_logits.std(1, keepdim=True).clamp_min(.05)
        order = original_logits.argsort(1, descending=True)
        rank = torch.empty_like(original_logits)
        rank.scatter_(1, order, torch.linspace(1., 0., 20, device=order.device)[None].expand_as(original_logits))
        geometry = torch.cat((relative.flatten(2), speed, relative[:, :, -1], score_z[..., None], rank[..., None]), dim=-1)
        value = self.context(torch.cat((context, motion), dim=-1))[:, None]
        value = value + self.geometry(geometry) + self.mode(self.original_mode_embedding)[None]
        return original_logits + self.output(value).squeeze(-1)


def batch_tensors(arrays, ids, device):
    return tuple(torch.from_numpy(arrays[k][ids]).to(device=device, dtype=torch.float32)
                 for k in ("context", "motion", "last_position", "candidates", "original_logits", "future"))


def cost(candidates, future):
    error = torch.linalg.vector_norm(candidates - future[:, None], dim=-1)
    return error.mean(-1) + .25 * error[..., -1]


def train_scorer(arrays, embedding, device, seed, epochs, batch_size, lr, temperature,
                 val_fraction, patience, hidden_dim):
    rng = np.random.default_rng(seed); order = rng.permutation(len(arrays["future"]))
    val_n = max(1, round(len(order) * val_fraction)); val_ids, train_ids = order[:val_n], order[val_n:]
    model = FrozenCandidateScorer(arrays["context"].shape[1], arrays["motion"].shape[1], embedding,
                                  arrays["future"].shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_loss, stale, history = None, math.inf, 0, []
    for epoch in range(1, epochs + 1):
        model.train(); total = seen = 0; shuffled = rng.permutation(train_ids)
        for start in range(0, len(shuffled), batch_size):
            ids = shuffled[start:start + batch_size]
            context, motion, last, candidates, logits, future = batch_tensors(arrays, ids, device)
            target = torch.softmax(-cost(candidates, future).detach() / temperature, dim=1)
            loss = -(target * torch.log_softmax(model(context, motion, last, candidates, logits), dim=1)).sum(1).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); total += float(loss.detach()) * len(ids); seen += len(ids)
        model.eval(); val_total = 0.
        with torch.inference_mode():
            for start in range(0, len(val_ids), batch_size):
                ids = val_ids[start:start + batch_size]
                context, motion, last, candidates, logits, future = batch_tensors(arrays, ids, device)
                target = torch.softmax(-cost(candidates, future) / temperature, dim=1)
                loss = -(target * torch.log_softmax(model(context, motion, last, candidates, logits), dim=1)).sum(1).mean()
                val_total += float(loss) * len(ids)
        val_loss = val_total / len(val_ids); row = {"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss}
        history.append(row); print("[scorer] " + json.dumps(row), flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, stale = val_loss, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best)
    return model, history, {"train_samples": len(train_ids), "validation_samples": len(val_ids), "best_validation_loss": best_loss}


def rank_rows(values):
    order = np.argsort(values, axis=1, kind="stable"); ranks = np.empty_like(order, dtype=np.float64)
    ranks[np.arange(len(values))[:, None], order] = np.arange(values.shape[1])[None]
    return ranks


def audit(ade, fde, scores):
    rows = np.arange(len(ade)); c = ade + .25 * fde; top = scores.argmax(1); best = c.argmin(1)
    x, y = rank_rows(scores), rank_rows(-c); x -= x.mean(1, keepdims=True); y -= y.mean(1, keepdims=True)
    rho = (x*y).sum(1) / np.sqrt((x*x).sum(1)*(y*y).sum(1)).clip(1e-12)
    shifted = scores - scores.max(1, keepdims=True); p = np.exp(shifted); p /= p.sum(1, keepdims=True)
    return {"top1_ADE": float(ade[rows, top].mean()), "top1_FDE": float(fde[rows, top].mean()),
            "top1_combined_cost": float(c[rows, top].mean()), "mean_regret": float((c[rows, top]-c[rows, best]).mean()),
            "p90_regret": float(np.quantile(c[rows, top]-c[rows, best], .9)),
            "combined_winner_hit": float((top == best).mean()), "spearman_mean": float(rho.mean()),
            "score_entropy": float(-(p*np.log(p.clip(1e-12))).sum(1).mean()),
            "active_modes": int(np.unique(top).size)}


def evaluate(model, arrays, device, batch_size):
    model.eval(); learned = []
    with torch.inference_mode():
        ids_all = np.arange(len(arrays["future"]))
        for start in range(0, len(ids_all), batch_size):
            ids = ids_all[start:start + batch_size]
            context, motion, last, candidates, logits, future = batch_tensors(arrays, ids, device)
            learned.append(model(context, motion, last, candidates, logits).cpu().numpy())
    candidates, future = arrays["candidates"], arrays["future"]
    distance = np.linalg.norm(candidates - future[:, None], axis=-1); ade = distance.mean(-1); fde = distance[..., -1]
    rows = np.arange(len(ade)); best_ade = ade.argmin(1)
    oracle = {"minADE": float(ade.min(1).mean()), "minFDE": float(fde.min(1).mean()),
              "same_path_FDE": float(fde[rows, best_ade].mean())}
    raw, synced = audit(ade, fde, arrays["original_logits"]), audit(ade, fde, np.concatenate(learned))
    improvement = {"top1_ADE": 1-synced["top1_ADE"]/raw["top1_ADE"],
                   "top1_FDE": 1-synced["top1_FDE"]/raw["top1_FDE"],
                   "mean_regret": 1-synced["mean_regret"]/max(raw["mean_regret"], 1e-12)}
    entropy_retention = synced["score_entropy"] / max(raw["score_entropy"], 1e-12)
    passed = (improvement["top1_ADE"] >= .05 and improvement["top1_FDE"] >= .05
              and improvement["mean_regret"] >= .15 and entropy_retention >= .90)
    return {"samples": len(ade), "oracle": oracle, "raw_score": raw, "synced_score": synced,
            "relative_improvement": improvement, "entropy_retention": float(entropy_retention),
            "sync_gate_passed": bool(passed)}


def macro(rows, family):
    chosen = [r for r in rows if r["family"] == family]
    out = {"folds": len(chosen), "all_fold_sync_gates_passed": all(r["sync_gate_passed"] for r in chosen)}
    for section in ("oracle", "raw_score", "synced_score"):
        keys = chosen[0][section]
        out[section] = {k: float(np.mean([r[section][k] for r in chosen])) for k in keys}
    out["relative_improvement"] = {
        "top1_ADE": 1-out["synced_score"]["top1_ADE"]/out["raw_score"]["top1_ADE"],
        "top1_FDE": 1-out["synced_score"]["top1_FDE"]/out["raw_score"]["top1_FDE"],
        "mean_regret": 1-out["synced_score"]["mean_regret"]/max(out["raw_score"]["mean_regret"], 1e-12),
    }
    out["entropy_retention"] = out["synced_score"]["score_entropy"] / max(out["raw_score"]["score_entropy"], 1e-12)
    out["macro_sync_gate_passed"] = (all(out["relative_improvement"][k] >= v for k, v in
                                         (("top1_ADE", .05), ("top1_FDE", .05), ("mean_regret", .15)))
                                      and out["entropy_retention"] >= .90)
    return out


def choose(v2, m2):
    if not v2["macro_sync_gate_passed"] and not m2["macro_sync_gate_passed"]:
        return "UNRESOLVED", "Neither family passes the EXP-161 macro synchronization gate."
    if v2["macro_sync_gate_passed"] and not m2["macro_sync_gate_passed"]:
        return "V2", "Only V2 passes the score-synchronization gate."
    if m2["macro_sync_gate_passed"] and not v2["macro_sync_gate_passed"]:
        return "M2", "Only M2 passes the score-synchronization gate."
    oracle_gain = .5*((1-m2["oracle"]["minADE"]/v2["oracle"]["minADE"])+(1-m2["oracle"]["minFDE"]/v2["oracle"]["minFDE"]))
    top_v2 = v2["synced_score"]["top1_combined_cost"]; top_m2 = m2["synced_score"]["top1_combined_cost"]
    if oracle_gain >= .01 and top_m2 <= 1.02*top_v2:
        return "M2", "M2 improves mean oracle quality by at least 1% without >2% synced Top-1 cost regression."
    if top_m2 <= .98*top_v2 and m2["oracle"]["minADE"] <= 1.01*v2["oracle"]["minADE"] and m2["oracle"]["minFDE"] <= 1.01*v2["oracle"]["minFDE"]:
        return "M2", "M2 improves synced Top-1 cost by at least 2% without material oracle regression."
    return "V2", "M2 does not provide enough post-synchronization gain to justify its extra refinement stage."


def write_analysis(path, summary):
    lines = ["# EXP-161 Result Analysis", "", f"## Decision: {summary['decision']}", "",
             f"Recommended candidate family: **{summary['recommended_family']}**", "", summary["decision_reason"], ""]
    for family in FAMILIES:
        x = summary["macro"][family]; imp = x["relative_improvement"]
        lines += [f"## {family.upper()}", "",
                  f"Oracle minADE/minFDE: {x['oracle']['minADE']:.4f}/{x['oracle']['minFDE']:.4f}",
                  f"Synced Top-1 ADE/FDE: {x['synced_score']['top1_ADE']:.4f}/{x['synced_score']['top1_FDE']:.4f}",
                  f"Top-1 ADE/FDE improvement: {100*imp['top1_ADE']:.2f}%/{100*imp['top1_FDE']:.2f}%",
                  f"Mean-regret improvement: {100*imp['mean_regret']:.2f}%",
                  f"Score-entropy retention: {100*x['entropy_retention']:.2f}%", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0"); p.add_argument("--folds", nargs="+", default=["all"], choices=("all", *FOLDS))
    p.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--candidate-batch-size", type=int, default=64)
    p.add_argument("--scorer-batch-size", type=int, default=256); p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=.10); p.add_argument("--validation-fraction", type=float, default=.10)
    p.add_argument("--patience", type=int, default=5); p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--max-train-samples", type=int, default=0); p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT); p.add_argument("--force-cache", action="store_true")
    p.add_argument("--force", action="store_true"); p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(); folds = FOLDS if "all" in a.folds else tuple(a.folds); families = tuple(a.families)
    base = load_module("exp161_base", BASE_SCRIPT); base.preflight(folds, a.seed)
    if a.dry_run:
        print(json.dumps({"status": "dry-run passed", "folds": folds, "families": families,
                          "candidate_K": 20, "encoder_decoder_frozen": True, "test_fold_used_for_training": False}))
        return
    if a.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(a.device); seed_all(a.seed); cfg = base.configure(device)
    from front_encoder_models import FrontEncoderASTRA
    helper = load_module("exp161_helper", base.TRAINER); a.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = a.output_dir / "cache"; ckpt_dir = a.output_dir / "scorers"; result_dir = a.output_dir / "fold_results"
    rows = []; contract = vars(a).copy(); contract["output_dir"] = str(contract["output_dir"])
    for family in families:
        for fold_index, fold in enumerate(folds):
            print(f"[run] family={family} fold={fold}", flush=True)
            result_path = result_dir / f"{fold}_{family}_seed_{a.seed}.json"; ckpt_path = ckpt_dir / f"{fold}_{family}_seed_{a.seed}.pth"
            if result_path.is_file() and ckpt_path.is_file() and not a.force and not a.force_cache:
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved.get("contract") == contract:
                    print(f"[resume] {family}/{fold}", flush=True); rows.append(saved["report"]); continue
            train, embedding = get_cache(base, helper, FrontEncoderASTRA, cfg, fold, family, "train", a.seed, device,
                                         a.candidate_batch_size, a.max_train_samples, cache_dir, a.force_cache)
            test, test_embedding = get_cache(base, helper, FrontEncoderASTRA, cfg, fold, family, "test", a.seed, device,
                                              a.candidate_batch_size, a.max_test_samples, cache_dir, a.force_cache)
            if not np.allclose(embedding, test_embedding): raise RuntimeError(f"Mode embedding mismatch: {family}/{fold}")
            model, history, split = train_scorer(train, embedding, device, a.seed + fold_index, a.epochs,
                                                 a.scorer_batch_size, a.learning_rate, a.temperature,
                                                 a.validation_fraction, a.patience, a.hidden_dim)
            report = evaluate(model, test, device, a.scorer_batch_size)
            report.update({"fold": fold, "family": family, "seed": a.seed,
                           "scorer_parameters": sum(p.numel() for p in model.parameters()),
                           "epochs_ran": len(history), "training": split})
            rows.append(report); ckpt_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "history": history, "training": split,
                        "family": family, "fold": fold, "contract": contract}, ckpt_path)
            result_path.write_text(json.dumps({"contract": contract, "report": report}, indent=2), encoding="utf-8")
            del model, train, test
            if device.type == "cuda": torch.cuda.empty_cache()
    if set(families) == set(FAMILIES) and set(folds) == set(FOLDS):
        macros = {family: macro(rows, family) for family in FAMILIES}; recommendation, reason = choose(macros["v2"], macros["m2"])
        decision = "PASSED" if recommendation in FAMILIES else "REJECTED"
    else:
        macros = {family: macro(rows, family) for family in families}; recommendation, reason, decision = "INCOMPLETE", "Run both families on all five folds for a final choice.", "CONDITIONAL"
    summary = {"experiment": "EXP-161 Frozen Candidate Listwise Scorer", "decision": decision,
               "recommended_family": recommendation, "decision_reason": reason, "macro": macros, "per_fold": rows,
               "protocol": {"candidate_K": 20, "encoder_decoder_frozen": True, "candidate_geometry_frozen": True,
                            "scorer_only": True, "test_fold_used_for_training": False, "seed": a.seed}}
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    flat = []
    for r in rows:
        flat.append({"family": r["family"], "fold": r["fold"], "samples": r["samples"],
                     "minADE": r["oracle"]["minADE"], "minFDE": r["oracle"]["minFDE"],
                     "raw_top1_ADE": r["raw_score"]["top1_ADE"], "raw_top1_FDE": r["raw_score"]["top1_FDE"],
                     "synced_top1_ADE": r["synced_score"]["top1_ADE"], "synced_top1_FDE": r["synced_score"]["top1_FDE"],
                     "ADE_improvement": r["relative_improvement"]["top1_ADE"], "FDE_improvement": r["relative_improvement"]["top1_FDE"],
                     "regret_improvement": r["relative_improvement"]["mean_regret"], "sync_gate_passed": r["sync_gate_passed"]})
    with (a.output_dir / "per_fold_summary.csv").open("w", newline="", encoding="utf-8-sig") as h:
        w = csv.DictWriter(h, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    write_analysis(a.output_dir / "RESULT_ANALYSIS.md", summary)
    print(json.dumps({"experiment": "EXP-161", "decision": decision, "recommended_family": recommendation,
                      "output": str(a.output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
