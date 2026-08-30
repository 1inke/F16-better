"""EXP-158B: isolate K20/V2 complementarity under identical encoded tokens."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
BASE_SCRIPT = WORKSPACE / "CANDIDATE_COMPLEMENTARITY_SERIES" / "run_exp158_exp160_joint_audit.py"
DEFAULT_OUT = HERE / "EXP_158B_SHARED_ENCODER_COMPLEMENTARITY"
HOSTS = ("v2", "k20")
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_shared(host_model, donor_decoder, host_family: str, loader, helper, device: torch.device):
    captured: dict[str, torch.Tensor] = {}

    def capture_inputs(_module, args):
        captured["tokens"] = args[0]
        captured["past"] = args[1]

    handle = host_model.decoder.register_forward_pre_hook(capture_inputs)
    parts = {
        "past": [], "future": [], "num_valid": [],
        "k20_candidates": [], "k20_logits": [],
        "v2_candidates": [], "v2_logits": [],
    }
    donor_family = "k20" if host_family == "v2" else "v2"
    try:
        with torch.inference_mode():
            for past, future, num_valid, _, _, _ in loader:
                past = past.to(device)
                future = future.to(device)
                num_valid = num_valid.to(device).int()
                past, future = helper.normalize_to_focal_heading(past, future)
                center = past.mean(dim=-2, keepdim=True)
                host_prediction = host_model(
                    past - center, None, None, mode="testing",
                    num_valid=num_valid, graph_loc=past,
                )[2] + center.unsqueeze(-3)
                if "tokens" not in captured:
                    raise RuntimeError("Decoder input hook did not capture shared encoded tokens")
                donor_prediction, donor_logits = donor_decoder(captured["tokens"], captured["past"])
                donor_prediction = donor_prediction + center.unsqueeze(-3)
                native_logits = host_model.last_mode_logits
                predictions = {
                    host_family: host_prediction[:, 0],
                    donor_family: donor_prediction[:, 0],
                }
                logits = {
                    host_family: native_logits[:, 0],
                    donor_family: donor_logits[:, 0],
                }
                parts["past"].append(past.float().cpu().numpy())
                parts["future"].append(future[:, 0].float().cpu().numpy())
                parts["num_valid"].append(num_valid.cpu().numpy())
                for family in ("k20", "v2"):
                    parts[f"{family}_candidates"].append(predictions[family].float().cpu().numpy())
                    parts[f"{family}_logits"].append(logits[family].float().cpu().numpy())
                captured.clear()
    finally:
        handle.remove()
    return {key: np.concatenate(value) for key, value in parts.items()}


def pair_metrics(base, bundle: dict) -> dict:
    future = bundle["future"]
    errors = {}
    singles = {}
    for family in ("k20", "v2"):
        ade, fde = base.errors(bundle[f"{family}_candidates"], future)
        errors[family] = (ade, fde)
        singles[family] = base.oracle_metrics(ade, fde)
    union_ade = np.concatenate((errors["k20"][0], errors["v2"][0]), axis=1)
    union_fde = np.concatenate((errors["k20"][1], errors["v2"][1]), axis=1)
    union = base.oracle_metrics(union_ade, union_fde)
    winner = {
        "ADE": {
            "k20": float((errors["k20"][0].min(1) <= errors["v2"][0].min(1)).mean()),
            "v2": float((errors["v2"][0].min(1) < errors["k20"][0].min(1)).mean()),
        },
        "FDE": {
            "k20": float((errors["k20"][1].min(1) <= errors["v2"][1].min(1)).mean()),
            "v2": float((errors["v2"][1].min(1) < errors["k20"][1].min(1)).mean()),
        },
    }
    gains = {
        metric: (min(singles["k20"][metric], singles["v2"][metric]) - union[metric])
        / max(min(singles["k20"][metric], singles["v2"][metric]), 1.0e-12)
        for metric in ("minADE", "minFDE", "same_path_FDE")
    }
    return {
        "families": singles,
        "union": union,
        "relative_union_gain": gains,
        "winner_family": winner,
        "overlap": base.overlap_metrics(bundle["k20_candidates"], bundle["v2_candidates"]),
    }


def average(rows: list[dict], host: str, path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        if row["host"] != host:
            continue
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    if not values:
        return float("nan")
    return float(np.mean(values))


def decide(rows: list[dict], folds: tuple[str, ...]) -> tuple[str, dict]:
    primary = [row for row in rows if row["host"] == "v2"]
    if len(primary) != len(folds):
        return "CONDITIONAL", {"reason": "V2-host was not evaluated on every requested fold"}
    macro = {
        metric: float(np.mean([row["relative_union_gain"][metric] for row in primary]))
        for metric in ("minADE", "minFDE", "same_path_FDE")
    }
    all_fold_positive = all(
        row["relative_union_gain"]["minADE"] > 0
        and row["relative_union_gain"]["minFDE"] > 0
        for row in primary
    )
    if macro["minADE"] >= 0.05 and macro["minFDE"] >= 0.05 and all_fold_positive:
        decision = "PASSED"
    elif macro["minADE"] >= 0.02 and macro["minFDE"] >= 0.02:
        decision = "CONDITIONAL"
    else:
        decision = "REJECTED"
    return decision, {"v2_host_macro_relative_union_gain": macro, "all_fold_ADE_FDE_gain_positive": all_fold_positive}


def flatten(prefix: str, value: dict) -> dict:
    result = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result.update(flatten(f"{prefix}{key}_", item))
        else:
            result[f"{prefix}{key}"] = item
    return result


def write_analysis(path: Path, decision: str, detail: dict) -> None:
    gain = detail.get("v2_host_macro_relative_union_gain", {})
    if decision == "PASSED":
        next_step = "共享编码器下互补性成立。进入 EXP-159，在 V2-host 上搜索固定 K=20 的 K20/V2 配额。"
    elif decision == "CONDITIONAL":
        next_step = "证据有限。先检查逐折与 transplanted decoder 的退化，再决定是否进行 EXP-159。"
    else:
        next_step = "共享编码器后互补性不足。停止 candidate mixing，原 EXP-158 的主要收益应视为模型集成收益。"
    text = (
        f"# EXP-158B Result Analysis\n\n## Decision: {decision}\n\n"
        f"V2-host 宏平均 union 增益：minADE {100 * gain.get('minADE', float('nan')):.2f}%，"
        f"minFDE {100 * gain.get('minFDE', float('nan')):.2f}%，"
        f"same-path FDE {100 * gain.get('same_path_FDE', float('nan')):.2f}%。\n\n"
        f"五折 minADE/minFDE 是否全部正增益：{detail.get('all_fold_ADE_FDE_gain_positive')}。\n\n"
        f"## Next step\n\n{next_step}\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", nargs="+", choices=("all", *FOLDS), default=["all"])
    parser.add_argument("--hosts", nargs="+", choices=HOSTS, default=list(HOSTS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-save-candidates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_module("exp158b_base", BASE_SCRIPT)
    folds = FOLDS if "all" in args.folds else tuple(args.folds)
    base.preflight(folds, args.seed)
    if args.dry_run:
        print(json.dumps({
            "status": "dry-run passed", "folds": folds, "hosts": args.hosts,
            "seed": args.seed, "shared_decoder_calls": len(folds) * len(args.hosts) * 2,
        }))
        return
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    cfg = base.configure(device)
    from front_encoder_models import FrontEncoderASTRA
    helper = load_module("exp158b_training_helper", base.TRAINER)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "per_sample"
    if not args.no_save_candidates:
        cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold in folds:
        dataset = helper.TrajectoryDataset(base.DATA / f"{fold}_data_test.npy", base.DATA / f"{fold}_num_test.npy", cfg)
        if args.max_samples:
            dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for host_family in args.hosts:
            donor_family = "k20" if host_family == "v2" else "v2"
            print(f"[collect] fold={fold} shared_host={host_family} donor_decoder={donor_family}", flush=True)
            host_model = base.build_model(fold, host_family, args.seed, cfg, FrontEncoderASTRA, device)
            donor_model = base.build_model(fold, donor_family, args.seed, cfg, FrontEncoderASTRA, device)
            bundle = collect_shared(host_model, donor_model.decoder, host_family, loader, helper, device)
            metrics = pair_metrics(base, bundle)
            row = {"fold": fold, "host": host_family, "seed": args.seed, "samples": len(bundle["future"]), **metrics}
            rows.append(row)
            if not args.no_save_candidates:
                np.savez_compressed(cache_dir / f"{fold}_{host_family}_host_seed_{args.seed}.npz", **bundle)
            del host_model, donor_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[done] fold={fold} host={host_family} gain_ADE={metrics['relative_union_gain']['minADE']:.4f} gain_FDE={metrics['relative_union_gain']['minFDE']:.4f}", flush=True)

    decision, detail = decide(rows, folds)
    macro_by_host = {}
    for host in args.hosts:
        macro_by_host[host] = {
            "families": {
                family: {
                    metric: average(rows, host, ("families", family, metric))
                    for metric in ("minADE", "minFDE", "same_path_FDE", "combined_cost")
                }
                for family in ("k20", "v2")
            },
            "union": {
                metric: average(rows, host, ("union", metric))
                for metric in ("minADE", "minFDE", "same_path_FDE", "combined_cost")
            },
        }
    summary = {
        "experiment": "EXP-158B Shared-Encoder Complementarity Control",
        "protocol": "official ETH/UCY 8/12 at 2.5 Hz",
        "seed": args.seed, "folds": list(folds), "hosts": args.hosts,
        "decision": decision, **detail, "macro_by_host": macro_by_host,
        "per_fold": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    flat_rows = [{"fold": row["fold"], "host": row["host"], "seed": row["seed"], "samples": row["samples"], **flatten("", {key: row[key] for key in ("families", "union", "relative_union_gain", "winner_family", "overlap")})} for row in rows]
    if flat_rows:
        with (args.output_dir / "per_fold_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    write_analysis(args.output_dir / "RESULT_ANALYSIS.md", decision, detail)
    print(json.dumps({"experiment": "EXP-158B", "decision": decision, "output": str(args.output_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
