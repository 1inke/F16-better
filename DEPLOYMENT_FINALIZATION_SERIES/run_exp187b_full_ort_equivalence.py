"""EXP-187B full-dataset PyTorch/ORT CUDA FP32 equivalence audit."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
ASTRA = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
MODEL_FILE = WORKSPACE / "FRONT_ENCODER_SERIES" / "front_encoder_models.py"
TRAINER_FILE = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
ADAPTER_FILE = WORKSPACE / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
EXPORT_HELPER_FILE = WORKSPACE / "DEPLOYMENT_OPTIMIZATION_SERIES" / "export_onnx.py"
CFG_FILE = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
EXP173 = WORKSPACE / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END" / "results"
SDD_BASE = WORKSPACE / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175S0_FINAL_JOINT_V2_SDD_BASELINE" / "results"
ETH_DATA = ASTRA / "datasets" / "eth_ucy" / "processed_data_diverse"
SDD_DATA = WORKSPACE / "SDD_STANDARD_POTENTIAL_SERIES" / "processed_data"
OUT = ROOT / "EXP_187B_FULL_ORT_EQUIVALENCE"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
PROFILES = (*FOLDS, "sdd")
VARIANT = "f16_temporal_social_cross"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint(profile: str, stage: str) -> Path:
    if profile == "sdd":
        folder = "h3_anchor_proposal" if stage == "h3" else "joint_v2"
        return SDD_BASE / folder / VARIANT / "seed_1" / "val_scene_1" / "best_selected.pth"
    return EXP173 / profile / stage / "e3" / VARIANT / "seed_1" / "val_scene_-1" / "best_selected.pth"


def configure(profile: str, device: torch.device):
    for path in (ASTRA, ASTRA / "scripts", WORKSPACE / "FRONT_ENCODER_SERIES"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.chdir(ASTRA)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    cfg = load_locked_q20(CFG_FILE)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 16 if profile == "sdd" else 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device)
    cfg.device_list = [device.index or 0]
    return cfg


def build(profile: str, device: torch.device):
    cfg = configure(profile, device)
    trainer = load_module(f"exp187b_trainer_{profile}", TRAINER_FILE)
    models = load_module(f"exp187b_models_{profile}", MODEL_FILE)
    adapter = load_module(f"exp187b_adapter_{profile}", ADAPTER_FILE)
    helper = load_module(f"exp187b_export_{profile}", EXPORT_HELPER_FILE)
    h3, final = checkpoint(profile, "h3"), checkpoint(profile, "v2")
    for path in (h3, final):
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(h3)
    os.environ["EXP173_DECODER_INIT_SEED"] = "1"
    model = models.FrontEncoderASTRA(cfg, VARIANT)
    adapter.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")
    model.load_state_dict(torch.load(final, map_location="cpu", weights_only=True), strict=True)
    wrapper = helper.TrajectoryDeploymentWrapper(model).to(device).eval()
    if profile == "sdd":
        dataset = trainer.TrajectoryDataset(SDD_DATA / "sdd_standard_test_data.npy", SDD_DATA / "sdd_standard_test_num.npy", cfg)
    else:
        dataset = trainer.TrajectoryDataset(ETH_DATA / f"{profile}_data_test.npy", ETH_DATA / f"{profile}_num_test.npy", cfg)
    return wrapper, dataset, helper


def sample(dataset, index: int, device: torch.device):
    item = dataset[index]
    past = torch.as_tensor(item[0], dtype=torch.float32, device=device).unsqueeze(0)
    future = torch.as_tensor(item[1], dtype=torch.float32, device=device)[0]
    valid = torch.as_tensor(item[2], dtype=torch.long, device=device).reshape(1)
    return past, future, valid


def export(wrapper, dataset, helper, profile: str, device: torch.device, output: Path, force: bool) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and not force:
        return output
    past, _, valid = sample(dataset, 0, device)
    # Keep export outside inference_mode: on current PyTorch, inference_mode
    # selects aten::_transformer_encoder_layer_fwd, which the legacy ONNX
    # exporter cannot lower. The exported eval graph is otherwise identical.
    torch.onnx.export(wrapper, (past, valid), output, input_names=["past", "num_valid"], output_names=["candidates", "logits"], opset_version=17, do_constant_folding=True, dynamo=False)
    graph = helper.stabilize_exported_atan2(helper.fold_fixed_eyelike(onnx.load(output)))
    onnx.checker.check_model(graph)
    onnx.save(graph, output)
    return output


def make_session(path: Path) -> ort.InferenceSession:
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"CUDAExecutionProvider unavailable: {providers}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


class Accumulator:
    def __init__(self):
        self.samples = 0
        self.pt = {name: 0.0 for name in ("minADE", "minFDE", "same_path_FDE", "top1_ADE", "top1_FDE")}
        self.ort = dict(self.pt)
        self.candidate_abs_sum = self.logit_abs_sum = 0.0
        self.candidate_count = self.logit_count = 0
        self.candidate_max_abs = self.logit_max_abs = 0.0
        self.top1_disagreements = 0

    @staticmethod
    def metrics(candidate: np.ndarray, future: np.ndarray, logits: np.ndarray) -> dict[str, float]:
        distance = np.linalg.norm(candidate - future[None], axis=-1)
        ade, fde = distance.mean(-1), distance[:, -1]
        best_ade, best_fde, top1 = int(ade.argmin()), int(fde.argmin()), int(logits.argmax())
        return {"minADE": float(ade[best_ade]), "minFDE": float(fde[best_fde]), "same_path_FDE": float(fde[best_ade]), "top1_ADE": float(ade[top1]), "top1_FDE": float(fde[top1])}

    def add(self, pt_c, pt_l, ort_c, ort_l, future):
        pt_c, pt_l, ort_c, ort_l = pt_c[0], pt_l[0], ort_c[0], ort_l[0]
        for target, values in ((self.pt, self.metrics(pt_c, future, pt_l)), (self.ort, self.metrics(ort_c, future, ort_l))):
            for key, value in values.items():
                target[key] += value
        dc, dl = np.abs(pt_c - ort_c), np.abs(pt_l - ort_l)
        self.candidate_abs_sum += float(dc.sum(dtype=np.float64)); self.candidate_count += dc.size
        self.logit_abs_sum += float(dl.sum(dtype=np.float64)); self.logit_count += dl.size
        self.candidate_max_abs = max(self.candidate_max_abs, float(dc.max()))
        self.logit_max_abs = max(self.logit_max_abs, float(dl.max()))
        self.top1_disagreements += int(pt_l.argmax() != ort_l.argmax())
        self.samples += 1

    def finish(self, profile: str, elapsed: float) -> dict:
        row = {"profile": profile, "samples": self.samples, "elapsed_seconds": elapsed}
        for backend, source in (("pytorch", self.pt), ("ort", self.ort)):
            for key, value in source.items():
                row[f"{backend}_{key}"] = value / self.samples
            row[f"{backend}_top1_J"] = row[f"{backend}_top1_ADE"] + 0.5 * row[f"{backend}_top1_FDE"]
        for key in (*self.pt.keys(), "top1_J"):
            base = row[f"pytorch_{key}"]
            row[f"relative_{key}_delta_pct"] = abs(row[f"ort_{key}"] - base) / max(abs(base), 1e-12) * 100.0
        row.update({
            "candidate_mean_abs_diff": self.candidate_abs_sum / self.candidate_count,
            "candidate_max_abs_diff": self.candidate_max_abs,
            "logit_mean_abs_diff": self.logit_abs_sum / self.logit_count,
            "logit_max_abs_diff": self.logit_max_abs,
            "top1_disagreement_rate": self.top1_disagreements / self.samples,
        })
        row["passed"] = bool(row["relative_minADE_delta_pct"] < 0.2 and row["relative_minFDE_delta_pct"] < 0.2)
        return row


def audit_profile(profile: str, args, output_dir: Path, device: torch.device) -> dict:
    cached = output_dir / "profiles" / f"{profile}.json"
    if cached.is_file() and not args.force:
        print(f"[reuse] {profile}", flush=True)
        return json.loads(cached.read_text(encoding="utf-8"))
    wrapper, dataset, helper = build(profile, device)
    artifact_dir = output_dir / "artifacts"
    model_path = export(wrapper, dataset, helper, profile, device, artifact_dir / f"{profile}_fp32.onnx", args.force_export)
    session = make_session(model_path)
    limit = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)
    accumulator, started = Accumulator(), time.perf_counter()
    with torch.inference_mode():
        for index in range(limit):
            past, future, valid = sample(dataset, index, device)
            pt_c, pt_l = wrapper(past, valid)
            ort_c, ort_l = session.run(["candidates", "logits"], {"past": past.cpu().numpy(), "num_valid": valid.cpu().numpy()})
            accumulator.add(pt_c.cpu().numpy(), pt_l.cpu().numpy(), ort_c, ort_l, future.cpu().numpy())
            if (index + 1) % args.progress_every == 0 or index + 1 == limit:
                print(f"[{profile}] {index + 1}/{limit}", flush=True)
    row = accumulator.finish(profile, time.perf_counter() - started)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate(rows: list[dict], output_dir: Path, smoke: bool) -> dict:
    # The guide requires both metric equivalence and no fold-level abnormal
    # tensor divergence. These deliberately loose audit limits only catch
    # gross FP32 mismatches; they are not accuracy-tuning thresholds.
    for row in rows:
        row["metric_pass"] = bool(
            row["relative_minADE_delta_pct"] < 0.2
            and row["relative_minFDE_delta_pct"] < 0.2
        )
        reasons = []
        if row["candidate_max_abs_diff"] > 0.5:
            reasons.append("candidate_max_abs_diff>0.5")
        if row["logit_max_abs_diff"] > 0.5:
            reasons.append("logit_max_abs_diff>0.5")
        if row["top1_disagreement_rate"] > 0.01:
            reasons.append("top1_disagreement_rate>1%")
        row["tensor_anomaly"] = bool(reasons)
        row["anomaly_reasons"] = ";".join(reasons)
        row["passed"] = bool(row["metric_pass"] and not row["tensor_anomaly"])
    eth = [row for row in rows if row["profile"] in FOLDS]
    macro = {}
    if len(eth) == len(FOLDS):
        for backend in ("pytorch", "ort"):
            for key in ("minADE", "minFDE", "same_path_FDE", "top1_ADE", "top1_FDE", "top1_J"):
                macro[f"{backend}_{key}"] = float(np.mean([row[f"{backend}_{key}"] for row in eth]))
        for key in ("minADE", "minFDE", "same_path_FDE", "top1_ADE", "top1_FDE", "top1_J"):
            macro[f"relative_{key}_delta_pct"] = abs(macro[f"ort_{key}"] - macro[f"pytorch_{key}"]) / max(abs(macro[f"pytorch_{key}"]), 1e-12) * 100.0
    complete = set(row["profile"] for row in rows) == set(PROFILES)
    metric_pass = complete and all(row["metric_pass"] for row in rows) and macro.get("relative_minADE_delta_pct", 1.0) < 0.2 and macro.get("relative_minFDE_delta_pct", 1.0) < 0.2
    tensor_anomalies = [row["profile"] for row in rows if row["tensor_anomaly"]]
    passed = metric_pass and not tensor_anomalies
    if passed:
        decision = "FINAL_DEPLOYMENT_BACKEND_ORT_CUDA_FP32"
    elif metric_pass and tensor_anomalies:
        decision = "METRIC_PASS_TENSOR_ANOMALY_AUDIT_REQUIRED"
    else:
        decision = "PREFLIGHT_OK" if smoke else "REVIEW_EQUIVALENCE_FAILURE"
    summary = {"experiment": "EXP-187B", "model": "F16+H3+Joint-V2+original score+K20 seed1", "complete": complete, "smoke": smoke, "metric_gate_passed": metric_pass, "tensor_anomaly_profiles": tensor_anomalies, "profiles": rows, "eth_ucy_macro": macro, "decision": decision}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "eth_ucy.csv", eth)
    write_csv(output_dir / "sdd.csv", [row for row in rows if row["profile"] == "sdd"])
    write_csv(output_dir / "tensor_diff.csv", [{key: value for key, value in row.items() if key in {"profile", "samples", "candidate_mean_abs_diff", "candidate_max_abs_diff", "logit_mean_abs_diff", "logit_max_abs_diff", "top1_disagreement_rate", "passed"}} for row in rows])
    lines = ["# EXP-187B Full ORT Equivalence", "", f"Decision: **{decision}**", "", "| Profile | Samples | ΔminADE | ΔminFDE | Candidate mean/max | Logit mean/max | Pass |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['profile']} | {row['samples']} | {row['relative_minADE_delta_pct']:.4f}% | {row['relative_minFDE_delta_pct']:.4f}% | {row['candidate_mean_abs_diff']:.6g}/{row['candidate_max_abs_diff']:.6g} | {row['logit_mean_abs_diff']:.6g}/{row['logit_max_abs_diff']:.6g} | {'PASS' if row['passed'] else 'REVIEW'} |")
    if tensor_anomalies:
        lines += ["", "## Tensor anomaly audit", ""]
        for row in rows:
            if row["tensor_anomaly"]:
                lines.append(f"- {row['profile']}: {row['anomaly_reasons']}")
    (output_dir / "RESULT_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profiles", nargs="+", choices=("all", *PROFILES), default=["all"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    profiles = PROFILES if "all" in args.profiles else tuple(args.profiles)
    if args.smoke:
        args.max_samples = min(args.max_samples or 2, 100)
        output_dir = OUT / "smoke"
    else:
        output_dir = OUT
    required = (MODEL_FILE, TRAINER_FILE, ADAPTER_FILE, EXPORT_HELPER_FILE, CFG_FILE)
    missing = [str(path) for path in required if not path.is_file()]
    missing += [str(checkpoint(profile, stage)) for profile in profiles for stage in ("h3", "v2") if not checkpoint(profile, stage).is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))
    rows = [audit_profile(profile, args, output_dir, device) for profile in profiles]
    summary = aggregate(rows, output_dir, args.smoke)
    print(json.dumps({"complete": summary["complete"], "decision": summary["decision"]}, ensure_ascii=False), flush=True)
    print(f"[done] {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
