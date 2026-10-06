"""Zero-data RGDM/C00 integration preflight. This script never starts training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASTRA = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DEFAULT_C00 = (
    ROOT / "QUALITY_AWARE_F16_SERIES_V2" / "results" / "p0_protocol"
    / "C0_EXP173_EXACT" / "f16_temporal_social_cross" / "seed_1"
    / "val_scene_-1" / "best_selected.pth"
)
DEFAULT_H3 = (
    ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"
    / "results" / "eth" / "h3" / "e3" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1" / "best_selected.pth"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="rgdm_b4_full")
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_C00)
    parser.add_argument("--h3-checkpoint", type=Path, default=DEFAULT_H3)
    args = parser.parse_args()
    source = args.source_checkpoint.resolve()
    h3 = args.h3_checkpoint.resolve()
    for path in (source, h3):
        if not path.is_file():
            raise FileNotFoundError(path)

    os.chdir(ASTRA)
    sys.path[:0] = [str(ASTRA), str(ASTRA / "scripts"), str(ROOT / "FRONT_ENCODER_SERIES"), str(HERE)]
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20

    cfg_path = ROOT / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
    cfg = load_locked_q20(cfg_path)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = "cpu"
    cfg.device_list = ["cpu"]

    front = load_module("rgdm_preflight_front", ROOT / "FRONT_ENCODER_SERIES" / "front_encoder_models.py")
    base_adapter = load_module(
        "rgdm_preflight_base", ROOT / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
    )
    rgdm = load_module("rgdm_preflight_models", HERE / "rgdm_models.py")
    os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(h3)
    os.environ["EXP173_DECODER_INIT_SEED"] = "1"
    os.environ["RGDM_SOURCE_CHECKPOINT"] = str(source)

    base_cfg, treatment_cfg = copy.deepcopy(cfg), copy.deepcopy(cfg)
    baseline = front.FrontEncoderASTRA(base_cfg, "f16_temporal_social_cross")
    base_adapter.replace_decoder(baseline, base_cfg, "v2_joint_wide_endpoint_correction")
    baseline.load_state_dict(torch.load(source, map_location="cpu", weights_only=True), strict=True)
    baseline.eval()

    treatment = front.FrontEncoderASTRA(treatment_cfg, "f16_temporal_social_cross")
    rgdm.replace_decoder(treatment, treatment_cfg, args.variant)
    treatment.eval()

    torch.manual_seed(198)
    past = 0.05 * torch.randn(2, 9, 8, 2).cumsum(dim=-2)
    num_valid = torch.tensor([3, 5], dtype=torch.int32)
    with torch.inference_mode():
        baseline_trajectory = baseline(
            past, None, None, mode="testing", num_valid=num_valid, graph_loc=past
        )[2]
        treatment_trajectory = treatment(
            past, None, None, mode="testing", num_valid=num_valid, graph_loc=past
        )[2]
    trajectory_error = float((baseline_trajectory - treatment_trajectory).abs().max())
    logit_error = float((baseline.last_mode_logits - treatment.last_mode_logits).abs().max())
    # Separate full-model CPU calls can differ at the last reduction bit even
    # when the decoder addition is an exact identity; keep the numerical gate
    # tighter than the existing 2e-6 project equivalence convention.
    if trajectory_error > 2.0e-6 or logit_error > 2.0e-6:
        raise RuntimeError(
            f"RGDM identity gate failed: trajectory={trajectory_error}, logits={logit_error}"
        )
    trainable = {
        name: parameter.numel() for name, parameter in treatment.named_parameters()
        if parameter.requires_grad
    }
    allowed = ("decoder.generation_rgdm.", "decoder.selection_rgdm.")
    if not trainable or any(not name.startswith(allowed) for name in trainable):
        raise RuntimeError(f"unexpected trainable scope: {sorted(trainable)}")
    result = {
        "status": "PASS_NO_TRAINING",
        "variant": args.variant,
        "source_checkpoint": str(source),
        "source_sha256": sha256(source),
        "trajectory_max_abs_error": trajectory_error,
        "logit_max_abs_error": logit_error,
        "total_parameters": sum(parameter.numel() for parameter in treatment.parameters()),
        "trainable_parameters": sum(trainable.values()),
        "trainable_prefixes": sorted({name.split(".")[1] for name in trainable}),
        "synthetic_observation_only": True,
        "future_or_test_data_accessed": False,
        "experiment_started": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
