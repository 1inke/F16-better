"""One-small-batch forward/backward validation; it does not train formal runs."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
DECODER_PATH = (
    WORKSPACE / "AUTONOMOUS_EXPERIMENTS" /
    "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mirror(values: torch.Tensor) -> torch.Tensor:
    return values * values.new_tensor((1.0, -1.0))


def smoke(name: str, encoder_name: str | None, use_eq: bool, cfg, modules) -> dict:
    FrontEncoderASTRA, encoder_module, decoder_module = modules
    torch.manual_seed(7)
    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    if encoder_name:
        encoder_module.replace_encoder(model, cfg, encoder_name)
    decoder_module.replace_decoder(model, cfg, "kinematic_k20")
    model.train()
    batch = 2
    agents = model.num_pedestrians
    obs = model.obs_len
    pred = model.pred_len
    increments = torch.randn(batch, agents, obs, 2) * 0.03
    past = increments.cumsum(dim=2)
    future = past[:, :, -1:] + torch.randn(batch, agents, pred, 2).cumsum(dim=2) * 0.03
    num_valid = torch.tensor([min(agents, 5), min(agents, 3)], dtype=torch.int32)
    center = past.mean(dim=-2, keepdim=True)
    predictions = model(
        past - center, future - center, None, mode="train",
        num_valid=num_valid, graph_loc=past,
    )[2]
    logits = model.last_mode_logits
    loss = predictions[:, :1].square().mean() + 0.01 * logits[:, :1].square().mean()
    if use_eq:
        mirrored = model(
            mirror(past - center), mirror(future - center), None, mode="train",
            num_valid=num_valid, graph_loc=mirror(past),
        )[2]
        restored = mirror(mirrored[:, 0])
        pairwise = torch.linalg.vector_norm(
            predictions[:, 0].detach()[:, :, None] - restored[:, None], dim=-1
        ).mean(dim=-1)
        loss = loss + 0.02 * 0.5 * (
            pairwise.min(dim=-1).values.mean() + pairwise.min(dim=-2).values.mean()
        )
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise RuntimeError(f"non-finite gradient in {name}")
    return {
        "experiment": name,
        "prediction_shape": list(predictions.shape),
        "logit_shape": list(logits.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "loss": float(loss.detach()),
        "forward_backward": "passed",
    }


def main() -> None:
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA

    cfg = load_locked_q20(PROJECT / "configs/motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.device = "cpu"
    cfg.device_list = ["cpu"]
    encoder_module = load_module("encoder_precision_models", ROOT / "encoder_precision_models.py")
    decoder_module = load_module("kinematic_decoder_models", DECODER_PATH)
    modules = (FrontEncoderASTRA, encoder_module, decoder_module)
    cases = [
        ("baseline", None, False),
        ("eq_mirror", None, True),
        ("state_gate", "state_channel_gate", False),
        ("relation_gate", "relation_channel_gate", False),
        ("tcn_residual", "tcn_residual", False),
        ("temporal_rope", "temporal_rope", False),
    ]
    results = [smoke(name, encoder, eq, cfg, modules) for name, encoder, eq in cases]
    output = ROOT / "smoke_test_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"[passed] {output}")


if __name__ == "__main__":
    main()
