from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
WORKSPACE = Path(__file__).resolve().parents[1]
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
os.chdir(PROJECT)

import sitecustomize  # noqa: E402,F401
from experiment_config import load_locked_q20  # noqa: E402
from front_encoder_models import FrontEncoderASTRA  # noqa: E402


def forward(model: torch.nn.Module, past_absolute: torch.Tensor, num_valid: torch.Tensor):
    center = past_absolute.mean(dim=-2, keepdim=True)
    with torch.inference_mode():
        prediction = model(
            past_absolute - center, None, None, mode="testing",
            num_valid=num_valid, graph_loc=past_absolute,
        )[2]
    return prediction, model.last_mode_logits


def main() -> None:
    cfg = load_locked_q20(PROJECT / "configs/motif_rbf_univ.yaml")
    cfg.device = "cpu"
    cfg.device_list = ["cpu"]
    torch.manual_seed(7)
    past = torch.randn(3, 9, 8, 2)
    num_valid = torch.tensor([9, 6, 3])

    dense = FrontEncoderASTRA(cfg, "f16_temporal_social_cross").eval()
    edge = FrontEncoderASTRA(cfg, "f41_edge_list_attention").eval()
    edge.load_state_dict(dense.state_dict(), strict=True)
    dense_prediction, dense_logits = forward(dense, past, num_valid)
    edge_prediction, edge_logits = forward(edge, past, num_valid)
    prediction_error = (dense_prediction - edge_prediction).abs().max().item()
    logit_error = (dense_logits - edge_logits).abs().max().item()
    print(f"F16/F41 max prediction difference: {prediction_error:.9g}")
    print(f"F16/F41 max logit difference: {logit_error:.9g}")
    if prediction_error > 2.0e-5 or logit_error > 2.0e-5:
        raise RuntimeError("Edge-list attention is not numerically equivalent to F16.")


if __name__ == "__main__":
    main()
