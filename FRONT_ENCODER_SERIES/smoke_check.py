"""Fast shape/gradient check for every front-encoder variant."""

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(HERE)]
os.chdir(REPO)

import sitecustomize  # noqa: E402,F401
from front_encoder_models import FrontEncoderASTRA, VARIANTS  # noqa: E402
from experiment_config import load_locked_q20  # noqa: E402
from locked_mode_query_loss import ModeQueryLoss  # noqa: E402

cfg = load_locked_q20(REPO / "configs/motif_rbf_univ.yaml")
cfg.device = "cpu"
cfg.device_list = ["cpu"]
batch, agents, history = 2, 9, 8
past_absolute = torch.randn(batch, agents, history, 2)
center = past_absolute.mean(dim=-2, keepdim=True)
past = past_absolute - center
target = torch.randn(batch, agents, 12, 2)
num_valid = torch.tensor([9, 5])

parser = argparse.ArgumentParser()
parser.add_argument("--variants", nargs="*", default=None)
args = parser.parse_args()
selected = args.variants or sorted(VARIANTS)

for name in selected:
    if name not in VARIANTS:
        raise ValueError(f"Unknown variant: {name}")
    torch.manual_seed(1)
    model = FrontEncoderASTRA(cfg, name)
    prediction = model(
        past, target, None, mode="train",
        num_valid=num_valid, graph_loc=past_absolute,
    )[2]
    logits = model.last_mode_logits
    loss, _ = ModeQueryLoss(cfg)(
        prediction[:, :1], target[:, :1], logits[:, :1], cfg
    )
    loss.backward()
    decoder_parameters = sum(
        parameter.numel() for parameter in model.decoder.parameters()
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        name, tuple(prediction.shape), tuple(logits.shape),
        f"loss={float(loss):.4f}", f"params={total_parameters}",
        f"decoder={decoder_parameters}",
    )
