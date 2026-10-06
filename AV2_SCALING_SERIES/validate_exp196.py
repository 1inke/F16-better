"""Fast shape/checkpoint validation without training or AV2 preparation."""
from __future__ import annotations

import argparse

import torch

from exp196_common import build_model, model_forward, temporal_weight_mass
from exp196_constants import A1_CHECKPOINT, VARIANTS, config_for
from materialize_configs import materialize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=["a1c", "l4", "d4", "j2"])
    args = parser.parse_args()
    materialize()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for name in args.variants:
        spec = VARIANTS[name]
        model, _ = build_model(config_for(name), A1_CHECKPOINT, device)
        model.eval()
        past = torch.randn(
            1, int(spec["max_agents"]), int(spec["history_steps"]), 2,
            device=device,
        )
        num = torch.tensor([int(spec["max_agents"])], device=device)
        with torch.inference_mode():
            prediction, logits = model_forward(model, past, num)
        print({
            "config": name,
            "prediction": tuple(prediction.shape),
            "logits": tuple(logits.shape),
            "history_weight_mass": temporal_weight_mass(
                int(spec["history_steps"]), float(spec["history_hz"])
            ),
        })


if __name__ == "__main__":
    main()
