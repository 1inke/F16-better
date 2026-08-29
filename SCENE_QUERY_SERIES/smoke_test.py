"""CPU smoke test for the semantic anchor query decoder."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent


def main() -> None:
    smoke = ROOT / "_smoke"
    smoke.mkdir(parents=True, exist_ok=True)
    asset = smoke / "synthetic_bank.npz"
    maps = np.zeros((60, 6, 32, 32), dtype=np.uint8)
    maps[:, 0, :, :16] = 255
    maps[:, 1, :, 16:] = 255
    maps[:, 5] = np.linspace(0, 255, 32, dtype=np.uint8)[None, None, :]
    np.savez_compressed(
        asset, maps=maps, image_wh=np.tile(np.asarray([[640, 480]], np.float32), (60, 1)),
        video_names=np.asarray([f"scene_{index}" for index in range(60)]), channel_names=np.asarray(["x"] * 6),
        pixel_scale=np.asarray([20.0], np.float32),
    )
    os.environ["EXP147_HEADING_NORMALIZED"] = "1"
    from scene_query_models import SemanticAnchorQueryDecoder

    decoder = SemanticAnchorQueryDecoder(token_dim=32, pred_len=12, asset_path=asset)
    batch, agents = 3, 4
    encoded = torch.randn(batch, agents, 8, 32, requires_grad=True)
    past = torch.randn(batch, agents, 8, 2) * 0.1
    absolute = past + torch.tensor([[[[12.0, 10.0]]]])
    decoder.set_scene_batch(torch.tensor([0, 1, 0]), absolute)
    trajectories, logits = decoder(encoded, past)
    assert trajectories.shape == (batch, agents, 20, 12, 2)
    assert logits.shape == (batch, agents, 20)
    loss = trajectories[:, 0].square().mean() + logits.square().mean()
    loss.backward()
    gradient = decoder.scene_correction[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    print("SMOKE_OK", trajectories.shape, logits.shape, float(loss.detach()))


if __name__ == "__main__":
    main()
