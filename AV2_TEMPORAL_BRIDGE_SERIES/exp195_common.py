"""Shared EXP-195 model, data, and metric utilities."""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.utils.data import Dataset

from exp195_constants import (
    ASTRA_PROJECT, ASTRA_PYTHON, CONFIGS, DECODER_SOURCE,
    DEFAULT_ETH_CHECKPOINT, FRONT_DIR, FUTURE_HZ, FUTURE_STEPS,
    MAX_AGENTS, REFERENCE_HZ, ROOT, TRAINER, WORKSPACE, require_paths,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_history_positions(obs_len: int, history_hz: float, device, dtype) -> torch.Tensor:
    """End-align PE to the pretrained A1 clock (A1 remains exactly 0..7)."""
    reference_last = float(CONFIGS["a1"]["history_steps"] - 1)
    relative = torch.arange(obs_len, device=device, dtype=dtype) - float(obs_len - 1)
    return reference_last + relative * (REFERENCE_HZ / float(history_hz))


def patch_physical_time(model, history_hz: float) -> None:
    """Patch non-parametric PE and decoder motion units without changing weights."""
    history_hz = float(history_hz)

    def temporal_encoder(owner, batch_size: int):
        position = physical_history_positions(
            owner.obs_len, history_hz, owner.device, torch.float32
        ).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, owner.temp_dim, 2, device=position.device).float()
            * (-math.log(10000.0) / owner.temp_dim)
        )
        encoded = torch.zeros(owner.obs_len, owner.temp_dim, device=position.device)
        encoded[:, 0::2] = torch.sin(position * div)
        encoded[:, 1::2] = torch.cos(position * div)
        return encoded[None, :, None].expand(
            batch_size, -1, owner.num_pedestrians, -1
        )

    def future_temporal_encoder(owner, batch_size: int):
        # 12 future labels are always on the 2.5 Hz reference clock.
        start = float(CONFIGS["a1"]["history_steps"])
        position = torch.arange(
            start, start + FUTURE_STEPS, device=owner.device, dtype=torch.float32
        ).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, owner.temp_dim, 2, device=position.device).float()
            * (-math.log(10000.0) / owner.temp_dim)
        )
        encoded = torch.zeros(FUTURE_STEPS, owner.temp_dim, device=position.device)
        encoded[:, 0::2] = torch.sin(position * div)
        encoded[:, 1::2] = torch.cos(position * div)
        return encoded[None, :, None].expand(
            batch_size, -1, owner.num_pedestrians, -1
        )

    def reference_motion_features(_decoder, past_loc: torch.Tensor) -> torch.Tensor:
        # Match the pretrained decoder's 2.5 Hz displacement scale while using
        # physical velocities from every history rate.
        scale = history_hz / REFERENCE_HZ
        short = (past_loc[:, :, -1] - past_loc[:, :, -2]) * scale
        lag = min(3, past_loc.shape[2] - 1)
        long = (
            (past_loc[:, :, -1] - past_loc[:, :, -1 - lag])
            / float(lag) * scale
        )
        acceleration = short - long
        speed = torch.linalg.vector_norm(short, dim=-1, keepdim=True)
        return torch.cat((short, long, acceleration, speed), dim=-1)

    model.temporal_encoder = MethodType(temporal_encoder, model)
    model.future_temporal_encoder = MethodType(future_temporal_encoder, model)
    model.decoder.motion_features = MethodType(reference_motion_features, model.decoder)
    model.exp195_history_hz = history_hz


def import_runtime():
    require_paths(ASTRA_PROJECT, FRONT_DIR, DECODER_SOURCE)
    sys.path[:0] = [str(ASTRA_PROJECT), str(ASTRA_PROJECT / "scripts"), str(FRONT_DIR)]
    os.chdir(ASTRA_PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA

    decoder_module = load_module("exp195_grouped_decoder", DECODER_SOURCE)
    return load_locked_q20, FrontEncoderASTRA, decoder_module.GroupedModeTemplateDecoder


def build_model(config_path: Path, checkpoint: Path, device: torch.device):
    load_config, model_class, decoder_class = import_runtime()
    cfg = load_config(config_path)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.device = str(device)
    cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    model = model_class(cfg, "f16_temporal_social_cross")
    model.decoder = decoder_class(model.token_dim, model.pred_len, use_template=True, use_groups=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    patch_physical_time(model, float(cfg.DATA.FREQUENCY))
    model.decoder.cv_reference = False
    return model.to(device), cfg


class BridgeArrayDataset(Dataset):
    def __init__(self, data_path: Path, num_path: Path, history_steps: int):
        self.data = np.load(data_path, mmap_mode="r")
        self.num = np.load(num_path, mmap_mode="r")
        self.history_steps = int(history_steps)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        row = np.asarray(self.data[index, :, :, -2:], dtype=np.float32)
        return (
            torch.from_numpy(row[:, : self.history_steps].copy()),
            torch.from_numpy(row[:, self.history_steps :].copy()),
            torch.tensor(min(int(self.num[index]), MAX_AGENTS), dtype=torch.int64),
        )


def normalize_to_focal_heading(past: torch.Tensor, future: torch.Tensor):
    lag = min(3, past.shape[-2] - 1)
    displacement = past[:, 0, -1] - past[:, 0, -1 - lag]
    speed = torch.linalg.vector_norm(displacement, dim=-1)
    default = torch.zeros_like(displacement)
    default[:, 0] = 1.0
    heading = torch.where(
        (speed > 0.03)[:, None],
        displacement / speed[:, None].clamp_min(1.0e-6),
        default,
    )
    cosine, sine = heading[:, 0], heading[:, 1]
    matrix = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        past.shape[0], 1, 1, 2, 2
    )
    origin = past[:, :1, -1:]

    def rotate(values):
        return torch.matmul((values - origin).unsqueeze(-2), matrix).squeeze(-2) + origin

    return rotate(past), rotate(future)


def model_forward(model, past: torch.Tensor, num_valid: torch.Tensor):
    center = past.mean(dim=-2, keepdim=True)
    prediction = model(
        past - center, None, None, mode="testing",
        num_valid=num_valid.int(), graph_loc=past,
    )[2] + center.unsqueeze(-3)
    return prediction[:, 0], model.last_mode_logits[:, 0]
