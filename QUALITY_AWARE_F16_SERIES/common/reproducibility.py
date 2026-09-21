"""
Reproducibility utilities for Quality-Aware F16 Training Series.
"""
from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np
import torch


def stable_seed(base_seed: int, *keys: Any) -> int:
    """
    Derives deterministic 32-bit integer seed from base_seed and string representation of keys.
    """
    raw_str = f"{base_seed}:" + ":".join(str(k) for k in keys)
    digest = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def set_deterministic_seed(seed: int) -> None:
    """
    Sets global deterministic random seeds across Python, NumPy, and PyTorch (CPU & CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
