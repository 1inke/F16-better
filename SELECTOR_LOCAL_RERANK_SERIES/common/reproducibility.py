"""
Deterministic seed utilities for reproducible ranking experiments.
"""
from __future__ import annotations

import hashlib
import random
import numpy as np
import torch


def stable_seed(base_seed: int, *parts: str | int) -> int:
    """Derives a deterministic 31-bit integer seed from a base seed and variable parts."""
    raw = "::".join([str(base_seed), *[str(x) for x in parts]])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def set_deterministic_seed(seed: int) -> None:
    """Sets deterministic seed across random, numpy, torch CPU and CUDA with cuDNN deterministic flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
