"""
Protocol Identity and Verification Utilities for F16 Confidence Refinement V2.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def compute_sha256(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path).resolve()
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_protocol_identity(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, tuple[Any, Any]]:
    """
    Compares baseline and candidate protocol identities.
    Allowed differences are only objective-specific hyperparameters.
    Any other difference raises an unexpected mismatch.
    """
    allowed_diff = {
        "decoder_module",
        "decoder_module_sha256",
        "objective_mode",
        "fixed_tau",
        "hard_weight",
        "command",
        "output_root",
        "run_dir",
        "elapsed_seconds",
    }

    unexpected = {}
    for key, base_val in baseline.items():
        if key in allowed_diff:
            continue
        cand_val = candidate.get(key)
        if base_val != cand_val:
            unexpected[key] = (base_val, cand_val)

    for key, cand_val in candidate.items():
        if key in allowed_diff or key in baseline:
            continue
        unexpected[key] = (None, cand_val)

    return unexpected
