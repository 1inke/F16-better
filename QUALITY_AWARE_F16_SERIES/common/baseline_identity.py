"""
Baseline Identity management for Quality-Aware F16 Series.

Enforces Rule 2 (Baseline Identity Lock):
Records complete provenance metadata into baseline_identity.json.
Provides equality checking to prevent illicit cross-version comparisons.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def create_baseline_identity(
    architecture: str,
    variant: str,
    decoder: str,
    source_checkpoint: str,
    seed: int,
    fold: str,
    candidate_count: int,
    total_parameter_count: int,
    trainable_parameter_count: int,
    original_loss_components: List[str],
    quality_loss_type: str,
    lambda_q: float,
    quality_tau: float,
    optimizer_name: str,
    learning_rate_groups: Dict[str, float],
    epochs: int,
    batch_size: int,
    validation_split: str,
    checkpoint_criterion: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    identity = {
        "architecture": architecture,
        "variant": variant,
        "decoder": decoder,
        "source_checkpoint": str(source_checkpoint),
        "seed": seed,
        "fold": fold,
        "candidate_count": candidate_count,
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "original_loss_components": original_loss_components,
        "quality_loss_type": quality_loss_type,
        "lambda_q": lambda_q,
        "quality_tau": quality_tau,
        "optimizer": optimizer_name,
        "learning_rate_groups": learning_rate_groups,
        "epochs": epochs,
        "batch_size": batch_size,
        "validation_split": validation_split,
        "checkpoint_criterion": checkpoint_criterion,
    }
    if extra_metadata:
        identity.update(extra_metadata)
    return identity


def save_baseline_identity(path: Path, identity: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(identity, fp, indent=2, ensure_ascii=False)


def verify_identities_match(id_a: Dict[str, Any], id_b: Dict[str, Any], ignore_keys: Optional[List[str]] = None) -> List[str]:
    """
    Returns list of mismatched fields between two baseline identities.
    """
    ignore = set(ignore_keys or ["source_checkpoint", "timestamp"])
    mismatches = []
    all_keys = set(id_a.keys()).union(set(id_b.keys()))
    for k in all_keys:
        if k in ignore:
            continue
        if id_a.get(k) != id_b.get(k):
            mismatches.append(f"{k}: '{id_a.get(k)}' != '{id_b.get(k)}'")
    return mismatches
