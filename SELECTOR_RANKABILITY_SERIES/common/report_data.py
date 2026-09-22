"""
Report Data Layer for Rankability Series.

Loads verified JSON/CSV artifacts directly from disk to prevent hardcoded numbers
in generated Markdown reports and orchestrators.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json_safe(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_safe(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        return list(reader)


def load_s1b_summary(base_dir: Path) -> Dict[str, Any]:
    p = base_dir / "S1_RANKABILITY_AUDIT" / "results" / "oracle_margin_summary.json"
    return load_json_safe(p)


def load_s4_summary(base_dir: Path) -> Dict[str, Any]:
    p = base_dir / "S4_REGRET_GATE" / "results" / "regret_summary.json"
    return load_json_safe(p)


def load_s4_budget_curve(base_dir: Path) -> List[Dict[str, str]]:
    p = base_dir / "S4_REGRET_GATE" / "results" / "conditional_budget_curve.csv"
    return load_csv_safe(p)


def load_s1a_v2_results(base_dir: Path) -> List[Dict[str, str]]:
    p = base_dir / "S1_RANKABILITY_AUDIT" / "results_v2" / "train_test_rankability_v2.csv"
    return load_csv_safe(p)


def load_s2_summary(base_dir: Path) -> List[Dict[str, str]]:
    p = base_dir / "S2_LATENT_PROBE" / "results" / "latent_probe_summary.csv"
    return load_csv_safe(p)


def load_s5_distribution(base_dir: Path) -> List[Dict[str, str]]:
    p = base_dir / "S5_TOPK_AUDIT" / "results" / "native_oracle_rank_distribution.csv"
    return load_csv_safe(p)


def load_s5_near_oracle_recall(base_dir: Path) -> List[Dict[str, str]]:
    p = base_dir / "S5_TOPK_AUDIT" / "results" / "near_oracle_recall_at_k.csv"
    return load_csv_safe(p)


def load_s5_topk_metrics(base_dir: Path) -> List[Dict[str, str]]:
    p = base_dir / "S5_TOPK_AUDIT" / "results" / "topk_ranking_metrics.csv"
    return load_csv_safe(p)


def load_s1a_v3_results(base_dir: Path) -> List[Dict[str, str]]:
    p = base_dir / "S1_RANKABILITY_AUDIT" / "results_v3" / "train_test_rankability_v3.csv"
    return load_csv_safe(p)
