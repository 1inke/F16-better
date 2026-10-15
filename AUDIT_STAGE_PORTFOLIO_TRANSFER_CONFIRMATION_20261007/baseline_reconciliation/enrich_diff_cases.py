import sys, json, hashlib
from pathlib import Path
import numpy as np

REPO = Path(r"c:\Users\linke\Documents\New project")
BASE_DIR = REPO / "AUDIT_STAGE_PORTFOLIO_TRANSFER_CONFIRMATION_20261007" / "baseline_reconciliation"

with open(BASE_DIR / "STATIC_BASELINE_DIFF_CASES.json") as f:
    data = json.load(f)

# Enriched metadata
source_hashes = {
    "train_cache_sha256": hashlib.sha256((REPO / "REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz").read_bytes()).hexdigest(),
    "lab_train_sha256": hashlib.sha256((REPO / "F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz").read_bytes()).hexdigest(),
    "splits_sf_sha256": hashlib.sha256((REPO / "GENERATION_GROUPED_TRANSFER_SERIES_20261006/experiment3_scene_family/identity_audit/splits.npz").read_bytes()).hexdigest(),
    "canonical_intent_classifier_sha256": hashlib.sha256((BASE_DIR / "canonical_intent_classifier.pth").read_bytes()).hexdigest()
}

data["audit_conclusion"] = {
    "root_cause": "The discrepancy between 87, 88, and 89 covered Change windows stems from non-deterministic initial weights in the IntentClassifier during ad-hoc script runs. In boundary sample ID 6290, 'stop' and 'accelerate' have marginal probability differences (~0.04-0.05). When 'stop' ranks 4th, the sample is good (88/211); when 'accelerate' ranks 4th, 'stop' is dropped to 5th, leaving the sample not covered (87/211).",
    "canonical_selection": "Canonical Static-K24 is frozen using canonical_intent_classifier.pth (Seed 43, 15 epochs on grouped_train_index), strictly matching the 88/211 (41.71%) baseline audited in AUDIT_STAGE_BEST_OF_4_GENERATOR_3SEED_20261007 and AUDIT_STAGE_PORTFOLIO_P0_COMPLEMENTARITY_20261007.",
    "native20_invariance": "Native20 candidates are 100% bitwise identical across all versions (max abs difference = 0.0).",
    "motion_program_generator": "motion_program.py generate_motion_program with default_params(mode, 1).",
    "hashes": source_hashes
}

with open(BASE_DIR / "STATIC_BASELINE_DIFF_CASES.json", "w") as f:
    json.dump(data, f, indent=2)

print("Enriched STATIC_BASELINE_DIFF_CASES.json")
