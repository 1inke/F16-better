"""Dependency-free paths and protocol constants for EXP-195 launch scripts."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
ASTRA_PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
ASTRA_PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
PREP_PYTHON = Path(r"C:\coding_toobox\python.exe")
FRONT_DIR = WORKSPACE / "FRONT_ENCODER_SERIES"
TRAINER = FRONT_DIR / "train_front_variant.py"
DECODER_SOURCE = WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" / "grouped_mode_template_models.py"
DEFAULT_ETH_CHECKPOINT = (
    WORKSPACE / "FORMAL_ENCODER_ABLATION_SERIES" / "EXP_154_FORMAL_ENCODER_ABLATION"
    / "results" / "general" / "d_f16_full_temporal"
    / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1"
    / "best_selected.pth"
)
REFERENCE_HZ = 2.5
FUTURE_HZ = 2.5
FUTURE_STEPS = 12
MAX_AGENTS = 9
CONFIGS = {
    "a1": {"history_hz": 2.5, "history_steps": 8},
    "a2": {"history_hz": 5.0, "history_steps": 16},
    "a3": {"history_hz": 10.0, "history_steps": 32},
}
GENERATED_CONFIGS = {
    name: ROOT / "generated_configs" / f"exp195a_{name}.yaml" for name in CONFIGS
}


def require_paths(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
