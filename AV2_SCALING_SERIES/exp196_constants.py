"""Dependency-free paths and experiment matrix for EXP-196."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
EXP195_ROOT = WORKSPACE / "AV2_TEMPORAL_BRIDGE_SERIES"
ASTRA_PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
ASTRA_PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
PREP_PYTHON = Path(r"C:\coding_toobox\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DECODER_SOURCE = WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" / "grouped_mode_template_models.py"
A1_CHECKPOINT = (
    EXP195_ROOT / "results" / "a1" / "f16_temporal_social_cross"
    / "seed_1" / "val_scene_-1" / "best_selected.pth"
)
BASE_CONFIG = EXP195_ROOT / "configs" / "exp195a_a1_2p5hz.yaml"
EXP195_MANIFESTS = EXP195_ROOT / "manifests"
EXP195_PREPARED = EXP195_ROOT / "prepared"

REFERENCE_HZ = 2.5
FUTURE_HZ = 2.5
FUTURE_STEPS = 12
FUTURE_INDICES = tuple(range(53, 98, 4))
MAX_SOURCE_AGENTS = 24

VARIANTS = {
    # EXP-196A/B: common A1 checkpoint and equal four-epoch continuation.
    "a1c": {"stage": "fair", "history_hz": 2.5, "history_steps": 8, "max_agents": 9, "top_k": 4, "data_source": "exp195_a1"},
    "a2": {"stage": "fair", "history_hz": 5.0, "history_steps": 16, "max_agents": 9, "top_k": 4, "data_source": "exp195_a2"},
    "a3": {"stage": "fair", "history_hz": 10.0, "history_steps": 32, "max_agents": 9, "top_k": 4, "data_source": "exp195_a3"},
    # EXP-196C: fixed 10 Hz and common full-history target pool.
    "l1": {"stage": "long", "history_hz": 10.0, "history_steps": 16, "max_agents": 9, "top_k": 4, "data_source": "temporal"},
    "l2": {"stage": "long", "history_hz": 10.0, "history_steps": 32, "max_agents": 9, "top_k": 4, "data_source": "temporal"},
    "l3": {"stage": "long", "history_hz": 10.0, "history_steps": 40, "max_agents": 9, "top_k": 4, "data_source": "temporal"},
    "l4": {"stage": "long", "history_hz": 10.0, "history_steps": 50, "max_agents": 9, "top_k": 4, "data_source": "temporal"},
    # EXP-196D: fixed T=32; d5/d6 isolate top-k at matched N.
    "d1": {"stage": "dense", "history_hz": 10.0, "history_steps": 32, "max_agents": 5, "top_k": 4, "data_source": "dense"},
    "d2": {"stage": "dense", "history_hz": 10.0, "history_steps": 32, "max_agents": 9, "top_k": 4, "data_source": "dense"},
    "d3": {"stage": "dense", "history_hz": 10.0, "history_steps": 32, "max_agents": 16, "top_k": 4, "data_source": "dense"},
    "d4": {"stage": "dense", "history_hz": 10.0, "history_steps": 32, "max_agents": 24, "top_k": 4, "data_source": "dense"},
    "d5": {"stage": "dense", "history_hz": 10.0, "history_steps": 32, "max_agents": 16, "top_k": 8, "data_source": "dense"},
    "d6": {"stage": "dense", "history_hz": 10.0, "history_steps": 32, "max_agents": 24, "top_k": 8, "data_source": "dense"},
    # EXP-196E: joint stress points.
    "j1": {"stage": "joint", "history_hz": 10.0, "history_steps": 50, "max_agents": 16, "top_k": 4, "data_source": "joint"},
    "j2": {"stage": "joint", "history_hz": 10.0, "history_steps": 50, "max_agents": 24, "top_k": 4, "data_source": "joint"},
}

STAGE_VARIANTS = {
    "fair": ("a1c", "a2", "a3"),
    "long": ("l1", "l2", "l3", "l4"),
    "dense": ("d1", "d2", "d3", "d4", "d5", "d6"),
    "joint": ("j1", "j2"),
}
LOW_COST_VARIANTS = ("a1c", "a2", "a3", "l2", "l4", "d2", "d4", "j2")


def checkpoint_for(name: str, seed: int) -> Path:
    return ROOT / "results" / name / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"


def config_for(name: str) -> Path:
    return ROOT / "generated_configs" / f"exp196_{name}.yaml"


def data_paths(name: str, split: str) -> tuple[Path, Path]:
    source = VARIANTS[name]["data_source"]
    if str(source).startswith("exp195_"):
        old = str(source).split("_", 1)[1]
        return (
            EXP195_PREPARED / f"{split}_{old}_data.npy",
            EXP195_PREPARED / f"{split}_{old}_num.npy",
        )
    return (
        ROOT / "prepared" / f"{split}_{name}_data.npy",
        ROOT / "prepared" / f"{split}_{name}_num.npy",
    )


def metadata_path(name: str, split: str) -> Path:
    source = VARIANTS[name]["data_source"]
    if str(source).startswith("exp195_"):
        return EXP195_MANIFESTS / f"{split}_targets_exp195.json"
    return ROOT / "prepared" / f"{split}_{source}_metadata.json"


def require_paths(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
