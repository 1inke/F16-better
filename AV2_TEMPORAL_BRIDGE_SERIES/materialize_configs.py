"""Materialize complete A1/A2/A3 configs without duplicating the locked YAML."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
BASE = CONFIG_DIR / "exp195a_a1_2p5hz.yaml"
SETTINGS = {
    "a1": (2.5, 3.2, 4.8, 0.25),
    "a2": (5.0, 3.2, 2.4, 0.125),
    "a3": (10.0, 3.2, 1.2, 0.0625),
}


def materialize(output_dir: Path | None = None) -> dict[str, Path]:
    output_dir = output_dir or (ROOT / "generated_configs")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    paths = {}
    for name, (frequency, obs_time, pred_time, gamma) in SETTINGS.items():
        config = copy.deepcopy(base)
        config["DATA"]["FREQUENCY"] = frequency
        config["PREDICTION"]["OBS_TIME"] = obs_time
        # This makes the model tensor length 12; labels still represent 4.8 s.
        config["PREDICTION"]["PRED_TIME"] = pred_time
        config["MODEL"]["GRAPH"]["HISTORY_GAMMA"] = gamma
        path = output_dir / f"exp195a_{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths[name] = path
    return paths


if __name__ == "__main__":
    for key, value in materialize().items():
        print(f"{key}: {value}")
