"""Create matched EXP-196 configs from the locked EXP-195 config."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from exp196_constants import BASE_CONFIG, FUTURE_STEPS, ROOT, VARIANTS


def materialize(output_dir: Path | None = None) -> dict[str, Path]:
    output_dir = output_dir or (ROOT / "generated_configs")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    paths = {}
    for name, spec in VARIANTS.items():
        config = copy.deepcopy(base)
        hz = float(spec["history_hz"])
        config["SUBSET"] = f"av2_exp196_{name}"
        config["PREDICTION"]["OBS_TIME"] = int(spec["history_steps"]) / hz
        # Keep the decoder at exactly 12 outputs; labels remain 0.4...4.8 s.
        config["PREDICTION"]["PRED_TIME"] = FUTURE_STEPS / hz
        config["DATA"]["FREQUENCY"] = hz
        config["DATA"]["MAX_AGENTS"] = int(spec["max_agents"])
        config["MODEL"]["GRAPH"]["TOP_K"] = int(spec["top_k"])
        config["MODEL"]["GRAPH"]["HISTORY_GAMMA"] = 0.25 * 2.5 / hz
        path = output_dir / f"exp196_{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths[name] = path
    return paths


if __name__ == "__main__":
    for key, value in materialize().items():
        print(f"{key}: {value}")
