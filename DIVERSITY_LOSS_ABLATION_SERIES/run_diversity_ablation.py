"""Run controlled EXP-173/F16 diversity-loss ablations.

Default P1 sweep (recommended first):
    weight = [0.000, 0.010, 0.025, 0.050], margin = 0.400, seed = 1

Optional P2 sweep after reviewing P1:
    margin = [0.100, 0.200, 0.300, 0.400], weight = 0.050, seed = 1

Run this script with the same Python environment used for ASTRA training.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAINER = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
ADAPTER = HERE / "diversity_ablation_adapter.py"
CONFIG = REPO / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
EXP173 = REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END"


def encode(x: float) -> str:
    return f"{x:.3f}".replace(".", "p")


def exp_name(weight: float, margin: float) -> str:
    return f"divw_{encode(weight)}_m_{encode(margin)}"


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def run_one(args, weight: float, margin: float, seed: int) -> Path:
    astra = args.astra_root.resolve()
    data_dir = astra / "datasets" / "eth_ucy" / "processed_data_diverse"
    data = data_dir / "eth_data_train.npy"
    num = data_dir / "eth_num_train.npy"
    h3 = (
        EXP173 / "results" / "eth" / "h3" / "e3" / "f16_temporal_social_cross"
        / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"
    )
    for p in [TRAINER, ADAPTER, CONFIG, data, num, h3]:
        if not p.is_file():
            raise FileNotFoundError(p)

    name = exp_name(weight, margin)
    condition_root = args.output_root.resolve() / name
    run_dir = condition_root / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"
    metrics = run_dir / "metrics.json"

    if metrics.is_file() and not args.force:
        print(f"[SKIP] {name} seed={seed}: {metrics}")
        return run_dir

    env = os.environ.copy()
    env["EXP173_H3_SOURCE_CHECKPOINT"] = str(h3)
    env["EXP173_DECODER_INIT_SEED"] = str(seed)

    cmd = [
        sys.executable, "-u", str(TRAINER),
        "--project-root", str(astra),
        "--config-file", str(CONFIG),
        "--data-path", str(data),
        "--num-path", str(num),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", name,
        "--decoder-module", str(ADAPTER),
        "--seed", str(seed),
        "--val-scene-id", "-1",
        "--validation-fraction", "0.05",
        "--validation-seed", "20260830",
        "--validation-split-mode", "scene_tail",
        "--epochs", str(args.epochs),
        "--val-freq", "5",
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--max-agents", "9",
        "--graph-top-k", "4",
        "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine",
        "--warmup-epochs", "1",
        "--learning-rate", "0.00025",
        "--output-root", str(condition_root),
        "--oracle-coverage-weight", "0.20",
        "--oracle-coverage-temperature", "0.10",
        "--oracle-coverage-fde-weight", "0.25",
        "--checkpoint-criterion", "oracle_combo",
        "--ema-decay", "0.995",
        "--focal-heading-normalize",
        "--augment-reflection",
        "--augment-speed-min", "0.9",
        "--augment-speed-max", "1.1",
        "--augment-probability", "0.5",
    ]
    if args.force:
        cmd.append("--force")

    print("\n" + "=" * 100)
    print(f"[RUN] {name} | seed={seed} | weight={weight:.3f} | margin={margin:.3f}")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, env=env, check=True, cwd=str(REPO))

    spec = {
        "experiment_name": name,
        "seed": seed,
        "diversity_weight": weight,
        "diversity_margin": margin,
        "baseline_weight": 0.05,
        "baseline_margin": 0.4,
        "only_intended_intervention": [
            "cfg.MODEL.DECODER.DIVERSITY_WEIGHT",
            "cfg.MODEL.DECODER.DIVERSITY_MARGIN",
        ],
        "h3_source_checkpoint": str(h3),
        "command": cmd,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "diversity_ablation_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--astra-root", type=Path,
        default=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=HERE / "results" / "weight_sweep",
    )
    parser.add_argument("--stage", choices=["weight", "margin"], default="weight")
    parser.add_argument("--weights", default="0.000,0.010,0.025,0.050")
    parser.add_argument("--margins", default="0.100,0.200,0.300,0.400")
    parser.add_argument("--fixed-weight", type=float, default=0.050)
    parser.add_argument("--fixed-margin", type=float, default=0.400)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.stage == "weight":
        conditions = [(w, args.fixed_margin) for w in parse_csv_floats(args.weights)]
    else:
        conditions = [(args.fixed_weight, m) for m in parse_csv_floats(args.margins)]

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seed in seeds:
        for weight, margin in conditions:
            run_dir = run_one(args, weight, margin, seed)
            manifest.append({
                "seed": seed,
                "weight": weight,
                "margin": margin,
                "run_dir": str(run_dir),
            })

    (args.output_root / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\n[DONE] Manifest: {args.output_root / 'sweep_manifest.json'}")


if __name__ == "__main__":
    main()
