"""Run the F0-F5 encoder series and aggregate matched comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALL_VARIANTS = [
    "f0_baseline", "f1_robust_state", "f2_temporal_queries",
    "f3_factorized", "f4_relation_tokens", "f5_dual_graph",
]


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"))
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS, choices=ALL_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--val-scenes", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--python-executable", type=Path, default=None,
        help="Python used for training. By default a PyTorch/CUDA environment is detected.",
    )
    parser.add_argument("--output-root", default="results/front_encoder_series")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def select_training_python(requested: Path | None, device: str) -> Path:
    """Find an interpreter with torch, preferring CUDA when requested."""
    candidates = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend(
        [
            Path(sys.executable),
            Path(r"C:\coding_toobox\envs\astra2\python.exe"),
            Path(r"C:\coding_toobox\envs\astra\python.exe"),
            Path(r"C:\coding_toobox\python.exe"),
        ]
    )
    require_cuda = device.lower().startswith("cuda")
    seen = set()
    failures = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if str(candidate).lower() in seen or not candidate.is_file():
            continue
        seen.add(str(candidate).lower())
        probe = (
            "import torch,yaml,numpy; "
            "assert (not " + repr(require_cuda) + ") or torch.cuda.is_available(); "
            "print(torch.__version__, torch.cuda.is_available())"
        )
        completed = subprocess.run(
            [str(candidate), "-c", probe], capture_output=True, text=True
        )
        if completed.returncode == 0:
            print(f"[python] {candidate} | torch {completed.stdout.strip()}")
            return candidate
        failures.append(f"{candidate}: {completed.stderr.strip().splitlines()[-1:]}")
    requirement = "CUDA PyTorch" if require_cuda else "PyTorch"
    raise RuntimeError(
        f"No Python interpreter with {requirement} was found. "
        "Pass --python-executable explicitly. Probes: " + "; ".join(failures)
    )


def flat_row(payload: dict) -> dict:
    evaluation = payload["evaluation"]
    original, oracle, fused = evaluation["original"], evaluation["oracle"], evaluation.get("fused")
    row = {
        "variant": payload["variant"], "seed": payload["seed"],
        "val_scene": payload["validation_scene_id"], "epochs": payload["epochs"],
        "parameters": payload["locked_backend"]["parameter_counts"]["total"],
        "decoder_parameters": payload["locked_backend"]["parameter_counts"]["decoder"],
        "original_ade": original["ade"], "original_fde": original["fde"],
        "min_ade_20": oracle["min_ade_20"], "same_path_fde": oracle["same_path_fde"],
        "min_fde_20": oracle["min_fde_20"], "hit_best_ade": original["hit_best_ade"],
        "regret": original["regret"], "elapsed_seconds": payload["elapsed_seconds"],
    }
    row["fused_ade"] = fused["ade"] if fused else None
    row["fused_fde"] = fused["fde"] if fused else None
    row["fused_hit_best_ade"] = fused["hit_best_ade"] if fused else None
    row["fused_regret"] = fused["regret"] if fused else None
    return row


def run() -> None:
    args = parse()
    project = args.project_root.resolve()
    training_python = select_training_python(args.python_executable, args.device)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project / output_root
    for scene in args.val_scenes:
        for seed in args.seeds:
            for variant in args.variants:
                command = [
                    str(training_python), str(HERE / "train_front_variant.py"),
                    "--project-root", str(project), "--variant", variant,
                    "--seed", str(seed), "--val-scene-id", str(scene),
                    "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
                    "--batch-size", str(args.batch_size), "--device", args.device,
                    "--output-root", str(output_root),
                ]
                if args.force:
                    command.append("--force")
                log_dir = output_root / variant / f"seed_{seed}" / f"val_scene_{scene}"
                log_dir.mkdir(parents=True, exist_ok=True)
                print(f"[run] {variant} seed={seed} val_scene={scene}")
                with (log_dir / "console.log").open("w", encoding="utf-8") as log:
                    completed = subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f"Failed: {variant} seed={seed} scene={scene}; see {log_dir / 'console.log'}")

    rows = []
    for path in output_root.glob("*/seed_*/val_scene_*/metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["variant"] in args.variants and payload["seed"] in args.seeds and payload["validation_scene_id"] in args.val_scenes:
            rows.append(flat_row(payload))
    baseline = {(r["seed"], r["val_scene"]): r for r in rows if r["variant"] == "f0_baseline"}
    for row in rows:
        base = baseline.get((row["seed"], row["val_scene"]))
        row["delta_original_ade_vs_f0"] = row["original_ade"] - base["original_ade"] if base else None
        row["delta_original_fde_vs_f0"] = row["original_fde"] - base["original_fde"] if base else None
        row["delta_fused_ade_vs_f0"] = (
            row["fused_ade"] - base["fused_ade"] if base and row["fused_ade"] is not None and base["fused_ade"] is not None else None
        )
    rows.sort(key=lambda r: (r["val_scene"], r["seed"], ALL_VARIANTS.index(r["variant"])))
    summary_dir = output_root / "batch_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "matched_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows: writer.writeheader(); writer.writerows(rows)
    (summary_dir / "matched_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {summary_dir / 'matched_results.csv'}")


if __name__ == "__main__":
    run()
