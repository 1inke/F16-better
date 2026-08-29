"""Run MA1 on General/Challenge seed1 and compare with the exact matched B0."""

from __future__ import annotations
import argparse, csv, json, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
CSRS = WORKSPACE / "COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"
SOURCE_ROOT = WORKSPACE / "INTERACTION_RESIDUAL_SUPERVISION_SERIES" / "results"
BASELINE_ROOT = CSRS / "results"
SIDECAR = CSRS / "teacher_cache" / "sample_index_encoded.npy"
RESULTS = ROOT / "results"
DIAGNOSTIC = ROOT / "run_ma1_diagnostic.py"
SPLITS = {"general": ("stratified_random", .05), "challenge": ("motion_challenge", .10)}
METRICS = ("min_ade_20", "min_fde_20", "same_path_fde", "top1_ade", "top1_fde")


def source(split, seed):
    return SOURCE_ROOT / split / "baseline" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"


def run_dir(split, seed):
    return RESULTS / split / "ma1_interaction_aware_top5" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def baseline_dir(split, seed):
    return BASELINE_ROOT / split / "baseline" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"


def command(args, split):
    mode, fraction = SPLITS[split]
    return [
        str(PYTHON), "-u", str(TRAINER), "--project-root", str(PROJECT),
        "--config-file", str(PROJECT / "configs" / "motif_rbf_univ.yaml"),
        "--data-path", str(DATA / "eth_data_train.npy"), "--num-path", str(DATA / "eth_num_train.npy"),
        "--sample-scene-path", str(SIDECAR), "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", "ma1_interaction_aware_top5", "--decoder-module", str(ROOT / "iama_models.py"),
        "--seed", str(args.seed), "--val-scene-id", "-1", "--validation-fraction", str(fraction),
        "--validation-seed", str(args.validation_seed), "--validation-split-mode", mode,
        "--epochs", str(args.epochs), "--val-freq", str(args.val_freq), "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate), "--device", args.device, "--max-agents", "9",
        "--graph-top-k", "4", "--graph-radius", "6.0", "--scheduler", "monotonic_cosine",
        "--warmup-epochs", "1", "--output-root", str(RESULTS / split / "ma1_interaction_aware_top5"),
        "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
        "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995", "--focal-heading-normalize",
    ]


def read(path, name, split, seed):
    raw = json.loads(path.read_text(encoding="utf-8")); ev = raw["evaluation_best_selected"]
    pc = raw["locked_backend"]["parameter_counts"]
    return {"experiment": name, "split": split, "seed": seed,
            "min_ade_20": ev["oracle"]["min_ade_20"], "min_fde_20": ev["oracle"]["min_fde_20"],
            "same_path_fde": ev["oracle"]["same_path_fde"], "top1_ade": ev["original"]["ade"],
            "top1_fde": ev["original"]["fde"], "parameters": pc["total"],
            "trainable_parameters": pc["trainable"], "elapsed_seconds": raw["elapsed_seconds"]}


def summarize(seed, splits):
    rows, paired = [], []
    for split in splits:
        bp, mp = baseline_dir(split, seed) / "metrics.json", run_dir(split, seed) / "metrics.json"
        if bp.is_file(): rows.append(read(bp, "baseline_reused_exact_match", split, seed))
        if mp.is_file(): rows.append(read(mp, "ma1_interaction_aware_top5", split, seed))
        b = next((r for r in rows if r["split"] == split and r["experiment"].startswith("baseline")), None)
        m = next((r for r in rows if r["split"] == split and r["experiment"].startswith("ma1")), None)
        if b and m: paired.append({"split": split, "percent_change": {k: 100 * (m[k] / b[k] - 1) for k in METRICS}})
    result = {"description": "MA1 interaction-aware Top-5 soft mode allocation",
              "baseline_policy": "reuse exact CSRS1 B0", "rows": rows, "paired_comparisons": paired}
    (ROOT / "seed1_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (ROOT / "all_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return result


def run_logged(cmd, path, env=None, cwd=ROOT):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
        if proc.wait(): raise RuntimeError(f"failed; see {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--val-freq", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--diagnostic-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=.00025); parser.add_argument("--validation-seed", type=int, default=20260828)
    parser.add_argument("--splits", nargs="+", choices=("all", *SPLITS), default=["all"])
    parser.add_argument("--force", action="store_true"); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-diagnostic", action="store_true"); args = parser.parse_args()
    splits = list(SPLITS) if "all" in args.splits else args.splits
    required = [PYTHON, TRAINER, DIAGNOSTIC, ROOT / "iama_models.py", SIDECAR,
                DATA / "eth_data_train.npy", DATA / "eth_num_train.npy"]
    for path in required:
        if not path.is_file(): raise FileNotFoundError(path)
    for split in splits:
        for path in (source(split, args.seed), baseline_dir(split, args.seed) / "metrics.json"):
            if not path.is_file(): raise FileNotFoundError(path)
        dest = run_dir(split, args.seed); cmd = command(args, split)
        if args.force: cmd.append("--force")
        if args.dry_run: print("[dry-run] " + json.dumps(cmd)); continue
        if (dest / "metrics.json").is_file() and not args.force:
            print(f"[skip] {split}/MA1", flush=True); continue
        env = os.environ.copy(); env["IAMA_SOURCE_CHECKPOINT"] = str(source(split, args.seed))
        print(f"[run] {split}/MA1 seed={args.seed}", flush=True)
        run_logged(cmd, dest / "console.log", env, cwd=PROJECT); summarize(args.seed, splits)
    if args.dry_run: print("[done] dry-run passed"); return
    result = summarize(args.seed, splits); print(f"[done] rows={len(result['rows'])}", flush=True)
    if not args.skip_diagnostic and set(splits) == set(SPLITS):
        run_logged([str(PYTHON), "-u", str(DIAGNOSTIC), "--device", args.device,
                    "--batch-size", str(args.diagnostic_batch_size), "--seed", str(args.seed), "--splits", "all"],
                   ROOT / "diagnostic.log")
    elif not args.skip_diagnostic: print("[diagnostic] skipped: requires both splits")


if __name__ == "__main__":
    main()
