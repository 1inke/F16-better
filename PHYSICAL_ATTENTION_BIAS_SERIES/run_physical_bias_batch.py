"""Run the controlled time-resolved physical attention-bias series."""
from __future__ import annotations
import argparse, csv, json, os, statistics, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
CUSTOM = ROOT / "physical_bias_models.py"
DECODER = ROOT / "current_mode_anchor_adapter.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
SOURCE_ROOT = (
    WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" /
    "EXP_146B_ANCHORED_MODE_TEMPLATE" / "results"
)
RESULTS_ROOT = ROOT / "results_current_mode_anchor"

EXPERIMENTS = {
    "baseline": None,
    "pb1_add": "pb1_time_bias_add",
    "pb2_replace": "pb2_time_bias_replace",
    "pb3_bias_only": "pb3_bias_only_interaction",
}
SPLITS = {"general": ("stratified_random", 0.05), "challenge": ("motion_challenge", 0.10)}
METRICS = ("min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde","elapsed_seconds")

def result_dir(name, split, seed):
    return RESULTS_ROOT / split / name / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"

def source_checkpoint(split, seed):
    return (
        SOURCE_ROOT / split / "f16_temporal_social_cross" / f"seed_{seed}" /
        "val_scene_-1" / "best_selected.pth"
    )

def command_for(args, name, split, seed):
    split_mode, validation_fraction = SPLITS[split]
    cmd = [
        str(PYTHON), "-u", str(TRAINER),
        "--project-root", str(PROJECT),
        "--config-file", str(PROJECT / "configs" / "motif_rbf_univ.yaml"),
        "--data-path", str(DATA / "eth_data_train.npy"),
        "--num-path", str(DATA / "eth_num_train.npy"),
        "--variant", "f16_temporal_social_cross",
        "--decoder-experiment", "current_mode_anchor_template",
        "--decoder-module", str(DECODER),
        "--seed", str(seed), "--val-scene-id", "-1",
        "--validation-fraction", str(validation_fraction),
        "--validation-seed", str(args.validation_seed),
        "--validation-split-mode", split_mode,
        "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
        "--batch-size", str(args.batch_size), "--device", args.device,
        "--max-agents", "9", "--graph-top-k", str(args.graph_top_k), "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine", "--warmup-epochs", "1",
        "--learning-rate", str(args.learning_rate),
        "--output-root", str(RESULTS_ROOT / split / name),
        "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
        "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
        "--focal-heading-normalize", "--augment-reflection",
        "--augment-speed-min", "0.9", "--augment-speed-max", "1.1", "--augment-probability", "0.5",
    ]
    encoder_experiment = EXPERIMENTS[name]
    if encoder_experiment is not None:
        cmd += ["--encoder-experiment", encoder_experiment, "--encoder-module", str(CUSTOM)]
    if args.force: cmd.append("--force")
    return cmd

def metric_row(path, experiment, split, seed):
    raw = json.loads(path.read_text(encoding="utf-8"))
    ev = raw["evaluation_best_selected"]
    return {
        "experiment": experiment, "split": split, "seed": seed,
        "min_ade_20": ev["oracle"]["min_ade_20"],
        "min_fde_20": ev["oracle"]["min_fde_20"],
        "same_path_fde": ev["oracle"]["same_path_fde"],
        "top1_ade": ev["original"]["ade"], "top1_fde": ev["original"]["fde"],
        "parameters": raw["locked_backend"]["parameter_counts"]["total"],
        "elapsed_seconds": raw["elapsed_seconds"],
    }

def summarize(seeds, names, splits):
    rows = []
    for split in splits:
        for name in names:
            for seed in seeds:
                p = result_dir(name, split, seed) / "metrics.json"
                if p.is_file(): rows.append(metric_row(p, name, split, seed))

    aggregate, paired = [], []
    for split in splits:
        baseline = {int(r["seed"]): r for r in rows if r["split"] == split and r["experiment"] == "baseline"}
        for name in names:
            selected = sorted((r for r in rows if r["split"] == split and r["experiment"] == name), key=lambda r: int(r["seed"]))
            if not selected: continue
            aggregate.append({
                "split": split, "experiment": name, "parameters": int(selected[0]["parameters"]),
                "metrics": {m: {
                    "mean": statistics.fmean(float(r[m]) for r in selected),
                    "std": statistics.stdev(float(r[m]) for r in selected) if len(selected) > 1 else 0.0,
                    "values": [float(r[m]) for r in selected],
                } for m in METRICS},
            })
            if name != "baseline" and baseline:
                changes = {}
                for metric in METRICS[:-1]:
                    vals = [
                        100.0 * (float(r[metric]) / float(baseline[int(r["seed"])][metric]) - 1.0)
                        for r in selected if int(r["seed"]) in baseline
                    ]
                    if vals:
                        changes[metric] = {
                            "mean": statistics.fmean(vals),
                            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                            "values": vals,
                        }
                paired.append({"split": split, "experiment": name, "paired_percent_change_vs_baseline": changes})

    result = {
        "description":"Physical attention bias on locked EXP-146B mode/three-anchor decoder",
        "seeds":seeds, "experiments":names, "splits":splits,
        "rows":rows, "aggregate":aggregate, "paired_comparisons":paired,
    }
    (ROOT / "multiseed_summary_current_mode_anchor.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (ROOT / "all_runs_current_mode_anchor.csv").open("w", encoding="utf-8-sig", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--val-freq", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.00025)
    p.add_argument("--seeds", type=int, nargs="+", default=[1,2,3])
    p.add_argument("--validation-seed", type=int, default=20260828)
    p.add_argument("--graph-top-k", type=int, default=4)
    p.add_argument("--variants", nargs="+", choices=("all", *EXPERIMENTS), default=["all"])
    p.add_argument("--splits", nargs="+", choices=("all", *SPLITS), default=["all"])
    p.add_argument("--force", action="store_true"); p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for path in (PYTHON, TRAINER, CUSTOM, DECODER, DATA/"eth_data_train.npy", DATA/"eth_num_train.npy"):
        if not path.is_file(): raise FileNotFoundError(path)
    names = list(EXPERIMENTS) if "all" in args.variants else args.variants
    splits = list(SPLITS) if "all" in args.splits else args.splits
    for split in splits:
        for name in names:
            for seed in args.seeds:
                source = source_checkpoint(split, seed)
                if not source.is_file(): raise FileNotFoundError(source)
                dest = result_dir(name, split, seed); metrics = dest/"metrics.json"
                if metrics.is_file() and not args.force:
                    print(f"[skip] {split} {name} seed={seed}", flush=True); continue
                cmd = command_for(args, name, split, seed)
                if args.dry_run:
                    print("[dry-run] source="+str(source)+" command="+json.dumps(cmd), flush=True); continue
                dest.mkdir(parents=True, exist_ok=True)
                print(f"[run] {split} {name} seed={seed}", flush=True)
                env = os.environ.copy()
                env["PHYSICAL_BIAS_SOURCE_CHECKPOINT"] = str(source)
                proc = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
                assert proc.stdout is not None
                with (dest/"console.log").open("w", encoding="utf-8") as log:
                    for line in proc.stdout:
                        print(f"[{split}/{name}] {line}", end="", flush=True); log.write(line); log.flush()
                if proc.wait(): raise RuntimeError(f"Experiment failed; see {dest/'console.log'}")
                summarize(args.seeds, names, splits)
    if args.dry_run: print("[done] dry-run passed", flush=True)
    else:
        result = summarize(args.seeds, names, splits)
        print(f"[done] completed {len(result['rows'])}/{len(names)*len(splits)*len(args.seeds)}; summary={ROOT/'multiseed_summary_current_mode_anchor.json'}", flush=True)

if __name__ == "__main__":
    main()
