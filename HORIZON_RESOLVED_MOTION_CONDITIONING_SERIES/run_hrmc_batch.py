"""Train P1 proposal residual and C1 correction residual on official five folds."""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent; WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER = WORKSPACE / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
CONFIG = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
MODEL = ROOT / "motion_conditioned_decoder_models.py"; EVALUATOR = ROOT / "evaluate_hrmc.py"
SOURCE = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "EXP_156B_FOLD_LOCAL_CURRENT_DECODER" / "results"
RESULTS = ROOT / "results"; FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
VARIANTS = {"p1": "p1_horizon_motion_proposal_residual", "c1": "c1_horizon_motion_correction_residual"}

def source_dir(fold, stage, seed): return SOURCE / fold / stage / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"
def run_dir(variant, fold, seed): return RESULTS / variant / fold / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1"
def command(args, variant, fold):
    return [str(PYTHON), "-u", str(TRAINER), "--project-root", str(PROJECT), "--config-file", str(CONFIG),
        "--data-path", str(DATA / f"{fold}_data_train.npy"), "--num-path", str(DATA / f"{fold}_num_train.npy"),
        "--test-data-path", str(DATA / f"{fold}_data_test.npy"), "--test-num-path", str(DATA / f"{fold}_num_test.npy"),
        "--variant", "f16_temporal_social_cross", "--decoder-experiment", VARIANTS[variant], "--decoder-module", str(MODEL),
        "--seed", str(args.seed), "--val-scene-id", "-1", "--validation-fraction", "0.05", "--validation-seed", "20260830",
        "--validation-split-mode", "scene_tail", "--epochs", str(args.epochs), "--val-freq", str(args.val_freq),
        "--batch-size", str(args.batch_size), "--device", args.device, "--max-agents", "9", "--graph-top-k", "4", "--graph-radius", "6.0",
        "--scheduler", "monotonic_cosine", "--warmup-epochs", "1", "--learning-rate", str(args.learning_rate),
        "--output-root", str(RESULTS / variant / fold), "--oracle-coverage-weight", "0.20", "--oracle-coverage-temperature", "0.10",
        "--oracle-coverage-fde-weight", "0.25", "--checkpoint-criterion", "oracle_combo", "--ema-decay", "0.995",
        "--focal-heading-normalize", "--augment-reflection", "--augment-speed-min", "0.9", "--augment-speed-max", "1.1", "--augment-probability", "0.5"]

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--device",default="cuda:0");parser.add_argument("--folds",nargs="+",choices=("all",*FOLDS),default=["all"])
    parser.add_argument("--variants",nargs="+",choices=("all",*VARIANTS),default=["all"]);parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--epochs",type=int,default=20);parser.add_argument("--batch-size",type=int,default=64);parser.add_argument("--val-freq",type=int,default=5)
    parser.add_argument("--learning-rate",type=float,default=2.5e-4);parser.add_argument("--dry-run",action="store_true");parser.add_argument("--force",action="store_true");args=parser.parse_args()
    folds=FOLDS if "all" in args.folds else tuple(args.folds);variants=tuple(VARIANTS) if "all" in args.variants else tuple(args.variants)
    required=[PYTHON,TRAINER,CONFIG,MODEL,EVALUATOR]
    for fold in folds: required += [source_dir(fold,"h3",args.seed)/"best_selected.pth",source_dir(fold,"v2",args.seed)/"best_selected.pth",DATA/f"{fold}_data_train.npy",DATA/f"{fold}_num_train.npy",DATA/f"{fold}_data_test.npy",DATA/f"{fold}_num_test.npy"]
    missing=[str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError("Missing:\n"+"\n".join(missing))
    for variant in variants:
      for fold in folds:
        destination=run_dir(variant,fold,args.seed);metrics=destination/"metrics.json";cmd=command(args,variant,fold)
        if args.force: cmd.append("--force")
        if args.dry_run: print(f"[dry-run] {variant}/{fold} "+json.dumps(cmd));continue
        if metrics.is_file() and not args.force: print(f"[skip] {variant}/{fold}");continue
        env=os.environ.copy();env["HRMC_H3_CHECKPOINT"]=str(source_dir(fold,"h3",args.seed)/"best_selected.pth");env["HRMC_V2_CHECKPOINT"]=str(source_dir(fold,"v2",args.seed)/"best_selected.pth")
        destination.mkdir(parents=True,exist_ok=True);print(f"[run] {variant}/{fold}",flush=True)
        with (destination/"console.log").open("w",encoding="utf-8") as log:
          job=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1);assert job.stdout
          for line in job.stdout: print(f"[{variant}/{fold}] {line}",end="",flush=True);log.write(line);log.flush()
        if job.wait(): raise RuntimeError(f"failed: {destination/'console.log'}")
    if args.dry_run: print("[done] dry-run passed");return
    evaluate=[str(PYTHON),"-u",str(EVALUATOR),"--device",args.device,"--folds",*folds,"--variants",*variants,"--seed",str(args.seed),"--batch-size",str(args.batch_size)]
    if subprocess.run(evaluate,cwd=PROJECT).returncode: raise RuntimeError("HRMC evaluation failed")
    print("[done] HRMC training and conditioned evaluation completed")
if __name__=="__main__":main()
