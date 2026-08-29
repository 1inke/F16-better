"""Run FCRA1 General/Challenge seed1 and compare with the exact existing matched B0."""

from __future__ import annotations
import argparse,csv,json,os,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent;WORKSPACE=ROOT.parent;PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON=Path(r"C:\coding_toobox\envs\astra2\python.exe");TRAINER=WORKSPACE/"FRONT_ENCODER_SERIES"/"train_front_variant.py";DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse"
CSRS=WORKSPACE/"COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES";SOURCE_ROOT=WORKSPACE/"INTERACTION_RESIDUAL_SUPERVISION_SERIES"/"results"
BASELINE_ROOT=CSRS/"results";CACHE_ROOT=CSRS/"teacher_cache";SIDECAR=CACHE_ROOT/"sample_index_encoded.npy";RESULTS=ROOT/"results";DIAGNOSTIC=ROOT/"run_fcra1_diagnostic.py"
SPLITS={"general":("stratified_random",.05),"challenge":("motion_challenge",.10)};METRICS=("min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde")

def source(split,seed):return SOURCE_ROOT/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"
def cache(split,seed):return CACHE_ROOT/f"{split}_seed_{seed}.npz"
def run_dir(split,seed):return RESULTS/split/"fcra1_full_consistent_top3"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def baseline_dir(split,seed):return BASELINE_ROOT/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"

def command(a,split):
    mode,fraction=SPLITS[split]
    return [str(PYTHON),"-u",str(TRAINER),"--project-root",str(PROJECT),"--config-file",str(PROJECT/"configs"/"motif_rbf_univ.yaml"),
            "--data-path",str(DATA/"eth_data_train.npy"),"--num-path",str(DATA/"eth_num_train.npy"),"--sample-scene-path",str(SIDECAR),
            "--variant","f16_temporal_social_cross","--decoder-experiment","fcra1_full_consistent_top3","--decoder-module",str(ROOT/"fcra_models.py"),
            "--seed",str(a.seed),"--val-scene-id","-1","--validation-fraction",str(fraction),"--validation-seed",str(a.validation_seed),"--validation-split-mode",mode,
            "--epochs",str(a.epochs),"--val-freq",str(a.val_freq),"--batch-size",str(a.batch_size),"--learning-rate",str(a.learning_rate),"--device",a.device,
            "--max-agents","9","--graph-top-k","4","--graph-radius","6.0","--scheduler","monotonic_cosine","--warmup-epochs","1",
            "--output-root",str(RESULTS/split/"fcra1_full_consistent_top3"),"--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10",
            "--checkpoint-criterion","oracle_combo","--ema-decay","0.995","--focal-heading-normalize"]

def read(path,name,split,seed):
    raw=json.loads(path.read_text(encoding="utf-8"));ev=raw["evaluation_best_selected"];pc=raw["locked_backend"]["parameter_counts"]
    return {"experiment":name,"split":split,"seed":seed,"min_ade_20":ev["oracle"]["min_ade_20"],"min_fde_20":ev["oracle"]["min_fde_20"],
            "same_path_fde":ev["oracle"]["same_path_fde"],"top1_ade":ev["original"]["ade"],"top1_fde":ev["original"]["fde"],
            "parameters":pc["total"],"trainable_parameters":pc["trainable"],"elapsed_seconds":raw["elapsed_seconds"]}

def summarize(seed,splits):
    rows=[];paired=[]
    for split in splits:
        bp=baseline_dir(split,seed)/"metrics.json";fp=run_dir(split,seed)/"metrics.json"
        if bp.is_file():rows.append(read(bp,"baseline_reused_exact_match",split,seed))
        if fp.is_file():rows.append(read(fp,"fcra1_full_consistent_top3",split,seed))
        b=next((r for r in rows if r["split"]==split and r["experiment"].startswith("baseline")),None);f=next((r for r in rows if r["split"]==split and r["experiment"].startswith("fcra")),None)
        if b and f:paired.append({"split":split,"percent_change":{m:100*(f[m]/b[m]-1) for m in METRICS}})
    result={"description":"FCRA1 online full-consistent Top-3 correction","baseline_policy":"reuse exact CSRS1 B0: identical source/split/seed/training/no-augmentation",
            "rows":rows,"paired_comparisons":paired};(ROOT/"seed1_summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if rows:
        with (ROOT/"all_runs.csv").open("w",newline="",encoding="utf-8-sig") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return result

def run_logged(cmd,path,env=None,cwd=ROOT):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as log:
        p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1);assert p.stdout
        for line in p.stdout:print(line,end="",flush=True);log.write(line);log.flush()
        if p.wait():raise RuntimeError(f"failed; see {path}")

def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=1);p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--val-freq",type=int,default=3);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--diagnostic-batch-size",type=int,default=128)
    p.add_argument("--learning-rate",type=float,default=.00025);p.add_argument("--validation-seed",type=int,default=20260828);p.add_argument("--splits",nargs="+",choices=("all",*SPLITS),default=["all"])
    p.add_argument("--force",action="store_true");p.add_argument("--dry-run",action="store_true");p.add_argument("--skip-diagnostic",action="store_true");a=p.parse_args()
    splits=list(SPLITS) if "all" in a.splits else a.splits
    for path in (PYTHON,TRAINER,DIAGNOSTIC,ROOT/"fcra_models.py",CSRS/"prepare_index_sidecar.py",DATA/"eth_data_train.npy",DATA/"eth_num_train.npy"):
        if not path.is_file():raise FileNotFoundError(path)
    side=[str(PYTHON),str(CSRS/"prepare_index_sidecar.py"),"--data",str(DATA/"eth_data_train.npy"),"--output",str(SIDECAR)]
    if a.dry_run:print("[dry-run] "+json.dumps(side))
    else:subprocess.run(side,cwd=ROOT,check=True)
    for split in splits:
        for path in (source(split,a.seed),cache(split,a.seed),baseline_dir(split,a.seed)/"metrics.json"):
            if not path.is_file():raise FileNotFoundError(path)
        dest=run_dir(split,a.seed)
        if (dest/"metrics.json").is_file() and not a.force:print(f"[skip] {split}/FCRA1",flush=True);continue
        cmd=command(a,split)
        if a.force:cmd.append("--force")
        if a.dry_run:print("[dry-run] "+json.dumps(cmd));continue
        env=os.environ.copy();env["FCRA_SOURCE_CHECKPOINT"]=str(source(split,a.seed));env["FCRA_NEED_CACHE"]=str(cache(split,a.seed))
        print(f"[run] {split}/FCRA1 seed={a.seed}",flush=True);run_logged(cmd,dest/"console.log",env,cwd=PROJECT);summarize(a.seed,splits)
    if a.dry_run:print("[done] dry-run passed");return
    result=summarize(a.seed,splits);print(f"[done] rows={len(result['rows'])}",flush=True)
    if not a.skip_diagnostic and set(splits)==set(SPLITS):run_logged([str(PYTHON),"-u",str(DIAGNOSTIC),"--device",a.device,"--batch-size",str(a.diagnostic_batch_size),"--seed",str(a.seed),"--splits","all"],ROOT/"diagnostic.log")
    elif not a.skip_diagnostic:print("[diagnostic] skipped: requires both splits")

if __name__=="__main__":main()

