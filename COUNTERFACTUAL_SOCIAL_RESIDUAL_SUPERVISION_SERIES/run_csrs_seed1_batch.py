"""Teacher cache -> matched B0/CSRS1 seed1 -> mechanism diagnostic."""

from __future__ import annotations
import argparse,csv,json,os,statistics,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent; WORKSPACE=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"); PYTHON=Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER=WORKSPACE/"FRONT_ENCODER_SERIES"/"train_front_variant.py"; DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse"
SOURCE_ROOT=WORKSPACE/"INTERACTION_RESIDUAL_SUPERVISION_SERIES"/"results"; RESULTS=ROOT/"results"; CACHE=ROOT/"teacher_cache"
INDEX_SIDECAR=ROOT/"teacher_cache"/"sample_index_encoded.npy"; DIAGNOSTIC=ROOT/"run_csrs_diagnostic.py"
EXPERIMENTS={"baseline":("current_mode_anchor_template",ROOT/"current_baseline_adapter.py"),
             "csrs1_actual_gain_residual":("csrs1_actual_gain_residual",ROOT/"csrs_models.py")}
SPLITS={"general":("stratified_random",0.05),"challenge":("motion_challenge",0.10)}
METRICS=("min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde")


def source_checkpoint(split,seed): return SOURCE_ROOT/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"
def teacher_cache(split,seed): return CACHE/f"{split}_seed_{seed}.npz"
def run_dir(name,split,seed): return RESULTS/split/name/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"


def command(a,name,split,seed):
    split_mode,fraction=SPLITS[split]; experiment,module=EXPERIMENTS[name]
    return [str(PYTHON),"-u",str(TRAINER),"--project-root",str(PROJECT),"--config-file",str(PROJECT/"configs"/"motif_rbf_univ.yaml"),
            "--data-path",str(DATA/"eth_data_train.npy"),"--num-path",str(DATA/"eth_num_train.npy"),"--sample-scene-path",str(INDEX_SIDECAR),
            "--variant","f16_temporal_social_cross","--decoder-experiment",experiment,"--decoder-module",str(module),
            "--seed",str(seed),"--val-scene-id","-1","--validation-fraction",str(fraction),"--validation-seed",str(a.validation_seed),
            "--validation-split-mode",split_mode,"--epochs",str(a.epochs),"--val-freq",str(a.val_freq),"--batch-size",str(a.batch_size),
            "--learning-rate",str(a.learning_rate),"--device",a.device,"--max-agents","9","--graph-top-k","4","--graph-radius","6.0",
            "--scheduler","monotonic_cosine","--warmup-epochs","1","--output-root",str(RESULTS/split/name),
            "--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10","--checkpoint-criterion","oracle_combo",
            "--ema-decay","0.995","--focal-heading-normalize"]


def read_row(path,name,split,seed):
    raw=json.loads(path.read_text(encoding="utf-8")); ev=raw["evaluation_best_selected"]; pc=raw["locked_backend"]["parameter_counts"]
    return {"experiment":name,"split":split,"seed":seed,"min_ade_20":ev["oracle"]["min_ade_20"],"min_fde_20":ev["oracle"]["min_fde_20"],
            "same_path_fde":ev["oracle"]["same_path_fde"],"top1_ade":ev["original"]["ade"],"top1_fde":ev["original"]["fde"],
            "parameters":pc["total"],"trainable_parameters":pc["trainable"],"elapsed_seconds":raw["elapsed_seconds"]}


def summarize(seed,names,splits):
    rows=[]
    for split in splits:
        for name in names:
            path=run_dir(name,split,seed)/"metrics.json"
            if path.is_file(): rows.append(read_row(path,name,split,seed))
    paired=[]
    for split in splits:
        base=next((r for r in rows if r["split"]==split and r["experiment"]=="baseline"),None)
        test=next((r for r in rows if r["split"]==split and r["experiment"]=="csrs1_actual_gain_residual"),None)
        if base and test: paired.append({"split":split,"percent_change":{m:100*(test[m]/base[m]-1) for m in METRICS}})
    result={"description":"CSRS1 actual-gain counterfactual social residual seed1","rows":rows,"paired_comparisons":paired}
    (ROOT/"seed1_summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if rows:
        with (ROOT/"all_runs.csv").open("w",newline="",encoding="utf-8-sig") as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return result


def run_logged(cmd,log_path,env=None,cwd=ROOT):
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open("w",encoding="utf-8") as log:
        process=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1)
        assert process.stdout is not None
        for line in process.stdout: print(line,end="",flush=True); log.write(line); log.flush()
        if process.wait(): raise RuntimeError(f"failed; see {log_path}")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda:0"); p.add_argument("--seed",type=int,default=1)
    p.add_argument("--epochs",type=int,default=12); p.add_argument("--val-freq",type=int,default=3); p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--teacher-batch-size",type=int,default=128); p.add_argument("--diagnostic-batch-size",type=int,default=128)
    p.add_argument("--learning-rate",type=float,default=.00025); p.add_argument("--validation-seed",type=int,default=20260828)
    p.add_argument("--variants",nargs="+",choices=("all",*EXPERIMENTS),default=["all"]); p.add_argument("--splits",nargs="+",choices=("all",*SPLITS),default=["all"])
    p.add_argument("--force",action="store_true"); p.add_argument("--force-cache",action="store_true"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--skip-diagnostic",action="store_true"); a=p.parse_args()
    names=list(EXPERIMENTS) if "all" in a.variants else a.variants; splits=list(SPLITS) if "all" in a.splits else a.splits
    for path in (PYTHON,TRAINER,DIAGNOSTIC,DATA/"eth_data_train.npy",DATA/"eth_num_train.npy",*(v[1] for v in EXPERIMENTS.values())):
        if not path.is_file(): raise FileNotFoundError(path)
    sidecar_cmd=[str(PYTHON),str(ROOT/"prepare_index_sidecar.py"),"--data",str(DATA/"eth_data_train.npy"),"--output",str(INDEX_SIDECAR)]
    teacher_cmd=[str(PYTHON),"-u",str(ROOT/"generate_teacher_cache.py"),"--device",a.device,"--batch-size",str(a.teacher_batch_size),"--seed",str(a.seed),"--splits",*splits]
    if a.force_cache: teacher_cmd.append("--force")
    if a.dry_run:
        print("[dry-run] "+json.dumps(sidecar_cmd)); print("[dry-run] "+json.dumps(teacher_cmd))
    else:
        subprocess.run(sidecar_cmd,cwd=ROOT,check=True); run_logged(teacher_cmd,ROOT/"teacher_cache.log")
    for split in splits:
        source=source_checkpoint(split,a.seed); cache=teacher_cache(split,a.seed)
        if not source.is_file(): raise FileNotFoundError(source)
        if not a.dry_run and not cache.is_file(): raise FileNotFoundError(cache)
        for name in names:
            destination=run_dir(name,split,a.seed)
            if (destination/"metrics.json").is_file() and not a.force: print(f"[skip] {split}/{name}",flush=True); continue
            cmd=command(a,name,split,a.seed)
            if a.force: cmd.append("--force")
            if a.dry_run: print("[dry-run] "+json.dumps(cmd)); continue
            env=os.environ.copy(); env["CSRS_SOURCE_CHECKPOINT"]=str(source); env["CSRS_TEACHER_CACHE"]=str(cache)
            print(f"[run] {split}/{name} seed={a.seed}",flush=True); run_logged(cmd,destination/"console.log",env=env,cwd=PROJECT)
            summarize(a.seed,names,splits)
    if a.dry_run: print("[done] dry-run passed"); return
    result=summarize(a.seed,names,splits); print(f"[done] completed {len(result['rows'])}/{len(names)*len(splits)}",flush=True)
    can=not a.skip_diagnostic and set(names)==set(EXPERIMENTS) and set(splits)==set(SPLITS)
    if can: run_logged([str(PYTHON),"-u",str(DIAGNOSTIC),"--device",a.device,"--batch-size",str(a.diagnostic_batch_size),"--seed",str(a.seed),"--splits","all"],ROOT/"diagnostic.log")
    elif not a.skip_diagnostic: print("[diagnostic] skipped: requires both variants and splits")


if __name__=="__main__": main()

