"""Run matched B0/IR1 seed-1 experiments, then response/geometry diagnostics."""

from __future__ import annotations
import argparse, csv, json, os, statistics, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent; WORKSPACE=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON=Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER=WORKSPACE/"FRONT_ENCODER_SERIES"/"train_front_variant.py"
DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse"
SOURCE_ROOT=WORKSPACE/"MODE_TEMPLATE_GROUP_SERIES"/"EXP_146B_ANCHORED_MODE_TEMPLATE"/"results"
RESULTS=ROOT/"results"; DIAGNOSTIC=ROOT/"run_response_diagnostic.py"
EXPERIMENTS={
    "baseline":("current_mode_anchor_template",ROOT/"current_baseline_adapter.py"),
    "ir1_future_response_supervision":("ir1_future_response_supervision",ROOT/"irs_models.py"),
}
SPLITS={"general":("stratified_random",0.05),"challenge":("motion_challenge",0.10)}
METRICS=("min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde")

def run_dir(name,split,seed): return RESULTS/split/name/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def source_checkpoint(split,seed): return SOURCE_ROOT/split/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"

def command(args,name,split,seed):
    split_mode,fraction=SPLITS[split]; experiment,module=EXPERIMENTS[name]
    cmd=[str(PYTHON),"-u",str(TRAINER),"--project-root",str(PROJECT),"--config-file",str(PROJECT/"configs"/"motif_rbf_univ.yaml"),
         "--data-path",str(DATA/"eth_data_train.npy"),"--num-path",str(DATA/"eth_num_train.npy"),
         "--variant","f16_temporal_social_cross","--decoder-experiment",experiment,"--decoder-module",str(module),
         "--seed",str(seed),"--val-scene-id","-1","--validation-fraction",str(fraction),"--validation-seed",str(args.validation_seed),
         "--validation-split-mode",split_mode,"--epochs",str(args.epochs),"--val-freq",str(args.val_freq),
         "--batch-size",str(args.batch_size),"--learning-rate",str(args.learning_rate),"--device",args.device,
         "--max-agents","9","--graph-top-k","4","--graph-radius","6.0","--scheduler","monotonic_cosine",
         "--warmup-epochs","1","--output-root",str(RESULTS/split/name),"--oracle-coverage-weight","0.20",
         "--oracle-coverage-temperature","0.10","--checkpoint-criterion","oracle_combo","--ema-decay","0.995",
         "--focal-heading-normalize","--augment-reflection","--augment-speed-min","0.9","--augment-speed-max","1.1","--augment-probability","0.5"]
    if args.force: cmd.append("--force")
    return cmd

def read_row(path,name,split,seed):
    raw=json.loads(path.read_text(encoding="utf-8")); ev=raw["evaluation_best_selected"]; pc=raw["locked_backend"]["parameter_counts"]
    return {"experiment":name,"split":split,"seed":seed,"min_ade_20":ev["oracle"]["min_ade_20"],
            "min_fde_20":ev["oracle"]["min_fde_20"],"same_path_fde":ev["oracle"]["same_path_fde"],
            "top1_ade":ev["original"]["ade"],"top1_fde":ev["original"]["fde"],
            "parameters":pc["total"],"trainable_parameters":pc["trainable"],"elapsed_seconds":raw["elapsed_seconds"]}

def summarize(seeds,names,splits):
    rows=[]
    for split in splits:
        for name in names:
            for seed in seeds:
                p=run_dir(name,split,seed)/"metrics.json"
                if p.is_file(): rows.append(read_row(p,name,split,seed))
    aggregate=[]; paired=[]
    for split in splits:
        baseline={r["seed"]:r for r in rows if r["split"]==split and r["experiment"]=="baseline"}
        for name in names:
            group=[r for r in rows if r["split"]==split and r["experiment"]==name]
            if not group: continue
            aggregate.append({"split":split,"experiment":name,"parameters":group[0]["parameters"],"trainable_parameters":group[0]["trainable_parameters"],
                              "metrics":{m:{"mean":statistics.fmean(float(r[m]) for r in group),"std":statistics.stdev(float(r[m]) for r in group) if len(group)>1 else 0.0,"values":[float(r[m]) for r in group]} for m in METRICS}})
            if name!="baseline":
                changes={m:[100*(r[m]/baseline[r["seed"]][m]-1) for r in group if r["seed"] in baseline] for m in METRICS}
                paired.append({"split":split,"experiment":name,"paired_percent_change":{m:{"mean":statistics.fmean(v),"values":v} for m,v in changes.items() if v}})
    result={"description":"IR1 future-response supervised social semantic bridge","seeds":seeds,"experiments":names,"splits":splits,"rows":rows,"aggregate":aggregate,"paired_comparisons":paired}
    (ROOT/"multiseed_summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    if rows:
        with (ROOT/"all_runs.csv").open("w",newline="",encoding="utf-8-sig") as h:
            w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda:0"); p.add_argument("--epochs",type=int,default=12)
    p.add_argument("--val-freq",type=int,default=3); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--diagnostic-batch-size",type=int,default=128)
    p.add_argument("--learning-rate",type=float,default=0.00025); p.add_argument("--validation-seed",type=int,default=20260828)
    p.add_argument("--seeds",type=int,nargs="+",default=[1]); p.add_argument("--variants",nargs="+",choices=("all",*EXPERIMENTS),default=["all"])
    p.add_argument("--splits",nargs="+",choices=("all",*SPLITS),default=["all"]); p.add_argument("--force",action="store_true")
    p.add_argument("--dry-run",action="store_true"); p.add_argument("--skip-diagnostic",action="store_true"); args=p.parse_args()
    names=list(EXPERIMENTS) if "all" in args.variants else args.variants; splits=list(SPLITS) if "all" in args.splits else args.splits
    for path in (PYTHON,TRAINER,DIAGNOSTIC,DATA/"eth_data_train.npy",DATA/"eth_num_train.npy",*(v[1] for v in EXPERIMENTS.values())):
        if not path.is_file(): raise FileNotFoundError(path)
    for split in splits:
        for name in names:
            for seed in args.seeds:
                source=source_checkpoint(split,seed); destination=run_dir(name,split,seed)
                if not source.is_file(): raise FileNotFoundError(source)
                if (destination/"metrics.json").is_file() and not args.force: print(f"[skip] {split} {name} seed={seed}",flush=True); continue
                cmd=command(args,name,split,seed)
                if args.dry_run: print("[dry-run] source="+str(source)+" command="+json.dumps(cmd)); continue
                destination.mkdir(parents=True,exist_ok=True); env=os.environ.copy(); env["IRS_SOURCE_CHECKPOINT"]=str(source)
                print(f"[run] {split} {name} seed={seed}",flush=True)
                with (destination/"console.log").open("w",encoding="utf-8") as log:
                    proc=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1)
                    assert proc.stdout is not None
                    for line in proc.stdout: print(f"[{split}/{name}] {line}",end="",flush=True); log.write(line); log.flush()
                    if proc.wait(): raise RuntimeError(f"failed; see {destination/'console.log'}")
                summarize(args.seeds,names,splits)
    if args.dry_run: print("[done] dry-run passed"); return
    result=summarize(args.seeds,names,splits); print(f"[done] completed {len(result['rows'])}/{len(names)*len(splits)*len(args.seeds)}")
    can=not args.skip_diagnostic and args.seeds==[1] and set(names)==set(EXPERIMENTS) and set(splits)==set(SPLITS)
    if can: subprocess.run([str(PYTHON),"-u",str(DIAGNOSTIC),"--device",args.device,"--batch-size",str(args.diagnostic_batch_size),"--splits","all"],cwd=WORKSPACE,check=True)
    elif not args.skip_diagnostic: print("[diagnostic] skipped: requires seed1, both variants and both splits")

if __name__=="__main__": main()
