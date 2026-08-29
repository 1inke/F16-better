"""Matched seed-1 B0/SDRE1 training chain with the decoder locked."""
from __future__ import annotations
import argparse, csv, json, os, statistics, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent; W=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON=Path(r"C:\coding_toobox\envs\astra2\python.exe"); TRAINER=W/"FRONT_ENCODER_SERIES"/"train_front_variant.py"
DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse"; MODELS=ROOT/"sdre1_models.py"
DECODER=W/"COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"/"current_baseline_adapter.py"
SOURCE_ROOT=W/"MODE_TEMPLATE_GROUP_SERIES"/"EXP_146B_ANCHORED_MODE_TEMPLATE"/"results"; RESULTS=ROOT/"results"
SPLITS={"general":("stratified_random",.05),"challenge":("motion_challenge",.10)}; NAMES=("baseline","sdre1")
METRICS=("min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde")

def source(split,seed): return SOURCE_ROOT/split/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"
def run_dir(split,name,seed): return RESULTS/split/name/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def command(a,split,name):
    mode,fraction=SPLITS[split]
    cmd=[str(PYTHON),"-u",str(TRAINER),"--project-root",str(PROJECT),"--config-file",str(PROJECT/"configs"/"motif_rbf_univ.yaml"),"--data-path",str(DATA/"eth_data_train.npy"),"--num-path",str(DATA/"eth_num_train.npy"),"--variant","f16_temporal_social_cross","--decoder-experiment","current_mode_anchor_template","--decoder-module",str(DECODER),"--seed",str(a.seed),"--val-scene-id","-1","--validation-fraction",str(fraction),"--validation-seed",str(a.validation_seed),"--validation-split-mode",mode,"--epochs",str(a.epochs),"--val-freq",str(a.val_freq),"--batch-size",str(a.batch_size),"--learning-rate",str(a.learning_rate),"--device",a.device,"--max-agents","9","--graph-top-k","4","--graph-radius","6.0","--scheduler","monotonic_cosine","--warmup-epochs","1","--output-root",str(RESULTS/split/name),"--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10","--checkpoint-criterion","oracle_combo","--ema-decay","0.995","--focal-heading-normalize","--augment-reflection","--augment-speed-min","0.9","--augment-speed-max","1.1","--augment-probability","0.5"]
    if name=="sdre1": cmd += ["--encoder-experiment","sdre1_scale_decoupled","--encoder-module",str(MODELS)]
    return cmd
def row(path,split,name,seed):
    raw=json.loads(path.read_text(encoding="utf-8")); ev=raw["evaluation_best_selected"]
    return {"split":split,"experiment":name,"seed":seed,**{m:ev["oracle"][m] if m in ev["oracle"] else ev["original"][m.replace("top1_","")] for m in METRICS},"parameters":raw["locked_backend"]["parameter_counts"]["total"],"elapsed_seconds":raw["elapsed_seconds"]}
def summarize(seed):
    rows=[]
    for split in SPLITS:
        for name in NAMES:
            p=run_dir(split,name,seed)/"metrics.json"
            if p.is_file(): rows.append(row(p,split,name,seed))
    paired=[]
    for split in SPLITS:
        base=next((r for r in rows if r["split"]==split and r["experiment"]=="baseline"),None); test=next((r for r in rows if r["split"]==split and r["experiment"]=="sdre1"),None)
        if base and test: paired.append({"split":split,"percent_change":{m:100*(test[m]/base[m]-1) for m in METRICS}})
    out={"description":"SDRE1 scale-decoupled robust F16 encoder, locked mode/three-anchor decoder","rows":rows,"paired_comparisons":paired}
    (ROOT/"seed1_summary.json").write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
    if rows:
        with (ROOT/"all_runs.csv").open("w",newline="",encoding="utf-8-sig") as h: w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=1);p.add_argument("--epochs",type=int,default=12);p.add_argument("--val-freq",type=int,default=3);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--learning-rate",type=float,default=.00025);p.add_argument("--validation-seed",type=int,default=20260828);p.add_argument("--splits",nargs="+",choices=("all",*SPLITS),default=["all"]);p.add_argument("--variants",nargs="+",choices=("all",*NAMES),default=["all"]);p.add_argument("--force",action="store_true");p.add_argument("--dry-run",action="store_true");a=p.parse_args()
    splits=list(SPLITS) if "all" in a.splits else a.splits; names=list(NAMES) if "all" in a.variants else a.variants
    for path in (PYTHON,TRAINER,MODELS,DECODER,DATA/"eth_data_train.npy",DATA/"eth_num_train.npy"):
        if not path.is_file(): raise FileNotFoundError(path)
    for split in splits:
        src=source(split,a.seed)
        if not src.is_file(): raise FileNotFoundError(src)
        for name in names:
            dest=run_dir(split,name,a.seed); metrics=dest/"metrics.json"
            if metrics.is_file() and not a.force: print(f"[skip] {split}/{name}",flush=True);continue
            cmd=command(a,split,name)
            if a.force:cmd.append("--force")
            if a.dry_run: print("[dry-run] "+json.dumps(cmd));continue
            dest.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env["CSRS_SOURCE_CHECKPOINT"]=str(src)
            print(f"[run] {split}/{name}",flush=True)
            with (dest/"console.log").open("w",encoding="utf-8") as log:
                proc=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1);assert proc.stdout
                for line in proc.stdout:print(f"[{split}/{name}] {line}",end="",flush=True);log.write(line);log.flush()
                if proc.wait():raise RuntimeError(f"failed: {dest/'console.log'}")
            summarize(a.seed)
    if a.dry_run:print("[done] dry-run passed")
    else: print(f"[done] {len(summarize(a.seed)['rows'])} completed runs")
if __name__=="__main__":main()
