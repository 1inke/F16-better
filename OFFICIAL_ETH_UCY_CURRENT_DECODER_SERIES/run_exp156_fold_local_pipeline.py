"""Leakage-free H3 -> V2 -> M2 current-decoder training on official folds."""
from __future__ import annotations
import argparse,csv,json,os,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent;WORKSPACE=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON=Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER=WORKSPACE/"FRONT_ENCODER_SERIES"/"train_front_variant.py"
DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse"
CONFIG=ROOT/"motif_rbf_official_4p8.yaml"
H3_MODULE=WORKSPACE/"HIERARCHICAL_DECODER_SERIES"/"hierarchical_decoder_models.py"
V2_MODULE=WORKSPACE/"ANCHOR_CORRECTION_VALIDATION_SERIES"/"anchor_correction_validation_models.py"
M2_MODULE=WORKSPACE/"MODE_TEMPLATE_GROUP_SERIES"/"grouped_mode_template_models.py"
AUDIT=ROOT/"EXP_156A_PROTOCOL_AUDIT"/"protocol_audit.json"
OUT=ROOT/"EXP_156B_FOLD_LOCAL_CURRENT_DECODER"/"results"
FOLDS=("eth","hotel","univ","zara01","zara02")
STAGES=("h3","v2","m2")

def run_dir(fold,stage,seed):return OUT/fold/stage/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def common(a,fold,seed,stage,epochs):
    modules={"h3":(H3_MODULE,"h3_anchor_proposal"),"v2":(V2_MODULE,"v2_wide_endpoint_correction"),"m2":(M2_MODULE,"m2_anchored_mode_template")}
    module,name=modules[stage]
    c=[str(PYTHON),"-u",str(TRAINER),"--project-root",str(PROJECT),"--config-file",str(CONFIG),
       "--data-path",str(DATA/f"{fold}_data_train.npy"),"--num-path",str(DATA/f"{fold}_num_train.npy"),
       "--variant","f16_temporal_social_cross","--decoder-experiment",name,"--decoder-module",str(module),
       "--seed",str(seed),"--val-scene-id","-1","--validation-fraction","0.05","--validation-seed","20260830",
       "--validation-split-mode","scene_tail","--epochs",str(epochs),"--val-freq",str(a.val_freq),
       "--batch-size",str(a.batch_size),"--device",a.device,"--max-agents","9","--graph-top-k","4","--graph-radius","6.0",
       "--scheduler","monotonic_cosine","--warmup-epochs","1","--learning-rate","0.00025",
       "--output-root",str(OUT/fold/stage),"--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10",
       "--oracle-coverage-fde-weight","0.25","--checkpoint-criterion","oracle_combo","--ema-decay","0.995",
       "--focal-heading-normalize","--augment-reflection","--augment-speed-min","0.9","--augment-speed-max","1.1","--augment-probability","0.5"]
    if stage=="m2":c += ["--test-data-path",str(DATA/f"{fold}_data_test.npy"),"--test-num-path",str(DATA/f"{fold}_num_test.npy")]
    return c

def summarize(seeds):
    rows=[]
    for fold in FOLDS:
      for seed in seeds:
       p=run_dir(fold,"m2",seed)/"metrics.json"
       if not p.is_file():continue
       j=json.loads(p.read_text(encoding="utf-8"));e=j.get("evaluation_test_best_selected")
       if not e:continue
       rows.append({"fold":fold,"seed":seed,"test_samples":e["samples"],"min_ade_20":e["oracle"]["min_ade_20"],
                    "min_fde_20":e["oracle"]["min_fde_20"],"same_path_fde":e["oracle"]["same_path_fde"],
                    "top1_ade":e["original"]["ade"],"top1_fde":e["original"]["fde"],
                    "parameters":j["locked_backend"]["parameter_counts"]["total"]})
    target=ROOT/"EXP_156B_FOLD_LOCAL_CURRENT_DECODER";target.mkdir(parents=True,exist_ok=True)
    (target/"summary.json").write_text(json.dumps({"rows":rows},indent=2),encoding="utf-8")
    if rows:
      with (target/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--folds",nargs="+",choices=("all",*FOLDS),default=["eth"])
    p.add_argument("--seeds",nargs="+",type=int,default=[1]);p.add_argument("--stages",nargs="+",choices=("all",*STAGES),default=["all"])
    p.add_argument("--h3-epochs",type=int,default=30);p.add_argument("--v2-epochs",type=int,default=20);p.add_argument("--m2-epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=64);p.add_argument("--val-freq",type=int,default=5);p.add_argument("--dry-run",action="store_true");p.add_argument("--force",action="store_true");a=p.parse_args()
    required=(PYTHON,TRAINER,CONFIG,H3_MODULE,V2_MODULE,M2_MODULE,AUDIT)
    for x in required:
      if not x.is_file():raise FileNotFoundError(x)
    audit=json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit.get("passed"):raise RuntimeError("EXP-156A audit did not pass")
    folds=FOLDS if "all" in a.folds else tuple(a.folds);stages=STAGES if "all" in a.stages else tuple(a.stages);epochs={"h3":a.h3_epochs,"v2":a.v2_epochs,"m2":a.m2_epochs}
    for fold in folds:
      for seed in a.seeds:
       for stage in stages:
        destination=run_dir(fold,stage,seed);metrics=destination/"metrics.json"
        if metrics.is_file() and not a.force:print(f"[skip] {fold}/{stage}/seed{seed}");continue
        env=os.environ.copy()
        if stage=="v2":
          source=run_dir(fold,"h3",seed)/"best_selected.pth";env["EXP143_SOURCE_CHECKPOINT"]=str(source)
        elif stage=="m2":
          source=run_dir(fold,"v2",seed)/"best_selected.pth";env["EXP146_SOURCE_CHECKPOINT"]=str(source)
        else:source=None
        if source is not None and not source.is_file() and not a.dry_run:raise FileNotFoundError(source)
        cmd=common(a,fold,seed,stage,epochs[stage]);
        if a.force:cmd.append("--force")
        if a.dry_run:print(f"[dry-run] {fold}/{stage}/seed{seed} "+json.dumps(cmd));continue
        destination.mkdir(parents=True,exist_ok=True);print(f"[run] {fold}/{stage}/seed{seed}",flush=True)
        with (destination/"console.log").open("w",encoding="utf-8") as log:
          job=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1);assert job.stdout
          for line in job.stdout:print(f"[{fold}/{stage}] {line}",end="",flush=True);log.write(line);log.flush()
        if job.wait():raise RuntimeError(f"failed: {destination/'console.log'}")
        summarize(a.seeds)
    if a.dry_run:print("[done] dry-run passed")
    else:summarize(a.seeds);print("[done] requested fold-local pipeline completed")
if __name__=="__main__":main()
