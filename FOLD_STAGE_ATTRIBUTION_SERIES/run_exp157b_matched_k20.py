"""EXP-157B: matched-protocol K20 control on official folds."""
from __future__ import annotations
import argparse,csv,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WS=ROOT.parent;PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");PY=Path(r"C:\coding_toobox\envs\astra2\python.exe")
DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse";CONFIG=WS/"OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES"/"motif_rbf_official_4p8.yaml";TRAIN=WS/"FRONT_ENCODER_SERIES"/"train_front_variant.py";DECODER=WS/"AUTONOMOUS_EXPERIMENTS"/"EXP_002_LIGHT_KINEMATIC_DECODER"/"decoder_models.py";OUT=ROOT/"EXP_157B_MATCHED_K20"/"results";FOLDS=("eth","hotel","univ","zara01","zara02")
def run_dir(fold,seed):return OUT/fold/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def summarize(seeds):
 rows=[]
 for fold in FOLDS:
  for seed in seeds:
   p=run_dir(fold,seed)/"metrics.json"
   if not p.is_file():continue
   j=json.loads(p.read_text(encoding="utf-8"));e=j.get("evaluation_test_best_selected")
   if e:rows.append({"fold":fold,"seed":seed,"samples":e["samples"],"min_ade_20":e["oracle"]["min_ade_20"],"min_fde_20":e["oracle"]["min_fde_20"],"same_path_fde":e["oracle"]["same_path_fde"],"top1_ade":e["original"]["ade"],"top1_fde":e["original"]["fde"],"parameters":j["locked_backend"]["parameter_counts"]["total"]})
 target=ROOT/"EXP_157B_MATCHED_K20";target.mkdir(parents=True,exist_ok=True);(target/"summary.json").write_text(json.dumps({"rows":rows},indent=2),encoding="utf-8")
 if rows:
  with (target/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--folds",nargs="+",choices=("all",*FOLDS),default=["all"]);p.add_argument("--seeds",nargs="+",type=int,default=[1]);p.add_argument("--epochs",type=int,default=30);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--val-freq",type=int,default=5);p.add_argument("--dry-run",action="store_true");p.add_argument("--force",action="store_true");a=p.parse_args()
 for x in (PY,TRAIN,DECODER,CONFIG):
  if not x.is_file():raise FileNotFoundError(x)
 folds=FOLDS if "all" in a.folds else tuple(a.folds)
 for fold in folds:
  for seed in a.seeds:
   dest=run_dir(fold,seed);metrics=dest/"metrics.json"
   if metrics.is_file() and not a.force:print(f"[skip] {fold}/seed{seed}");continue
   cmd=[str(PY),"-u",str(TRAIN),"--project-root",str(PROJECT),"--config-file",str(CONFIG),"--data-path",str(DATA/f"{fold}_data_train.npy"),"--num-path",str(DATA/f"{fold}_num_train.npy"),"--test-data-path",str(DATA/f"{fold}_data_test.npy"),"--test-num-path",str(DATA/f"{fold}_num_test.npy"),"--variant","f16_temporal_social_cross","--decoder-experiment","kinematic_k20","--decoder-module",str(DECODER),"--seed",str(seed),"--val-scene-id","-1","--validation-fraction","0.05","--validation-seed","20260830","--validation-split-mode","scene_tail","--epochs",str(a.epochs),"--val-freq",str(a.val_freq),"--batch-size",str(a.batch_size),"--device",a.device,"--max-agents","9","--graph-top-k","4","--graph-radius","6.0","--scheduler","monotonic_cosine","--warmup-epochs","1","--learning-rate","0.00025","--output-root",str(OUT/fold),"--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10","--oracle-coverage-fde-weight","0.25","--checkpoint-criterion","oracle_combo","--ema-decay","0.995","--focal-heading-normalize","--augment-reflection","--augment-speed-min","0.9","--augment-speed-max","1.1","--augment-probability","0.5"]
   if a.force:cmd.append("--force")
   if a.dry_run:print("[dry-run] "+json.dumps(cmd));continue
   dest.mkdir(parents=True,exist_ok=True);print(f"[run] {fold}/seed{seed}",flush=True)
   with (dest/"console.log").open("w",encoding="utf-8") as log:
    job=subprocess.Popen(cmd,cwd=PROJECT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1);assert job.stdout
    for line in job.stdout:print(f"[{fold}] {line}",end="",flush=True);log.write(line);log.flush()
   if job.wait():raise RuntimeError(f"failed: {dest/'console.log'}")
   summarize(a.seeds)
 if a.dry_run:print("[done] dry-run passed")
 else:summarize(a.seeds);print("[done] matched K20 completed")
if __name__=="__main__":main()
