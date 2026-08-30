"""Stage-A formal encoder ablation: same decoder, split, loss and schedule."""
from __future__ import annotations
import argparse,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WS=ROOT.parent;PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");PY=Path(r"C:\coding_toobox\envs\astra2\python.exe");TRAIN=WS/"FRONT_ENCODER_SERIES"/"train_front_variant.py";ADAPTER=ROOT/"formal_decoder_adapter.py";DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse";SOURCE=WS/"MODE_TEMPLATE_GROUP_SERIES"/"EXP_146B_ANCHORED_MODE_TEMPLATE"/"results";OUT=ROOT/"EXP_154_FORMAL_ENCODER_ABLATION"/"results"
VARIANTS={"a_trajectory_only":"f0_baseline","b_sparse_graph":"f3_factorized","c_relation_prior":"f4_relation_tokens","d_f16_full_temporal":"f16_temporal_social_cross","e_f35_half_pool":"f35_temporal_half_pool"};SPLITS={"general":("stratified_random",.05),"challenge":("motion_challenge",.10)}
def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--seeds",nargs="+",type=int,default=[1]);p.add_argument("--splits",nargs="+",choices=tuple(SPLITS),default=list(SPLITS));p.add_argument("--variants",nargs="+",choices=("all",*VARIANTS),default=["all"]);p.add_argument("--epochs",type=int,default=30);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--dry-run",action="store_true");p.add_argument("--force",action="store_true");a=p.parse_args();names=list(VARIANTS) if "all" in a.variants else a.variants
 for x in (PY,TRAIN,ADAPTER,DATA/"eth_data_train.npy",DATA/"eth_num_train.npy"):
  if not x.is_file():raise FileNotFoundError(x)
 for split in a.splits:
  mode,fraction=SPLITS[split]
  for name in names:
   for seed in a.seeds:
    source=SOURCE/split/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth";run=OUT/split/name/VARIANTS[name]/f"seed_{seed}"/"val_scene_-1";metrics=run/"metrics.json"
    if not source.is_file():raise FileNotFoundError(source)
    if metrics.is_file() and not a.force:print(f"[skip] {split}/{name}/seed{seed}");continue
    cmd=[str(PY),"-u",str(TRAIN),"--project-root",str(PROJECT),"--config-file",str(PROJECT/"configs"/"motif_rbf_univ.yaml"),"--data-path",str(DATA/"eth_data_train.npy"),"--num-path",str(DATA/"eth_num_train.npy"),"--variant",VARIANTS[name],"--decoder-experiment","fixed_current_mode_anchor","--decoder-module",str(ADAPTER),"--seed",str(seed),"--val-scene-id","-1","--validation-fraction",str(fraction),"--validation-seed","20260828","--validation-split-mode",mode,"--epochs",str(a.epochs),"--val-freq","3","--batch-size",str(a.batch_size),"--device",a.device,"--max-agents","9","--graph-top-k","4","--graph-radius","6.0","--scheduler","monotonic_cosine","--warmup-epochs","1","--learning-rate","0.00025","--output-root",str(OUT/split/name),"--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10","--checkpoint-criterion","oracle_combo","--ema-decay","0.995","--focal-heading-normalize","--augment-reflection","--augment-speed-min","0.9","--augment-speed-max","1.1","--augment-probability","0.5"]
    if a.force:cmd.append("--force")
    if a.dry_run:print("[dry-run] "+json.dumps(cmd));continue
    run.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env["FORMAL_DECODER_SOURCE"]=str(source);print(f"[run] {split}/{name}/seed{seed}",flush=True)
    with (run/"console.log").open("w",encoding="utf-8") as log:
     job=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1);assert job.stdout
     for line in job.stdout:print(line,end="",flush=True);log.write(line);log.flush()
    if job.wait():raise RuntimeError(f"failed: {run/'console.log'}")
if __name__=="__main__":main()
