"""Run the controlled TECM1 General/Challenge screening protocol."""
from __future__ import annotations
import argparse,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WS=ROOT.parent;PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");PY=Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAIN=WS/"FRONT_ENCODER_SERIES"/"train_front_variant.py";MODEL=ROOT/"tecm_models.py";DEC=ROOT/"tecm_current_decoder_adapter.py";DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse";SOURCE=WS/"MODE_TEMPLATE_GROUP_SERIES"/"EXP_146B_ANCHORED_MODE_TEMPLATE"/"results";OUT=ROOT/"EXP_152_TECM1"/"results";SPLITS={"general":("stratified_random",.05),"challenge":("motion_challenge",.10)}
def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--seeds",nargs="+",type=int,default=[1]);p.add_argument("--splits",nargs="+",choices=tuple(SPLITS),default=list(SPLITS));p.add_argument("--epochs",type=int,default=12);p.add_argument("--val-freq",type=int,default=3);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--learning-rate",type=float,default=2.5e-4);p.add_argument("--force",action="store_true");a=p.parse_args(); rows=[]
 for x in (PY,TRAIN,MODEL,DEC,DATA/"eth_data_train.npy",DATA/"eth_num_train.npy"):
  if not x.is_file():raise FileNotFoundError(x)
 for split in a.splits:
  split_mode,fraction=SPLITS[split]
  for seed in a.seeds:
   source=SOURCE/split/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth";run=OUT/split/"tecm1"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1";metrics=run/"metrics.json"
   if not source.is_file():raise FileNotFoundError(source)
   if metrics.is_file() and not a.force: print(f"[skip] {split} seed={seed}");continue
   cmd=[str(PY),"-u",str(TRAIN),"--project-root",str(PROJECT),"--config-file",str(PROJECT/"configs"/"motif_rbf_univ.yaml"),"--data-path",str(DATA/"eth_data_train.npy"),"--num-path",str(DATA/"eth_num_train.npy"),"--variant","f16_temporal_social_cross","--encoder-experiment","tecm1_time_resolved_edge_message","--encoder-module",str(MODEL),"--decoder-experiment","tecm_current_mode_anchor","--decoder-module",str(DEC),"--seed",str(seed),"--val-scene-id","-1","--validation-fraction",str(fraction),"--validation-seed","20260828","--validation-split-mode",split_mode,"--epochs",str(a.epochs),"--val-freq",str(a.val_freq),"--batch-size",str(a.batch_size),"--device",a.device,"--max-agents","9","--graph-top-k","4","--graph-radius","6.0","--scheduler","monotonic_cosine","--warmup-epochs","1","--learning-rate",str(a.learning_rate),"--output-root",str(OUT/split/"tecm1"),"--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10","--checkpoint-criterion","oracle_combo","--ema-decay","0.995","--focal-heading-normalize","--augment-reflection","--augment-speed-min","0.9","--augment-speed-max","1.1","--augment-probability","0.5"]
   if a.force:cmd.append("--force")
   run.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env["TECM_SOURCE_CHECKPOINT"]=str(source);print(f"[run] TECM1 {split} seed={seed}",flush=True)
   with (run/"console.log").open("w",encoding="utf-8") as log:
    proc=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1);assert proc.stdout
    for line in proc.stdout:print(f"[{split}/tecm1] {line}",end="",flush=True);log.write(line);log.flush()
   if proc.wait():raise RuntimeError(f"failed: {run/'console.log'}")
   ev=json.loads(metrics.read_text(encoding="utf-8"))["evaluation_best_selected"];rows.append({"split":split,"seed":seed,"minADE20":ev["oracle"]["min_ade_20"],"minFDE20":ev["oracle"]["min_fde_20"],"same_path_FDE":ev["oracle"]["same_path_fde"],"top1_ADE":ev["original"]["ade"],"top1_FDE":ev["original"]["fde"]})
 out=ROOT/"EXP_152_TECM1";out.mkdir(exist_ok=True);(out/"summary.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding="utf-8");print("[done] "+str(out/"summary.json"))
if __name__=="__main__":main()
