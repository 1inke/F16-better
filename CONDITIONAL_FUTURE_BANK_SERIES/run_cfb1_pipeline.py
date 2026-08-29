"""Audit-gated CFB1 pipeline: train only when the future-bank oracle passes."""
from __future__ import annotations
import argparse,csv,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORKSPACE=ROOT.parent;PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");PYTHON=Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAINER=WORKSPACE/"FRONT_ENCODER_SERIES"/"train_front_variant.py";DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse";CSRS=WORKSPACE/"COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES";IR1=WORKSPACE/"INTERACTION_RESIDUAL_SUPERVISION_SERIES"
AUDITOR=ROOT/"prepare_and_audit_future_bank.py";DIAGNOSTIC=ROOT/"run_cfb1_diagnostic.py";SIDECAR=CSRS/"teacher_cache"/"sample_index_encoded.npy";RESULTS=ROOT/"results";SPLITS={"general":("stratified_random",.05),"challenge":("motion_challenge",.10)};METRICS=("min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde")
def source(split,seed):return IR1/"results"/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"
def cache(split,seed):return ROOT/"cache"/f"{split}_seed_{seed}.npz"
def baseline(split,seed):return CSRS/"results"/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def run_dir(split,seed):return RESULTS/split/"cfb1_cross_scene_three_future"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def logged(cmd,path,env=None,cwd=ROOT):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",encoding="utf-8") as h:
  p=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1);assert p.stdout
  for line in p.stdout:print(line,end="",flush=True);h.write(line);h.flush()
  if p.wait():raise RuntimeError(f"failed; see {path}")
def command(a,split):
 mode,fraction=SPLITS[split]
 return [str(PYTHON),"-u",str(TRAINER),"--project-root",str(PROJECT),"--config-file",str(PROJECT/"configs"/"motif_rbf_univ.yaml"),"--data-path",str(DATA/"eth_data_train.npy"),"--num-path",str(DATA/"eth_num_train.npy"),"--sample-scene-path",str(SIDECAR),"--variant","f16_temporal_social_cross","--decoder-experiment","cfb1_cross_scene_three_future","--decoder-module",str(ROOT/"cfb_models.py"),"--seed",str(a.seed),"--val-scene-id","-1","--validation-fraction",str(fraction),"--validation-seed",str(a.validation_seed),"--validation-split-mode",mode,"--epochs",str(a.epochs),"--val-freq",str(a.val_freq),"--batch-size",str(a.batch_size),"--learning-rate",str(a.learning_rate),"--device",a.device,"--max-agents","9","--graph-top-k","4","--graph-radius","6.0","--scheduler","monotonic_cosine","--warmup-epochs","1","--output-root",str(RESULTS/split/"cfb1_cross_scene_three_future"),"--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10","--checkpoint-criterion","oracle_combo","--ema-decay","0.995","--focal-heading-normalize"]
def read(path,name,split,seed):
 raw=json.loads(path.read_text(encoding="utf-8"));ev=raw["evaluation_best_selected"];pc=raw["locked_backend"]["parameter_counts"]
 return {"experiment":name,"split":split,"seed":seed,"min_ade_20":ev["oracle"]["min_ade_20"],"min_fde_20":ev["oracle"]["min_fde_20"],"same_path_fde":ev["oracle"]["same_path_fde"],"top1_ade":ev["original"]["ade"],"top1_fde":ev["original"]["fde"],"parameters":pc["total"],"trainable_parameters":pc["trainable"],"elapsed_seconds":raw["elapsed_seconds"]}
def summarize(seed):
 rows=[];paired=[]
 for split in SPLITS:
  b=read(baseline(split,seed)/"metrics.json","baseline_reused_exact_match",split,seed);rows.append(b);path=run_dir(split,seed)/"metrics.json"
  if path.is_file():
   v=read(path,"cfb1_cross_scene_three_future",split,seed);rows.append(v);paired.append({"split":split,"percent_change":{m:100*(v[m]/b[m]-1) for m in METRICS}})
 result={"description":"CFB1 cross-scene conditional future distillation","rows":rows,"paired_comparisons":paired};(ROOT/"seed1_summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
 if rows:
  with (ROOT/"all_runs.csv").open("w",newline="",encoding="utf-8-sig") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=1);p.add_argument("--epochs",type=int,default=12);p.add_argument("--val-freq",type=int,default=3);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--audit-batch-size",type=int,default=128);p.add_argument("--learning-rate",type=float,default=.00025);p.add_argument("--validation-seed",type=int,default=20260828);p.add_argument("--audit-only",action="store_true");p.add_argument("--force",action="store_true");p.add_argument("--dry-run",action="store_true");a=p.parse_args()
 for path in (PYTHON,TRAINER,AUDITOR,DIAGNOSTIC,ROOT/"cfb_models.py",SIDECAR,DATA/"eth_data_train.npy",DATA/"eth_num_train.npy"):
  if not path.is_file():raise FileNotFoundError(path)
 audit_cmd=[str(PYTHON),"-u",str(AUDITOR),"--device",a.device,"--seed",str(a.seed),"--batch-size",str(a.audit_batch_size)]+(["--force"] if a.force else [])
 if a.dry_run:print("[dry-run] audit "+json.dumps(audit_cmd));[print("[dry-run] train "+json.dumps(command(a,s))) for s in SPLITS];return
 logged(audit_cmd,ROOT/"audit.log");audit=json.loads((ROOT/"audit"/"combined_summary.json").read_text(encoding="utf-8"))
 if not audit["gate_passed"]:print("[stop] CFB-Audit did not pass. CFB1 training was not started.");return
 if a.audit_only:print("[done] audit passed; training intentionally skipped");return
 for split in SPLITS:
  for path in (source(split,a.seed),cache(split,a.seed),baseline(split,a.seed)/"metrics.json"):
   if not path.is_file():raise FileNotFoundError(path)
  dest=run_dir(split,a.seed)
  if (dest/"metrics.json").is_file() and not a.force:print(f"[skip] {split}/CFB1");continue
  env=os.environ.copy();env["CFB_SOURCE_CHECKPOINT"]=str(source(split,a.seed));env["CFB_CACHE"]=str(cache(split,a.seed));cmd=command(a,split)+(["--force"] if a.force else []);print(f"[run] {split}/CFB1 seed={a.seed}");logged(cmd,dest/"console.log",env,cwd=PROJECT);summarize(a.seed)
 summarize(a.seed);logged([str(PYTHON),"-u",str(DIAGNOSTIC),"--device",a.device,"--batch-size",str(a.audit_batch_size),"--seed",str(a.seed)],ROOT/"diagnostic.log")
if __name__=="__main__":main()
