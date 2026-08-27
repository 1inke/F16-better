from __future__ import annotations
import json, subprocess
from pathlib import Path
R=Path(__file__).resolve().parent; W=R.parent
P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"); PY=Path(r"C:\coding_toobox\envs\astra2\python.exe")
T=W/"FRONT_ENCODER_SERIES"/"train_front_variant.py"; D=W/"AUTONOMOUS_EXPERIMENTS"/"EXP_002_LIGHT_KINEMATIC_DECODER"/"decoder_models.py"
jobs=[
 ("EXP_101_FOCAL_HEADING_NORMALIZATION",[]),
 ("EXP_102_HEADING_MIRROR_SPEED",["--augment-reflection","--augment-speed-min","0.9","--augment-speed-max","1.1","--augment-probability","0.5"]),
]
summary_path=R/"part_a_screen_summary.json"; rows=json.loads(summary_path.read_text()) if summary_path.is_file() else []
for name,extra in jobs:
 for scene in (1,3):
  O=R/name/"results"; rd=O/"f16_temporal_social_cross"/"seed_1"/f"val_scene_{scene}"; rd.mkdir(parents=True,exist_ok=True); mp=rd/"metrics.json"
  if mp.is_file() and any(x["experiment"]==name and x["scene"]==scene for x in rows): print(f"[skip] {name} scene={scene}",flush=True); continue
  cmd=[str(PY),str(T),"--project-root",str(P),"--variant","f16_temporal_social_cross","--decoder-experiment","kinematic_k20","--decoder-module",str(D),"--seed","1","--val-scene-id",str(scene),"--epochs","30","--val-freq","5","--device","cuda:0","--scheduler","warm_restarts","--output-root",str(O),"--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10","--checkpoint-criterion","oracle_combo","--ema-decay","0.995","--focal-heading-normalize","--force",*extra]
  print(f"[run] {name} scene={scene}",flush=True)
  with (rd/"console.log").open("w",encoding="utf-8") as log: result=subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode: raise RuntimeError(rd/"console.log")
  m=json.loads(mp.read_text()); v=m["evaluation_best_selected"]
  rows=[x for x in rows if not(x["experiment"]==name and x["scene"]==scene)]
  rows.append({"experiment":name,"scene":scene,"samples":v["samples"],**v["oracle"],"top1_ade":v["original"]["ade"],"top1_fde":v["original"]["fde"],"parameters":m["locked_backend"]["parameter_counts"]})
  summary_path.write_text(json.dumps(rows,indent=2))
print("[done] part A",flush=True)
