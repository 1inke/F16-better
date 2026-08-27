from __future__ import annotations
import json, subprocess
from pathlib import Path
R=Path(__file__).resolve().parent; W=R.parent; E=R/"EXP_091_HALF_AUGMENT_REPLICATION"
P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"); PY=Path(r"C:\coding_toobox\envs\astra2\python.exe")
T=W/"FRONT_ENCODER_SERIES"/"train_front_variant.py"; D=W/"AUTONOMOUS_EXPERIMENTS"/"EXP_002_LIGHT_KINEMATIC_DECODER"/"decoder_models.py"
jobs=[(2,1),(3,1),(1,0),(1,2),(1,3),(1,4)]
summary_path=E/"raw_summary.json"
rows=json.loads(summary_path.read_text()) if summary_path.is_file() else []
for seed,scene in jobs:
 O=E/"results"; rd=O/"f16_temporal_social_cross"/f"seed_{seed}"/f"val_scene_{scene}"; rd.mkdir(parents=True,exist_ok=True)
 metrics_path=rd/"metrics.json"
 existing=next((row for row in rows if row["seed"]==seed and row["scene"]==scene),None)
 if existing is not None and metrics_path.is_file():
  print(f"[skip] half augment seed={seed} scene={scene}",flush=True)
  continue
 cmd=[str(PY),str(T),"--project-root",str(P),"--variant","f16_temporal_social_cross","--decoder-experiment","kinematic_k20","--decoder-module",str(D),"--seed",str(seed),"--val-scene-id",str(scene),"--epochs","30","--val-freq","5","--device","cuda:0","--scheduler","warm_restarts","--output-root",str(O),"--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10","--checkpoint-criterion","oracle_combo","--ema-decay","0.995","--augment-rotation","--augment-reflection","--augment-speed-min","0.9","--augment-speed-max","1.1","--augment-probability","0.5","--force"]
 print(f"[run] half augment seed={seed} scene={scene}",flush=True)
 with (rd/"console.log").open("w",encoding="utf-8") as log: result=subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode: raise RuntimeError(rd/"console.log")
 m=json.loads(metrics_path.read_text()); v=m["evaluation_best_selected"]
 rows=[row for row in rows if not (row["seed"]==seed and row["scene"]==scene)]
 rows.append({"seed":seed,"scene":scene,"samples":v["samples"],**v["oracle"],"top1_ade":v["original"]["ade"],"top1_fde":v["original"]["fde"]})
 summary_path.write_text(json.dumps(rows,indent=2))
print("[done] half augment replication",flush=True)
