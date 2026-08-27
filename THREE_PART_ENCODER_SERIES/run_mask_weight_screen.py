from __future__ import annotations
import json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent; W=R.parent
P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"); PY=Path(r"C:\coding_toobox\envs\astra2\python.exe")
T=W/"FRONT_ENCODER_SERIES"/"train_front_variant.py"; D=W/"AUTONOMOUS_EXPERIMENTS"/"EXP_002_LIGHT_KINEMATIC_DECODER"/"decoder_models.py"
jobs=[("EXP_107_MASKED_HISTORY_W002","0.02"),("EXP_108_MASKED_HISTORY_W010","0.10")]
sp=R/"mask_weight_summary.json"; rows=json.loads(sp.read_text()) if sp.is_file() else []
for name,weight in jobs:
 O=R/name/"results"; rd=O/"f16_temporal_social_cross"/"seed_1"/"val_scene_1"; rd.mkdir(parents=True,exist_ok=True); mp=rd/"metrics.json"
 if mp.is_file() and any(x["experiment"]==name for x in rows): print(f"[skip] {name}",flush=True); continue
 cmd=[str(PY),str(T),"--project-root",str(P),"--variant","f16_temporal_social_cross","--decoder-experiment","kinematic_k20","--decoder-module",str(D),"--seed","1","--val-scene-id","1","--epochs","30","--val-freq","5","--device","cuda:0","--scheduler","warm_restarts","--output-root",str(O),"--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10","--checkpoint-criterion","oracle_combo","--ema-decay","0.995","--augment-rotation","--augment-reflection","--augment-speed-min","0.9","--augment-speed-max","1.1","--augment-probability","0.5","--masked-history-weight",weight,"--masked-history-probability","0.15","--force"]
 print(f"[run] {name}",flush=True)
 with (rd/"console.log").open("w",encoding="utf-8") as log: result=subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode: raise RuntimeError(rd/"console.log")
 m=json.loads(mp.read_text()); v=m["evaluation_best_selected"]
 rows=[x for x in rows if x["experiment"]!=name]; rows.append({"experiment":name,**v["oracle"],"top1_ade":v["original"]["ade"],"top1_fde":v["original"]["fde"]})
 sp.write_text(json.dumps(rows,indent=2))
print("[done] mask weights",flush=True)
