from __future__ import annotations
import json, subprocess
from pathlib import Path
R=Path(__file__).resolve().parent; W=R.parent
P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"); PY=Path(r"C:\coding_toobox\envs\astra2\python.exe")
T=W/"FRONT_ENCODER_SERIES"/"train_front_variant.py"; D=W/"AUTONOMOUS_EXPERIMENTS"/"EXP_002_LIGHT_KINEMATIC_DECODER"/"decoder_models.py"
jobs=[
 ("EXP_088_HALF_PROBABILITY_AUGMENT",["--augment-rotation","--augment-reflection","--augment-speed-min","0.9","--augment-speed-max","1.1","--augment-probability","0.5"]),
 ("EXP_089_RIGID_ONLY_AUGMENT",["--augment-rotation","--augment-reflection"]),
 ("EXP_090_ENDPOINT_WEIGHTED_COVERAGE",["--augment-rotation","--augment-reflection","--augment-speed-min","0.9","--augment-speed-max","1.1","--oracle-coverage-fde-weight","0.50"]),
]
rows=[]
for name,extra in jobs:
 O=R/name/"results"; rd=O/"f16_temporal_social_cross"/"seed_1"/"val_scene_1"; rd.mkdir(parents=True,exist_ok=True)
 cmd=[str(PY),str(T),"--project-root",str(P),"--variant","f16_temporal_social_cross","--decoder-experiment","kinematic_k20","--decoder-module",str(D),"--seed","1","--val-scene-id","1","--epochs","30","--val-freq","5","--device","cuda:0","--scheduler","warm_restarts","--output-root",str(O),"--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10","--checkpoint-criterion","oracle_combo","--ema-decay","0.995","--force",*extra]
 print(f"[run] {name}",flush=True)
 with (rd/"console.log").open("w",encoding="utf-8") as log: result=subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode: raise RuntimeError(rd/"console.log")
 m=json.loads((rd/"metrics.json").read_text()); v=m["evaluation_best_selected"]
 rows.append({"experiment":name,**v["oracle"],"top1_ade":v["original"]["ade"],"top1_fde":v["original"]["fde"]})
 (R/"stage4_refinement_summary.json").write_text(json.dumps(rows,indent=2))
print("[done] refinement",flush=True)
