import subprocess
from pathlib import Path

R=Path(__file__).resolve().parent
W=R.parents[1]
P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PY=Path(r"C:\coding_toobox\envs\astra2\python.exe")
T=W/"FRONT_ENCODER_SERIES"/"train_front_variant.py"
D=W/"AUTONOMOUS_EXPERIMENTS"/"EXP_002_LIGHT_KINEMATIC_DECODER"/"decoder_models.py"
O=R/"results"

for variant in ("f4_relation_tokens","f16_temporal_social_cross"):
 for scene in (0,2,3,4):
  rd=O/variant/"seed_1"/f"val_scene_{scene}"
  if (rd/"metrics.json").is_file():
   print(f"[skip] {variant} scene={scene}",flush=True);continue
  rd.mkdir(parents=True,exist_ok=True)
  cmd=[str(PY),str(T),"--project-root",str(P),"--variant",variant,"--decoder-experiment","kinematic_k20","--decoder-module",str(D),"--seed","1","--val-scene-id",str(scene),"--epochs","30","--device","cuda:0","--output-root",str(O)]
  print(f"[run] {variant} scene={scene}",flush=True)
  with (rd/"console.log").open("w",encoding="utf-8") as log:result=subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(rd/"console.log")
print("[cross-scene done]",flush=True)
