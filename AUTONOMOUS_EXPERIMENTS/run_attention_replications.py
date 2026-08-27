import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON=Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAIN=WORKSPACE/"FRONT_ENCODER_SERIES"/"train_front_variant.py"
DECODER=ROOT/"EXP_002_LIGHT_KINEMATIC_DECODER"/"decoder_models.py"
EXPERIMENTS=[
 ("EXP_034_DYNAMIC_ROUTING_REPLICATION","f13_dynamic_routing"),
 ("EXP_035_TEMPORAL_SOCIAL_CROSS_REPLICATION","f16_temporal_social_cross"),
 ("EXP_036_SPARSEMAX_REPLICATION","f17_sparsemax_attention"),
 ("EXP_037_RELATION_QK_REPLICATION","f18_relation_qk"),
]

for folder,variant in EXPERIMENTS:
 output=ROOT/folder/"results"
 for seed in (2,3):
  run_dir=output/variant/f"seed_{seed}"/"val_scene_1"
  if (run_dir/"metrics.json").is_file():
   print(f"[skip] {variant} seed={seed}",flush=True);continue
  run_dir.mkdir(parents=True,exist_ok=True)
  cmd=[str(PYTHON),str(TRAIN),"--project-root",str(PROJECT),"--variant",variant,"--decoder-experiment","kinematic_k20","--decoder-module",str(DECODER),"--seed",str(seed),"--val-scene-id","1","--epochs","30","--device","cuda:0","--output-root",str(output)]
  print(f"[run] {variant} seed={seed}",flush=True)
  with (run_dir/"console.log").open("w",encoding="utf-8") as log: result=subprocess.run(cmd,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode: raise RuntimeError(run_dir/"console.log")
print("[replications done]",flush=True)
