import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent; W=ROOT.parents[1]
P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"); PY=Path(r"C:\coding_toobox\envs\astra2\python.exe")
T=W/"FRONT_ENCODER_SERIES"/"train_front_variant.py"; D=W/"AUTONOMOUS_EXPERIMENTS"/"EXP_002_LIGHT_KINEMATIC_DECODER"/"decoder_models.py"; O=ROOT/"results"
for scene in (0,2,3,4):
    rd=O/"f16_temporal_social_cross"/"seed_1"/f"val_scene_{scene}"; rd.mkdir(parents=True,exist_ok=True)
    cmd=[str(PY),str(T),"--project-root",str(P),"--variant","f16_temporal_social_cross","--decoder-experiment","kinematic_k20","--decoder-module",str(D),"--seed","1","--val-scene-id",str(scene),"--epochs","30","--val-freq","5","--device","cuda:0","--scheduler","warm_restarts","--output-root",str(O),"--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10","--force"]
    print(f"[run] scene={scene}",flush=True)
    with (rd/"console.log").open("w",encoding="utf-8") as log: result=subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(rd/"console.log")
print("[done]",flush=True)
