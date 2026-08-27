from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent; W=ROOT.parent
P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PY=Path(r"C:\coding_toobox\envs\astra2\python.exe")
T=W/"FRONT_ENCODER_SERIES"/"train_front_variant.py"
D=W/"AUTONOMOUS_EXPERIMENTS"/"EXP_002_LIGHT_KINEMATIC_DECODER"/"decoder_models.py"
E=ROOT/"EXP_087_AUGMENT_REPLICATION"

jobs=[]
for seed in (2,3):
    jobs += [("baseline", seed, 1), ("augment", seed, 1)]
for scene in (0,2,3,4):
    jobs += [("baseline", 1, scene), ("augment", 1, scene)]

rows=[]
for label,seed,scene in jobs:
    output=E/("results_"+label)
    run_dir=output/"f16_temporal_social_cross"/f"seed_{seed}"/f"val_scene_{scene}"
    run_dir.mkdir(parents=True,exist_ok=True)
    cmd=[str(PY),str(T),"--project-root",str(P),"--variant","f16_temporal_social_cross",
         "--decoder-experiment","kinematic_k20","--decoder-module",str(D),"--seed",str(seed),
         "--val-scene-id",str(scene),"--epochs","30","--val-freq","5","--device","cuda:0",
         "--scheduler","warm_restarts","--output-root",str(output),"--oracle-coverage-weight","0.20",
         "--oracle-coverage-temperature","0.10","--checkpoint-criterion","oracle_combo","--force"]
    if label=="augment":
        cmd += ["--ema-decay","0.995","--augment-rotation","--augment-reflection",
                "--augment-speed-min","0.9","--augment-speed-max","1.1"]
    print(f"[run] {label} seed={seed} scene={scene}",flush=True)
    with (run_dir/"console.log").open("w",encoding="utf-8") as log:
        result=subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(run_dir/"console.log")
    m=json.loads((run_dir/"metrics.json").read_text(encoding="utf-8"))
    v=m["evaluation_best_selected"]
    rows.append({"method":label,"seed":seed,"scene":scene,"samples":v["samples"],
                 **v["oracle"],"top1_ade":v["original"]["ade"],"top1_fde":v["original"]["fde"]})
    (E/"raw_summary.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
print("[done] winner replication",flush=True)
