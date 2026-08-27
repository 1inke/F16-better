import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent; W=HERE.parent
P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PY=Path(r"C:\coding_toobox\envs\astra2\python.exe")
T=W/"FRONT_ENCODER_SERIES"/"train_front_variant.py"
D=W/"AUTONOMOUS_EXPERIMENTS"/"EXP_002_LIGHT_KINEMATIC_DECODER"/"decoder_models.py"
jobs=[
    (HERE/"EXP_070_COVERAGE_WEIGHT_SWEEP"/"results_w0p05",0.05,0.0,"w0p05"),
    (HERE/"EXP_070_COVERAGE_WEIGHT_SWEEP"/"results_w0p10",0.10,0.0,"w0p10"),
    (HERE/"EXP_070_COVERAGE_WEIGHT_SWEEP"/"results_w0p20",0.20,0.0,"w0p20"),
    (HERE/"EXP_071_COVERAGE_WARMUP"/"results",0.15,5.0,"w0p15_warm5"),
]
rows=[]
for output,weight,warmup,label in jobs:
    rd=output/"f16_temporal_social_cross"/"seed_1"/"val_scene_1"; rd.mkdir(parents=True,exist_ok=True)
    cmd=[str(PY),str(T),"--project-root",str(P),"--variant","f16_temporal_social_cross","--decoder-experiment","kinematic_k20","--decoder-module",str(D),"--seed","1","--val-scene-id","1","--epochs","30","--val-freq","5","--device","cuda:0","--scheduler","warm_restarts","--output-root",str(output),"--oracle-coverage-weight",str(weight),"--oracle-coverage-temperature","0.10","--oracle-coverage-warmup-epochs",str(warmup),"--force"]
    print(f"[run] {label}",flush=True)
    with (rd/"console.log").open("w",encoding="utf-8") as log: result=subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(rd/"console.log")
    m=json.loads((rd/"metrics.json").read_text(encoding="utf-8")); o=m["evaluation"]["oracle"]
    rows.append({"label":label,"weight":weight,"warmup":warmup,**o,"top1_ade":m["evaluation"]["original"]["ade"],"top1_fde":m["evaluation"]["original"]["fde"]})
(HERE/"coverage_tuning_summary.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
print("[done]",flush=True)
