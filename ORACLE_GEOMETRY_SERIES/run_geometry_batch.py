"""Run the Oracle candidate-geometry screen with a locked F16 encoder."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
W=HERE.parent
P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PY=Path(r"C:\coding_toobox\envs\astra2\python.exe")
TRAIN=W/"FRONT_ENCODER_SERIES"/"train_front_variant.py"
BASE_DECODER=W/"AUTONOMOUS_EXPERIMENTS"/"EXP_002_LIGHT_KINEMATIC_DECODER"/"decoder_models.py"

EXPERIMENTS={
    "conditional_rotation_k20": (HERE/"EXP_066_CONDITIONAL_ANCHOR_ROTATION", "decoder_models.py", []),
    "mid_control_k20": (HERE/"EXP_067_MID_CONTROL_POINT", "decoder_models.py", []),
    "learnable_basis_k20": (HERE/"EXP_068_LEARNABLE_MOTION_BASIS", "decoder_models.py", []),
    "coverage_k20": (HERE/"EXP_069_SOFT_ORACLE_COVERAGE", None, ["--oracle-coverage-weight","0.15","--oracle-coverage-temperature","0.10"]),
}


def args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--experiments",nargs="+",choices=list(EXPERIMENTS),default=list(EXPERIMENTS))
    parser.add_argument("--seeds",nargs="+",type=int,default=[1])
    parser.add_argument("--val-scenes",nargs="+",type=int,default=[1])
    parser.add_argument("--epochs",type=int,default=30)
    parser.add_argument("--val-freq",type=int,default=5)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--force",action="store_true")
    return parser.parse_args()


def flatten(payload):
    oracle=payload["evaluation"]["oracle"]; original=payload["evaluation"]["original"]
    return {"experiment":payload["decoder_experiment"],"seed":payload["seed"],"scene":payload["validation_scene_id"],
        "parameters":payload["locked_backend"]["parameter_counts"]["total"],"decoder_parameters":payload["locked_backend"]["parameter_counts"]["decoder"],
        "min_ade_20":oracle["min_ade_20"],"same_path_fde":oracle["same_path_fde"],"min_fde_20":oracle["min_fde_20"],
        "original_ade":original["ade"],"original_fde":original["fde"],"elapsed_seconds":payload["elapsed_seconds"]}


def main():
    a=args(); rows=[]
    for name in a.experiments:
        folder,module_name,extra=EXPERIMENTS[name]
        decoder=folder/module_name if module_name else BASE_DECODER
        decoder_name=name if name != "coverage_k20" else "kinematic_k20"
        output=folder/"results"
        for scene in a.val_scenes:
            for seed in a.seeds:
                rd=output/"f16_temporal_social_cross"/f"seed_{seed}"/f"val_scene_{scene}"
                rd.mkdir(parents=True,exist_ok=True)
                cmd=[str(PY),str(TRAIN),"--project-root",str(P),"--variant","f16_temporal_social_cross",
                    "--decoder-experiment",decoder_name,"--decoder-module",str(decoder),"--seed",str(seed),"--val-scene-id",str(scene),
                    "--epochs",str(a.epochs),"--val-freq",str(a.val_freq),"--device",a.device,"--scheduler","warm_restarts","--output-root",str(output),*extra]
                if a.force: cmd.append("--force")
                print(f"[run] {name} seed={seed} scene={scene}",flush=True)
                with (rd/"console.log").open("w",encoding="utf-8") as log:
                    result=subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode: raise RuntimeError(f"Failed {name}; see {rd/'console.log'}")
                rows.append(flatten(json.loads((rd/"metrics.json").read_text(encoding="utf-8"))))
        (folder/"summary.json").write_text(json.dumps([r for r in rows if r["experiment"]==decoder_name],indent=2),encoding="utf-8")
    (HERE/"batch_summary.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    print("[done]",flush=True)


if __name__=="__main__": main()
