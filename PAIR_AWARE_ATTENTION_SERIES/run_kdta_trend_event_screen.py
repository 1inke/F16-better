from __future__ import annotations
import json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent; W=ROOT.parent; E=ROOT/"EXP_115_KDTA_TREND_EVENT_RECENT"
P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"); PY=Path(r"C:\coding_toobox\envs\astra2\python.exe")
T=W/"FRONT_ENCODER_SERIES"/"train_front_variant.py"; D=W/"AUTONOMOUS_EXPERIMENTS"/"EXP_002_LIGHT_KINEMATIC_DECODER"/"decoder_models.py"
def run():
 rows=[]; output=E/"results"; variant="f54_kdta_trend_event_recent"
 for scene in (1,3):
  rd=output/variant/"seed_1"/f"val_scene_{scene}"; rd.mkdir(parents=True,exist_ok=True); mp=rd/"metrics.json"
  if not mp.is_file():
   cmd=[str(PY),str(T),"--project-root",str(P),"--variant",variant,"--decoder-experiment","kinematic_k20","--decoder-module",str(D),"--seed","1","--val-scene-id",str(scene),"--epochs","30","--val-freq","5","--device","cuda:0","--scheduler","warm_restarts","--output-root",str(output),"--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10","--checkpoint-criterion","oracle_combo","--ema-decay","0.995","--focal-heading-normalize","--augment-reflection","--augment-speed-min","0.9","--augment-speed-max","1.1","--augment-probability","0.5","--force"]
   print(f"[run] KDTA trend-event-recent scene={scene}",flush=True)
   with (rd/"console.log").open("w",encoding="utf-8") as log: result=subprocess.run(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT)
   if result.returncode: raise RuntimeError(rd/"console.log")
  m=json.loads(mp.read_text()); s=m["evaluation_best_selected"]
  rows.append({"variant":variant,"seed":1,"scene":scene,"samples":s["samples"],**s["oracle"],"top1_ade":s["original"]["ade"],"top1_fde":s["original"]["fde"],"fused_ade":s["fused"]["ade"],"fused_fde":s["fused"]["fde"]})
  (E/"screen_summary.json").write_text(json.dumps(rows,indent=2))
 print("[done] KDTA trend-event-recent",flush=True)
if __name__=="__main__": run()
