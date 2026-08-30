"""EXP-157A: evaluate frozen H3/V2/M2 checkpoints on all held-out folds."""
from __future__ import annotations
import argparse,csv,importlib.util,json,os,statistics,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset

ROOT=Path(__file__).resolve().parent;WS=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse"
CONFIG=WS/"OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES"/"motif_rbf_official_4p8.yaml"
SOURCE=WS/"OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES"/"EXP_156B_FOLD_LOCAL_CURRENT_DECODER"/"results"
TRAINER=WS/"FRONT_ENCODER_SERIES"/"train_front_variant.py"
MODULES={"h3":WS/"HIERARCHICAL_DECODER_SERIES"/"hierarchical_decoder_models.py",
         "v2":WS/"ANCHOR_CORRECTION_VALIDATION_SERIES"/"anchor_correction_validation_models.py",
         "m2":WS/"MODE_TEMPLATE_GROUP_SERIES"/"grouped_mode_template_models.py"}
NAMES={"h3":"h3_anchor_proposal","v2":"v2_wide_endpoint_correction","m2":"m2_anchored_mode_template"}
FOLDS=("eth","hotel","univ","zara01","zara02");STAGES=("h3","v2","m2")
OUT=ROOT/"EXP_157A_FROZEN_STAGE_ATTRIBUTION"

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);assert spec and spec.loader;spec.loader.exec_module(m);return m
def run_dir(fold,stage,seed):return SOURCE/fold/stage/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def build(fold,stage,seed,cfg,FrontEncoderASTRA,device):
    checkpoint=run_dir(fold,stage,seed)/"best_selected.pth"
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if stage=="v2":os.environ["EXP143_SOURCE_CHECKPOINT"]=str(run_dir(fold,"h3",seed)/"best_selected.pth")
    if stage=="m2":os.environ["EXP146_SOURCE_CHECKPOINT"]=str(run_dir(fold,"v2",seed)/"best_selected.pth")
    model=FrontEncoderASTRA(cfg,"f16_temporal_social_cross");module=load_module(f"exp157_{stage}_{fold}_{seed}",MODULES[stage]);module.replace_decoder(model,cfg,NAMES[stage])
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
    return model.to(device).eval()
def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--folds",nargs="+",choices=("all",*FOLDS),default=["all"]);p.add_argument("--stages",nargs="+",choices=("all",*STAGES),default=["all"]);p.add_argument("--seed",type=int,default=1);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--max-samples",type=int,default=0);a=p.parse_args()
    for x in (CONFIG,TRAINER,*MODULES.values()):
        if not x.is_file():raise FileNotFoundError(x)
    sys.path[:0]=[str(PROJECT),str(PROJECT/"scripts"),str(WS/"FRONT_ENCODER_SERIES")];os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from train_front_variant import TrajectoryDataset,evaluate
    device=torch.device(a.device);cfg=load_locked_q20(CONFIG);cfg.MODEL.USE_PRETRAINED_UNET=False;cfg.MODEL.USE_CACHED_UNET_FEATURES=False;cfg.DATA.MAX_AGENTS=9;cfg.MODEL.GRAPH.TOP_K=4;cfg.MODEL.GRAPH.RADIUS=6.0;cfg.device=str(device);cfg.device_list=[device.index or 0]
    folds=FOLDS if "all" in a.folds else tuple(a.folds);stages=STAGES if "all" in a.stages else tuple(a.stages);rows=[]
    for fold in folds:
      dataset=TrajectoryDataset(DATA/f"{fold}_data_test.npy",DATA/f"{fold}_num_test.npy",cfg)
      if a.max_samples:dataset=Subset(dataset,range(min(a.max_samples,len(dataset))))
      loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=False,num_workers=0)
      for stage in stages:
       model=build(fold,stage,a.seed,cfg,FrontEncoderASTRA,device);e=evaluate(model,loader,device,None,.30,True);o=e["oracle"];s=e["original"];d=e.get("proposal_diagnostics",{})
       row={"fold":fold,"stage":stage,"seed":a.seed,"samples":e["samples"],"min_ade_20":o["min_ade_20"],"min_fde_20":o["min_fde_20"],"same_path_fde":o["same_path_fde"],"top1_ade":s["ade"],"top1_fde":s["fde"],"hit_best_ade":s["hit_best_ade"],"regret":s["regret"],"active_modes":s["active_modes"],"coarse_min_ade_20":d.get("coarse_min_ade_20"),"refinement_min_ade_delta":d.get("refinement_min_ade_delta"),"refinement_min_fde_delta":d.get("refinement_min_fde_delta")}
       rows.append(row);print("[done] "+json.dumps(row),flush=True);del model;torch.cuda.empty_cache()
    aggregate=[]
    for stage in stages:
      group=[r for r in rows if r["stage"]==stage]
      if group:aggregate.append({"stage":stage,**{m:statistics.fmean(r[m] for r in group) for m in ("min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde")}})
    OUT.mkdir(parents=True,exist_ok=True);tag="smoke" if a.max_samples else "full";(OUT/f"stage_attribution_{tag}.json").write_text(json.dumps({"rows":rows,"macro":aggregate},indent=2),encoding="utf-8")
    if rows:
      with (OUT/f"stage_attribution_{tag}.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if __name__=="__main__":main()
