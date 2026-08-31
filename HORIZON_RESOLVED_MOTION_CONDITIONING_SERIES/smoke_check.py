"""Real-data initialization/backward smoke test for both HRMC adapters."""
from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parent;WS=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse"
CONFIG=WS/"OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES"/"motif_rbf_official_4p8.yaml";SOURCE=WS/"OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES"/"EXP_156B_FOLD_LOCAL_CURRENT_DECODER"/"results"/"eth"

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def source(stage):return SOURCE/stage/"f16_temporal_social_cross"/"seed_1"/"val_scene_-1"/"best_selected.pth"
def main():
 sys.path[:0]=[str(PROJECT),str(PROJECT/"scripts"),str(WS/"FRONT_ENCODER_SERIES")];os.chdir(PROJECT);import sitecustomize
 from experiment_config import load_locked_q20
 from front_encoder_models import FrontEncoderASTRA
 from train_front_variant import TrajectoryDataset,normalize_to_focal_heading
 helper=load("hrmc_smoke_helper",WS/"INDIVIDUAL_MOTION_REPRESENTATION_AUDIT"/"run_motion_representation_audit.py")
 module=load("hrmc_smoke_models",ROOT/"motion_conditioned_decoder_models.py")
 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu");cfg=load_locked_q20(CONFIG);cfg.MODEL.USE_PRETRAINED_UNET=False;cfg.MODEL.USE_CACHED_UNET_FEATURES=False;cfg.DATA.MAX_AGENTS=9;cfg.MODEL.GRAPH.TOP_K=4;cfg.MODEL.GRAPH.RADIUS=6.;cfg.device=str(device);cfg.device_list=[device.index or 0]
 dataset=TrajectoryDataset(DATA/"eth_data_test.npy",DATA/"eth_num_test.npy",cfg);past,future,num,*_=next(iter(DataLoader(Subset(dataset,list(range(8))),batch_size=8,shuffle=False,num_workers=0)));past,future,num=past.to(device),future.to(device),num.to(device).int();past,future=normalize_to_focal_heading(past,future);center=past.mean(dim=-2,keepdim=True)
 baseline=helper.build_model("eth",1,cfg,FrontEncoderASTRA,device)
 with torch.no_grad():reference=baseline(past-center,None,None,mode="testing",num_valid=num,graph_loc=past)[2]
 for short,name in (("p1","p1_horizon_motion_proposal_residual"),("c1","c1_horizon_motion_correction_residual")):
  os.environ["HRMC_H3_CHECKPOINT"]=str(source("h3"));os.environ["HRMC_V2_CHECKPOINT"]=str(source("v2"));model=FrontEncoderASTRA(cfg,"f16_temporal_social_cross");module.replace_decoder(model,cfg,name);model=model.to(device).eval()
  with torch.no_grad():initial=model(past-center,None,None,mode="testing",num_valid=num,graph_loc=past)[2]
  difference=float((initial-reference).abs().max());assert difference<2e-5,(short,difference)
  trainable=[p for p in model.parameters() if p.requires_grad];count=sum(p.numel() for p in trainable);assert count>0 and all(p.requires_grad for p in model.decoder.adapter.parameters())
  optimizer=torch.optim.AdamW(trainable,lr=2.5e-4);model.decoder.apply_training_policy(model);prediction=model(past-center,future-center,None,mode="train",num_valid=num,graph_loc=past)[2];loss=torch.linalg.vector_norm(prediction[:,0]-future[:,0,None]+center[:,0,0,None,None],dim=-1).mean();optimizer.zero_grad();loss.backward();gradient=sum(float(p.grad.abs().sum()) for p in trainable if p.grad is not None);assert gradient>0,(short,gradient);optimizer.step()
  model.eval()
  with torch.no_grad():updated=model(past-center,None,None,mode="testing",num_valid=num,graph_loc=past)[2]
  change=float((updated-initial).abs().max());assert change>0,(short,change);print(f"[smoke-ok] {short} params={count} init_diff={difference:.3g} grad={gradient:.3g} step_change={change:.3g}")
if __name__=="__main__":main()
