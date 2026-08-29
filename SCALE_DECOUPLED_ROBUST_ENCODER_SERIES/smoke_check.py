"""Forward/backward contract check for SDRE1 with the locked decoder."""
from __future__ import annotations
import copy, importlib.util, os, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent; W=ROOT.parent; PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FRONT=W/"FRONT_ENCODER_SERIES"; ADAPTER=W/"COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"/"current_baseline_adapter.py"
SOURCE=W/"MODE_TEMPLATE_GROUP_SERIES"/"EXP_146B_ANCHORED_MODE_TEMPLATE"/"results"/"general"/"f16_temporal_social_cross"/"seed_1"/"val_scene_-1"/"best_selected.pth"
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(m);return m
def main():
 sys.path[:0]=[str(PROJECT),str(PROJECT/"scripts"),str(FRONT)];os.chdir(PROJECT);import sitecustomize # noqa
 from experiment_config import load_locked_q20
 from front_encoder_models import FrontEncoderASTRA
 cfg=load_locked_q20(PROJECT/"configs"/"motif_rbf_univ.yaml");cfg.MODEL.USE_PRETRAINED_UNET=False;cfg.MODEL.USE_CACHED_UNET_FEATURES=False;cfg.DATA.MAX_AGENTS=9;cfg.MODEL.GRAPH.TOP_K=4;cfg.MODEL.GRAPH.RADIUS=6.;cfg.MODEL.DECODER.NUM_MODES=20;cfg.device="cpu";cfg.device_list=["cpu"]
 os.environ["CSRS_SOURCE_CHECKPOINT"]=str(SOURCE);model=FrontEncoderASTRA(cfg,"f16_temporal_social_cross");load("sdre",ROOT/"sdre1_models.py").replace_encoder(model,cfg,"sdre1_scale_decoupled");load("adapter",ADAPTER).replace_decoder(model,cfg,"current_mode_anchor_template")
 # Match ModelEMA: clone before the first forward, then use a different batch.
 ema=copy.deepcopy(model).eval()
 past=torch.randn(2,9,8,2);future=torch.randn(2,9,12,2);valid=torch.tensor([9,4]);out=model(past,future,None,mode="train",num_valid=valid,graph_loc=past)[2];loss=out.square().mean();loss.backward()
 ema_past=torch.randn(3,9,8,2);ema_valid=torch.tensor([9,6,3]);ema_out=ema(ema_past,None,None,mode="testing",num_valid=ema_valid,graph_loc=ema_past)[2]
 enc=model.agent_aware_trans_encoder;required=("last_sdre_scale","last_sdre_condition","last_sdre_relation_gate")
 if not all(hasattr(enc,k) for k in required) or not hasattr(enc.temporal_queries,"last_sdre_shape_gate"):raise RuntimeError("SDRE diagnostics missing")
 trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); grads=sum(p.numel() for p in model.parameters() if p.requires_grad and p.grad is not None)
 if not torch.isfinite(out).all() or not torch.isfinite(ema_out).all() or not torch.isfinite(loss):raise RuntimeError("nonfinite SDRE forward")
 print(f"[ok] output={tuple(out.shape)} ema_output={tuple(ema_out.shape)} loss={loss.item():.6f} trainable={trainable} grad_params={grads} relation_gate={enc.last_sdre_relation_gate.mean().item():.4f} shape_gate={enc.temporal_queries.last_sdre_shape_gate.mean().item():.4f}")
if __name__=="__main__":main()
