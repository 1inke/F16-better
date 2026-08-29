from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent; W=ROOT.parent; P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"); F=W/"FRONT_ENCODER_SERIES"
SOURCE=W/"INTERACTION_RESIDUAL_SUPERVISION_SERIES"/"results"/"general"/"baseline"/"f16_temporal_social_cross"/"seed_1"/"val_scene_-1"/"best_selected.pth"
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
def config():
 from experiment_config import load_locked_q20
 cfg=load_locked_q20(P/"configs"/"motif_rbf_univ.yaml"); cfg.MODEL.USE_PRETRAINED_UNET=False; cfg.MODEL.USE_CACHED_UNET_FEATURES=False
 cfg.DATA.MAX_AGENTS=9; cfg.MODEL.GRAPH.TOP_K=4; cfg.MODEL.GRAPH.RADIUS=6.; cfg.MODEL.DECODER.NUM_MODES=20; cfg.device="cpu"; cfg.device_list=["cpu"]; return cfg
def main():
 sys.path[:0]=[str(P),str(P/"scripts"),str(F)]; os.chdir(P); import sitecustomize  # noqa
 from front_encoder_models import FrontEncoderASTRA
 torch.manual_seed(7); past=torch.randn(2,9,8,2)*.1; future=torch.randn(2,1,12,2)*.2; num=torch.tensor([4,6])
 cfg_b=config(); baseline=FrontEncoderASTRA(cfg_b,"f16_temporal_social_cross"); os.environ["CSRS_SOURCE_CHECKPOINT"]=str(SOURCE)
 bmod=load("ma_smoke_b0",W/"COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"/"current_baseline_adapter.py"); bmod.replace_decoder(baseline,cfg_b,"current_mode_anchor_template"); baseline.eval()
 with torch.no_grad(): bpred=baseline(past,future,None,mode="train",num_valid=num,graph_loc=past)[2]
 cfg=config(); model=FrontEncoderASTRA(cfg,"f16_temporal_social_cross"); os.environ["IAMA_SOURCE_CHECKPOINT"]=str(SOURCE)
 module=load("ma_smoke",ROOT/"iama_models.py"); module.replace_decoder(model,cfg,"ma1_interaction_aware_top5"); model.eval(); model.decoder.compatibility.train(True)
 output=model(past,future,None,mode="train",num_valid=num,graph_loc=past)[2]; logits=model.last_mode_logits
 max_difference=float((output-bpred).abs().max())
 assert output.shape==(2,9,20,12,2) and max_difference<2e-6, max_difference
 assert float(model.decoder.last_compatibility_logits.abs().max())==0.0 and model.decoder.cv_reference is False
 loss_fn=module.build_loss(cfg,model); loss,_=loss_fn(output[:,:1],future,logits[:,:1],cfg); loss.backward()
 target=model.decoder.last_allocation_target; assert torch.all((target>0).sum(-1)==5) and torch.allclose(target.sum(-1),torch.ones_like(target.sum(-1)))
 head_grad=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith("decoder.compatibility.") and p.grad is not None)
 encoder_grad=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith("agent_aware_trans_encoder.") and p.grad is not None)
 added=sum(p.numel() for n,p in model.named_parameters() if n.startswith("decoder.compatibility."))
 assert torch.isfinite(loss) and head_grad>0 and encoder_grad>0
 print(f"[ok] shape={tuple(output.shape)} parity=True top5=True added={added} loss={float(loss):.6f} head_grad={head_grad:.4f} encoder_grad={encoder_grad:.4f} cv=False")
if __name__=="__main__": main()
