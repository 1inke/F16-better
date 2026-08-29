from __future__ import annotations
import importlib.util,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;W=ROOT.parent;P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");F=W/"FRONT_ENCODER_SERIES"
SOURCE=W/"INTERACTION_RESIDUAL_SUPERVISION_SERIES"/"results"/"general"/"baseline"/"f16_temporal_social_cross"/"seed_1"/"val_scene_-1"/"best_selected.pth"
def load(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def cfg():
 from experiment_config import load_locked_q20
 c=load_locked_q20(P/"configs"/"motif_rbf_univ.yaml");c.MODEL.USE_PRETRAINED_UNET=False;c.MODEL.USE_CACHED_UNET_FEATURES=False;c.DATA.MAX_AGENTS=9;c.MODEL.GRAPH.TOP_K=4;c.MODEL.GRAPH.RADIUS=6.;c.MODEL.DECODER.NUM_MODES=20;c.device="cpu";c.device_list=["cpu"];return c
def main():
 sys.path[:0]=[str(P),str(P/"scripts"),str(F)];os.chdir(P);import sitecustomize  # noqa
 from front_encoder_models import FrontEncoderASTRA
 torch.manual_seed(11);past=torch.randn(3,9,8,2)*.1;future=torch.randn(3,1,12,2)*.2;num=torch.tensor([4,6,5]);os.environ["CSO_SOURCE_CHECKPOINT"]=str(SOURCE)
 base_cfg=cfg();baseline=FrontEncoderASTRA(base_cfg,"f16_temporal_social_cross");os.environ["CSRS_SOURCE_CHECKPOINT"]=str(SOURCE);b=load("cso_smoke_b",W/"COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"/"current_baseline_adapter.py");b.replace_decoder(baseline,base_cfg,"current_mode_anchor_template");baseline.eval()
 with torch.no_grad():reference=baseline(past,future,None,mode="train",num_valid=num,graph_loc=past)[2]
 for variant in ("cso1_winner_protected_repulsion","cso1_repulsion_prototype"):
  c=cfg();model=FrontEncoderASTRA(c,"f16_temporal_social_cross");module=load("cso_smoke_"+variant,ROOT/"cso_models.py");module.replace_decoder(model,c,variant);model.eval();output=model(past,future,None,mode="train",num_valid=num,graph_loc=past)[2];logits=model.last_mode_logits
  diff=float((output-reference).abs().max());assert output.shape==(3,9,20,12,2) and diff<2e-6,diff
  loss_fn=module.build_loss(c,model);loss,parts=loss_fn(output[:,:1],future,logits[:,:1],c)
  if "prototype" in variant:loss,parts=loss_fn(output[:,:1]+.01,future,logits[:,:1],c)
  loss.backward();grad=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith("agent_aware_trans_encoder.") and p.grad is not None)
  assert torch.isfinite(loss) and grad>0 and float(parts["diversity"])>=0
  if "prototype" in variant:assert int(model.decoder.prototype_seen.sum())>0 and int(model.decoder.prototype_updates.sum())==6 and float(parts["prototype"])>0
  print(f"[ok] {variant} parity=True shape={tuple(output.shape)} loss={float(loss):.6f} rep={float(parts['diversity']):.6f} proto={float(parts['prototype']):.6f} grad={grad:.3f}")
if __name__=="__main__":main()
