from __future__ import annotations
import importlib.util,json,os,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent;W=ROOT.parent;P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");F=W/"FRONT_ENCODER_SERIES";CSRS=W/"COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"
SOURCE=CSRS/"results"/"general"/"baseline"/"f16_temporal_social_cross"/"seed_1"/"val_scene_-1"/"best_selected.pth"
def load(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def cfg():
 from experiment_config import load_locked_q20
 c=load_locked_q20(P/"configs"/"motif_rbf_univ.yaml");c.MODEL.USE_PRETRAINED_UNET=False;c.MODEL.USE_CACHED_UNET_FEATURES=False;c.DATA.MAX_AGENTS=9;c.MODEL.GRAPH.TOP_K=4;c.MODEL.GRAPH.RADIUS=6.;c.MODEL.DECODER.NUM_MODES=20;c.device="cpu";c.device_list=["cpu"];return c
def main():
 sys.path[:0]=[str(P),str(P/"scripts"),str(F)];os.chdir(P);import sitecustomize  # noqa
 from front_encoder_models import FrontEncoderASTRA
 scale=ROOT/"_smoke_scale.json";scale.write_text(json.dumps({"selected_sigma":[.18,.24,.32]}),encoding="utf-8")
 try:
  torch.manual_seed(7);past=torch.randn(3,9,8,2)*.2;future=torch.randn(3,1,12,2)*.3;num=torch.tensor([3,5,7]);outputs=[];modules=[];models=[]
  for name in ("cda0_matched_continuation","cda1_fixed_scale_anchor_mixture"):
   c=cfg();model=FrontEncoderASTRA(c,"f16_temporal_social_cross");os.environ["CDA_SOURCE_CHECKPOINT"]=str(SOURCE);os.environ["CDA_SCALE_JSON"]=str(scale);module=load(f"cda_smoke_{name}",ROOT/"cda_models.py");module.replace_decoder(model,c,name);model.eval();out=model(past,future,None,mode="train",num_valid=num,graph_loc=past)[2];outputs.append(out);modules.append(module);models.append(model)
  assert float((outputs[0]-outputs[1]).abs().max())==0.0
  logits=models[1].last_mode_logits;aux=models[1].decoder.auxiliary_training_loss(outputs[1][:,:1],future,logits[:,:1]);aux.backward();encoder_grad=sum(float(p.grad.abs().sum()) for p in models[1].agent_aware_trans_encoder.parameters() if p.grad is not None);decoder_grad=sum(float(p.grad.abs().sum()) for p in models[1].decoder.parameters() if p.grad is not None);d=models[1].decoder.last_mixture_diagnostics
  assert torch.isfinite(aux) and encoder_grad>0 and decoder_grad==0 and torch.isfinite(d["effective_modes"]).all()
  print(f"[ok] init_parity=True aux={float(aux):.6f} N_eff={float(d['effective_modes'].mean()):.3f} encoder_grad={encoder_grad:.3f} frozen_decoder=True")
 finally:
  if scale.exists():scale.unlink()
if __name__=="__main__":main()
