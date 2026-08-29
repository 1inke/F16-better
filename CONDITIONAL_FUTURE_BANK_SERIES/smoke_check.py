from __future__ import annotations
import importlib.util,json,os,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;W=ROOT.parent;P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");F=W/"FRONT_ENCODER_SERIES";SOURCE=W/"INTERACTION_RESIDUAL_SUPERVISION_SERIES"/"results"/"general"/"baseline"/"f16_temporal_social_cross"/"seed_1"/"val_scene_-1"/"best_selected.pth"
def load(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def cfg():
 from experiment_config import load_locked_q20
 c=load_locked_q20(P/"configs"/"motif_rbf_univ.yaml");c.MODEL.USE_PRETRAINED_UNET=False;c.MODEL.USE_CACHED_UNET_FEATURES=False;c.DATA.MAX_AGENTS=9;c.MODEL.GRAPH.TOP_K=4;c.MODEL.GRAPH.RADIUS=6.;c.MODEL.DECODER.NUM_MODES=20;c.device="cpu";c.device_list=["cpu"];return c
def main():
 sys.path[:0]=[str(P),str(P/"scripts"),str(F)];os.chdir(P);import sitecustomize  # noqa
 from front_encoder_models import FrontEncoderASTRA
 torch.manual_seed(13);past=torch.randn(2,9,8,2)*.1;future=torch.randn(2,1,12,2)*.2;num=torch.tensor([4,6]);cache=ROOT/"_smoke_cache.npz";pseudo=np.random.default_rng(3).normal(0,.3,(4,3,12,2)).astype(np.float32);np.savez(cache,pseudo_futures=pseudo,metadata_json=np.array(json.dumps({"bank_policy":"smoke"})))
 try:
  base_cfg=cfg();base=FrontEncoderASTRA(base_cfg,"f16_temporal_social_cross");os.environ["CSRS_SOURCE_CHECKPOINT"]=str(SOURCE);b=load("cfb_smoke_b",W/"COUNTERFACTUAL_SOCIAL_RESIDUAL_SUPERVISION_SERIES"/"current_baseline_adapter.py");b.replace_decoder(base,base_cfg,"current_mode_anchor_template");base.eval()
  with torch.no_grad():reference=base(past,future,None,mode="train",num_valid=num,graph_loc=past)[2]
  c=cfg();model=FrontEncoderASTRA(c,"f16_temporal_social_cross");os.environ["CFB_SOURCE_CHECKPOINT"]=str(SOURCE);os.environ["CFB_CACHE"]=str(cache);module=load("cfb_smoke",ROOT/"cfb_models.py");module.replace_decoder(model,c,"cfb1_cross_scene_three_future");model.eval();model.decoder.set_scene_batch(torch.tensor([-2,-3]),past);output=model(past,future,None,mode="train",num_valid=num,graph_loc=past)[2];logits=model.last_mode_logits;diff=float((output-reference).abs().max());assert diff<2e-6,diff
  loss,parts=module.build_loss(c,model)(output[:,:1],future,logits[:,:1],c);assignment=parts["pseudo_assignment"];winner=parts["winner"];assert assignment.shape==(2,1,3) and torch.all(assignment.sort(-1).values.diff(dim=-1)>0) and not torch.any(assignment==winner[...,None]);loss.backward();grad=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith("agent_aware_trans_encoder.") and p.grad is not None);assert torch.isfinite(loss) and grad>0 and int(model.decoder.pseudo_assignment_total)==6
  print(f"[ok] parity=True shape={tuple(output.shape)} pseudo={float(parts['pseudo']):.6f} unique=True winner_excluded=True grad={grad:.3f}")
 finally:
  if cache.exists():cache.unlink()
if __name__=="__main__":main()
