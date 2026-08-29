from __future__ import annotations
import hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;W=ROOT.parent;P=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");F=W/"FRONT_ENCODER_SERIES"
SOURCE=W/"INTERACTION_RESIDUAL_SUPERVISION_SERIES"/"results"/"general"/"baseline"/"f16_temporal_social_cross"/"seed_1"/"val_scene_-1"/"best_selected.pth"
def sha(path):h=hashlib.sha256();h.update(path.read_bytes());return h.hexdigest()
def load(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 sys.path[:0]=[str(P),str(P/"scripts"),str(F)];os.chdir(P);import sitecustomize  # noqa
 from experiment_config import load_locked_q20
 from front_encoder_models import FrontEncoderASTRA
 cache=ROOT/"_smoke_need.npz";np.savez(cache,metadata_json=np.array(json.dumps({"source_checkpoint_sha256":sha(SOURCE)})),need_label=np.array([0,1,0,1,0,1,0,1],np.float32))
 try:
  cfg=load_locked_q20(P/"configs"/"motif_rbf_univ.yaml");cfg.MODEL.USE_PRETRAINED_UNET=False;cfg.MODEL.USE_CACHED_UNET_FEATURES=False;cfg.DATA.MAX_AGENTS=9;cfg.MODEL.GRAPH.TOP_K=4;cfg.MODEL.GRAPH.RADIUS=6.;cfg.MODEL.DECODER.NUM_MODES=20;cfg.device="cpu";cfg.device_list=["cpu"]
  model=FrontEncoderASTRA(cfg,"f16_temporal_social_cross");os.environ["FCRA_SOURCE_CHECKPOINT"]=str(SOURCE);os.environ["FCRA_NEED_CACHE"]=str(cache);module=load("fcra_smoke",ROOT/"fcra_models.py");module.replace_decoder(model,cfg,"fcra1_full_consistent_top3")
  past=torch.randn(2,9,8,2)*.1;future=torch.randn(2,1,12,2)*.2;num=torch.tensor([4,6]);model.decoder.set_scene_batch(torch.tensor([-2,-3]),past)
  output=model(past,future,None,mode="train",num_valid=num,graph_loc=past)[2];logits=model.last_mode_logits
  assert float(model.decoder.last_applied_anchor_delta.abs().max())==0.0
  loss=model.decoder.auxiliary_training_loss(output[:,:1],future,logits[:,:1]);loss.backward()
  groups={n.split('.')[2]:0. for n,p in model.named_parameters() if n.startswith("decoder.head.")}
  for n,p in model.named_parameters():
   if n.startswith("decoder.head.") and p.grad is not None:groups[n.split('.')[2]]+=float(p.grad.abs().sum())
  added=sum(p.numel() for n,p in model.named_parameters() if n.startswith("decoder.head."));resp=model.decoder.last_target_responsibility
  assert output.shape==(2,9,20,12,2) and torch.isfinite(loss) and all(v>0 for v in groups.values())
  assert torch.all((resp>0).sum(-1)==3) and torch.allclose(resp.sum(-1),torch.ones(2)) and model.decoder.cv_reference is False
  print(f"[ok] shape={tuple(output.shape)} added={added} loss={float(loss):.6f} grads={groups} top3=True zero_init=True cv=False")
 finally:
  if cache.exists():cache.unlink()
if __name__=="__main__":main()

