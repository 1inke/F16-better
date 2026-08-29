"""Architecture, cache-index, zero-init and gradient contract for CSRS1."""

from __future__ import annotations
import hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;WORKSPACE=ROOT.parent;PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20");FRONT=WORKSPACE/"FRONT_ENCODER_SERIES"
SOURCE=WORKSPACE/"INTERACTION_RESIDUAL_SUPERVISION_SERIES"/"results"/"general"/"baseline"/"f16_temporal_social_cross"/"seed_1"/"val_scene_-1"/"best_selected.pth"

def sha(path):
    h=hashlib.sha256();h.update(path.read_bytes());return h.hexdigest()

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    sys.path[:0]=[str(PROJECT),str(PROJECT/"scripts"),str(FRONT)];os.chdir(PROJECT);import sitecustomize  # noqa
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    cache_path=ROOT/"_smoke_cache.npz";count=8
    meta={"source_checkpoint_sha256":sha(SOURCE),"delta":.01,"tau":.10}
    np.savez(cache_path,metadata_json=np.array(json.dumps(meta)),need_label=np.array([0,1,0,1,0,1,0,1],np.float32),
             confidence=np.linspace(0,1,count,dtype=np.float32),residual_target=np.ones((count,3,2),np.float32)*.1,social_gain=np.linspace(-.1,.2,count,dtype=np.float32))
    try:
        cfg=load_locked_q20(PROJECT/"configs"/"motif_rbf_univ.yaml");cfg.MODEL.USE_PRETRAINED_UNET=False;cfg.MODEL.USE_CACHED_UNET_FEATURES=False
        cfg.DATA.MAX_AGENTS=9;cfg.MODEL.GRAPH.TOP_K=4;cfg.MODEL.GRAPH.RADIUS=6.;cfg.MODEL.DECODER.NUM_MODES=20;cfg.device="cpu";cfg.device_list=["cpu"]
        csrs=FrontEncoderASTRA(cfg,"f16_temporal_social_cross");os.environ["CSRS_SOURCE_CHECKPOINT"]=str(SOURCE);os.environ["CSRS_TEACHER_CACHE"]=str(cache_path)
        module=load("csrs_smoke_model",ROOT/"csrs_models.py");module.replace_decoder(csrs,cfg,"csrs1_actual_gain_residual");csrs.eval()
        past=torch.randn(2,9,8,2)*.1;num=torch.tensor([4,6]);csrs.decoder.set_scene_batch(torch.tensor([-2,-3]),past)
        output=csrs(past,None,None,mode="train",num_valid=num,graph_loc=past)[2];logits=csrs.last_mode_logits
        target=torch.randn(2,1,12,2)*.2;loss=csrs.decoder.auxiliary_training_loss(output[:,:1],target,logits[:,:1]);loss.backward()
        grad=sum(float(p.grad.abs().sum()) for p in csrs.decoder.head.parameters() if p.grad is not None)
        added=sum(p.numel() for n,p in csrs.named_parameters() if n.startswith(("decoder.head.","decoder.mode_gate.")))
        assert output.shape==(2,9,20,12,2) and torch.isfinite(loss) and grad>0
        assert float(csrs.decoder.last_applied_anchor_delta.abs().max())==0.0,"zero-init changed source geometry"
        assert getattr(csrs.decoder,"cv_reference",None) is False
        print(f"[ok] shape={tuple(output.shape)} added_params={added} aux={float(loss):.6f} grad={grad:.6e} zero_init=True cv=False")
    finally:
        if cache_path.exists():cache_path.unlink()

if __name__=="__main__":main()

