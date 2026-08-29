"""Current CV-free mode/three-anchor decoder loader for unmodified F16 audits."""
from __future__ import annotations
import importlib.util,os
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent;source_path=ROOT.parent/"MODE_TEMPLATE_GROUP_SERIES"/"grouped_mode_template_models.py"
spec=importlib.util.spec_from_file_location("fca_decoder_source",source_path);module=importlib.util.module_from_spec(spec);assert spec and spec.loader;spec.loader.exec_module(module)
class FrozenModeAnchorDecoder(module.GroupedModeTemplateDecoder):
 def __init__(self,token_dim,pred_len):super().__init__(token_dim,pred_len,use_template=True,use_groups=False)
 def apply_training_policy(self,owner:nn.Module):owner.train(True);self.eval()
def replace_decoder(model,cfg,name):
 if name!="fca_current_mode_anchor":raise ValueError(name)
 checkpoint=Path(os.environ.get("FCA_SOURCE_CHECKPOINT",""))
 if not checkpoint.is_file():raise FileNotFoundError(f"FCA_SOURCE_CHECKPOINT missing: {checkpoint}")
 model.decoder=FrozenModeAnchorDecoder(model.token_dim,model.pred_len);state=torch.load(checkpoint,map_location="cpu",weights_only=True)
 incompatible=model.load_state_dict(state,strict=False)
 if incompatible.missing_keys or incompatible.unexpected_keys:raise RuntimeError(f"checkpoint mismatch: {incompatible.missing_keys}, {incompatible.unexpected_keys}")
 for p in model.parameters():p.requires_grad_(False)
 model.decoder.source_checkpoint=str(checkpoint);model.decoder.cv_reference=False;cfg.MODEL.DECODER.NUM_MODES=20
