"""Load the matched E0 and train only TECM1's new edge-message adapter."""
from __future__ import annotations
import importlib.util,os
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent;source_module=ROOT.parent/"MODE_TEMPLATE_GROUP_SERIES"/"grouped_mode_template_models.py"
spec=importlib.util.spec_from_file_location("tecm_decoder_source",source_module); module=importlib.util.module_from_spec(spec);assert spec and spec.loader;spec.loader.exec_module(module)
class FrozenModeAnchorDecoder(module.GroupedModeTemplateDecoder):
 def __init__(self,token_dim,pred_len):super().__init__(token_dim,pred_len,use_template=True,use_groups=False)
 def apply_training_policy(self,owner:nn.Module): owner.train(True);self.eval()
def replace_decoder(model,cfg,name):
 if name!="tecm_current_mode_anchor":raise ValueError(name)
 checkpoint=Path(os.environ.get("TECM_SOURCE_CHECKPOINT",""));
 if not checkpoint.is_file():raise FileNotFoundError(f"TECM_SOURCE_CHECKPOINT missing: {checkpoint}")
 model.decoder=FrozenModeAnchorDecoder(model.token_dim,model.pred_len); source=torch.load(checkpoint,map_location="cpu",weights_only=True);target=model.state_dict()
 compatible={k:v for k,v in source.items() if k in target and tuple(v.shape)==tuple(target[k].shape)};incompatible=model.load_state_dict(compatible,strict=False)
 expected={k for k in target if k.startswith("agent_aware_trans_encoder.tecm_edge_message.")}
 if set(incompatible.missing_keys)!=expected or incompatible.unexpected_keys:raise RuntimeError(f"unexpected checkpoint mismatch: {incompatible.missing_keys}, {incompatible.unexpected_keys}")
 for p in model.parameters():p.requires_grad_(False)
 for p in model.agent_aware_trans_encoder.tecm_edge_message.parameters():p.requires_grad_(True)
 model.decoder.source_checkpoint=str(checkpoint);model.decoder.cv_reference=False;cfg.MODEL.DECODER.NUM_MODES=20
