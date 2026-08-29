"""Matched frozen-decoder B0 initialized from the CSNA checkpoint."""

from __future__ import annotations
import importlib.util, os
from pathlib import Path
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
SOURCE_PATH=ROOT.parent/"MODE_TEMPLATE_GROUP_SERIES"/"grouped_mode_template_models.py"
spec=importlib.util.spec_from_file_location("csrs_baseline_source",SOURCE_PATH)
if spec is None or spec.loader is None: raise ImportError(SOURCE_PATH)
SOURCE=importlib.util.module_from_spec(spec); spec.loader.exec_module(SOURCE)

class FrozenCurrentDecoder(SOURCE.GroupedModeTemplateDecoder):
    def __init__(self,token_dim:int,pred_len:int)->None: super().__init__(token_dim,pred_len,use_template=True,use_groups=False)
    def apply_training_policy(self,owner:nn.Module)->None: owner.train(True); self.eval()

def replace_decoder(model,cfg,name:str)->None:
    if name!="current_mode_anchor_template": raise ValueError(name)
    checkpoint=Path(os.environ.get("CSRS_SOURCE_CHECKPOINT",""))
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    model.decoder=FrozenCurrentDecoder(model.token_dim,model.pred_len)
    source=torch.load(checkpoint,map_location="cpu",weights_only=True); target=model.state_dict()
    compatible={k:v for k,v in source.items() if k in target and tuple(v.shape)==tuple(target[k].shape)}
    incompatible=model.load_state_dict(compatible,strict=False)
    missing=[k for k in incompatible.missing_keys if k.startswith("decoder.")]
    skipped=[k for k in source if k.startswith("decoder.") and k not in compatible]
    if missing or skipped: raise RuntimeError((missing,skipped))
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.agent_aware_trans_encoder.parameters(): p.requires_grad_(True)
    for p in model.decoder.parameters(): p.requires_grad_(False)
    model.decoder.source_checkpoint=str(checkpoint); model.decoder.cv_reference=False
    cfg.MODEL.DECODER.NUM_MODES=20

