"""Load E0 then freeze every parameter except ETSE-E1's new projection."""
from __future__ import annotations
import importlib.util, os
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "MODE_TEMPLATE_GROUP_SERIES" / "grouped_mode_template_models.py"
spec = importlib.util.spec_from_file_location("etse_source_decoder", SOURCE)
module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)

class FrozenModeAnchorDecoder(module.GroupedModeTemplateDecoder):
    def __init__(self, token_dim: int, pred_len: int) -> None:
        super().__init__(token_dim, pred_len, use_template=True, use_groups=False)
    def apply_training_policy(self, owner: nn.Module) -> None:
        owner.train(True); self.eval()

def replace_decoder(model, cfg, name: str) -> None:
    if name != "etse_current_mode_anchor": raise ValueError(name)
    source = Path(os.environ.get("ETSE_SOURCE_CHECKPOINT", ""))
    if not source.is_file(): raise FileNotFoundError(f"ETSE_SOURCE_CHECKPOINT missing: {source}")
    model.decoder = FrozenModeAnchorDecoder(model.token_dim, model.pred_len)
    state = torch.load(source, map_location="cpu", weights_only=True)
    target = model.state_dict()
    compatible = {k:v for k,v in state.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
    incompatible = model.load_state_dict(compatible, strict=False)
    expected = {k for k in target if k.startswith("agent_aware_trans_encoder.etse_time_bias.")}
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint mismatch: {incompatible.missing_keys}, {incompatible.unexpected_keys}")
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.agent_aware_trans_encoder.etse_time_bias.parameters(): p.requires_grad_(True)
    model.decoder.source_checkpoint = str(source)
    model.decoder.cv_reference = False
    cfg.MODEL.DECODER.NUM_MODES = 20
