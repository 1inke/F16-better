"""Fixed current mode/three-anchor decoder for formal encoder ablations."""
from __future__ import annotations
import importlib.util,os
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;source_path=ROOT.parent/"MODE_TEMPLATE_GROUP_SERIES"/"grouped_mode_template_models.py"
spec=importlib.util.spec_from_file_location("formal_decoder_source",source_path);module=importlib.util.module_from_spec(spec);assert spec and spec.loader;spec.loader.exec_module(module)
def replace_decoder(model,cfg,name):
 if name!="fixed_current_mode_anchor":raise ValueError(name)
 path=Path(os.environ.get("FORMAL_DECODER_SOURCE",""))
 if not path.is_file():raise FileNotFoundError(f"FORMAL_DECODER_SOURCE missing: {path}")
 model.decoder=module.GroupedModeTemplateDecoder(model.token_dim,model.pred_len,use_template=True,use_groups=False)
 source=torch.load(path,map_location="cpu",weights_only=True);decoder={k[len("decoder."):]:v for k,v in source.items() if k.startswith("decoder.")}
 incompatible=model.decoder.load_state_dict(decoder,strict=False)
 if incompatible.missing_keys or incompatible.unexpected_keys:raise RuntimeError(f"decoder mismatch: {incompatible.missing_keys}, {incompatible.unexpected_keys}")
 for p in model.decoder.parameters():p.requires_grad_(False)
 model.decoder.cv_reference=False;model.decoder.source_checkpoint=str(path);cfg.MODEL.DECODER.NUM_MODES=20
