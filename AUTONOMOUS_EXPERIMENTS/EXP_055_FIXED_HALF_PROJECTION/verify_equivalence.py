from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

W = Path(__file__).resolve().parents[2]
P = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
DECODER = W / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"
CHECKPOINT = W / "AUTONOMOUS_EXPERIMENTS" / "EXP_052_TEMPORAL_HALF_POOL" / "results" / "f35_temporal_half_pool" / "seed_1" / "val_scene_1" / "final_model.pth"

sys.path[:0] = [str(P), str(P / "scripts"), str(W / "FRONT_ENCODER_SERIES"), str(W / "ATTENTION_COMPUTE_SERIES")]
os.chdir(P)
import sitecustomize  # noqa: F401
from experiment_config import load_locked_q20
from front_encoder_models import FrontEncoderASTRA
from benchmark_attention_variants import make_batch

spec = importlib.util.spec_from_file_location("locked_decoder", DECODER)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
cfg = load_locked_q20(P / "configs" / "motif_rbf_univ.yaml")
cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
cfg.device = "cuda:0"; cfg.device_list = [0]
state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
models = []
for variant in ("f35_temporal_half_pool", "f36_fixed_half_projection"):
    model = FrontEncoderASTRA(cfg, variant)
    module.replace_decoder(model, cfg, "kinematic_k20")
    model.load_state_dict(state, strict=True)
    models.append(model.cuda().eval())
past, num_valid = make_batch(cfg, 64, dense=False)
past = past.cuda(); num_valid = num_valid.cuda(); center = past.mean(dim=-2, keepdim=True)
outputs = []
with torch.inference_mode():
    for model in models:
        prediction = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)[2]
        outputs.append((prediction, model.last_mode_logits.clone()))
result = {
    "prediction_max_abs_diff": float((outputs[0][0] - outputs[1][0]).abs().max()),
    "logit_max_abs_diff": float((outputs[0][1] - outputs[1][1]).abs().max()),
    "parameters_f35": sum(p.numel() for p in models[0].parameters()),
    "parameters_f36": sum(p.numel() for p in models[1].parameters()),
}
out = Path(__file__).resolve().parent / "equivalence.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
