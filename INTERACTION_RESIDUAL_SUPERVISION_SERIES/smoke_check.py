"""Zero-init, auxiliary-target, gradient, EMA and invalid-agent checks for IR1."""

from __future__ import annotations
import copy
import importlib.util
import os
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent; WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
SOURCE = WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" / "EXP_146B_ANCHORED_MODE_TEMPLATE" / "results" / "general" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def build(ir1, device):
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9; cfg.MODEL.GRAPH.TOP_K = 4; cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device = str(device); cfg.device_list = [device.index or 0] if device.type == "cuda" else ["cpu"]
    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    module = load("irs_smoke_ir1" if ir1 else "irs_smoke_b0", ROOT / ("irs_models.py" if ir1 else "current_baseline_adapter.py"))
    module.replace_decoder(model, cfg, "ir1_future_response_supervision" if ir1 else "current_mode_anchor_template")
    return model.to(device)

def main():
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(FRONT)]
    os.chdir(PROJECT); import sitecustomize  # noqa: F401
    os.environ["IRS_SOURCE_CHECKPOINT"] = str(SOURCE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    b0, ir1 = build(False, device).eval(), build(True, device).eval()
    past = torch.randn(3, 9, 8, 2, device=device) * 0.20
    past[..., 0] += torch.linspace(0.0, 0.7, 8, device=device)
    future = past[:, :, -1:, :] + torch.cumsum(torch.randn(3, 9, 12, 2, device=device) * 0.05, dim=2)
    valid = torch.tensor([4, 7, 9], device=device)
    with torch.inference_mode():
        b0_pred = b0(past, None, None, mode="testing", num_valid=valid, graph_loc=past)[2]
        b0_anchor = b0.decoder.last_corrected_anchors.clone()
        ir_pred = ir1(past, None, None, mode="testing", num_valid=valid, graph_loc=past)[2]
    if not torch.allclose(b0_pred, ir_pred, atol=1e-6, rtol=1e-6): raise AssertionError("zero-init mismatch")
    if not torch.allclose(b0_anchor, ir1.decoder.last_corrected_anchors, atol=1e-6, rtol=1e-6): raise AssertionError("anchor mismatch")
    if float(ir1.decoder.last_predicted_response.abs().max()) != 0.0: raise AssertionError("response head is not zero-init")
    if tuple(ir1.decoder.anchor_steps) != (4, 8, 12) or ir1.decoder.cv_reference: raise AssertionError("decoder contract")
    if not ((ir1.decoder.last_interaction_weight >= 0).all() and (ir1.decoder.last_interaction_weight <= 1).all()): raise AssertionError("interaction weight range")

    copied = copy.deepcopy(ir1).eval()
    with torch.inference_mode():
        copied_pred = copied(past, None, None, mode="testing", num_valid=valid, graph_loc=past)[2]
    if not torch.allclose(ir_pred, copied_pred, atol=1e-6, rtol=1e-6): raise AssertionError("EMA deepcopy transport")

    changed = past.clone(); changed[0, 4:] = torch.randn_like(changed[0, 4:]) * 50.0
    with torch.inference_mode():
        reference = ir1(past, None, None, mode="testing", num_valid=valid, graph_loc=past)[2]
        altered = ir1(changed, None, None, mode="testing", num_valid=valid, graph_loc=changed)[2]
    if not torch.allclose(reference[0, :4], altered[0, :4], atol=1e-5, rtol=1e-5): raise AssertionError("invalid padded agents leak")

    ir1.train(); ir1.decoder.apply_training_policy(ir1)
    prediction = ir1(past, future, None, mode="train", num_valid=valid, graph_loc=past)[2]
    auxiliary = ir1.decoder.auxiliary_training_loss(prediction[:, :1], future[:, :1], ir1.last_mode_logits[:, :1])
    if not torch.isfinite(auxiliary) or float(auxiliary) <= 0.0: raise AssertionError("invalid auxiliary")
    loss = prediction.square().mean() + auxiliary; loss.backward()
    gradient = ir1.decoder.response_head.output.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or float(gradient.norm()) <= 0.0: raise AssertionError("no response gradient")
    trainable = [n for n,p in ir1.decoder.named_parameters() if p.requires_grad]
    allowed = ("response_head.", "response_to_mode.", "mode_gate.")
    if not trainable or any(not n.startswith(allowed) for n in trainable): raise AssertionError(trainable)

    with torch.no_grad():
        torch.nn.init.normal_(ir1.decoder.response_head.output.weight, std=0.02)
        torch.nn.init.normal_(ir1.decoder.response_to_mode.weight, std=0.02)
        torch.nn.init.normal_(ir1.decoder.mode_gate.weight, std=0.02)
    ir1.eval()
    with torch.inference_mode():
        activated = ir1(past, None, None, mode="testing", num_valid=valid, graph_loc=past)[2]
        shift = torch.linalg.vector_norm(ir1.decoder.last_corrected_anchors[:,0] - b0_anchor[:,0], dim=-1)
        differential = ir1.decoder.last_semantic_mode_delta[:,0].norm(dim=-1).std(dim=-1).mean()
    if torch.allclose(activated, b0_pred) or float(shift.mean()) <= 0 or float(differential) <= 0: raise AssertionError("semantic bridge inactive")
    added = sum(p.numel() for n,p in ir1.decoder.named_parameters() if n.startswith(allowed))
    print(f"[ok] shape={tuple(prediction.shape)} anchors={ir1.decoder.anchor_steps} cv={ir1.decoder.cv_reference} added_params={added} aux={float(auxiliary):.6f} grad={float(gradient.norm()):.6e} forced_shift={float(shift.mean()):.6e}")

if __name__ == "__main__": main()
