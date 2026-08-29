"""Forward/backward checks on the selected EXP-146B mode/three-anchor model."""
from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
ADAPTER = ROOT / "current_mode_anchor_adapter.py"
SOURCE = (
    WORKSPACE / "MODE_TEMPLATE_GROUP_SERIES" /
    "EXP_146B_ANCHORED_MODE_TEMPLATE" / "results" / "general" /
    "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" /
    "best_selected.pth"
)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    sys.path[:0] = [str(PROJECT), str(PROJECT/"scripts"), str(FRONT)]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    custom = load_module("physical_bias_smoke", ROOT/"physical_bias_models.py")
    adapter = load_module("current_mode_anchor_smoke", ADAPTER)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ["PHYSICAL_BIAS_SOURCE_CHECKPOINT"] = str(SOURCE)
    for experiment in ("baseline", *sorted(custom.EXPERIMENTS)):
        cfg = load_locked_q20(PROJECT/"configs"/"motif_rbf_univ.yaml")
        cfg.MODEL.USE_PRETRAINED_UNET=False; cfg.MODEL.USE_CACHED_UNET_FEATURES=False
        cfg.DATA.MAX_AGENTS=9; cfg.MODEL.GRAPH.TOP_K=4; cfg.MODEL.GRAPH.RADIUS=6.0
        cfg.device=str(device); cfg.device_list=[device.index or 0] if device.type=="cuda" else ["cpu"]
        model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
        if experiment != "baseline":
            custom.replace_encoder(model, cfg, experiment)
        adapter.replace_decoder(model, cfg, "current_mode_anchor_template")
        model = model.to(device).train()
        model.decoder.apply_training_policy(model)
        if tuple(model.decoder.anchor_steps) != (4, 8, 12):
            raise AssertionError((experiment, model.decoder.anchor_steps))
        if not bool(model.decoder.use_mode_template) or bool(model.decoder.use_motion_groups):
            raise AssertionError(f"{experiment}: not the selected EXP-146B structure")
        if any(parameter.requires_grad for parameter in model.decoder.parameters()):
            raise AssertionError(f"{experiment}: decoder is not frozen")
        base = model.agent_aware_trans_encoder
        past = torch.randn(2,9,8,2,device=device)*0.25
        past[:,:,:,0] += torch.linspace(0.0,1.0,8,device=device)
        num_valid = torch.tensor([4,9],device=device)
        pred = model(past,None,None,mode="train",num_valid=num_valid,graph_loc=past)[2]
        if pred.shape != (2,9,20,12,2): raise AssertionError((experiment,pred.shape))
        smoke_loss = pred.square().mean() + model.last_mode_logits.square().mean()
        smoke_loss.backward()
        grad_norm = None
        if experiment != "baseline":
            module = base.physical_time_bias
            if float(module.last_max_abs_bias.item()) != 0.0:
                raise AssertionError(f"{experiment}: zero-init bias not zero")
            if module.weights.grad is None: raise AssertionError(f"{experiment}: no bias gradient")
            if not torch.isfinite(module.weights.grad).all(): raise AssertionError(f"{experiment}: non-finite bias gradient")
            grad_norm=float(module.weights.grad.norm().item())
            if grad_norm <= 0.0: raise AssertionError(f"{experiment}: zero bias gradient")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[ok] {experiment} shape={tuple(pred.shape)} anchors={model.decoder.anchor_steps} "
            f"cv={model.decoder.cv_reference} trainable={trainable} bias_grad_norm={grad_norm}"
        )

if __name__ == "__main__":
    main()
