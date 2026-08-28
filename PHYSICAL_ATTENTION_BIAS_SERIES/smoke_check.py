"""Forward/backward and activation checks for physical attention-bias variants."""
from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
DECODER = WORKSPACE / "AUTONOMOUS_EXPERIMENTS" / "EXP_002_LIGHT_KINEMATIC_DECODER" / "decoder_models.py"

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
    decoder = load_module("kinematic_decoder_smoke", DECODER)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for experiment in sorted(custom.EXPERIMENTS):
        cfg = load_locked_q20(PROJECT/"configs"/"motif_rbf_univ.yaml")
        cfg.MODEL.USE_PRETRAINED_UNET=False; cfg.MODEL.USE_CACHED_UNET_FEATURES=False
        cfg.DATA.MAX_AGENTS=9; cfg.MODEL.GRAPH.TOP_K=4; cfg.MODEL.GRAPH.RADIUS=6.0
        cfg.device=str(device); cfg.device_list=[device.index or 0] if device.type=="cuda" else ["cpu"]
        model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
        custom.replace_encoder(model, cfg, experiment); decoder.replace_decoder(model, cfg, "kinematic_k20")
        model = model.to(device).train()
        base = model.agent_aware_trans_encoder; module = base.physical_time_bias
        past = torch.randn(2,9,8,2,device=device)*0.25
        past[:,:,:,0] += torch.linspace(0.0,1.0,8,device=device)
        num_valid = torch.tensor([4,9],device=device)
        pred = model(past,None,None,mode="train",num_valid=num_valid,graph_loc=past)[2]
        if pred.shape != (2,9,20,12,2): raise AssertionError((experiment,pred.shape))
        if float(module.last_max_abs_bias.item()) != 0.0:
            raise AssertionError(f"{experiment}: zero-init bias not zero")
        pred.square().mean().backward()
        if module.weights.grad is None: raise AssertionError(f"{experiment}: no bias gradient")
        if not torch.isfinite(module.weights.grad).all(): raise AssertionError(f"{experiment}: non-finite bias gradient")
        grad_norm=float(module.weights.grad.norm().item())
        if grad_norm <= 0.0: raise AssertionError(f"{experiment}: zero bias gradient")
        print(f"[ok] {experiment} shape={tuple(pred.shape)} params={sum(p.numel() for p in model.parameters())} bias_grad_norm={grad_norm:.6e}")

if __name__ == "__main__":
    main()
