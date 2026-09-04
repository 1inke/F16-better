"""Real-batch forward/backward and H3->Joint-V2 reload preflight for E0/E1."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
if Path(sys.executable).resolve() != PYTHON.resolve():
    raise SystemExit(subprocess.run([str(PYTHON), *sys.argv]).returncode)

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
CONFIG = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
H3_PATH = ROOT / "exp187a_h3_adapter.py"
V2_PATH = WORKSPACE / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
VARIANTS = {"e0": "f0_baseline", "e1": "f3_factorized"}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grad_norm(parameters) -> float:
    return sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None) ** 0.5


def main() -> None:
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(WORKSPACE / "FRONT_ENCODER_SERIES")]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from train_front_variant import TrajectoryDataset, normalize_to_focal_heading

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_locked_q20(CONFIG)
    cfg.MODEL.USE_PRETRAINED_UNET = False
    cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9
    cfg.MODEL.GRAPH.TOP_K = 4
    cfg.MODEL.GRAPH.RADIUS = 6.0
    cfg.device, cfg.device_list = str(device), [device.index or 0]
    dataset = Subset(TrajectoryDataset(DATA / "eth_data_train.npy", DATA / "eth_num_train.npy", cfg), range(4))
    past, future, num_valid, *_ = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    past, future, num_valid = past.to(device), future.to(device), num_valid.to(device).int()
    past, future = normalize_to_focal_heading(past, future)
    center = past.mean(dim=-2, keepdim=True)
    h3, v2 = load("exp187a_h3", H3_PATH), load("exp187a_v2", V2_PATH)

    with tempfile.TemporaryDirectory(prefix="exp187a_", dir=ROOT) as temporary:
        temporary = Path(temporary)
        initial_h3, initial_v2 = {}, {}
        for label, variant in VARIANTS.items():
            torch.manual_seed(1)
            model = FrontEncoderASTRA(cfg, variant)
            os.environ["EXP187A_H3_INIT_SEED"] = "187001"
            h3.replace_decoder(model, cfg, "h3_anchor_proposal")
            initial_h3[label] = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
            model = model.to(device).train()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            output = model(past - center, None, None, mode="training", num_valid=num_valid, graph_loc=past)[2]
            (output.square().mean() + 0.01 * model.last_mode_logits.mean()).backward()
            optimizer.step()
            source = temporary / f"{label}_h3.pth"
            torch.save(model.state_dict(), source)
            del model

            torch.manual_seed(1)
            model = FrontEncoderASTRA(cfg, variant)
            os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(source)
            os.environ["EXP173_DECODER_INIT_SEED"] = "1"
            v2.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")
            initial_v2[label] = {k: v.detach().cpu().clone() for k, v in model.decoder.refiner.state_dict().items()}
            model = model.to(device).train()
            model.zero_grad(set_to_none=True)
            output = model(past - center, None, None, mode="training", num_valid=num_valid, graph_loc=past)[2]
            (output.square().mean() + 0.01 * model.last_mode_logits.mean()).backward()
            groups = {
                "encoder": [p for k, p in model.named_parameters() if not k.startswith("decoder.") and p.requires_grad],
                "h3": [p for k, p in model.decoder.named_parameters() if not k.startswith("refiner.") and not k.startswith("score_head.") and p.requires_grad],
                "score": [p for p in model.decoder.score_head.parameters() if p.requires_grad],
                "v2": [p for p in model.decoder.refiner.parameters() if p.requires_grad],
            }
            norms = {name: grad_norm(parameters) for name, parameters in groups.items()}
            if not all(value > 0 for value in norms.values()):
                raise RuntimeError(f"Missing joint gradient for {label}: {norms}")
            checkpoint = temporary / f"{label}_reload.pth"
            torch.save(model.state_dict(), checkpoint)
            reloaded = FrontEncoderASTRA(cfg, variant)
            v2.replace_decoder(reloaded, cfg, "v2_joint_wide_endpoint_correction")
            reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            print(f"[smoke-ok] {label} shape={tuple(output.shape)} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)} gradients={norms}")
            del model, reloaded
        for left, right in ((initial_h3["e0"], initial_h3["e1"]), (initial_v2["e0"], initial_v2["e1"])):
            assert left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left), "Decoder initialization mismatch"
    print("[done] E0/E1 real-batch forward, joint gradients, matched initialization and reload passed")


if __name__ == "__main__":
    main()
