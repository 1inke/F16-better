"""Real-batch contract check for the fixed-V2 encoder ablation."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

PYTHON = Path(r"C:\coding_toobox\envs\astra2\python.exe")
if str(Path(sys.executable).resolve()).casefold() != str(PYTHON.resolve()).casefold():
    raise SystemExit(subprocess.run([str(PYTHON), *sys.argv]).returncode)

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
CONFIG = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "motif_rbf_official_4p8.yaml"
ADAPTER = ROOT / "official_v2_decoder_adapter.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
SOURCE = WORKSPACE / "OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES" / "EXP_156B_FOLD_LOCAL_CURRENT_DECODER" / "results" / "eth" / "v2" / "f16_temporal_social_cross" / "seed_1" / "val_scene_-1" / "best_selected.pth"
VARIANTS = ("f0_baseline", "f3_factorized", "f4_relation_tokens", "f16_temporal_social_cross")


def load_adapter():
    spec = importlib.util.spec_from_file_location("official_v2_smoke_adapter", ADAPTER)
    if spec is None or spec.loader is None:
        raise ImportError(ADAPTER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def main():
    sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts"), str(WORKSPACE / "FRONT_ENCODER_SERIES")]
    os.chdir(PROJECT)
    import sitecustomize  # noqa: F401
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    from train_front_variant import TrajectoryDataset, normalize_to_focal_heading

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_locked_q20(CONFIG); cfg.MODEL.USE_PRETRAINED_UNET = False; cfg.MODEL.USE_CACHED_UNET_FEATURES = False
    cfg.DATA.MAX_AGENTS = 9; cfg.MODEL.GRAPH.TOP_K = 4; cfg.MODEL.GRAPH.RADIUS = 6.0; cfg.device = str(device); cfg.device_list = [device.index or 0]
    dataset = Subset(TrajectoryDataset(DATA / "eth_data_train.npy", DATA / "eth_num_train.npy", cfg), range(4))
    past, future, num, *_ = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    past, future, num = past.to(device), future.to(device), num.to(device).int()
    past, future = normalize_to_focal_heading(past, future); center = past.mean(dim=-2, keepdim=True)
    adapter = load_adapter(); os.environ["OFFICIAL_V2_DECODER_SOURCE"] = str(SOURCE)
    fingerprints = []
    for variant in VARIANTS:
        model = FrontEncoderASTRA(cfg, variant); adapter.replace_decoder(model, cfg, "fixed_official_v2"); model = model.to(device)
        assert all(not parameter.requires_grad for parameter in model.decoder.parameters())
        trainable = [parameter for name, parameter in model.named_parameters() if not name.startswith("decoder.") and parameter.requires_grad]
        assert trainable
        fingerprints.append(torch.cat([value.detach().cpu().flatten() for value in model.decoder.state_dict().values() if value.is_floating_point()]).sum().item())
        model.train(); model.decoder.apply_training_policy(model); model.zero_grad(set_to_none=True)
        prediction = model(past - center, None, None, mode="training", num_valid=num, graph_loc=past)[2]
        loss = prediction.square().mean(); loss.backward()
        gradient = sum(float(parameter.grad.detach().square().sum()) for parameter in trainable if parameter.grad is not None) ** 0.5
        assert gradient > 0.0
        assert all(parameter.grad is None for parameter in model.decoder.parameters())
        print(f"[smoke-ok] {variant} output={tuple(prediction.shape)} trainable={sum(p.numel() for p in trainable)} grad={gradient:.6g}")
        del model
    assert max(fingerprints) - min(fingerprints) == 0.0
    print("[done] exact decoder state is identical across all encoder variants")


if __name__ == "__main__":
    main()
