"""Real-data fairness, gradient and save/load smoke test for EXP-173."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
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
H3_MODULE = WORKSPACE / "HIERARCHICAL_DECODER_SERIES" / "hierarchical_decoder_models.py"
V2_MODULE = ROOT / "joint_v2_adapter.py"
DATA = PROJECT / "datasets" / "eth_ucy" / "processed_data_diverse"
VARIANTS = {"e2": "f4_relation_tokens", "e3": "f16_temporal_social_cross"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_state_equal(left: dict, right: dict) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


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
    cfg.device = str(device)
    cfg.device_list = [device.index or 0]
    dataset = Subset(TrajectoryDataset(DATA / "eth_data_train.npy", DATA / "eth_num_train.npy", cfg), range(4))
    past, future, num_valid, *_ = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
    past, future, num_valid = past.to(device), future.to(device), num_valid.to(device).int()
    past, future = normalize_to_focal_heading(past, future)
    center = past.mean(dim=-2, keepdim=True)
    h3_module = load_module("exp173_h3_smoke", H3_MODULE)
    v2_module = load_module("exp173_v2_smoke", V2_MODULE)

    with tempfile.TemporaryDirectory(prefix="exp173_smoke_", dir=ROOT) as temporary:
        temporary = Path(temporary)
        h3_decoder_states = {}
        sources = {}
        for label, variant in VARIANTS.items():
            torch.manual_seed(1)
            model = FrontEncoderASTRA(cfg, variant)
            h3_module.replace_decoder(model, cfg, "h3_anchor_proposal")
            h3_decoder_states[label] = {key: value.detach().cpu().clone() for key, value in model.decoder.state_dict().items()}
            # The H3 proposal heads intentionally start at zero, so the very
            # first batch cannot yet send a trajectory gradient to the encoder.
            # One Stage-A update opens that path before checking Stage B.
            model = model.to(device).train()
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1.0e-3
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                past - center, None, None, mode="training",
                num_valid=num_valid, graph_loc=past,
            )[2]
            stage_a_loss = prediction.square().mean() + 0.01 * model.last_mode_logits.mean()
            stage_a_loss.backward()
            optimizer.step()
            source = temporary / f"{label}_h3.pth"
            torch.save(model.state_dict(), source)
            sources[label] = source
            del model
        assert exact_state_equal(h3_decoder_states["e2"], h3_decoder_states["e3"]), "H3 initialization differs"

        refiner_states = {}
        for label, variant in VARIANTS.items():
            torch.manual_seed(1)
            model = FrontEncoderASTRA(cfg, variant)
            os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(sources[label])
            os.environ["EXP173_DECODER_INIT_SEED"] = "1"
            v2_module.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")
            refiner_states[label] = {key: value.detach().cpu().clone() for key, value in model.decoder.refiner.state_dict().items()}
            source_state = torch.load(sources[label], map_location="cpu", weights_only=True)
            for key, value in model.state_dict().items():
                if key.startswith("decoder.refiner."):
                    continue
                assert torch.equal(value.detach().cpu(), source_state[key]), f"H3 source changed before training: {label}/{key}"

            model = model.to(device)
            model.train()
            model.decoder.apply_training_policy(model)
            model.zero_grad(set_to_none=True)
            prediction = model(past - center, None, None, mode="training", num_valid=num_valid, graph_loc=past)[2]
            logits = model.last_mode_logits
            loss = prediction.square().mean() + 0.01 * logits.mean()
            loss.backward()
            groups = {
                "encoder": [p for key, p in model.named_parameters() if not key.startswith("decoder.") and p.requires_grad],
                "h3_proposal": [p for key, p in model.decoder.named_parameters() if not key.startswith("refiner.") and not key.startswith("score_head.") and p.requires_grad],
                "score": [p for p in model.decoder.score_head.parameters() if p.requires_grad],
                "v2": [p for p in model.decoder.refiner.parameters() if p.requires_grad],
            }
            norms = {name: grad_norm(parameters) for name, parameters in groups.items()}
            if not all(value > 0.0 for value in norms.values()):
                raise RuntimeError(f"Missing joint gradient for {label}: {norms}")
            checkpoint = temporary / f"{label}_joint_reload.pth"
            torch.save(model.state_dict(), checkpoint)
            reloaded = FrontEncoderASTRA(cfg, variant)
            os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(sources[label])
            v2_module.replace_decoder(reloaded, cfg, "v2_joint_wide_endpoint_correction")
            reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            print(f"[smoke-ok] {label} output={tuple(prediction.shape)} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)} gradients={norms}")
            del model, reloaded

        assert exact_state_equal(refiner_states["e2"], refiner_states["e3"]), "V2 initialization differs"
    print("[done] real batch, matched initialization, four trainable groups and checkpoint reload passed")


if __name__ == "__main__":
    main()
