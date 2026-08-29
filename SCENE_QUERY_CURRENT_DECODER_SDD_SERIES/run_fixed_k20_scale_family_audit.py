"""Fair K=20 post-hoc scale-family audit for the selected current SDD decoder.

Every variant retains exactly the original 20 candidates.  It is therefore a
candidate-set feasibility test, unlike EXP-148's 20x7 oracle upper bound.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PROJECT = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FRONT = WORKSPACE / "FRONT_ENCODER_SERIES"
DATA_DIR = WORKSPACE / "SDD_STANDARD_POTENTIAL_SERIES" / "processed_data"
ASSET_DIR = ROOT / "processed_scene_assets"
MODEL_MODULE = ROOT / "current_scene_query_models.py"
EXPERIMENT = ROOT / "EXP_149_FIXED_K20_SCALE_FAMILY_AUDIT"
CHECKPOINT = (ROOT / "EXP_147B_CURRENT_DECODER_SDD_SCENE_QUERY" / "results" / "s0_current" /
              "f16_temporal_social_cross" / "seed_1" / "val_scene_1" / "best_selected.pth")


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def metrics(pred: np.ndarray, target: np.ndarray, logits: np.ndarray, scenes: np.ndarray, video: dict[int, str]):
    error = np.linalg.norm(pred - target[:, None], axis=-1)
    ade, fde = error.mean(axis=-1), error[..., -1]
    rows = np.arange(len(ade)); best = ade.argmin(axis=1)
    result = {"samples": int(len(ade)), "candidate_count": int(pred.shape[1]),
              "minADE20_px": float(ade.min(axis=1).mean() * 20.0),
              "minFDE20_px": float(fde.min(axis=1).mean() * 20.0),
              "same_path_FDE_px": float(fde[rows, best].mean() * 20.0),
              "top1_ADE_px": float(ade[rows, logits.argmax(axis=1)].mean() * 20.0),
              "top1_FDE_px": float(fde[rows, logits.argmax(axis=1)].mean() * 20.0)}
    for label, mask in {"little": np.asarray([video[int(x)].startswith("little/") for x in scenes]),
                        "non_little": np.asarray([not video[int(x)].startswith("little/") for x in scenes])}.items():
        r, a, f = rows[mask], ade[mask], fde[mask]; b = a.argmin(axis=1); top = logits[mask].argmax(axis=1)
        result[label] = {"samples": int(mask.sum()), "minADE20_px": float(a.min(axis=1).mean()*20.0),
                         "minFDE20_px": float(f.min(axis=1).mean()*20.0), "same_path_FDE_px": float(f[np.arange(len(a)), b].mean()*20.0),
                         "top1_ADE_px": float(a[np.arange(len(a)), top].mean()*20.0), "top1_FDE_px": float(f[np.arange(len(f)), top].mean()*20.0)}
    return result


def apply_rank_family(pred: torch.Tensor, last: torch.Tensor, history_speed: torch.Tensor, gated: bool) -> torch.Tensor:
    """Use only predicted endpoint radii and observed history speed; K remains 20."""
    relative = pred - last[:, None, None]
    radius = torch.linalg.vector_norm(relative[..., -1, :], dim=-1)
    order = radius.argsort(dim=1)
    factors = torch.ones_like(radius)
    # Allocate 7 near, 6 standard and 7 far candidates by model-predicted radius.
    factors.scatter_(1, order[:, :7], 0.80)
    factors.scatter_(1, order[:, 13:], 1.25)
    if gated:
        threshold = torch.median(history_speed)
        factors = torch.where((history_speed >= threshold)[:, None], factors, torch.ones_like(factors))
    return last[:, None, None] + relative * factors[:, :, None, None]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--force", action="store_true"); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); output = EXPERIMENT / "audit_summary.json"
    if output.is_file() and not args.force: print(f"[skip] {output}; use --force"); return
    for path in (CHECKPOINT, DATA_DIR / "sdd_standard_test_data.npy", DATA_DIR / "sdd_standard_test_num.npy", ASSET_DIR / "sdd_test_scene_ids.npy", ASSET_DIR / "scene_asset_manifest.json"):
        if not path.is_file(): raise FileNotFoundError(path)
    if args.dry_run: print("[dry-run] fixed K=20 audit; checkpoint=" + str(CHECKPOINT)); return
    os.chdir(PROJECT); sys.path[:0] = [str(FRONT), str(PROJECT)]
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    trainer = module("exp149_helpers", FRONT / "train_front_variant.py")
    decoder = module("exp149_decoder", MODEL_MODULE)
    cfg = load_locked_q20(PROJECT / "configs" / "motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET=False; cfg.MODEL.USE_CACHED_UNET_FEATURES=False; cfg.DATA.MAX_AGENTS=16; cfg.MODEL.GRAPH.TOP_K=4; cfg.MODEL.GRAPH.RADIUS=6.0
    device = torch.device(args.device if torch.cuda.is_available() else "cpu"); cfg.device=str(device); cfg.device_list=[device.index or 0] if device.type=="cuda" else ["cpu"]
    model = FrontEncoderASTRA(cfg, "f16_temporal_social_cross"); decoder.replace_decoder(model, cfg, "s0_current")
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True); model=model.to(device).eval()
    dataset=trainer.TrajectoryDataset(DATA_DIR / "sdd_standard_test_data.npy", DATA_DIR / "sdd_standard_test_num.npy", cfg, ASSET_DIR / "sdd_test_scene_ids.npy")
    loader=DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    meta=json.loads((ASSET_DIR / "scene_asset_manifest.json").read_text(encoding="utf-8")); video={int(v):k for k,v in meta["video_to_scene_id"].items()}
    parts={key:[] for key in ("base","rank_family","history_gated_rank_family","target","logits","scenes")}
    with torch.inference_mode():
        for past,future,num,scene,_,_ in loader:
            raw=past.numpy(); past,future,num=past.to(device),future.to(device),num.to(device).int()
            local_past,local_future=trainer.normalize_to_focal_heading(past,future); center=local_past.mean(dim=-2,keepdim=True)
            pred=model(local_past-center,None,None,mode="testing",num_valid=num,graph_loc=local_past)[2][:,0]+center[:,0,None]
            last=local_past[:,0,-1]; speed=torch.linalg.vector_norm(local_past[:,0,-1]-local_past[:,0,-4],dim=-1)
            parts["base"].append(pred.cpu().numpy()); parts["rank_family"].append(apply_rank_family(pred,last,speed,False).cpu().numpy()); parts["history_gated_rank_family"].append(apply_rank_family(pred,last,speed,True).cpu().numpy())
            parts["target"].append(local_future[:,0].cpu().numpy()); parts["logits"].append(model.last_mode_logits[:,0].cpu().numpy()); parts["scenes"].append(scene.numpy())
    joined={key:np.concatenate(value) for key,value in parts.items()}
    result={"experiment":"EXP-149 fixed-K20 scale-family allocation audit","rule":"All variants preserve exactly 20 candidates. Rank-family uses only predicted endpoint radius; gated version additionally uses observed history speed.","variants":{}}
    for key in ("base","rank_family","history_gated_rank_family"):
        result["variants"][key]=metrics(joined[key],joined["target"],joined["logits"],joined["scenes"],video)
    base=result["variants"]["base"]
    for key,value in result["variants"].items():
        value["relative_to_base_pct"]={metric:float((value[metric]/base[metric]-1.0)*100.0) for metric in ("minADE20_px","minFDE20_px","same_path_FDE_px","top1_ADE_px","top1_FDE_px")}
    EXPERIMENT.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8"); print("[done] "+str(output))


if __name__ == "__main__": main()
