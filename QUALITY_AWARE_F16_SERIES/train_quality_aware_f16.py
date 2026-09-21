"""
Train Quality-Aware F16 Pipeline.

Implements F16 full unfreezing with Differential Learning Rate & Quality-Aware Probability Loss.
Supports:
- QA0: Standard reproduction (lambda_q = 0.0)
- QA1: Quality-Aware fine-tuning (lambda_q > 0.0, quality_tau)
Strictly adheres to:
- P0: quality target detached (no_grad)
- Rule 1: Baseline Identity Lock
- Differential LR: Encoder 0.1x, Decoder non-score 0.5x, Score head 1.0x
- Preserves all original F16 loss components identically
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [
    str(REPO),
    str(REPO / "SELECTOR_BENCHMARK_SERIES"),
    str(REPO / "FRONT_ENCODER_SERIES"),
    str(REPO / "E2_F16_END_TO_END_COMPARISON_SERIES"),
]

if sys.platform == "win32":
    torch_lib = Path(sys.executable).parent / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from build_candidate_cache import configure_astra_env, load_module
from QUALITY_AWARE_F16_SERIES.common.quality_targets import build_quality_target_ade
from QUALITY_AWARE_F16_SERIES.common.probability_metrics import (
    compute_generation_metrics,
    compute_selection_metrics,
    compute_probability_metrics,
)
from QUALITY_AWARE_F16_SERIES.common.baseline_identity import (
    create_baseline_identity,
    save_baseline_identity,
)
from QUALITY_AWARE_F16_SERIES.common.reproducibility import set_deterministic_seed


def setup_model(repo_root: Path, fold: str, seed: int, cfg, device: torch.device, source_type: str = "h3"):
    """
    Builds and initializes F16 model with exact EXP-173 weights.
    If source_type == 'h3': starts from matched H3 checkpoint (Stage V2 training).
    If source_type == 'v2': starts from matched V2 checkpoint (direct fine-tune).
    """
    exp173 = repo_root / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END" / "results"
    h3_ckpt = exp173 / fold / "h3" / "e3" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"
    v2_ckpt = exp173 / fold / "v2" / "e3" / "f16_temporal_social_cross" / f"seed_{seed}" / "val_scene_-1" / "best_selected.pth"

    if not h3_ckpt.is_file():
        raise FileNotFoundError(f"Missing H3 checkpoint: {h3_ckpt}")
    
    os.environ["EXP173_H3_SOURCE_CHECKPOINT"] = str(h3_ckpt)
    os.environ["EXP173_DECODER_INIT_SEED"] = str(seed)

    model_file = repo_root / "FRONT_ENCODER_SERIES" / "front_encoder_models.py"
    adapter_file = repo_root / "E2_F16_END_TO_END_COMPARISON_SERIES" / "joint_v2_adapter.py"
    models_mod = load_module("front_encoder_models", model_file)
    adapter_mod = load_module("joint_v2_adapter", adapter_file)

    model = models_mod.FrontEncoderASTRA(cfg, "f16_temporal_social_cross")
    adapter_mod.replace_decoder(model, cfg, "v2_joint_wide_endpoint_correction")

    if source_type == "v2":
        if not v2_ckpt.is_file():
            raise FileNotFoundError(f"Missing V2 checkpoint: {v2_ckpt}")
        state = torch.load(v2_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        source_used = v2_ckpt
    else:
        # replace_decoder already initialized with H3 checkpoint + V2 refiner initialization
        source_used = h3_ckpt

    # Unfreeze all parameters per Section 10
    for p in model.parameters():
        p.requires_grad = True

    model = model.to(device)
    return model, str(source_used)


def evaluate_f16(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    normalize_heading: bool = True,
    tau_eval: float = 0.10,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Full evaluation computing Generation, Selection, and Probability metrics.
    """
    model.eval()
    all_cands = []
    all_ades = []
    all_fdes = []
    all_logits = []
    all_probs = []
    all_selected_idx = []

    with torch.no_grad():
        for batch in val_loader:
            past, future, num_valid, *_ = batch
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()

            if normalize_heading:
                past, future = load_module(
                    "train_front_variant",
                    REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
                ).normalize_to_focal_heading(past, future)

            center = past.mean(dim=-2, keepdim=True)
            output = model(
                past - center, future - center, None, mode="val",
                num_valid=num_valid, graph_loc=past,
            )
            raw_pred = output[2] if isinstance(output, (tuple, list)) else output
            pred = raw_pred + center.unsqueeze(-3)
            logits = model.last_mode_logits

            focal_pred = pred[:, 0]     # [B, 20, 12, 2]
            focal_logits = logits[:, 0] # [B, 20]
            focal_gt = future[:, 0]     # [B, 12, 2]

            err = torch.linalg.vector_norm(focal_pred - focal_gt[:, None], dim=-1)
            ade = err.mean(dim=-1) # [B, 20]
            fde = err[:, :, -1]    # [B, 20]

            probs = torch.softmax(focal_logits, dim=-1)
            selected_idx = probs.argmax(dim=-1)

            all_cands.append(focal_pred.cpu().numpy())
            all_ades.append(ade.cpu().numpy())
            all_fdes.append(fde.cpu().numpy())
            all_logits.append(focal_logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_selected_idx.append(selected_idx.cpu().numpy())

    cands = np.concatenate(all_cands, axis=0)
    ades = np.concatenate(all_ades, axis=0)
    fdes = np.concatenate(all_fdes, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    selected_idx = np.concatenate(all_selected_idx, axis=0)

    gen_m = compute_generation_metrics(cands, ades, fdes)
    sel_m = compute_selection_metrics(selected_idx, ades, fdes)
    prob_m = compute_probability_metrics(probs, ades, tau_eval=tau_eval)

    return gen_m, sel_m, prob_m


def main():
    parser = argparse.ArgumentParser(description="Train Quality-Aware F16 Model")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", type=str, default="eth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-freq", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.00025, help="Base learning rate")
    parser.add_argument("--lambda-q", type=float, default=0.0, help="Weight for quality loss")
    parser.add_argument("--quality-tau", type=float, default=0.10, help="Temperature for soft quality target")
    parser.add_argument("--differential-lr", action="store_true", default=True, help="Use differential learning rate")
    parser.add_argument("--source-type", type=str, default="h3", choices=["h3", "v2"], help="Starting weights")
    parser.add_argument("--augment", action="store_true", default=True, help="Enable reflection/speed trajectory augmentation")
    parser.add_argument("--focal-heading-normalize", action="store_true", default=True, help="Enable focal heading normalization")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for checkpoints & logs")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    set_deterministic_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = (REPO / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[QA-F16] Initializing fold={args.fold}, seed={args.seed}, lambda_q={args.lambda_q}, tau={args.quality_tau}, device={device}...")

    astra_root = Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
    cfg = configure_astra_env(REPO, astra_root, device)
    cfg.TRAIN.LR = args.lr

    trainer_file = REPO / "FRONT_ENCODER_SERIES" / "train_front_variant.py"
    trainer_mod = load_module("train_front_variant", trainer_file)
    from locked_mode_query_loss import ModeQueryLoss

    loss_fn = ModeQueryLoss(cfg).to(device)

    # 1. Dataset & Loaders using exact EXP-173 split
    data_dir = astra_root / "datasets" / "eth_ucy" / "processed_data_diverse"
    base_dataset = trainer_mod.TrajectoryDataset(
        data_dir / f"{args.fold}_data_train.npy", data_dir / f"{args.fold}_num_train.npy", cfg
    )

    # Load existing locked split indices
    split_path = REPO / "E2_F16_END_TO_END_COMPARISON_SERIES" / "EXP_173_E2_F16_END_TO_END" / "results" / args.fold / "v2" / "e3" / "f16_temporal_social_cross" / f"seed_{args.seed}" / "val_scene_-1" / "split_indices.npz"
    if split_path.is_file():
        split_data = np.load(split_path)
        train_indices = split_data["train"]
        val_indices = split_data["validation"]
    else:
        raise FileNotFoundError(f"Missing EXP-173 locked split indices at {split_path}")

    train_loader = DataLoader(
        Subset(base_dataset, train_indices.tolist()), batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=device.type == "cuda", drop_last=False,
    )
    val_loader = DataLoader(
        Subset(base_dataset, val_indices.tolist()), batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=device.type == "cuda", drop_last=False,
    )

    test_dataset = trainer_mod.TrajectoryDataset(
        data_dir / f"{args.fold}_data_test.npy", data_dir / f"{args.fold}_num_test.npy", cfg
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda", drop_last=False,
    )

    # 2. Build model & verify parameters
    model, source_checkpoint = setup_model(REPO, args.fold, args.seed, cfg, device, source_type=args.source_type)
    ema = trainer_mod.ModelEMA(model, 0.995)

    # Group parameters for Differential Learning Rate per Section 11
    encoder_params = []
    decoder_non_score_params = []
    score_head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not name.startswith("decoder."):
            encoder_params.append(param)
        elif name.startswith("decoder.score_head."):
            score_head_params.append(param)
        else:
            decoder_non_score_params.append(param)

    if args.differential_lr:
        param_groups = [
            {"params": encoder_params, "lr": args.lr * 0.1},
            {"params": decoder_non_score_params, "lr": args.lr * 0.5},
            {"params": score_head_params, "lr": args.lr * 1.0},
        ]
        lr_groups_dict = {
            "encoder": args.lr * 0.1,
            "decoder_non_score": args.lr * 0.5,
            "score_head": args.lr * 1.0,
        }
    else:
        param_groups = [{"params": model.parameters(), "lr": args.lr}]
        lr_groups_dict = {"all": args.lr}

    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(cfg.TRAIN.W_DECAY))

    # Scheduler: Monotonic Cosine
    total_steps = max(args.epochs * len(train_loader), 1)
    warmup_steps = max(int(1 * len(train_loader)), 1)

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1)), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)

    # Record Baseline Identity
    tot_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    identity = create_baseline_identity(
        architecture="FrontEncoderASTRA",
        variant="f16_temporal_social_cross",
        decoder="v2_joint_wide_endpoint_correction",
        source_checkpoint=source_checkpoint,
        seed=args.seed,
        fold=args.fold,
        candidate_count=20,
        total_parameter_count=tot_params,
        trainable_parameter_count=train_params,
        original_loss_components=[
            "WeightedTrajectoryLoss", "ModeQueryLoss.classification", "ModeQueryLoss.endpoint",
            "ModeQueryLoss.diversity", "ModeQueryLoss.short_horizon", "oracle_coverage_attraction",
        ],
        quality_loss_type="ade_soft_distribution" if args.lambda_q > 0 else "none",
        lambda_q=args.lambda_q,
        quality_tau=args.quality_tau,
        optimizer_name="AdamW",
        learning_rate_groups=lr_groups_dict,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split="scene_tail_holdout_0.05",
        checkpoint_criterion="val_selected_ade_with_minade_guard",
    )
    save_baseline_identity(out_dir / "baseline_identity.json", identity)

    # 3. Training Loop
    history = []
    best_val_selected_ade = float("inf")
    best_checkpoint_metrics = None

    augment_args = argparse.Namespace(
        augment_rotation=False,
        augment_reflection=True,
        augment_speed_min=0.9,
        augment_speed_max=1.1,
        augment_curvature_max=0.0,
        augment_probability=0.5,
    )

    # Pre-training evaluation (Epoch 0)
    print("[QA-F16] Running pre-training evaluation (Epoch 0)...")
    val_gen, val_sel, val_prob = evaluate_f16(
        model, val_loader, device, normalize_heading=True, tau_eval=0.10
    )
    init_row = {
        "epoch": 0,
        "train_loss_total": None,
        "train_loss_orig": None,
        "train_loss_quality": None,
        "train_mean_max_prob": None,
        "train_mean_entropy": None,
        "lr_score_head": optimizer.param_groups[-1]["lr"],
        "val_selected_ade": val_sel["selected_ade"],
        "val_selected_fde": val_sel["selected_fde"],
        "val_selection_regret": val_sel["selection_regret"],
        "val_min_ade20": val_gen["min_ade20"],
        "val_min_fde20": val_gen["min_fde20"],
        "val_oracle_nll": val_prob["oracle_nll"],
        "val_soft_ce": val_prob["soft_ce"],
        "val_top1_ece": val_prob["top1_ece"],
        "is_best": False,
    }
    best_val_selected_ade = float("inf")
    best_checkpoint_metrics = None
    history.append(init_row)
    print(f"[Epoch 00/{args.epochs}] val_selected_ade: {val_sel['selected_ade']:.4f} | val_min_ade20: {val_gen['min_ade20']:.4f} | val_selection_regret: {val_sel['selection_regret']:.4f}")

    started = time.perf_counter()
    print(f"[QA-F16] Starting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        losses_total = []
        losses_orig = []
        losses_qual = []

        max_probs = []
        entropies = []

        for batch in train_loader:
            past, future, num_valid, *_ = batch
            past = past.to(device)
            future = future.to(device)
            num_valid = num_valid.to(device).int()

            # Data Augmentation (exact match to EXP-173)
            if args.augment:
                past, future = trainer_mod.augment_trajectories(past, future, augment_args)

            # Heading normalization & center
            if args.focal_heading_normalize:
                past, future = trainer_mod.normalize_to_focal_heading(past, future)
            center = past.mean(dim=-2, keepdim=True)

            output = model(
                past - center, future - center, None, mode="train",
                num_valid=num_valid, graph_loc=past,
            )
            raw_pred = output[2] if isinstance(output, (tuple, list)) else output
            logits = model.last_mode_logits

            # Focal agent extraction (all in CENTERED coordinates for loss)
            focal_raw_pred = raw_pred[:, :1]    # [B, 1, 20, 12, 2]
            focal_logits = logits[:, :1]         # [B, 1, 20]
            focal_target = (future - center)[:, :1] # [B, 1, 12, 2]

            # 1. Original Loss: ModeQueryLoss + Oracle Coverage Attraction
            loss_orig, _ = loss_fn(
                focal_raw_pred, focal_target, focal_logits, cfg
            )

            # Oracle coverage attraction (exact match to EXP-173 train_front_variant.py)
            target = focal_target.unsqueeze(-3) # [B, 1, 1, 12, 2]
            distances = torch.linalg.vector_norm(focal_raw_pred - target, dim=-1) # [B, 1, 20, 12]
            quality = distances.mean(dim=-1) + 0.25 * distances[..., -1] # [B, 1, 20]
            temp = 0.10
            detached_q = quality.detach()
            resp_logits = -detached_q / temp
            resp = torch.softmax(resp_logits, dim=-1)
            coverage = (resp * quality).sum(dim=-1).mean()
            loss_orig = loss_orig + 0.20 * coverage

            # 2. Quality Loss (P0: target detached)
            loss_quality = torch.tensor(0.0, device=device)
            if args.lambda_q > 0.0:
                focal_ade = distances[:, 0].mean(dim=-1) # [B, 20]
                target_q = torch.softmax(-focal_ade.detach() / float(args.quality_tau), dim=-1)
                log_p = torch.log_softmax(focal_logits[:, 0], dim=-1)
                loss_quality = -(target_q * log_p).sum(dim=-1).mean()

            loss_total = loss_orig + args.lambda_q * loss_quality

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            losses_total.append(float(loss_total.detach().cpu()))
            losses_orig.append(float(loss_orig.detach().cpu()))
            losses_qual.append(float(loss_quality.detach().cpu()))

            with torch.no_grad():
                probs_b = torch.softmax(focal_logits[:, 0], dim=-1)
                max_probs.append(float(probs_b.max(dim=-1).values.mean().cpu()))
                ent_b = -(probs_b * torch.log(probs_b.clamp_min(1e-12))).sum(dim=-1).mean()
                entropies.append(float(ent_b.cpu()))

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss_total": float(np.mean(losses_total)),
            "train_loss_orig": float(np.mean(losses_orig)),
            "train_loss_quality": float(np.mean(losses_qual)),
            "train_mean_max_prob": float(np.mean(max_probs)),
            "train_mean_entropy": float(np.mean(entropies)),
            "lr_score_head": optimizer.param_groups[-1]["lr"],
        }

        # Validation frequency check
        if (epoch + 1) % args.val_freq == 0 or epoch + 1 == args.epochs:
            val_gen, val_sel, val_prob = evaluate_f16(
                ema.module, val_loader, device, normalize_heading=True, tau_eval=0.10
            )
            epoch_row.update({
                "val_selected_ade": val_sel["selected_ade"],
                "val_selected_fde": val_sel["selected_fde"],
                "val_selection_regret": val_sel["selection_regret"],
                "val_min_ade20": val_gen["min_ade20"],
                "val_min_fde20": val_gen["min_fde20"],
                "val_oracle_nll": val_prob["oracle_nll"],
                "val_soft_ce": val_prob["soft_ce"],
                "val_top1_ece": val_prob["top1_ece"],
            })

            # Checkpoint Selection per Section 23:
            # 1. Validation Selected ADE lowest
            # 2. If diff < 1mm, minADE20 lower
            # 3. If tie, oracle NLL lower
            cur_sel_ade = val_sel["selected_ade"]
            is_best = False
            if cur_sel_ade < best_val_selected_ade - 0.001:
                is_best = True
            elif abs(cur_sel_ade - best_val_selected_ade) <= 0.001 and best_checkpoint_metrics is not None:
                if val_gen["min_ade20"] < best_checkpoint_metrics["val_min_ade20"]:
                    is_best = True
                elif val_prob["oracle_nll"] < best_checkpoint_metrics["val_oracle_nll"]:
                    is_best = True
            elif best_checkpoint_metrics is None:
                is_best = True

            if is_best:
                best_val_selected_ade = cur_sel_ade
                best_checkpoint_metrics = epoch_row.copy()
                torch.save(ema.module.state_dict(), out_dir / "best_selected.pth")
                epoch_row["is_best"] = True

        history.append(epoch_row)
        print(f"[Epoch {epoch+1:02d}/{args.epochs}] " + " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in epoch_row.items() if k in ["train_loss_total", "val_selected_ade", "val_min_ade20", "val_selection_regret"]
        ))

    # Save final model
    torch.save(ema.module.state_dict(), out_dir / "final_model.pth")
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    # 4. Final Evaluation of Best Checkpoint on Validation & Test
    print(f"[QA-F16] Loading best selected checkpoint for final evaluation...")
    best_state = torch.load(out_dir / "best_selected.pth", map_location=device, weights_only=True)
    ema.module.load_state_dict(best_state)

    val_gen, val_sel, val_prob = evaluate_f16(ema.module, val_loader, device, normalize_heading=True)
    test_gen, test_sel, test_prob = evaluate_f16(ema.module, test_loader, device, normalize_heading=True)

    summary_metrics = {
        "fold": args.fold,
        "seed": args.seed,
        "lambda_q": args.lambda_q,
        "quality_tau": args.quality_tau,
        "elapsed_seconds": time.perf_counter() - started,
        "val": {
            "generation": val_gen,
            "selection": val_sel,
            "probability": val_prob,
        },
        "test": {
            "generation": test_gen,
            "selection": test_sel,
            "probability": test_prob,
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(summary_metrics, fp, indent=2)

    print(f"[QA-F16 DONE] Val Selected ADE: {val_sel['selected_ade']:.4f}m | MinADE20: {val_gen['min_ade20']:.4f}m | Regret: {val_sel['selection_regret']:.4f}m")
    print(f"[QA-F16 DONE] Test Selected ADE: {test_sel['selected_ade']:.4f}m | MinADE20: {test_gen['min_ade20']:.4f}m | Regret: {test_sel['selection_regret']:.4f}m")


if __name__ == "__main__":
    main()
