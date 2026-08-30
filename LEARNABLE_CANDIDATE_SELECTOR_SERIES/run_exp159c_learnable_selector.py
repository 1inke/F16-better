"""EXP-159C: fold-local listwise scorer for dynamic K40-to-K20 compression."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
SHARED_SCRIPT = WORKSPACE / "SHARED_ENCODER_COMPLEMENTARITY_SERIES" / "run_exp158b_shared_encoder_control.py"
BASE_AUDIT_SCRIPT = WORKSPACE / "CANDIDATE_COMPLEMENTARITY_SERIES" / "run_exp158_exp160_joint_audit.py"
MIXTURE_SCRIPT = WORKSPACE / "FIXED_K20_MIXTURE_SERIES" / "run_exp159_fixed_k20_mixture.py"
DEFAULT_OUT = HERE / "EXP_159C_LEARNABLE_K40_TO_K20_SELECTOR"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def observation_features(past: torch.Tensor, num_valid: torch.Tensor) -> torch.Tensor:
    focal = past[:, 0]
    velocity = focal[:, 1:] - focal[:, :-1]
    recent = velocity[:, -1]
    lag = min(3, velocity.shape[1])
    long = velocity[:, -lag:].mean(dim=1)
    acceleration = recent - long
    speed = torch.linalg.vector_norm(recent, dim=-1, keepdim=True)
    history = focal[:, -1] - focal[:, 0]
    last = past[:, :, -1]
    distance = torch.linalg.vector_norm(last - last[:, :1], dim=-1)
    valid = torch.arange(past.shape[1], device=past.device)[None] < num_valid[:, None]
    density = ((distance <= 2.0) & valid).sum(dim=1, keepdim=True).float() - 1.0
    neighbours = (num_valid.float() - 1.0).clamp_min(0.0)[:, None]
    return torch.cat((recent, long, acceleration, speed, history, neighbours, density), dim=-1)


def collect_cache(
    shared, base, fold: str, split: str, seed: int, cfg, model_class, helper,
    device: torch.device, batch_size: int, max_samples: int,
) -> dict[str, np.ndarray]:
    dataset = helper.TrajectoryDataset(
        base.DATA / f"{fold}_data_{split}.npy",
        base.DATA / f"{fold}_num_{split}.npy", cfg,
    )
    if max_samples:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    host = base.build_model(fold, "v2", seed, cfg, model_class, device)
    donor = base.build_model(fold, "k20", seed, cfg, model_class, device)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, args):
        captured["tokens"] = args[0]
        captured["past"] = args[1]

    handle = host.decoder.register_forward_pre_hook(hook)
    parts = {key: [] for key in (
        "context", "motion", "last_position", "future",
        "candidates", "original_logits",
    )}
    try:
        with torch.inference_mode():
            for past, future, num_valid, _, _, _ in loader:
                past = past.to(device)
                future = future.to(device)
                num_valid = num_valid.to(device).int()
                past, future = helper.normalize_to_focal_heading(past, future)
                center = past.mean(dim=-2, keepdim=True)
                v2_prediction = host(
                    past - center, None, None, mode="testing",
                    num_valid=num_valid, graph_loc=past,
                )[2] + center.unsqueeze(-3)
                if "tokens" not in captured:
                    raise RuntimeError("Failed to capture shared encoded tokens")
                k20_prediction, k20_logits = donor.decoder(captured["tokens"], captured["past"])
                k20_prediction = k20_prediction + center.unsqueeze(-3)
                tokens = captured["tokens"][:, 0]
                context = torch.cat((tokens[:, -1], tokens.mean(dim=1)), dim=-1)
                candidates = torch.cat((k20_prediction[:, 0], v2_prediction[:, 0]), dim=1)
                logits = torch.cat((k20_logits[:, 0], host.last_mode_logits[:, 0]), dim=1)
                parts["context"].append(context.float().cpu().numpy())
                parts["motion"].append(observation_features(past, num_valid).float().cpu().numpy())
                parts["last_position"].append(past[:, 0, -1].float().cpu().numpy())
                parts["future"].append(future[:, 0].float().cpu().numpy())
                parts["candidates"].append(candidates.float().cpu().numpy())
                parts["original_logits"].append(logits.float().cpu().numpy())
                captured.clear()
    finally:
        handle.remove()
    del host, donor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {key: np.concatenate(value) for key, value in parts.items()}


def load_or_collect(
    path: Path, force: bool, shared, base, fold: str, split: str, seed: int,
    cfg, model_class, helper, device: torch.device, batch_size: int,
    max_samples: int,
) -> dict[str, np.ndarray]:
    if path.is_file() and not force:
        with np.load(path) as raw:
            cached = {key: raw[key] for key in raw.files}
        total = int(np.load(base.DATA / f"{fold}_data_{split}.npy", mmap_mode="r").shape[0])
        expected = min(max_samples, total) if max_samples else total
        if len(cached["future"]) == expected:
            print(f"[cache] reuse {path}", flush=True)
            return cached
        print(
            f"[cache] stale sample count {len(cached['future'])} != {expected}; rebuild {path}",
            flush=True,
        )
    print(f"[cache] collect fold={fold} split={split}", flush=True)
    arrays = collect_cache(
        shared, base, fold, split, seed, cfg, model_class, helper, device,
        batch_size, max_samples,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return arrays


class CandidateSelector(nn.Module):
    def __init__(self, context_dim: int, motion_dim: int, hidden_dim: int = 64, pred_len: int = 12) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.context = nn.Sequential(
            nn.LayerNorm(context_dim + motion_dim),
            nn.Linear(context_dim + motion_dim, hidden_dim), nn.GELU(),
        )
        geometry_dim = pred_len * 2 + pred_len + 2 + 1 + 1 + 2
        self.geometry = nn.Sequential(
            nn.LayerNorm(geometry_dim), nn.Linear(geometry_dim, hidden_dim), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def normalized_old_score(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        halves, ranks = [], []
        for values in (logits[:, :20], logits[:, 20:]):
            normalized = (values - values.mean(dim=1, keepdim=True)) / values.std(dim=1, keepdim=True).clamp_min(0.05)
            order = values.argsort(dim=1, descending=True)
            rank = torch.empty_like(values)
            rank.scatter_(1, order, torch.linspace(1.0, 0.0, 20, device=values.device)[None].expand_as(values))
            halves.append(normalized); ranks.append(rank)
        return torch.cat(halves, dim=1), torch.cat(ranks, dim=1)

    def forward(
        self, context: torch.Tensor, motion: torch.Tensor,
        last_position: torch.Tensor, candidates: torch.Tensor,
        original_logits: torch.Tensor,
    ) -> torch.Tensor:
        relative = candidates - last_position[:, None, None]
        first = candidates[:, :, :1] - last_position[:, None, None]
        displacement = torch.cat((first, candidates[:, :, 1:] - candidates[:, :, :-1]), dim=2)
        speed_profile = torch.linalg.vector_norm(displacement, dim=-1)
        endpoint = relative[:, :, -1]
        old_score, rank = self.normalized_old_score(original_logits)
        family = F.one_hot(
            torch.cat((torch.zeros(20, device=candidates.device, dtype=torch.long), torch.ones(20, device=candidates.device, dtype=torch.long))),
            num_classes=2,
        ).float()[None].expand(len(candidates), -1, -1)
        geometry = torch.cat((
            relative.flatten(2), speed_profile, endpoint,
            old_score[..., None], rank[..., None], family,
        ), dim=-1)
        shared = self.context(torch.cat((context, motion), dim=-1))[:, None]
        return self.fusion(shared + self.geometry(geometry)).squeeze(-1)


def tensors(arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device):
    return tuple(torch.from_numpy(arrays[key][indices]).to(device=device, dtype=torch.float32) for key in (
        "context", "motion", "last_position", "candidates", "original_logits", "future",
    ))


def listwise_loss(scores: torch.Tensor, candidates: torch.Tensor, future: torch.Tensor, temperature: float) -> torch.Tensor:
    error = torch.linalg.vector_norm(candidates - future[:, None], dim=-1)
    cost = error.mean(dim=-1) + 0.25 * error[..., -1]
    target = torch.softmax(-cost.detach() / temperature, dim=1)
    return -(target * torch.log_softmax(scores, dim=1)).sum(dim=1).mean()


def validation_loss(model, arrays, indices, device, batch_size, temperature) -> float:
    model.eval(); total = 0.0; count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            context, motion, last, candidates, logits, future = tensors(arrays, ids, device)
            loss = listwise_loss(model(context, motion, last, candidates, logits), candidates, future, temperature)
            total += float(loss) * len(ids); count += len(ids)
    return total / count


def train_selector(arrays, device, seed, epochs, batch_size, lr, temperature, validation_fraction, patience, hidden_dim):
    count = len(arrays["future"])
    if count < 4:
        raise RuntimeError("Too few training samples for selector training")
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    val_count = max(1, int(round(count * validation_fraction)))
    val_indices, train_indices = order[:val_count], order[val_count:]
    model = CandidateSelector(arrays["context"].shape[1], arrays["motion"].shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    best_state = None; best_val = math.inf; stale = 0; history = []
    for epoch in range(1, epochs + 1):
        model.train(); shuffled = rng.permutation(train_indices); total = 0.0; seen = 0
        for start in range(0, len(shuffled), batch_size):
            ids = shuffled[start:start + batch_size]
            context, motion, last, candidates, logits, future = tensors(arrays, ids, device)
            score = model(context, motion, last, candidates, logits)
            loss = listwise_loss(score, candidates, future, temperature)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); total += float(loss.detach()) * len(ids); seen += len(ids)
        val = validation_loss(model, arrays, val_indices, device, batch_size, temperature)
        row = {"epoch": epoch, "train_loss": total / seen, "val_loss": val}
        history.append(row); print("[selector] " + json.dumps(row), flush=True)
        if val < best_val - 1.0e-5:
            best_val = val; stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Selector did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history, {"train_samples": len(train_indices), "validation_samples": len(val_indices), "best_validation_loss": best_val}


def oracle_from_errors(ade: np.ndarray, fde: np.ndarray) -> dict:
    rows = np.arange(len(ade)); best = ade.argmin(axis=1)
    return {"minADE": float(ade.min(1).mean()), "minFDE": float(fde.min(1).mean()), "same_path_FDE": float(fde[rows, best].mean()), "combined_cost": float((ade + .25 * fde).min(1).mean())}


def evaluate_selector(model, arrays, device, batch_size):
    model.eval(); ade_parts=[]; fde_parts=[]; score_parts=[]
    with torch.inference_mode():
        all_indices = np.arange(len(arrays["future"]))
        for start in range(0, len(all_indices), batch_size):
            ids=all_indices[start:start+batch_size]
            context,motion,last,candidates,logits,future=tensors(arrays,ids,device)
            score=model(context,motion,last,candidates,logits)
            error=torch.linalg.vector_norm(candidates-future[:,None],dim=-1)
            ade_parts.append(error.mean(-1).cpu().numpy());fde_parts.append(error[...,-1].cpu().numpy());score_parts.append(score.cpu().numpy())
    ade=np.concatenate(ade_parts);fde=np.concatenate(fde_parts);score=np.concatenate(score_parts);rows=np.arange(len(ade))
    selected=np.argsort(-score,axis=1)[:,:20];sel_ade=np.take_along_axis(ade,selected,axis=1);sel_fde=np.take_along_axis(fde,selected,axis=1)
    full=oracle_from_errors(ade,fde);top20=oracle_from_errors(sel_ade,sel_fde);v2=oracle_from_errors(ade[:,20:],fde[:,20:])
    top=score.argmax(1);best_ade=ade.argmin(1);best_fde=fde.argmin(1);cost=ade+.25*fde;best_cost=cost.argmin(1)
    selected_k20=(selected<20).sum(1);shift=score-score.max(1,keepdims=True);prob=np.exp(shift)/np.exp(shift).sum(1,keepdims=True)
    report={
        "samples":len(ade),"k40_oracle":full,"learned_top20_oracle":top20,"v2_oracle":v2,
        "top1":{"ADE":float(ade[rows,top].mean()),"FDE":float(fde[rows,top].mean()),"combined_winner_hit":float((top==best_cost).mean()),"best_ADE_hit":float((top==best_ade).mean()),"best_FDE_hit":float((top==best_fde).mean())},
        "retention":{"best_ADE":float((selected==best_ade[:,None]).any(1).mean()),"best_FDE":float((selected==best_fde[:,None]).any(1).mean()),"best_combined":float((selected==best_cost[:,None]).any(1).mean())},
        "compression_loss":{"minADE":top20["minADE"]/full["minADE"]-1.,"minFDE":top20["minFDE"]/full["minFDE"]-1.,"same_path_FDE":top20["same_path_FDE"]/full["same_path_FDE"]-1.},
        "selected_K20":{"mean":float(selected_k20.mean()),"std":float(selected_k20.std()),"histogram":np.bincount(selected_k20,minlength=21).astype(int).tolist()},
        "score_entropy":float(-(prob*np.log(np.maximum(prob,1.e-12))).sum(1).mean()),
    }
    return report, selected


def decide(rows, baselines, folds):
    m2_ade=float(np.mean([baselines["m2"][f]["min_ade_20"] for f in folds]));m2_fde=float(np.mean([baselines["m2"][f]["min_fde_20"] for f in folds]))
    macro_ade=float(np.mean([r["learned_top20_oracle"]["minADE"] for r in rows]));macro_fde=float(np.mean([r["learned_top20_oracle"]["minFDE"] for r in rows]));macro_same=float(np.mean([r["learned_top20_oracle"]["same_path_FDE"] for r in rows]))
    max_ade=max(r["learned_top20_oracle"]["minADE"]/baselines["k20"][r["fold"]]["min_ade_20"]-1 for r in rows);max_fde=max(r["learned_top20_oracle"]["minFDE"]/baselines["k20"][r["fold"]]["min_fde_20"]-1 for r in rows)
    hard=max_ade<=.02 and max_fde<=.02;m2=macro_ade<=m2_ade and macro_fde<=m2_fde
    if hard and m2:decision="PASSED"
    elif max_ade<=.05 and max_fde<=.05 and macro_ade<=1.02*m2_ade and macro_fde<=1.02*m2_fde:decision="CONDITIONAL"
    else:decision="REJECTED"
    return decision,{"macro_minADE":macro_ade,"macro_minFDE":macro_fde,"macro_same_path_FDE":macro_same,"max_fold_ADE_regression_vs_K20":float(max_ade),"max_fold_FDE_regression_vs_K20":float(max_fde),"hard_fold_gate":bool(hard),"m2_macro_gate":bool(m2)}


def save_selected(path, arrays, selected, scores=None):
    rows=np.arange(len(selected))[:,None];path.parent.mkdir(parents=True,exist_ok=True)
    payload={"context":arrays["context"],"motion":arrays["motion"],"last_position":arrays["last_position"],"future":arrays["future"],"candidates":arrays["candidates"][rows,selected],"source_index":selected.astype(np.int8),"family_id":(selected>=20).astype(np.int8)}
    np.savez_compressed(path,**payload)


def write_analysis(path,decision,aggregate):
    route="进入最终score同步与多随机种子验证。" if decision=="PASSED" else ("检查退化fold并决定是否小幅校准选择器。" if decision=="CONDITIONAL" else "learned selector仍未保留K40优势，停止双候选族主线并退回M2/V2。")
    path.write_text(f"# EXP-159C Result Analysis\n\n## Decision: {decision}\n\n宏平均 learned Top-20 minADE/minFDE/same-path FDE = {aggregate['macro_minADE']:.4f}/{aggregate['macro_minFDE']:.4f}/{aggregate['macro_same_path_FDE']:.4f}。\n\n相对matched K20最坏逐折ADE/FDE退化 = {100*aggregate['max_fold_ADE_regression_vs_K20']:.2f}%/{100*aggregate['max_fold_FDE_regression_vs_K20']:.2f}%。\n\n## Next step\n\n{route}\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--folds",nargs="+",choices=("all",*FOLDS),default=["all"]);p.add_argument("--seed",type=int,default=1);p.add_argument("--epochs",type=int,default=20);p.add_argument("--candidate-batch-size",type=int,default=64);p.add_argument("--selector-batch-size",type=int,default=256);p.add_argument("--learning-rate",type=float,default=1.e-3);p.add_argument("--temperature",type=float,default=.10);p.add_argument("--validation-fraction",type=float,default=.10);p.add_argument("--patience",type=int,default=5);p.add_argument("--hidden-dim",type=int,default=64);p.add_argument("--max-train-samples",type=int,default=0);p.add_argument("--max-test-samples",type=int,default=0);p.add_argument("--output-dir",type=Path,default=DEFAULT_OUT);p.add_argument("--force-cache",action="store_true");p.add_argument("--force",action="store_true");p.add_argument("--dry-run",action="store_true");a=p.parse_args()
    shared=load_module("exp159c_shared",SHARED_SCRIPT);base=load_module("exp159c_base",BASE_AUDIT_SCRIPT);mixture=load_module("exp159c_mixture",MIXTURE_SCRIPT);folds=FOLDS if "all" in a.folds else tuple(a.folds);base.preflight(folds,a.seed)
    if a.dry_run:print(json.dumps({"status":"dry-run passed","folds":folds,"seed":a.seed,"test_fold_used_for_training":False,"final_K":20}));return
    if a.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    seed_everything(a.seed);device=torch.device(a.device);cfg=base.configure(device)
    from front_encoder_models import FrontEncoderASTRA
    helper=load_module("exp159c_helper",base.TRAINER);baselines=mixture.load_baselines();a.output_dir.mkdir(parents=True,exist_ok=True);cache_dir=a.output_dir/"cache";checkpoint_dir=a.output_dir/"scorers";fold_result_dir=a.output_dir/"fold_results";selection_dir=a.output_dir/"selection_indices";rows=[];selections={};test_arrays={}
    run_contract={"seed":a.seed,"epochs":a.epochs,"selector_batch_size":a.selector_batch_size,"learning_rate":a.learning_rate,"temperature":a.temperature,"validation_fraction":a.validation_fraction,"patience":a.patience,"hidden_dim":a.hidden_dim,"max_train_samples":a.max_train_samples,"max_test_samples":a.max_test_samples}
    for fold_index,fold in enumerate(folds):
        print(f"[fold] {fold}",flush=True);fold_result_path=fold_result_dir/f"{fold}_seed_{a.seed}.json";selection_path=selection_dir/f"{fold}_seed_{a.seed}.npy";checkpoint_path=checkpoint_dir/f"{fold}_seed_{a.seed}.pth"
        if fold_result_path.is_file() and selection_path.is_file() and checkpoint_path.is_file() and not a.force and not a.force_cache:
            saved=json.loads(fold_result_path.read_text(encoding="utf-8"))
            if saved.get("run_contract")==run_contract:
                print(f"[resume] reuse completed fold={fold}",flush=True);rows.append(saved["report"]);selections[fold]=np.load(selection_path);test_arrays[fold]=load_or_collect(cache_dir/f"{fold}_test_seed_{a.seed}.npz",False,shared,base,fold,"test",a.seed,cfg,FrontEncoderASTRA,helper,device,a.candidate_batch_size,a.max_test_samples);continue
        train=load_or_collect(cache_dir/f"{fold}_train_seed_{a.seed}.npz",a.force_cache,shared,base,fold,"train",a.seed,cfg,FrontEncoderASTRA,helper,device,a.candidate_batch_size,a.max_train_samples);test=load_or_collect(cache_dir/f"{fold}_test_seed_{a.seed}.npz",a.force_cache,shared,base,fold,"test",a.seed,cfg,FrontEncoderASTRA,helper,device,a.candidate_batch_size,a.max_test_samples)
        model,history,split=train_selector(train,device,a.seed+fold_index,a.epochs,a.selector_batch_size,a.learning_rate,a.temperature,a.validation_fraction,a.patience,a.hidden_dim);report,selected=evaluate_selector(model,test,device,a.selector_batch_size);report.update({"fold":fold,"seed":a.seed,"selector_parameters":sum(x.numel() for x in model.parameters()),"training":split,"epochs_ran":len(history)});rows.append(report);selections[fold]=selected;test_arrays[fold]=test
        checkpoint_dir.mkdir(parents=True,exist_ok=True);torch.save({"state_dict":model.state_dict(),"context_dim":train["context"].shape[1],"motion_dim":train["motion"].shape[1],"hidden_dim":a.hidden_dim,"history":history,"training":split},checkpoint_path);fold_result_dir.mkdir(parents=True,exist_ok=True);selection_dir.mkdir(parents=True,exist_ok=True);fold_result_path.write_text(json.dumps({"run_contract":run_contract,"report":report},indent=2,ensure_ascii=False),encoding="utf-8");np.save(selection_path,selected);del model,train
        if device.type=="cuda":torch.cuda.empty_cache()
    decision,aggregate=decide(rows,baselines,folds);summary={"experiment":"EXP-159C Learnable K40-to-K20 Candidate Selector","protocol":"fold-local train/validation; held-out fold evaluation only","decision":decision,"aggregate":aggregate,"per_fold":rows,"training_contract":{"test_fold_used_for_training":False,"encoder_decoder_frozen":True,"selector_only":True,"temperature":a.temperature,"hidden_dim":a.hidden_dim}}
    (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    flat=[]
    for r in rows:flat.append({"fold":r["fold"],"samples":r["samples"],"top20_minADE":r["learned_top20_oracle"]["minADE"],"top20_minFDE":r["learned_top20_oracle"]["minFDE"],"top20_same_path_FDE":r["learned_top20_oracle"]["same_path_FDE"],"k40_minADE":r["k40_oracle"]["minADE"],"k40_minFDE":r["k40_oracle"]["minFDE"],"top1_ADE":r["top1"]["ADE"],"top1_FDE":r["top1"]["FDE"],"retention_ADE":r["retention"]["best_ADE"],"retention_FDE":r["retention"]["best_FDE"],"mean_selected_K20":r["selected_K20"]["mean"],"selector_parameters":r["selector_parameters"]})
    with (a.output_dir/"per_fold_summary.csv").open("w",newline="",encoding="utf-8-sig") as h:w=csv.DictWriter(h,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    write_analysis(a.output_dir/"RESULT_ANALYSIS.md",decision,aggregate)
    if decision!="REJECTED" and not a.max_test_samples:
        for fold in folds:save_selected(a.output_dir/"selected_candidates"/f"{fold}_seed_{a.seed}.npz",test_arrays[fold],selections[fold])
    print(json.dumps({"experiment":"EXP-159C","decision":decision,"aggregate":aggregate,"output":str(a.output_dir)},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
