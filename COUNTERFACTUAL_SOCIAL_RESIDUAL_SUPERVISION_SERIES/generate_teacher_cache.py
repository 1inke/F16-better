"""Generate strict Full/Self B0 teacher cache for CSRS1."""

from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT=Path(__file__).resolve().parent; WORKSPACE=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
FRONT=WORKSPACE/"FRONT_ENCODER_SERIES"; IR1=WORKSPACE/"INTERACTION_RESIDUAL_SUPERVISION_SERIES"
SOURCE_ROOT=IR1/"results"; OUTPUT=ROOT/"teacher_cache"; SPLITS=("general","challenge"); LAMBDA_FDE=0.5


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def source_checkpoint(split,seed):
    return SOURCE_ROOT/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"


def run_dir(split,seed): return source_checkpoint(split,seed).parent


def prepare(device):
    sys.path[:0]=[str(PROJECT),str(PROJECT/"scripts"),str(FRONT),str(WORKSPACE/"COUNTERFACTUAL_SOCIAL_NECESSITY_AUDIT")]
    os.chdir(PROJECT); import sitecustomize  # noqa
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    cfg=load_locked_q20(PROJECT/"configs"/"motif_rbf_univ.yaml")
    cfg.MODEL.USE_PRETRAINED_UNET=False; cfg.MODEL.USE_CACHED_UNET_FEATURES=False
    cfg.DATA.MAX_AGENTS=9; cfg.MODEL.GRAPH.TOP_K=4; cfg.MODEL.GRAPH.RADIUS=6.0; cfg.MODEL.DECODER.NUM_MODES=20
    cfg.device=str(device); cfg.device_list=[device.index or 0] if device.type=="cuda" else ["cpu"]
    return cfg,FrontEncoderASTRA


def build_model(split,seed,cfg,model_class,device):
    checkpoint=source_checkpoint(split,seed)
    os.environ["IRS_SOURCE_CHECKPOINT"]=str(checkpoint)
    model=model_class(cfg,"f16_temporal_social_cross")
    adapter=load_module(f"csrs_teacher_adapter_{split}",IR1/"current_baseline_adapter.py")
    adapter.replace_decoder(model,cfg,"current_mode_anchor_template")
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def infer(model,loader,device,self_only):
    from train_front_variant import normalize_to_focal_heading
    from csna_mask import focal_self_only_intervention
    context=focal_self_only_intervention(model) if self_only else NullContext(model)
    parts={k:[] for k in ("index","candidate","anchor","ade","fde","best_mode","gt_anchor")}
    with context:
        for past,future,num_valid,indices,_ in loader:
            past=past.to(device,non_blocking=True); future=future.to(device,non_blocking=True); num_valid=num_valid.to(device).int()
            past,future=normalize_to_focal_heading(past,future); center=past.mean(dim=-2,keepdim=True)
            prediction=model(past-center,None,None,mode="testing",num_valid=num_valid,graph_loc=past)[2]+center.unsqueeze(-3)
            candidate=prediction[:,0]; target=future[:,0]
            errors=torch.linalg.vector_norm(candidate-target[:,None],dim=-1); ade=errors.mean(-1); fde=errors[...,-1]
            best=(ade+LAMBDA_FDE*fde).argmin(-1)
            values={"index":indices.numpy(),"candidate":candidate.cpu().numpy().astype(np.float16),
                    "anchor":model.decoder.last_corrected_anchors[:,0].cpu().numpy().astype(np.float16),
                    "ade":ade.cpu().numpy().astype(np.float32),"fde":fde.cpu().numpy().astype(np.float32),
                    "best_mode":best.cpu().numpy().astype(np.int16),
                    "gt_anchor":target[:,[3,7,11]].cpu().numpy().astype(np.float32)}
            for k,v in values.items(): parts[k].append(v)
    return {k:np.concatenate(v) for k,v in parts.items()}


class NullContext:
    def __init__(self,value): self.value=value
    def __enter__(self): return self.value
    def __exit__(self,*args): return False


def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda:0"); p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--seed",type=int,default=1); p.add_argument("--splits",nargs="+",choices=("all",*SPLITS),default=["all"])
    p.add_argument("--delta",type=float,default=0.01); p.add_argument("--tau",type=float,default=0.10); p.add_argument("--force",action="store_true"); a=p.parse_args()
    device=torch.device(a.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    splits=list(SPLITS) if "all" in a.splits else a.splits; OUTPUT.mkdir(parents=True,exist_ok=True)
    cfg,model_class=prepare(device)
    geometry=load_module("csrs_teacher_geometry",WORKSPACE/"INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES"/"run_geometry_diagnostic.py")
    for split in splits:
        output=OUTPUT/f"{split}_seed_{a.seed}.npz"
        if output.is_file() and not a.force: print(f"[skip] cache exists {output}",flush=True); continue
        run=run_dir(split,a.seed); split_arrays=np.load(run/"split_indices.npz")
        train=np.asarray(split_arrays["train"],dtype=np.int64); validation=np.asarray(split_arrays["validation"],dtype=np.int64)
        info=json.loads((run/"split.json").read_text(encoding="utf-8")); data_path,num_path=Path(info["data_path"]),Path(info["num_path"])
        indices=np.arange(len(np.load(data_path,mmap_mode="r")),dtype=np.int64)
        dataset=geometry.IndexedDataset(data_path,num_path,indices,cfg)
        loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda")
        model=build_model(split,a.seed,cfg,model_class,device)
        print(f"[teacher] {split}/full samples={len(dataset)}",flush=True); full=infer(model,loader,device,False)
        print(f"[teacher] {split}/self samples={len(dataset)}",flush=True); self_only=infer(model,loader,device,True)
        if not np.array_equal(full["index"],indices) or not np.array_equal(self_only["index"],indices): raise RuntimeError("teacher pairing failed")
        rows=np.arange(len(indices)); full_cost=full["ade"]+LAMBDA_FDE*full["fde"]; self_cost=self_only["ade"]+LAMBDA_FDE*self_only["fde"]
        full_best=full_cost.min(1); self_best=self_cost.min(1); gain=self_best-full_best
        self_anchor=self_only["anchor"].astype(np.float32)[rows,self_only["best_mode"]]
        residual=full["gt_anchor"]-self_anchor; need=(gain>a.delta).astype(np.float32)
        confidence=np.clip(np.maximum(gain,0.0)/a.tau,0.0,1.0).astype(np.float32)
        role=np.zeros(len(indices),dtype=np.int8); role[train]=1; role[validation]=2
        checkpoint=source_checkpoint(split,a.seed)
        metadata={"version":1,"split":split,"seed":a.seed,"delta":a.delta,"tau":a.tau,"lambda_fde":LAMBDA_FDE,
                  "source_checkpoint":str(checkpoint),"source_checkpoint_sha256":sha256(checkpoint),
                  "data_path":str(data_path),"samples":len(indices),"train_samples":len(train),"validation_samples":len(validation),
                  "train_indices_sha256":hashlib.sha256(train.tobytes()).hexdigest(),"validation_indices_sha256":hashlib.sha256(validation.tobytes()).hexdigest(),
                  "train_positive_rate":float(need[train].mean()),"validation_positive_rate":float(need[validation].mean())}
        np.savez_compressed(output,metadata_json=np.array(json.dumps(metadata,ensure_ascii=False)),indices=indices,role=role,
                            full_candidates=full["candidate"],self_candidates=self_only["candidate"],full_anchors=full["anchor"],self_anchors=self_only["anchor"],
                            full_ade=full["ade"],full_fde=full["fde"],self_ade=self_only["ade"],self_fde=self_only["fde"],
                            full_best_mode=full["best_mode"],self_best_mode=self_only["best_mode"],gt_anchor=full["gt_anchor"],
                            social_gain=gain.astype(np.float32),need_label=need,confidence=confidence,residual_target=residual.astype(np.float32))
        check=np.load(output,allow_pickle=False); loaded=json.loads(str(check["metadata_json"].item()))
        if loaded["source_checkpoint_sha256"]!=sha256(checkpoint) or len(check["indices"])!=len(indices): raise RuntimeError("cache verification failed")
        print(f"[done] {output} train_positive={metadata['train_positive_rate']:.4f} val_positive={metadata['validation_positive_rate']:.4f}",flush=True)
        del model
        if device.type=="cuda": torch.cuda.empty_cache()


if __name__=="__main__": main()

