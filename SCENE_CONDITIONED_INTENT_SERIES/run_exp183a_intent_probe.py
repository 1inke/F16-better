"""EXP-183A end-to-end frozen feature, intent probe, and scene-swap audit."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

# Avoid a harmless Windows physical-core probe warning from joblib/sklearn.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import timm

from intent_probe_models import build_probe, probe_forward


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "EXP_183A_SCENE_CONDITIONED_INTENT_PROBE"
CACHE = OUT / "cache"
DATA = ROOT / "SDD_STANDARD_POTENTIAL_SERIES" / "processed_data"
ASSETS = ROOT / "SCENE_QUERY_CURRENT_DECODER_SDD_SERIES" / "processed_scene_assets"
TOPOLOGY = ROOT / "SCENE_TOPOLOGY_SERIES" / "EXP_178A_TOPOLOGY_REACHABILITY" / "assets" / "topology_bank.npz"
S0 = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175S0_FINAL_JOINT_V2_SDD_BASELINE"
BASE = ROOT / "THREE_BRANCH_FOUNDATION_SERIES" / "EXP_175A_SCENE_UTILITY_AUDIT" / "run_exp175a_scene_utility_audit.py"
SDD_ROOT = Path(r"D:\Datasets\Stanford_Drone\stanford_campus_dataset")
PIXEL_SCALE = 20.0
VARIANTS = ["P0", "P1_lite", "P1_rich", "P2_lite", "P2_rich", "P3_lite", "P3_rich"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def split_loader(split, cfg, trainer, batch_size, max_samples):
    if split in {"train", "validation"}:
        prefix = "sdd_standard_train_val"; sidecar = ASSETS / "sdd_train_val_scene_ids.npy"
        ids = np.load(sidecar, mmap_mode="r")
        keep = np.flatnonzero(ids < 31) if split == "train" else np.flatnonzero((ids >= 31) & (ids <= 42))
    else:
        prefix = "sdd_standard_test"; sidecar = ASSETS / "sdd_test_scene_ids.npy"
        ids = np.load(sidecar, mmap_mode="r"); keep = np.arange(len(ids))
    dataset = trainer.TrajectoryDataset(DATA / f"{prefix}_data.npy", DATA / f"{prefix}_num.npy", cfg, sample_scene_path=sidecar)
    if max_samples and len(keep) > max_samples:
        keep = keep[np.linspace(0, len(keep) - 1, max_samples, dtype=np.int64)]
    return DataLoader(Subset(dataset, keep.tolist()), batch_size=batch_size, shuffle=False, num_workers=0)


def motion_descriptors(past_abs):
    velocity = past_abs[:, 0, 1:] - past_abs[:, 0, :-1]
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    mean_speed = speed.mean(-1)
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    mean_acceleration = torch.linalg.vector_norm(acceleration, dim=-1).mean(-1)
    a, b = velocity[:, -2], velocity[:, -1]
    cosine = (a * b).sum(-1) / (torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(b, dim=-1)).clamp_min(1e-6)
    turn = torch.acos(cosine.clamp(-1, 1))
    return torch.stack((mean_speed, mean_acceleration, turn), dim=-1)


def extract_motion_cache(device, batch_size, max_samples, force=False):
    cache_dir = CACHE; cache_dir.mkdir(parents=True, exist_ok=True)
    expected = [cache_dir / f"motion_{x}.npz" for x in ("train", "validation", "test")]
    if not force and all(x.is_file() for x in expected): return
    base = load_module("exp183a_base", BASE); model, cfg, trainer = base.build_model(1, device)
    model.eval()
    captured = {}
    def hook(_module, inputs, output):
        captured["m0"] = inputs[0].detach(); captured["m1"] = output.detach()
    handle = model.decoder.context.register_forward_hook(hook)
    for split in ("train", "validation", "test"):
        pieces = {}
        repeat_max = None
        def add(name, value): pieces.setdefault(name, []).append(value.detach().cpu().numpy())
        loader = split_loader(split, cfg, trainer, batch_size, max_samples)
        print(f"[motion] {split}: {len(loader.dataset)} samples", flush=True)
        with torch.inference_mode():
            for past_abs, future_abs, num_valid, scene_id, *_ in loader:
                past_abs = past_abs.float().to(device); future_abs = future_abs.float().to(device)
                num_valid = num_valid.int().to(device); scene_id = scene_id.long().to(device)
                past, future = trainer.normalize_to_focal_heading(past_abs, future_abs)
                center = past.mean(dim=-2, keepdim=True)
                output = model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
                if repeat_max is None:
                    first_m1 = captured["m1"].clone()
                    model(past - center, None, None, mode="testing", num_valid=num_valid, graph_loc=past)
                    repeat_max = float((first_m1 - captured["m1"]).abs().max().cpu())
                prediction = output[2] if isinstance(output, (tuple, list)) else output
                prediction = (prediction + center.unsqueeze(-3))[:, 0]
                endpoint = future[:, 0, -1] - past[:, 0, -1]
                radius = torch.linalg.vector_norm(endpoint, dim=-1)
                direction = endpoint / radius[:, None].clamp_min(1e-6)
                observed_speed = torch.linalg.vector_norm(past[:, 0, 1:] - past[:, 0, :-1], dim=-1).mean(-1)
                endpoint_scale = 12.0 * observed_speed
                radius_norm_raw = radius / endpoint_scale.clamp_min(1e-6)
                radius_norm = radius_norm_raw.clamp(0, 3)
                candidate_endpoint = prediction[:, :, -1] - past[:, 0, None, -1]
                candidate_unit = F.normalize(candidate_endpoint, dim=-1, eps=1e-6)
                ambiguity = 1.0 - torch.linalg.vector_norm(candidate_unit.mean(dim=1), dim=-1)
                lag = 3; heading = past_abs[:, 0, -1] - past_abs[:, 0, -1-lag]
                heading = F.normalize(heading, dim=-1, eps=1e-6)
                add("m0", captured["m0"][:, 0]); add("m1", captured["m1"][:, 0])
                add("target", torch.cat((direction, radius_norm[:, None]), dim=-1))
                add("target_endpoint", endpoint); add("radius_norm_raw", radius_norm_raw)
                add("endpoint_scale", endpoint_scale); add("scene_id", scene_id)
                add("center", past_abs[:, 0, -1]); add("heading", heading)
                add("physical", motion_descriptors(past_abs)); add("ambiguity", ambiguity)
        arrays = {k: np.concatenate(v, axis=0) for k, v in pieces.items()}
        reconstruction = arrays["target"][:, :2] * arrays["target"][:, 2:3] * arrays["endpoint_scale"][:, None]
        valid_reconstruction = (arrays["radius_norm_raw"] <= 3.0) & (arrays["endpoint_scale"] > 1e-6)
        error = np.abs(reconstruction[valid_reconstruction] - arrays["target_endpoint"][valid_reconstruction])
        arrays["target_reconstruction_max_error"] = np.asarray([error.max() if len(error) else 0.0], np.float32)
        arrays["radius_clipped_fraction"] = np.asarray([(arrays["radius_norm_raw"] > 3.0).mean()], np.float32)
        arrays["motion_repeat_max_abs"] = np.asarray([repeat_max or 0.0], np.float32)
        np.savez_compressed(cache_dir / f"motion_{split}.npz", **arrays)
    handle.remove(); del model
    if device.type == "cuda": torch.cuda.empty_cache()


def bilinear(map2d, x, y):
    h, w = map2d.shape; x = np.clip(x, 0, w-1); y = np.clip(y, 0, h-1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); x1 = np.minimum(x0+1, w-1); y1 = np.minimum(y0+1, h-1)
    wx = x-x0; wy = y-y0
    return ((1-wx)*(1-wy)*map2d[y0,x0] + wx*(1-wy)*map2d[y0,x1] + (1-wx)*wy*map2d[y1,x0] + wx*wy*map2d[y1,x1])


def build_lite_cache(force=False):
    cache_dir = CACHE; expected = [cache_dir / f"lite_{x}.npy" for x in ("train", "validation", "test")]
    if not force and all(x.is_file() for x in expected): return
    semantic = np.load(ASSETS / "sdd_semantic_bank.npz", allow_pickle=False)
    maps = semantic["maps"].astype(np.float32) / 255.0; wh = semantic["image_wh"].astype(np.float32)
    topology = np.load(TOPOLOGY, allow_pickle=False); orientation = topology["orientation"]; degree = topology["degree"]
    offsets = [(0,0), (32,0), (64,0), (0,-32), (0,32)]  # forward, lateral in original pixels
    for split in ("train", "validation", "test"):
        data = np.load(cache_dir / f"motion_{split}.npz"); center = data["center"] * PIXEL_SCALE
        heading = data["heading"]; right = np.stack((-heading[:,1], heading[:,0]), axis=-1); scene = data["scene_id"].astype(int)
        features = []
        for forward, lateral in offsets:
            point = center + forward*heading + lateral*right
            values = np.empty((len(scene), 6), np.float32)
            for channel in range(6):
                x = point[:,0] / np.maximum(wh[scene,0]-1,1) * 159; y = point[:,1] / np.maximum(wh[scene,1]-1,1) * 159
                values[:,channel] = np.asarray([bilinear(maps[s,channel], xx, yy) for s,xx,yy in zip(scene,x,y)])
            features.append(values)
        x0 = np.clip(np.rint(center[:,0] / np.maximum(wh[scene,0]-1,1)*159).astype(int),0,159)
        y0 = np.clip(np.rint(center[:,1] / np.maximum(wh[scene,1]-1,1)*159).astype(int),0,159)
        ori = orientation[scene,y0,x0]
        align = np.abs((ori*heading).sum(-1,keepdims=True)); branch = degree[scene,y0,x0,None].astype(np.float32)/8.0
        lite = np.concatenate(features+[align,branch],axis=1).astype(np.float32)
        if lite.shape[1] != 32: raise RuntimeError(lite.shape)
        np.save(cache_dir / f"lite_{split}.npy", lite)


def heading_crop(image, center_model, heading, size=224, span_pixels=224):
    center = np.asarray(center_model, np.float32) * PIXEL_SCALE; heading = np.asarray(heading, np.float32)
    right = np.asarray([-heading[1], heading[0]], np.float32)
    axis = (np.arange(size, dtype=np.float32) - (size-1)/2) * (span_pixels/size)
    xx, yy = np.meshgrid(axis, axis)
    map_x = center[0] + xx*right[0] - yy*heading[0]
    map_y = center[1] + xx*right[1] - yy*heading[1]
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def build_rich_cache(device, batch_size, force=False, untrained=False):
    cache_dir = CACHE; expected = [cache_dir / f"rich_raw_{x}.npy" for x in ("train", "validation", "test")]
    if not force and all(x.is_file() for x in expected): return
    manifest = json.loads((S0 / "video_ids" / "video_id_manifest.json").read_text(encoding="utf-8"))
    id_to_name = {int(v): k for k,v in manifest["mapping"].items()}
    images = {}
    for scene_id, name in id_to_name.items():
        path = SDD_ROOT / "annotations" / name / "reference.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(path)
        images[scene_id] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = timm.create_model("vit_small_patch14_dinov2", pretrained=not untrained, num_classes=0, img_size=224).to(device).eval()
    mean = torch.tensor((0.485,0.456,0.406),device=device).view(1,3,1,1); std = torch.tensor((0.229,0.224,0.225),device=device).view(1,3,1,1)
    for split in ("train", "validation", "test"):
        data = np.load(cache_dir / f"motion_{split}.npz"); scene = data["scene_id"].astype(int)
        outputs=[]; print(f"[DINOv2] {split}: {len(scene)} crops",flush=True)
        for start in range(0,len(scene),batch_size):
            crops=[heading_crop(images[int(s)],c,h) for s,c,h in zip(scene[start:start+batch_size],data["center"][start:start+batch_size],data["heading"][start:start+batch_size])]
            tensor=torch.from_numpy(np.stack(crops)).to(device=device,dtype=torch.float32).permute(0,3,1,2)/255.0
            with torch.inference_mode(), torch.autocast(device_type=device.type,enabled=device.type=="cuda",dtype=torch.float16):
                feature=model((tensor-mean)/std)
            outputs.append(feature.float().cpu().numpy())
        np.save(cache_dir/f"rich_raw_{split}.npy",np.concatenate(outputs).astype(np.float16))
    del model
    if device.type == "cuda": torch.cuda.empty_cache()


def fit_rich_pca(force=False):
    cache_dir=CACHE; expected=[cache_dir/f"rich_{x}.npy" for x in ("train","validation","test")]
    if not force and all(x.is_file() for x in expected): return
    train=np.load(cache_dir/"rich_raw_train.npy").astype(np.float32); components=min(64,train.shape[0]-1,train.shape[1])
    pca=PCA(n_components=components,svd_solver="randomized",random_state=183)
    pca.fit(train)
    for split in ("train","validation","test"):
        values=pca.transform(np.load(cache_dir/f"rich_raw_{split}.npy").astype(np.float32)).astype(np.float32)
        np.save(cache_dir/f"rich_{split}.npy",values)
    np.savez(cache_dir/"rich_pca.npz",mean=pca.mean_,components=pca.components_,explained_variance_ratio=pca.explained_variance_ratio_)


def load_split(split):
    cache=CACHE; motion=np.load(cache/f"motion_{split}.npz")
    return {k:motion[k] for k in motion.files} | {"lite":np.load(cache/f"lite_{split}.npy"),"rich":np.load(cache/f"rich_{split}.npy")}


def metrics(pred,target,scale):
    cosine=np.clip((pred[:,:2]*target[:,:2]).sum(-1),-1,1)
    angular=np.degrees(np.arccos(cosine)); radius=np.abs(pred[:,2]-target[:,2])
    endpoint=np.linalg.norm((pred[:,:2]*pred[:,2:3]-target[:,:2]*target[:,2:3])*scale[:,None],axis=-1)*PIXEL_SCALE
    return {"endpoint_error_px":float(endpoint.mean()),"angular_error_deg":float(angular.mean()),"radius_mae":float(radius.mean())},endpoint,angular


def predict(model,variant,data,device,batch_size,scene_override=None):
    outputs=[]; model.eval()
    with torch.inference_mode():
        for start in range(0,len(data["target"]),batch_size):
            sl=slice(start,start+batch_size); m=torch.from_numpy(data["m1"][sl]).float().to(device)
            lite_source=data["lite"] if scene_override is None else scene_override.get("lite",data["lite"])
            rich_source=data["rich"] if scene_override is None else scene_override.get("rich",data["rich"])
            lite=torch.from_numpy(lite_source[sl]).float().to(device); rich=torch.from_numpy(rich_source[sl]).float().to(device)
            outputs.append(probe_forward(model,variant,m,lite,rich).cpu().numpy())
    return np.concatenate(outputs)


def train_probe(variant,seed,train,val,device,batch_size,epochs,patience):
    seed_all(seed); model=build_probe(variant,train["m1"].shape[1],train["lite"].shape[1],train["rich"].shape[1]).to(device)
    parameters=sum(p.numel() for p in model.parameters())
    if parameters>100000: raise RuntimeError(f"Probe too large: {variant}={parameters}")
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    target=torch.from_numpy(train["target"]).float().to(device); motion=torch.from_numpy(train["m1"]).float().to(device)
    lite=torch.from_numpy(train["lite"]).float().to(device); rich=torch.from_numpy(train["rich"]).float().to(device)
    best=None; best_error=float("inf"); wait=0; best_epoch=0
    for epoch in range(1,epochs+1):
        model.train(); order=torch.randperm(len(target),device=device)
        for start in range(0,len(order),batch_size):
            idx=order[start:start+batch_size]; pred=probe_forward(model,variant,motion[idx],lite[idx],rich[idx])
            loss=(1-(pred[:,:2]*target[idx,:2]).sum(-1)).mean()+0.5*F.smooth_l1_loss(pred[:,2],target[idx,2])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        val_pred=predict(model,variant,val,device,batch_size); score=metrics(val_pred,val["target"],val["endpoint_scale"])[0]["endpoint_error_px"]
        if score<best_error-1e-6:
            best_error=score;best=copy.deepcopy(model.state_dict());best_epoch=epoch;wait=0
        else:
            wait+=1
            if wait>=patience:break
    model.load_state_dict(best)
    return model,parameters,best_epoch,best_error


def swap_indices(data,seed):
    physical=data["physical"].astype(np.float64); physical=(physical-physical.mean(0))/np.maximum(physical.std(0),1e-6)
    video=data["scene_id"].astype(int); n=len(video); nn_index=NearestNeighbors(n_neighbors=min(n,700)).fit(physical)
    neighbors=nn_index.kneighbors(physical, n_neighbors=min(n,700), return_distance=False); matched=np.empty(n,np.int64)
    for i,row in enumerate(neighbors):
        choices=row[video[row]!=video[i]]
        if not len(choices): raise RuntimeError(f"No cross-video neighbor for {i}")
        matched[i]=choices[0]
    rng=np.random.default_rng(seed); random_index=np.empty(n,np.int64)
    for i in range(n):
        choices=np.flatnonzero(video!=video[i]);random_index[i]=rng.choice(choices)
    return matched,random_index


def nearest_centroid_video_accuracy(features,video):
    train_idx=[];test_idx=[]
    for value in np.unique(video):
        ids=np.flatnonzero(video==value)
        if len(ids)<2:continue
        cut=max(1,int(len(ids)*0.8));train_idx.extend(ids[:cut]);test_idx.extend(ids[cut:])
    x=features.astype(np.float64);mean=x[train_idx].mean(0);std=np.maximum(x[train_idx].std(0),1e-6);x=(x-mean)/std
    labels=np.unique(video[train_idx]);centroids=np.stack([x[np.asarray(train_idx)[video[train_idx]==label]].mean(0) for label in labels])
    distance=((x[test_idx,None]-centroids[None])**2).mean(-1);prediction=labels[distance.argmin(1)]
    return float((prediction==video[test_idx]).mean()),len(test_idx)


def save_crop_alignment_preview(data, output_path):
    manifest=json.loads((S0/"video_ids"/"video_id_manifest.json").read_text(encoding="utf-8"));names={int(v):k for k,v in manifest["mapping"].items()}
    selected=np.linspace(0,len(data["scene_id"])-1,min(10,len(data["scene_id"])),dtype=int);tiles=[]
    for index in selected:
        name=names[int(data["scene_id"][index])];image=cv2.imread(str(SDD_ROOT/"annotations"/name/"reference.jpg"))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);crop=heading_crop(image,data["center"][index],data["heading"][index])
        crop=cv2.cvtColor(crop,cv2.COLOR_RGB2BGR);center=(112,112)
        cv2.arrowedLine(crop,center,(112,35),(0,255,0),2,tipLength=0.2)
        endpoint=data["target_endpoint"][index]*PIXEL_SCALE
        point=(int(np.clip(112+endpoint[1],0,223)),int(np.clip(112-endpoint[0],0,223)))
        cv2.circle(crop,point,5,(0,0,255),-1);cv2.putText(crop,name,(4,216),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1)
        tiles.append(crop)
    while len(tiles)<10:tiles.append(np.zeros_like(tiles[0]))
    sheet=np.concatenate([np.concatenate(tiles[:5],axis=1),np.concatenate(tiles[5:10],axis=1)],axis=0)
    output_path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(output_path),sheet)


def grouped_comparison(data, prediction_bank, seeds):
    p0=np.mean([prediction_bank[("P0",s)] for s in seeds],axis=0);p3=np.mean([prediction_bank[("P3_rich",s)] for s in seeds],axis=0)
    train=load_split("train"); speed_threshold=np.quantile(train["physical"][:,0],[1/3,2/3]); ambiguity_threshold=np.quantile(train["ambiguity"],[.25,.5,.75])
    speed=data["physical"][:,0];turn=np.degrees(data["physical"][:,2]);ambiguity=data["ambiguity"]
    groups={
        "speed_low":speed<=speed_threshold[0],"speed_medium":(speed>speed_threshold[0])&(speed<=speed_threshold[1]),"speed_high":speed>speed_threshold[1],
        "turn_straight":turn<10,"turn_moderate":(turn>=10)&(turn<30),"turn_strong":turn>=30,
    }
    edges=[-np.inf,*ambiguity_threshold,np.inf]
    for i in range(4):groups[f"ambiguity_Q{i+1}"]=(ambiguity>edges[i])&(ambiguity<=edges[i+1])
    rows=[]
    for name,mask in groups.items():
        if not mask.any():continue
        m0,_,_=metrics(p0[mask],data["target"][mask],data["endpoint_scale"][mask]);m3,_,_=metrics(p3[mask],data["target"][mask],data["endpoint_scale"][mask])
        rows.append({"group":name,"samples":int(mask.sum()),"P0_endpoint_error_px":m0["endpoint_error_px"],"P3_rich_endpoint_error_px":m3["endpoint_error_px"],"endpoint_gain_pct":(m0["endpoint_error_px"]-m3["endpoint_error_px"])/m0["endpoint_error_px"]*100,"P0_angular_error_deg":m0["angular_error_deg"],"P3_rich_angular_error_deg":m3["angular_error_deg"]})
    return rows


def write_csv(path,rows):
    if not rows:return
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def aggregate(rows,variant,subset="overall"):
    selected=[r for r in rows if r["variant"]==variant and r["subset"]==subset]
    result={"variant":variant,"subset":subset,"params":selected[0]["params"]}
    for key in ("endpoint_error_px","angular_error_deg","radius_mae"):
        values=np.asarray([r[key] for r in selected]);result[key+"_mean"]=float(values.mean());result[key+"_std"]=float(values.std(ddof=1)) if len(values)>1 else 0.0
    return result


def main():
    global CACHE
    parser=argparse.ArgumentParser(description="EXP-183A Scene-Conditioned Intent Probe")
    parser.add_argument("--device",default="cuda:0");parser.add_argument("--batch-size",type=int,default=512)
    parser.add_argument("--feature-batch-size",type=int,default=64);parser.add_argument("--rich-batch-size",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=50);parser.add_argument("--patience",type=int,default=8)
    parser.add_argument("--probe-seeds",type=int,nargs="+",default=[1,2,3]);parser.add_argument("--max-samples",type=int,default=0)
    parser.add_argument("--force-cache",action="store_true");parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--untrained-rich",action="store_true",help="Smoke-test only; formal runs reject this flag")
    args=parser.parse_args();device=torch.device(args.device)
    if args.untrained_rich and not args.smoke:raise RuntimeError("Untrained rich backbone is allowed only for smoke tests")
    if args.smoke:
        args.max_samples=args.max_samples or 32;args.epochs=min(args.epochs,2);args.patience=2;args.probe_seeds=args.probe_seeds[:1]
        CACHE=OUT/"preflight"/"cache"
    OUT.mkdir(parents=True,exist_ok=True)
    extract_motion_cache(device,args.feature_batch_size,args.max_samples,args.force_cache)
    build_lite_cache(args.force_cache);build_rich_cache(device,args.rich_batch_size,args.force_cache,args.untrained_rich);fit_rich_pca(args.force_cache)
    data={split:load_split(split) for split in ("train","validation","test")}
    reconstruction=float(data["test"]["target_reconstruction_max_error"][0])
    if reconstruction>1e-5:raise RuntimeError(f"Target reconstruction failed: {reconstruction}")
    output_dir=OUT/"preflight" if args.smoke else OUT;output_dir.mkdir(parents=True,exist_ok=True)
    if args.smoke:save_crop_alignment_preview(data["test"],output_dir/"crop_alignment.jpg")
    result_rows=[];prediction_bank={};swap_rows=[]
    matched,random_swap=swap_indices(data["test"],183)
    for variant in VARIANTS:
        for seed in args.probe_seeds:
            print(f"[probe] {variant} seed={seed}",flush=True)
            model,params,epoch,val_error=train_probe(variant,seed,data["train"],data["validation"],device,args.batch_size,args.epochs,args.patience)
            pred=predict(model,variant,data["test"],device,args.batch_size);prediction_bank[(variant,seed)]=pred
            for subset,mask in (("overall",np.ones(len(pred),bool)),("little",np.isin(data["test"]["scene_id"],[50,51,52,53])),("non_little",~np.isin(data["test"]["scene_id"],[50,51,52,53]))):
                values,_,_=metrics(pred[mask],data["test"]["target"][mask],data["test"]["endpoint_scale"][mask])
                result_rows.append({"variant":variant,"seed":seed,"subset":subset,"params":params,"best_epoch":epoch,"val_endpoint_error_px":val_error,**values})
            if variant=="P3_rich":
                alternatives={"correct":data["test"]["rich"],"matched_swap":data["test"]["rich"][matched],"random_swap":data["test"]["rich"][random_swap],"zero":np.zeros_like(data["test"]["rich"])}
                for kind,rich in alternatives.items():
                    p=predict(model,variant,data["test"],device,args.batch_size,{"rich":rich});value=metrics(p,data["test"]["target"],data["test"]["endpoint_scale"])[0]
                    swap_rows.append({"seed":seed,"condition":kind,**value})
    summary_rows=[aggregate(result_rows,v) for v in VARIANTS]
    per_video=[]
    manifest=json.loads((S0/"video_ids"/"video_id_manifest.json").read_text(encoding="utf-8"));names={int(v):k for k,v in manifest["mapping"].items()}
    for video in np.unique(data["test"]["scene_id"]).astype(int):
        mask=data["test"]["scene_id"]==video
        row={"video_id":video,"video":names[video],"samples":int(mask.sum())}
        for variant in ("P0","P2_lite","P2_rich","P3_lite","P3_rich"):
            values=[]
            for seed in args.probe_seeds:values.append(metrics(prediction_bank[(variant,seed)][mask],data["test"]["target"][mask],data["test"]["endpoint_scale"][mask])[0]["endpoint_error_px"])
            row[variant+"_endpoint_error_px"]=float(np.mean(values))
        per_video.append(row)
    p0=next(x for x in summary_rows if x["variant"]=="P0");p3=next(x for x in summary_rows if x["variant"]=="P3_rich")
    overall_gain=(p0["endpoint_error_px_mean"]-p3["endpoint_error_px_mean"])/p0["endpoint_error_px_mean"]*100
    wins=sum(x["P3_rich_endpoint_error_px"]<x["P0_endpoint_error_px"] for x in per_video)
    little0=aggregate(result_rows,"P0","little");little3=aggregate(result_rows,"P3_rich","little")
    little_endpoint_gain=(little0["endpoint_error_px_mean"]-little3["endpoint_error_px_mean"])/little0["endpoint_error_px_mean"]*100
    little_angular_gain=(little0["angular_error_deg_mean"]-little3["angular_error_deg_mean"])/little0["angular_error_deg_mean"]*100
    swap_correct=np.mean([x["endpoint_error_px"] for x in swap_rows if x["condition"]=="correct"]);swap_error=np.mean([x["endpoint_error_px"] for x in swap_rows if x["condition"]=="matched_swap"])
    swap_gain=(swap_error-swap_correct)/swap_error*100
    if overall_gain>=5 and wins>=14 and swap_gain>=5 and max(little_endpoint_gain,little_angular_gain)>0:decision="STRONG_PASS"
    elif overall_gain>=3 and wins>=12 and swap_gain>=3 and max(little_endpoint_gain,little_angular_gain)>=-1:decision="RUN_EXP183B"
    elif 1<=overall_gain<3 and swap_gain>=1:decision="WEAK_SIGNAL"
    else:decision="STOP_SCENE_INTENT"
    video_m1,video_n=nearest_centroid_video_accuracy(data["train"]["m1"],data["train"]["scene_id"]);video_rich,_=nearest_centroid_video_accuracy(data["train"]["rich"],data["train"]["scene_id"])
    group_rows=grouped_comparison(data["test"],prediction_bank,args.probe_seeds)
    summary={"experiment":"EXP-183A","dataset":"SDD","model_checkpoint":"EXP-175S0 seed1","smoke":bool(args.smoke),"samples":{k:int(len(v["target"])) for k,v in data.items()},"probe_seeds":[int(x) for x in args.probe_seeds],"primary_target":"endpoint_direction_plus_normalized_radius","variants":summary_rows,"primary_comparison":{"baseline":"P0","candidate":"P3_rich","endpoint_gain_pct":float(overall_gain)},"gates":{"overall_gain_ge_3pct":bool(overall_gain>=3),"positive_videos":int(wins),"positive_videos_ge_12_of_17":bool(wins>=12),"little_endpoint_gain_pct":float(little_endpoint_gain),"little_angular_gain_pct":float(little_angular_gain),"little_not_worse":bool(max(little_endpoint_gain,little_angular_gain)>=-1),"scene_swap_gain_pct":float(swap_gain),"scene_swap_gain_ge_3pct":bool(swap_gain>=3)},"leakage_diagnostic":{"train_internal_nearest_centroid_samples":int(video_n),"m1_video_accuracy":float(video_m1),"rich_video_accuracy":float(video_rich)},"target_reconstruction_max_error":float(reconstruction),"decision":decision}
    little_rows=[aggregate(result_rows,v,"little") for v in VARIANTS]+[aggregate(result_rows,v,"non_little") for v in VARIANTS]
    write_csv(output_dir/"probe_runs.csv",result_rows);write_csv(output_dir/"probe_summary.csv",summary_rows);write_csv(output_dir/"per_video.csv",per_video);write_csv(output_dir/"scene_swap.csv",swap_rows);write_csv(output_dir/"little_analysis.csv",little_rows);write_csv(output_dir/"motion_group_analysis.csv",group_rows)
    (output_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# EXP-183A Scene-Conditioned Intent Probe","",f"Decision: **{decision}**","",f"- P3-rich vs P0 endpoint gain: {overall_gain:.2f}%",f"- Positive test videos: {wins}/17",f"- Little endpoint/angular gain: {little_endpoint_gain:.2f}% / {little_angular_gain:.2f}%",f"- Correct-scene advantage over motion-matched swap: {swap_gain:.2f}%",f"- Train-only video-ID diagnostic: M1={video_m1:.3f}, S-rich={video_rich:.3f}","","Full metrics are stored in `probe_summary.csv`, `per_video.csv`, and `scene_swap.csv`."]
    (output_dir/"RESULT_ANALYSIS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"[done] {output_dir/'summary.json'} decision={decision}",flush=True)


if __name__=="__main__":main()
