"""EXP-162 nested scene-held-out GT-only verifier for frozen V2 candidates."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
BASE_SCRIPT = WORKSPACE / "CANDIDATE_COMPLEMENTARITY_SERIES" / "run_exp158_exp160_joint_audit.py"
VERIFIER_SCRIPT = WORKSPACE / "train_social_stimulus_verifier.py"
V2_CACHE = WORKSPACE / "SCORE_SYNCHRONIZATION_SERIES" / "EXP_161_FROZEN_CANDIDATE_LISTWISE_SCORER" / "cache"
OUT_A = HERE / "EXP_162A_V2_GT_ONLY_VERIFIER"
OUT_B = HERE / "EXP_162B_V2_VERIFIER_FUSION"
FOLDS = ("eth", "hotel", "univ", "zara01", "zara02")
SCENE_ID = {"eth": 0, "hotel": 1, "univ": 2, "zara01": 3, "zara02": 4}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class HeadingDataset(Dataset):
    def __init__(self, dataset, normalize): self.dataset, self.normalize = dataset, normalize
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index):
        item = self.dataset[index]; past, future, num_valid = item[0], item[1], item[2]
        past, future = self.normalize(past.unsqueeze(0), future.unsqueeze(0))
        return past[0], future[0], num_valid


def parse_floats(values):
    return tuple(float(x) for value in values for x in value.split(",") if x.strip())


def zscore(x):
    return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-6)


def ranks(x):
    order=np.argsort(x,axis=1,kind="stable"); result=np.empty_like(order,dtype=np.float64)
    result[np.arange(len(x))[:,None],order]=np.arange(x.shape[1])[None]; return result


def metrics(candidates, future, scores):
    distance=np.linalg.norm(candidates-future[:,None],axis=-1); ade=distance.mean(-1); fde=distance[...,-1]
    cost=ade+.25*fde; rows=np.arange(len(ade)); top=scores.argmax(1); best=cost.argmin(1); best_fde=fde.argmin(1)
    rx,ry=ranks(scores),ranks(-cost);rx-=rx.mean(1,keepdims=True);ry-=ry.mean(1,keepdims=True)
    rho=(rx*ry).sum(1)/np.sqrt((rx*rx).sum(1)*(ry*ry).sum(1)).clip(1e-12)
    shifted=scores-scores.max(1,keepdims=True);p=np.exp(shifted);p/=p.sum(1,keepdims=True)
    regret=cost[rows,top]-cost[rows,best]; fde_regret=fde[rows,top]-fde[rows,best_fde]
    order=np.argsort(-scores,axis=1); recovery={}
    for n in (1,3,5,10): recovery[f"hit@{n}"]=float((order[:,:n]==best[:,None]).any(1).mean())
    hist=np.bincount(top,minlength=20).astype(int).tolist(); best_ade=ade.argmin(1)
    return {"samples":len(ade),"top1_ADE":float(ade[rows,top].mean()),"top1_FDE":float(fde[rows,top].mean()),
            "top1_combined_cost":float(cost[rows,top].mean()),"spearman":float(rho.mean()),
            "mean_regret":float(regret.mean()),"p75_regret":float(np.quantile(regret,.75)),
            "p90_regret":float(np.quantile(regret,.90)),"p95_regret":float(np.quantile(regret,.95)),
            "mean_FDE_regret":float(fde_regret.mean()),"score_entropy":float(-(p*np.log(p.clip(1e-12))).sum(1).mean()),
            "active_modes":int(np.unique(top).size),"top1_mode_histogram":hist,"recovery":recovery,
            "oracle":{"minADE":float(ade.min(1).mean()),"minFDE":float(fde.min(1).mean()),
                      "same_path_FDE":float(fde[rows,best_ade].mean())}}


def macro(rows, method):
    selected=[r[method] for r in rows]; numeric=("top1_ADE","top1_FDE","top1_combined_cost","spearman","mean_regret",
                                                 "p75_regret","p90_regret","p95_regret","mean_FDE_regret","score_entropy","active_modes")
    result={k:float(np.mean([x[k] for x in selected])) for k in numeric}
    result["oracle"]={k:float(np.mean([x["oracle"][k] for x in selected])) for k in selected[0]["oracle"]}
    result["recovery"]={k:float(np.mean([x["recovery"][k] for x in selected])) for k in selected[0]["recovery"]}
    return result


def scene_indices(scene_ids, scene, maximum, seed):
    ids=np.flatnonzero(scene_ids==scene).astype(np.int64)
    if maximum and len(ids)>maximum:
        ids=np.sort(np.random.default_rng(seed+scene).choice(ids,maximum,replace=False))
    return ids


def load_v2_cache(fold, split, seed):
    path=v2_cache_path(fold,split,seed)
    if not path.is_file(): raise FileNotFoundError(f"Frozen V2 cache missing: {path}. Run EXP-161 cache collection first.")
    with np.load(path) as raw:
        return {k:raw[k] for k in ("future","candidates","original_logits")}


def v2_cache_path(fold,split,seed):
    return V2_CACHE/f"{fold}_v2_{split}_seed_{seed}.npz"


def make_dataset(base, helper, cfg, fold, split):
    raw_path=base.DATA/f"{fold}_data_{split}.npy"; num_path=base.DATA/f"{fold}_num_{split}.npy"
    raw=np.load(raw_path,mmap_mode="r"); scenes=np.asarray(raw[:,0,0,0],dtype=np.int64)
    dataset=helper.TrajectoryDataset(raw_path,num_path,cfg)
    if len(dataset)!=len(scenes): raise RuntimeError(f"Dataset/raw length mismatch {fold}/{split}")
    return HeadingDataset(dataset,helper.normalize_to_focal_heading),scenes


def verify_alignment(dataset, cache):
    for index in np.linspace(0,len(dataset)-1,min(5,len(dataset)),dtype=int):
        _,future,_=dataset[int(index)]
        delta=float(np.max(np.abs(future[0].numpy()-cache["future"][index])))
        if delta>2e-4: raise RuntimeError(f"Frozen candidate cache alignment failed at sample {index}: {delta}")


def loaders(dataset, train_ids, val_ids, batch_size, seed):
    generator=torch.Generator().manual_seed(seed)
    train=DataLoader(Subset(dataset,train_ids.tolist()),batch_size=batch_size,shuffle=True,num_workers=0,
                     pin_memory=True,drop_last=False,generator=generator)
    val=DataLoader(Subset(dataset,val_ids.tolist()),batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True)
    return train,val


def build_model(vm, dataset, cfg, min_scale, device):
    pred_len=int(dataset[0][1].shape[-2])
    config=vm.VerifierConfig(pred_len=pred_len,frequency=float(cfg.DATA.FREQUENCY),num_components=8,
                             self_hidden=96,social_hidden=64,decoder_hidden=192,min_scale=min_scale,dropout=.10)
    return vm.SocialStimulusVerifier(config).to(device),config


def train_inner(vm,dataset,train_ids,val_ids,cfg,min_scale,device,seed,epochs,patience,batch_size,path,contract,force):
    if path.is_file() and not force:
        saved=torch.load(path,map_location="cpu",weights_only=False)
        if saved.get("contract")==contract: return saved
    seed_all(seed);model,config=build_model(vm,dataset,cfg,min_scale,device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=4,min_lr=1e-6)
    train_loader,val_loader=loaders(dataset,train_ids,val_ids,batch_size,seed);best=None;best_nll=math.inf;stale=0;history=[]
    for epoch in range(1,epochs+1):
        tr=vm.run_epoch(model,train_loader,device,optimizer,5.,False)
        with torch.inference_mode(): va=vm.run_epoch(model,val_loader,device,None,5.,False)
        scheduler.step(va["nll"]);history.append({"epoch":epoch,"train":tr,"validation":va})
        print(f"[inner] epoch={epoch} train_nll={tr['nll']:.4f} val_nll={va['nll']:.4f}",flush=True)
        if va["nll"]<best_nll-1e-5:
            best_nll=va["nll"];stale=0;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch
        else:
            stale+=1
            if stale>=patience: break
    saved={"model_config":asdict(config),"model_state":best,"best_epoch":best_epoch,"best_validation_nll":best_nll,
           "history":history,"contract":contract,"train_samples":len(train_ids),"validation_samples":len(val_ids)}
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(saved,path);return saved


def train_final(vm,dataset,train_ids,cfg,min_scale,device,seed,epochs,batch_size,path,contract,force):
    if path.is_file() and not force:
        saved=torch.load(path,map_location="cpu",weights_only=False)
        if saved.get("contract")==contract:return saved
    seed_all(seed);model,config=build_model(vm,dataset,cfg,min_scale,device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    loader=DataLoader(Subset(dataset,train_ids.tolist()),batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True,
                      generator=torch.Generator().manual_seed(seed));history=[]
    for epoch in range(1,epochs+1):
        tr=vm.run_epoch(model,loader,device,optimizer,5.,False);history.append({"epoch":epoch,"train":tr})
        print(f"[final] epoch={epoch}/{epochs} nll={tr['nll']:.4f}",flush=True)
    saved={"model_config":asdict(config),"model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
           "epochs":epochs,"history":history,"contract":contract,"train_samples":len(train_ids)}
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(saved,path);return saved


def restore(vm,saved,device):
    model=vm.SocialStimulusVerifier(vm.VerifierConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["model_state"]);return model.eval()


def verifier_scores(vm,model,dataset,indices,cache,device,batch_size):
    scores=[]
    with torch.inference_mode():
        for start in range(0,len(indices),batch_size):
            ids=indices[start:start+batch_size]; batch=[dataset[int(i)] for i in ids]
            past=torch.stack([x[0] for x in batch]).to(device);num=torch.as_tensor([int(x[2]) for x in batch],device=device)
            candidates=torch.from_numpy(cache["candidates"][ids]).to(device)
            distribution=model.distribution(past,num);centered=candidates-distribution["origin"][:,None,None]
            local=vm.rotate_to_heading(centered,distribution["heading"])
            scores.append(model.score_candidates(distribution,local).cpu().numpy())
    return np.concatenate(scores)


def evaluate_indices(cache,indices,verifier,alpha):
    cand=cache["candidates"][indices];future=cache["future"][indices];orig=cache["original_logits"][indices]
    fused=(1-alpha)*zscore(orig)+alpha*zscore(verifier)
    return {"original":metrics(cand,future,orig),"verifier_only":metrics(cand,future,verifier),"fusion":metrics(cand,future,fused)}


def choose_inner(inner_by_scale,scales,alphas):
    candidates=[]
    for scale in scales:
        folds=inner_by_scale[scale]; original=macro([x["evaluation"] for x in folds],"original")
        for alpha in alphas:
            pseudo=[]
            for x in folds:
                e=x["evaluation_by_alpha"][f"{alpha:.2f}"];pseudo.append({"fusion":e})
            fused=macro(pseudo,"fusion")
            ratios=(fused["top1_ADE"]/original["top1_ADE"],fused["top1_FDE"]/original["top1_FDE"],
                    fused["mean_regret"]/max(original["mean_regret"],1e-12))
            safe=(ratios[0]<=1 and ratios[1]<=1 and fused["p95_regret"]<=1.10*original["p95_regret"])
            candidates.append({"min_scale":scale,"alpha":alpha,"safe":safe,"objective":float(np.mean(ratios)),
                               "macro_original":original,"macro_fusion":fused})
    pool=[x for x in candidates if x["safe"]] or candidates
    return min(pool,key=lambda x:(x["objective"],x["alpha"],x["min_scale"])),candidates


def experiment_decision(rows,method):
    original=macro(rows,"original");candidate=macro(rows,method)
    imp={"ADE":1-candidate["top1_ADE"]/original["top1_ADE"],"FDE":1-candidate["top1_FDE"]/original["top1_FDE"],
         "mean_regret":1-candidate["mean_regret"]/original["mean_regret"]}
    ade_non=sum(r[method]["top1_ADE"]<=r["original"]["top1_ADE"] for r in rows)
    fde_non=sum(r[method]["top1_FDE"]<=r["original"]["top1_FDE"] for r in rows)
    max_ade=max(r[method]["top1_ADE"]/r["original"]["top1_ADE"]-1 for r in rows)
    max_fde=max(r[method]["top1_FDE"]/r["original"]["top1_FDE"]-1 for r in rows)
    max_p95=max(r[method]["p95_regret"]/max(r["original"]["p95_regret"],1e-12)-1 for r in rows)
    hit_up=candidate["recovery"]["hit@1"]>original["recovery"]["hit@1"];regret_up=candidate["mean_regret"]>original["mean_regret"]
    passed=(imp["ADE"]>=.03 and imp["FDE"]>=.03 and imp["mean_regret"]>=.10 and ade_non>=4 and fde_non>=4
            and max_ade<=.05 and max_fde<=.05 and max_p95<=.10 and candidate["spearman"]>=original["spearman"]
            and not(hit_up and regret_up))
    return {"decision":"PASSED" if passed else "REJECTED","macro_original":original,"macro_candidate":candidate,
            "relative_improvement":imp,"ADE_non_regression_folds":ade_non,"FDE_non_regression_folds":fde_non,
            "max_fold_ADE_regression":float(max_ade),"max_fold_FDE_regression":float(max_fde),
            "max_fold_p95_regret_regression":float(max_p95),"winner_hit_up_but_regret_up":bool(hit_up and regret_up)}


def write_outputs(path,experiment,rows,decision,method,protocol):
    path.mkdir(parents=True,exist_ok=True);summary={"experiment":experiment,"method":method,**decision,"protocol":protocol,"per_fold":rows}
    (path/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    flat=[]
    for r in rows:
        x=r[method];o=r["original"];flat.append({"fold":r["fold"],"selected_min_scale":r["selected_min_scale"],
            "selected_alpha":r["selected_alpha"],"original_ADE":o["top1_ADE"],"original_FDE":o["top1_FDE"],
            f"{method}_ADE":x["top1_ADE"],f"{method}_FDE":x["top1_FDE"],"original_regret":o["mean_regret"],
            f"{method}_regret":x["mean_regret"],"original_spearman":o["spearman"],f"{method}_spearman":x["spearman"]})
    with (path/"per_fold_summary.csv").open("w",newline="",encoding="utf-8-sig") as h:
        w=csv.DictWriter(h,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    imp=decision["relative_improvement"]
    text=(f"# {experiment} Result Analysis\n\n## Decision: {decision['decision']}\n\n"
          f"Macro Top-1 ADE/FDE improvement: {100*imp['ADE']:.2f}%/{100*imp['FDE']:.2f}%.  \n"
          f"Macro mean-regret improvement: {100*imp['mean_regret']:.2f}%.  \n"
          f"ADE/FDE non-regression folds: {decision['ADE_non_regression_folds']}/5, {decision['FDE_non_regression_folds']}/5.\n")
    (path/"RESULT_ANALYSIS.md").write_text(text,encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--folds",nargs="+",default=["all"],choices=("all",*FOLDS))
    p.add_argument("--seed",type=int,default=1);p.add_argument("--min-scales",nargs="+",default=["0.03","0.05","0.08"])
    p.add_argument("--alphas",nargs="+",default=["0.0","0.2","0.4","0.6","0.8","1.0"])
    p.add_argument("--inner-epochs",type=int,default=60);p.add_argument("--patience",type=int,default=12)
    p.add_argument("--train-batch-size",type=int,default=128);p.add_argument("--score-batch-size",type=int,default=128)
    p.add_argument("--output-root",type=Path,default=HERE)
    p.add_argument("--max-samples-per-scene",type=int,default=0);p.add_argument("--force",action="store_true");p.add_argument("--dry-run",action="store_true")
    a=p.parse_args();folds=FOLDS if "all" in a.folds else tuple(a.folds);scales=parse_floats(a.min_scales);alphas=parse_floats(a.alphas)
    base=load_module("exp162_base",BASE_SCRIPT);base.preflight(folds,a.seed)
    for fold in folds:
        for split in ("train","test"):
            path=v2_cache_path(fold,split,a.seed)
            if not path.is_file():raise FileNotFoundError(f"Frozen V2 cache missing: {path}")
    if a.dry_run:
        print(json.dumps({"status":"dry-run passed","folds":folds,"min_scales":scales,"alphas":alphas,
                          "outer_test_used_for_tuning":False,"candidate_geometry_frozen":True}));return
    if a.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    device=torch.device(a.device);cfg=base.configure(device);helper=load_module("exp162_helper",base.TRAINER);vm=load_module("exp162_verifier",VERIFIER_SCRIPT)
    out_a=a.output_root/OUT_A.name;out_b=a.output_root/OUT_B.name
    rows=[];inner_root=out_a/"inner_cv";checkpoint_root=out_a/"final_verifiers"
    for outer_index,fold in enumerate(folds):
        print(f"[outer] {fold}",flush=True);train_ds,train_scene=make_dataset(base,helper,cfg,fold,"train")
        train_cache=load_v2_cache(fold,"train",a.seed);verify_alignment(train_ds,train_cache)
        outer_id=SCENE_ID[fold];source=tuple(sorted(set(train_scene.tolist())-{outer_id}))
        expected=tuple(sorted(set(SCENE_ID.values())-{outer_id}))
        if source!=expected:raise RuntimeError(f"Nested protocol violation {fold}: source scenes={source}, expected={expected}")
        source_ids={s:scene_indices(train_scene,s,a.max_samples_per_scene,a.seed+outer_index*100) for s in source}
        inner_by_scale={scale:[] for scale in scales}
        for scale in scales:
            for val_scene in source:
                train_ids=np.concatenate([source_ids[s] for s in source if s!=val_scene]);val_ids=source_ids[val_scene]
                tag=f"{scale:.2f}".replace(".","p");path=inner_root/fold/f"scale_{tag}"/f"val_scene_{val_scene}"/"verifier_best.pth"
                contract={"fold":fold,"outer_scene":outer_id,"val_scene":val_scene,"min_scale":scale,"seed":a.seed,
                          "epochs":a.inner_epochs,"patience":a.patience,"max_samples_per_scene":a.max_samples_per_scene}
                saved=train_inner(vm,train_ds,train_ids,val_ids,cfg,scale,device,a.seed+outer_index*10+val_scene,a.inner_epochs,
                                  a.patience,a.train_batch_size,path,contract,a.force);model=restore(vm,saved,device)
                vs=verifier_scores(vm,model,train_ds,val_ids,train_cache,device,a.score_batch_size);orig=train_cache["original_logits"][val_ids]
                eval_orig=metrics(train_cache["candidates"][val_ids],train_cache["future"][val_ids],orig)
                eval_ver=metrics(train_cache["candidates"][val_ids],train_cache["future"][val_ids],vs);by_alpha={}
                for alpha in alphas:
                    fused=(1-alpha)*zscore(orig)+alpha*zscore(vs)
                    by_alpha[f"{alpha:.2f}"]=metrics(train_cache["candidates"][val_ids],train_cache["future"][val_ids],fused)
                inner_by_scale[scale].append({"val_scene":val_scene,"best_epoch":saved["best_epoch"],
                                              "evaluation":{"original":eval_orig,"verifier_only":eval_ver},"evaluation_by_alpha":by_alpha})
                del model
                if device.type=="cuda":torch.cuda.empty_cache()
        selected,search=choose_inner(inner_by_scale,scales,alphas);scale=float(selected["min_scale"]);alpha=float(selected["alpha"])
        epochs=max(1,int(round(np.median([x["best_epoch"] for x in inner_by_scale[scale]]))))
        final_ids=np.concatenate([source_ids[s] for s in source]);final_path=checkpoint_root/fold/"verifier_final.pth"
        final_contract={"fold":fold,"outer_scene":outer_id,"source_scenes":source,"min_scale":scale,"alpha":alpha,"epochs":epochs,
                        "seed":a.seed,"max_samples_per_scene":a.max_samples_per_scene}
        saved=train_final(vm,train_ds,final_ids,cfg,scale,device,a.seed+1000+outer_index,epochs,a.train_batch_size,final_path,final_contract,a.force)
        # The outer test dataset and frozen candidate cache are first opened
        # here, after all hyperparameters and final training are locked.
        test_ds,test_scene=make_dataset(base,helper,cfg,fold,"test");test_cache=load_v2_cache(fold,"test",a.seed)
        verify_alignment(test_ds,test_cache)
        outer_test_ids=scene_indices(test_scene,outer_id,a.max_samples_per_scene,a.seed+outer_index*1000)
        model=restore(vm,saved,device);vs=verifier_scores(vm,model,test_ds,outer_test_ids,test_cache,device,a.score_batch_size)
        evaluation=evaluate_indices(test_cache,outer_test_ids,vs,alpha);oracle_values=[evaluation[x]["oracle"] for x in evaluation]
        if any(any(abs(v[k]-oracle_values[0][k])>1e-7 for k in v) for v in oracle_values[1:]):raise RuntimeError("Oracle changed across scores")
        row={"fold":fold,"outer_scene":outer_id,"source_scenes":source,"selected_min_scale":scale,"selected_alpha":alpha,
             "selected_final_epochs":epochs,"inner_selected_objective":selected["objective"],**evaluation};rows.append(row)
        out_fold=out_a/"outer_fold_results";out_fold.mkdir(parents=True,exist_ok=True)
        (out_fold/f"{fold}_seed_{a.seed}.json").write_text(json.dumps({"result":row,"inner_selected":selected,"inner_search":search},indent=2,ensure_ascii=False),encoding="utf-8")
        del model
        if device.type=="cuda":torch.cuda.empty_cache()
    protocol={"seed":a.seed,"nested_scene_held_out":True,"outer_test_used_for_tuning":False,"candidate_family":"V2",
              "candidate_K":20,"candidate_geometry_frozen":True,"winner_labels_used_for_training":False,
              "min_scales":scales,"alpha_grid":alphas,"max_samples_per_scene":a.max_samples_per_scene}
    if set(folds)==set(FOLDS) and not a.max_samples_per_scene:
        decision_a=experiment_decision(rows,"verifier_only");decision_b=experiment_decision(rows,"fusion")
    else:
        decision_a=experiment_decision(rows,"verifier_only");decision_b=experiment_decision(rows,"fusion")
        decision_a["decision"]=decision_b["decision"]="CONDITIONAL"
    write_outputs(out_a,"EXP-162A V2 GT-Only Verifier",rows,decision_a,"verifier_only",protocol)
    write_outputs(out_b,"EXP-162B V2 Verifier Fusion",rows,decision_b,"fusion",protocol)
    print(json.dumps({"EXP_162A":decision_a["decision"],"EXP_162B":decision_b["decision"],"outputs":[str(out_a),str(out_b)]},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
