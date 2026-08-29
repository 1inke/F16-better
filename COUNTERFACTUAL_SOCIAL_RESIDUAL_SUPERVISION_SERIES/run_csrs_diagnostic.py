"""Strict paired B0/CSRS1 precision and mechanism diagnostic."""

from __future__ import annotations
import argparse,csv,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score,average_precision_score,recall_score,roc_auc_score


ROOT=Path(__file__).resolve().parent; WORKSPACE=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20"); FRONT=WORKSPACE/"FRONT_ENCODER_SERIES"
SOURCE_ROOT=WORKSPACE/"INTERACTION_RESIDUAL_SUPERVISION_SERIES"/"results"; RESULTS=ROOT/"results"; CACHE=ROOT/"teacher_cache"; OUTPUT=ROOT/"diagnostic"
SPLITS=("general","challenge"); MODELS=("baseline","csrs1_actual_gain_residual"); METRICS=("min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde"); STEPS=(4,8,12)


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def run_dir(split,name,seed): return RESULTS/split/name/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def source_checkpoint(split,seed): return SOURCE_ROOT/split/"baseline"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"
def cache_path(split,seed): return CACHE/f"{split}_seed_{seed}.npz"


def prepare(device):
    sys.path[:0]=[str(PROJECT),str(PROJECT/"scripts"),str(FRONT)]; os.chdir(PROJECT); import sitecustomize  # noqa
    from experiment_config import load_locked_q20
    from front_encoder_models import FrontEncoderASTRA
    cfg=load_locked_q20(PROJECT/"configs"/"motif_rbf_univ.yaml"); cfg.MODEL.USE_PRETRAINED_UNET=False; cfg.MODEL.USE_CACHED_UNET_FEATURES=False
    cfg.DATA.MAX_AGENTS=9; cfg.MODEL.GRAPH.TOP_K=4; cfg.MODEL.GRAPH.RADIUS=6.; cfg.MODEL.DECODER.NUM_MODES=20
    cfg.device=str(device); cfg.device_list=[device.index or 0] if device.type=="cuda" else ["cpu"]
    return cfg,FrontEncoderASTRA


def build_model(split,name,seed,cfg,model_class,device):
    source=source_checkpoint(split,seed); checkpoint=run_dir(split,name,seed)/"best_selected.pth"
    if not source.is_file() or not checkpoint.is_file(): raise FileNotFoundError((source,checkpoint))
    os.environ["CSRS_SOURCE_CHECKPOINT"]=str(source); os.environ["CSRS_TEACHER_CACHE"]=str(cache_path(split,seed))
    model=model_class(cfg,"f16_temporal_social_cross")
    if name=="baseline": module=load_module(f"csrs_diag_b0_{split}",ROOT/"current_baseline_adapter.py"); experiment="current_mode_anchor_template"
    else: module=load_module(f"csrs_diag_model_{split}",ROOT/"csrs_models.py"); experiment="csrs1_actual_gain_residual"
    module.replace_decoder(model,cfg,experiment); model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def infer(model,loader,device,is_csrs):
    from train_front_variant import normalize_to_focal_heading
    keys=("index","scene_id","prediction","anchor","target",*METRICS,"winner")
    parts={k:[] for k in keys}; mechanism={k:[] for k in ("need_probability","predicted_residual","predicted_residual_norm","mode_gate_mean","mode_gate_std","applied_delta")}
    for past,future,num_valid,indices,scene_ids in loader:
        past=past.to(device,non_blocking=True); future=future.to(device,non_blocking=True); num_valid=num_valid.to(device).int()
        if is_csrs: model.decoder.set_scene_batch(-(indices.to(device).long()+2),past)
        past,future=normalize_to_focal_heading(past,future); center=past.mean(dim=-2,keepdim=True)
        prediction=model(past-center,None,None,mode="testing",num_valid=num_valid,graph_loc=past)[2]+center.unsqueeze(-3)
        candidates,target=prediction[:,0],future[:,0]; errors=torch.linalg.vector_norm(candidates-target[:,None],dim=-1)
        ade,fde=errors.mean(-1),errors[...,-1]; rows=torch.arange(len(ade),device=device); winner=ade.argmin(-1); top1=model.last_mode_logits[:,0].argmax(-1)
        values={"index":indices.numpy(),"scene_id":scene_ids.numpy(),"prediction":candidates.cpu().numpy(),"anchor":model.decoder.last_corrected_anchors[:,0].cpu().numpy(),
                "target":target.cpu().numpy(),"min_ade_20":ade.min(-1).values.cpu().numpy(),"min_fde_20":fde.min(-1).values.cpu().numpy(),
                "same_path_fde":fde[rows,winner].cpu().numpy(),"top1_ade":ade[rows,top1].cpu().numpy(),"top1_fde":fde[rows,top1].cpu().numpy(),"winner":winner.cpu().numpy()}
        for k,v in values.items(): parts[k].append(v)
        if is_csrs:
            mech={"need_probability":model.decoder.last_need_probability[:,0].cpu().numpy(),
                  "predicted_residual":model.decoder.last_predicted_residual[:,0].cpu().numpy(),
                  "predicted_residual_norm":model.decoder.last_predicted_residual[:,0].flatten(1).norm(dim=-1).cpu().numpy(),
                  "mode_gate_mean":model.decoder.last_mode_gates[:,0].mean(-1).cpu().numpy(),
                  "mode_gate_std":model.decoder.last_mode_gates[:,0].std(-1).cpu().numpy(),
                  "applied_delta":model.decoder.last_applied_anchor_delta[:,0].norm(dim=-1).mean(dim=(-1,-2)).cpu().numpy()}
            for k,v in mech.items(): mechanism[k].append(v)
    result={k:np.concatenate(v) for k,v in parts.items()}
    if is_csrs: result.update({k:np.concatenate(v) for k,v in mechanism.items()})
    return result


def metric_summary(v,mask): return {"samples":int(mask.sum()),**{m:float(v[m][mask].mean()) for m in METRICS}}
def change(base,test): return {m:100*(test[m]/base[m]-1) for m in METRICS}
def cosine(a,b):
    flat_a=a.reshape(len(a),-1); flat_b=b.reshape(len(b),-1); denom=np.linalg.norm(flat_a,axis=1)*np.linalg.norm(flat_b,axis=1)
    out=np.full(len(a),np.nan,dtype=np.float32); valid=denom>1e-8; out[valid]=(flat_a[valid]*flat_b[valid]).sum(1)/denom[valid]; return out
def safe_rho(a,b):
    mask=np.isfinite(a)&np.isfinite(b)
    if mask.sum()<3: return None
    v=spearmanr(a[mask],b[mask]).statistic; return float(v) if np.isfinite(v) else None
def describe(v):
    v=np.asarray(v); v=v[np.isfinite(v)]; return {"mean":float(v.mean()),"median":float(np.median(v)),"p10":float(np.quantile(v,.1)),"p90":float(np.quantile(v,.9)),"samples":len(v)}


def verify(split,name,seed,v):
    raw=json.loads((run_dir(split,name,seed)/"metrics.json").read_text(encoding="utf-8"))["evaluation_best_selected"]
    expected={"min_ade_20":raw["oracle"]["min_ade_20"],"min_fde_20":raw["oracle"]["min_fde_20"],"same_path_fde":raw["oracle"]["same_path_fde"],"top1_ade":raw["original"]["ade"],"top1_fde":raw["original"]["fde"]}
    result={}
    for m,e in expected.items():
        o=float(v[m].mean()); result[m]={"expected":float(e),"observed":o,"difference":o-float(e)}
        if abs(o-float(e))>2e-5: raise RuntimeError(f"reproduction failed {split}/{name}/{m}")
    return result


def write_csv(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf-8-sig") as h: w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyse(split,b0,csrs,cache,indices,output):
    labels=np.asarray(cache["need_label"])[indices].astype(int); gain=np.asarray(cache["social_gain"])[indices]
    target_res=np.asarray(cache["residual_target"])[indices]; probability=csrs["need_probability"]; pred_res=csrs["predicted_residual"]
    positive=labels==1; negative=~positive; predicted=probability>=.5
    classification={"positive_rate":float(positive.mean()),"auroc":float(roc_auc_score(labels,probability)),"auprc":float(average_precision_score(labels,probability)),
                    "accuracy":float(accuracy_score(labels,predicted)),"positive_recall":float(recall_score(labels,predicted,pos_label=1)),"negative_recall":float(recall_score(labels,predicted,pos_label=0)),
                    "spearman_probability_vs_gain":safe_rho(probability,gain)}
    cos=cosine(pred_res,target_res); delta_ade=csrs["min_ade_20"]-b0["min_ade_20"]; delta_fde=csrs["min_fde_20"]-b0["min_fde_20"]
    residual={"positive":describe(cos[positive]),"non_needed":describe(cos[negative]),
              "alignment_vs_csrs_delta_minade_positive":safe_rho(cos[positive],delta_ade[positive]),
              "alignment_vs_csrs_delta_minfde_positive":safe_rho(cos[positive],delta_fde[positive])}
    overall_b,overall_c=metric_summary(b0,np.ones(len(indices),bool)),metric_summary(csrs,np.ones(len(indices),bool))
    pos_b,pos_c=metric_summary(b0,positive),metric_summary(csrs,positive)
    target_anchor=b0["target"][:,np.asarray(STEPS)-1]
    coverage={}
    for label,v in (("baseline",b0),("csrs",csrs)):
        err=np.linalg.norm(v["anchor"]-target_anchor[:,None],axis=-1)
        coverage[label]={f"step_{step}":float(err[:,:,i].min(1).mean()) for i,step in enumerate(STEPS)}
        coverage[label]["endpoint"]=float(np.linalg.norm(v["prediction"][:,:,-1]-v["target"][:,None,-1],axis=-1).min(1).mean())
    coverage["change_pct"]={k:100*(coverage["csrs"][k]/coverage["baseline"][k]-1) for k in coverage["baseline"]}
    scene_level={}
    for scene in np.unique(b0["scene_id"]):
        mask=b0["scene_id"]==scene; pos=mask&positive
        scene_level[str(int(scene))]={"samples":int(mask.sum()),"positive_rate":float(positive[mask].mean()),
                                      "auroc":float(roc_auc_score(labels[mask],probability[mask])) if len(np.unique(labels[mask]))==2 else None,
                                      "positive_residual_cosine":describe(cos[pos]) if pos.any() else None,
                                      "metric_change_pct":change(metric_summary(b0,mask),metric_summary(csrs,mask))}
    rows=[]
    for i,index in enumerate(indices):
        row={"index":int(index),"scene_id":int(b0["scene_id"][i]),"need_label":int(labels[i]),"actual_social_gain":float(gain[i]),
             "need_probability":float(probability[i]),"residual_cosine":float(cos[i]),"mode_gate_mean":float(csrs["mode_gate_mean"][i]),"applied_delta":float(csrs["applied_delta"][i])}
        for m in METRICS: row[f"b0_{m}"]=float(b0[m][i]);row[f"csrs_{m}"]=float(csrs[m][i]);row[f"delta_{m}"]=float(csrs[m][i]-b0[m][i])
        rows.append(row)
    write_csv(output/"paired_samples.csv",rows)
    summary={"split":split,"samples":len(indices),"need_classification":classification,"residual_direction":residual,
             "overall":{"baseline":overall_b,"csrs":overall_c,"change_pct":change(overall_b,overall_c)},
             "actual_social_needed":{"baseline":pos_b,"csrs":pos_c,"change_pct":change(pos_b,pos_c)},"anchor_coverage":coverage,
             "activation":{"need_probability":describe(probability),"predicted_residual_norm":describe(csrs["predicted_residual_norm"]),
                           "mode_gate_mean":describe(csrs["mode_gate_mean"]),"mode_gate_std":describe(csrs["mode_gate_std"]),
                           "applied_anchor_delta":describe(csrs["applied_delta"])},"scene_level":scene_level}
    (output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"); return summary


def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=128);p.add_argument("--seed",type=int,default=1)
    p.add_argument("--splits",nargs="+",choices=("all",*SPLITS),default=["all"]);p.add_argument("--max-samples",type=int,default=None);a=p.parse_args()
    device=torch.device(a.device);splits=list(SPLITS) if "all" in a.splits else a.splits;OUTPUT.mkdir(parents=True,exist_ok=True)
    cfg,model_class=prepare(device);geometry=load_module("csrs_diag_geometry",WORKSPACE/"INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES"/"run_geometry_diagnostic.py")
    summaries=[]; reproduction={}
    for split in splits:
        run=run_dir(split,"baseline",a.seed); split_arrays=np.load(run/"split_indices.npz");indices=np.asarray(split_arrays["validation"],dtype=np.int64)
        if a.max_samples is not None: indices=indices[:a.max_samples]
        info=json.loads((run/"split.json").read_text(encoding="utf-8"));dataset=geometry.IndexedDataset(Path(info["data_path"]),Path(info["num_path"]),indices,cfg)
        loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda")
        values={};reproduction[split]={}
        for name in MODELS:
            print(f"[infer] {split}/{name} samples={len(dataset)}",flush=True);model=build_model(split,name,a.seed,cfg,model_class,device);values[name]=infer(model,loader,device,name!=MODELS[0])
            if a.max_samples is None: reproduction[split][name]=verify(split,name,a.seed,values[name])
            del model
            if device.type=="cuda":torch.cuda.empty_cache()
        cache=np.load(cache_path(split,a.seed),allow_pickle=False);out=OUTPUT/split;out.mkdir(parents=True,exist_ok=True)
        summaries.append(analyse(split,values[MODELS[0]],values[MODELS[1]],cache,indices,out))
    combined={"strict_pairing":True,"reproduction":reproduction,"results":summaries};(OUTPUT/"combined_summary.json").write_text(json.dumps(combined,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"[done] {OUTPUT}",flush=True)


if __name__=="__main__":main()
