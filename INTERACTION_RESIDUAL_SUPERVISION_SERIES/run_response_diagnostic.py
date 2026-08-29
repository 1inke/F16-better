"""Strict paired B0/IR1 response-semantics and candidate-geometry diagnostic."""

from __future__ import annotations
import argparse, csv, importlib.util, json, os
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parent; WORKSPACE=ROOT.parent
PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
SOURCE_ROOT=WORKSPACE/"MODE_TEMPLATE_GROUP_SERIES"/"EXP_146B_ANCHORED_MODE_TEMPLATE"/"results"
RESULTS=ROOT/"results"; OUTPUT=ROOT/"response_diagnostic"
SPLITS=("general","challenge"); MODELS=("baseline","ir1_future_response_supervision")
METRICS=("min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde"); ANCHOR_STEPS=(4,8,12)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

BASE=load("irs_geometry_helpers",WORKSPACE/"INTERACTION_CONDITIONED_MODE_ANCHOR_BRIDGE_SERIES"/"run_geometry_diagnostic.py")

def run_dir(split,name,seed): return RESULTS/split/name/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"

def build(split,name,seed,cfg,model_class,device):
    source=SOURCE_ROOT/split/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"
    checkpoint=run_dir(split,name,seed)/"best_selected.pth"
    if not source.is_file() or not checkpoint.is_file(): raise FileNotFoundError(f"{source} / {checkpoint}")
    os.environ["IRS_SOURCE_CHECKPOINT"]=str(source); model=model_class(cfg,"f16_temporal_social_cross")
    module=load(f"irs_diag_{split}_{name}",ROOT/("irs_models.py" if name!= "baseline" else "current_baseline_adapter.py"))
    module.replace_decoder(model,cfg,"ir1_future_response_supervision" if name!="baseline" else "current_mode_anchor_template")
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
    return model.to(device).eval()

@torch.inference_mode()
def infer(model,loader,device,is_ir1):
    from train_front_variant import normalize_to_focal_heading
    keys=("index","scene_id","prediction","anchors","target","min_ade_20","min_fde_20","same_path_fde","top1_ade","top1_fde","winner")
    parts={k:[] for k in keys}; extra={k:[] for k in ("predicted_response","inertia_anchors","interaction_weight","gate_mean","gate_std","semantic_delta_std")}
    for past,future,num_valid,indices,scene_ids in loader:
        past=past.to(device,non_blocking=True); future=future.to(device,non_blocking=True); num_valid=num_valid.to(device,non_blocking=True).int()
        past,future=normalize_to_focal_heading(past,future); center=past.mean(dim=-2,keepdim=True)
        prediction=model(past-center,None,None,mode="testing",num_valid=num_valid,graph_loc=past)[2]+center.unsqueeze(-3)
        candidates=prediction[:,0]; target=future[:,0]; anchors=model.decoder.last_corrected_anchors[:,0]
        errors=torch.linalg.vector_norm(candidates-target[:,None],dim=-1); ade=errors.mean(dim=-1); fde=errors[...,-1]
        rows=torch.arange(len(ade),device=device); winner=ade.argmin(-1); top1=model.last_mode_logits[:,0].argmax(-1)
        vals={"index":indices.numpy(),"scene_id":scene_ids.numpy(),"prediction":candidates.cpu().numpy(),"anchors":anchors.cpu().numpy(),"target":target.cpu().numpy(),
              "min_ade_20":ade.min(-1).values.cpu().numpy(),"min_fde_20":fde.min(-1).values.cpu().numpy(),"same_path_fde":fde[rows,winner].cpu().numpy(),
              "top1_ade":ade[rows,top1].cpu().numpy(),"top1_fde":fde[rows,top1].cpu().numpy(),"winner":winner.cpu().numpy()}
        for k,v in vals.items(): parts[k].append(v)
        if is_ir1:
            gates=model.decoder.last_mode_gates[:,0]; delta=model.decoder.last_semantic_mode_delta[:,0].norm(dim=-1)
            vals={"predicted_response":model.decoder.last_predicted_response[:,0].detach().cpu().numpy(),
                  "inertia_anchors":model.decoder.last_inertia_anchors[:,0].cpu().numpy(),"interaction_weight":model.decoder.last_interaction_weight[:,0].cpu().numpy(),
                  "gate_mean":gates.mean(-1).cpu().numpy(),"gate_std":gates.std(-1).cpu().numpy(),"semantic_delta_std":delta.std(-1).cpu().numpy()}
            for k,v in vals.items(): extra[k].append(v)
    out={k:np.concatenate(v) for k,v in parts.items()}
    if is_ir1: out.update({k:np.concatenate(v) for k,v in extra.items()})
    return out

def verify(split,name,v,seed):
    raw=json.loads((run_dir(split,name,seed)/"metrics.json").read_text(encoding="utf-8"))["evaluation_best_selected"]
    exp={"min_ade_20":raw["oracle"]["min_ade_20"],"min_fde_20":raw["oracle"]["min_fde_20"],"same_path_fde":raw["oracle"]["same_path_fde"],"top1_ade":raw["original"]["ade"],"top1_fde":raw["original"]["fde"]}
    out={}
    for metric in METRICS:
        observed=float(v[metric].mean()); difference=observed-float(exp[metric]); out[metric]={"expected":float(exp[metric]),"observed":observed,"absolute_difference":difference}
        if abs(difference)>2e-5: raise RuntimeError(f"reproduction failed {split}/{name}/{metric}")
    return out

def cosine(pred,target):
    dot=(pred*target).sum(-1); denom=np.linalg.norm(pred,axis=-1)*np.linalg.norm(target,axis=-1)
    valid=(np.linalg.norm(target,axis=-1)>0.05)&(np.linalg.norm(pred,axis=-1)>1e-6)
    out=np.full(dot.shape,np.nan,dtype=np.float32); out[valid]=dot[valid]/np.maximum(denom[valid],1e-8); return out,valid

def metric_group(b0,ir,mask):
    out={"samples":int(mask.sum())}
    for metric in METRICS:
        a=float(b0[metric][mask].mean()); b=float(ir[metric][mask].mean()); out[metric]={"baseline":a,"ir1":b,"change_pct":BASE.relative_change(a,b)}
    return out

def analyse(split,b0,ir,potential,train_potential,outdir):
    response_target=ir["target"][:,np.asarray(ANCHOR_STEPS)-1]-ir["inertia_anchors"]
    response_error=ir["predicted_response"]-response_target
    cos,valid=cosine(ir["predicted_response"],response_target)
    flat_cos,flat_valid=cosine(ir["predicted_response"].reshape(len(cos),-1),response_target.reshape(len(cos),-1))
    response={}
    for a,step in enumerate(ANCHOR_STEPS):
        response[f"step_{step}"]={"mae":float(np.abs(response_error[:,:,][...,a,:]).mean()),"l2":float(np.linalg.norm(response_error[:,a],axis=-1).mean()),
                                  "cosine_mean":float(np.nanmean(cos[:,a])),"cosine_median":float(np.nanmedian(cos[:,a])),"cosine_valid_samples":int(valid[:,a].sum())}
    response["six_dimensional"]={"cosine_mean":float(np.nanmean(flat_cos)),"cosine_median":float(np.nanmedian(flat_cos)),"cosine_valid_samples":int(flat_valid.sum())}
    q=np.quantile(ir["interaction_weight"],[1/3,2/3]); groups=[]
    masks=(("low",ir["interaction_weight"]<=q[0]),("medium",(ir["interaction_weight"]>q[0])&(ir["interaction_weight"]<=q[1])),("high",ir["interaction_weight"]>q[1]))
    for label,mask in masks:
        groups.append({"group":label,"samples":int(mask.sum()),"weight_min":float(ir["interaction_weight"][mask].min()),"weight_max":float(ir["interaction_weight"][mask].max()),
                       "cosine_6d_mean":float(np.nanmean(flat_cos[mask])),"mae":float(np.abs(response_error[mask]).mean()),"l2":float(np.linalg.norm(response_error[mask],axis=-1).mean()),"prediction":metric_group(b0,ir,mask)})
    correlations={}
    for metric in ("min_ade_20","min_fde_20","same_path_fde"):
        delta=ir[metric]-b0[metric]; use=np.isfinite(flat_cos)
        correlations[metric]={"spearman_alignment_vs_delta":float(spearmanr(flat_cos[use],delta[use]).statistic),"expected_sign":"negative"}
    shift=np.linalg.norm(ir["anchors"]-b0["anchors"],axis=-1)
    diversity_b=BASE.pairwise_diversity(b0["prediction"],b0["anchors"]); diversity_i=BASE.pairwise_diversity(ir["prediction"],ir["anchors"])
    target_anchors=ir["target"][:,np.asarray(ANCHOR_STEPS)-1]; coverage={}
    for a,step in enumerate(ANCHOR_STEPS):
        eb=np.linalg.norm(b0["anchors"][:,:,a]-target_anchors[:,None,a],axis=-1).min(1); ei=np.linalg.norm(ir["anchors"][:,:,a]-target_anchors[:,None,a],axis=-1).min(1)
        coverage[f"step_{step}"]={"baseline":BASE.describe(eb),"ir1":BASE.describe(ei),"mean_change_pct":BASE.relative_change(eb.mean(),ei.mean())}
    coverage["endpoint"]={"baseline":BASE.describe(b0["min_fde_20"]),"ir1":BASE.describe(ir["min_fde_20"]),"mean_change_pct":BASE.relative_change(b0["min_fde_20"].mean(),ir["min_fde_20"].mean())}
    high=potential>=float(np.quantile(train_potential,.75))
    summary={"split":split,"samples":len(b0["index"]),"response_prediction":response,"interaction_response_groups":groups,"alignment_gain_correlation":correlations,
             "anchor_shift":{f"step_{s}":BASE.describe(shift[:,:,a]) for a,s in enumerate(ANCHOR_STEPS)},"mode_differential":BASE.describe(shift.mean(-1).std(-1)),
             "oracle_winner_change_rate":float((b0["winner"]!=ir["winner"]).mean()),"coverage":coverage,
             "candidate_diversity":{k:{"baseline":BASE.describe(diversity_b[k]),"ir1":BASE.describe(diversity_i[k]),"mean_change_pct":BASE.relative_change(diversity_b[k].mean(),diversity_i[k].mean())} for k in diversity_b},
             "high_interaction":{"threshold_train_q75":float(np.quantile(train_potential,.75)),"metrics":metric_group(b0,ir,high)},
             "head_activation":{"predicted_response_norm":BASE.describe(np.linalg.norm(ir["predicted_response"].reshape(len(ir["index"]),-1),axis=-1)),"gate_mean":BASE.describe(ir["gate_mean"]),"gate_std":BASE.describe(ir["gate_std"]),"semantic_delta_std":BASE.describe(ir["semantic_delta_std"])}}
    rows=[]
    for i in range(len(b0["index"])):
        row={"index":int(b0["index"][i]),"scene_id":int(b0["scene_id"][i]),"interaction_weight":float(ir["interaction_weight"][i]),"potential_score":float(potential[i]),
             "response_cosine_6d":float(flat_cos[i]),"response_mae":float(np.abs(response_error[i]).mean()),"response_l2":float(np.linalg.norm(response_error[i],axis=-1).mean()),
             "winner_changed":bool(b0["winner"][i]!=ir["winner"][i]),"gate_mean":float(ir["gate_mean"][i]),"gate_std":float(ir["gate_std"][i]),"semantic_delta_std":float(ir["semantic_delta_std"][i])}
        for a,s in enumerate(ANCHOR_STEPS): row[f"response_cosine_step_{s}"]=float(cos[i,a]); row[f"anchor_shift_step_{s}"]=float(shift[i,:,a].mean())
        for metric in METRICS: row[f"b0_{metric}"]=float(b0[metric][i]); row[f"ir1_{metric}"]=float(ir[metric][i]); row[f"delta_{metric}"]=float(ir[metric][i]-b0[metric][i])
        rows.append(row)
    BASE.write_rows(outdir/"paired_response_samples.csv",rows); (outdir/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"); return summary

def plot(summaries,output):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2)); x=np.arange(3); labels=["low","medium","high"]
    for s in summaries: axes[0].plot(x,[g["cosine_6d_mean"] for g in s["interaction_response_groups"]],marker="o",label=s["split"])
    axes[0].axhline(0,color="black",lw=.8); axes[0].set_xticks(x,labels); axes[0].set_title("Future-response direction alignment"); axes[0].set_ylabel("Mean cosine"); axes[0].legend()
    keys=("step_4","step_8","step_12","endpoint")
    for s in summaries: axes[1].plot(np.arange(4),[s["coverage"][k]["mean_change_pct"] for k in keys],marker="o",label=s["split"])
    axes[1].axhline(0,color="black",lw=.8); axes[1].set_xticks(np.arange(4),keys); axes[1].set_title("IR1 coverage error change"); axes[1].set_ylabel("Change (%)"); axes[1].legend(); fig.tight_layout(); fig.savefig(output/"response_alignment_and_coverage.png",dpi=180); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda:0"); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--seed",type=int,default=1)
    p.add_argument("--splits",nargs="+",choices=("all",*SPLITS),default=["all"]); p.add_argument("--output",type=Path,default=OUTPUT); p.add_argument("--max-samples",type=int,default=None); args=p.parse_args()
    device=torch.device(args.device); splits=list(SPLITS) if "all" in args.splits else args.splits; output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    cfg,model_class=BASE.prepare(device); summaries=[]; reproduction={}
    for split in splits:
        bdir=run_dir(split,"baseline",args.seed); idir=run_dir(split,"ir1_future_response_supervision",args.seed); bz=np.load(bdir/"split_indices.npz"); iz=np.load(idir/"split_indices.npz")
        train=np.asarray(bz["train"],dtype=np.int64); val=np.asarray(bz["validation"],dtype=np.int64)
        if not np.array_equal(train,iz["train"]) or not np.array_equal(val,iz["validation"]): raise RuntimeError("split mismatch")
        if args.max_samples is not None: val=val[:args.max_samples]
        info=json.loads((bdir/"split.json").read_text(encoding="utf-8")); data,num=Path(info["data_path"]),Path(info["num_path"])
        ds=BASE.IndexedDataset(data,num,val,cfg); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda")
        values={}; reproduction[split]={}
        for name in MODELS:
            print(f"[infer] {split}/{name} samples={len(ds)}",flush=True); model=build(split,name,args.seed,cfg,model_class,device); values[name]=infer(model,loader,device,name!="baseline")
            if args.max_samples is None: reproduction[split][name]=verify(split,name,values[name],args.seed)
            del model
            if device.type=="cuda": torch.cuda.empty_cache()
        potential=BASE.potential_score(data,num,val,cfg); train_potential=BASE.potential_score(data,num,train,cfg); outdir=output/split; outdir.mkdir(parents=True,exist_ok=True)
        summaries.append(analyse(split,values[MODELS[0]],values[MODELS[1]],potential,train_potential,outdir))
    combined={"training_performed":False,"strict_pairing":True,"aggregate_reproduction":reproduction,"results":summaries}; (output/"combined_summary.json").write_text(json.dumps(combined,indent=2,ensure_ascii=False),encoding="utf-8"); plot(summaries,output); print(f"[done] {output}")

if __name__=="__main__": main()
