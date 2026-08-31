"""Prepare, train and summarize EXP-163 explicit prototype K20 on official folds."""
from __future__ import annotations
import argparse,csv,json,os,subprocess
from pathlib import Path
import numpy as np


HERE=Path(__file__).resolve().parent;WORKSPACE=HERE.parent;PROJECT=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20")
PYTHON=Path(r"C:\coding_toobox\envs\astra2\python.exe");TRAINER=WORKSPACE/"FRONT_ENCODER_SERIES"/"train_front_variant.py"
BUILDER=HERE/"build_exp163_prototypes.py";DECODER=HERE/"prototype_decoder_models.py";CONFIG=WORKSPACE/"OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES"/"motif_rbf_official_4p8.yaml"
DATA=PROJECT/"datasets"/"eth_ucy"/"processed_data_diverse";V2_ROOT=WORKSPACE/"OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES"/"EXP_156B_FOLD_LOCAL_CURRENT_DECODER"/"results"
K20_SUMMARY=WORKSPACE/"FOLD_STAGE_ATTRIBUTION_SERIES"/"EXP_157B_MATCHED_K20"/"summary.json";V2_SUMMARY=WORKSPACE/"SCORE_SYNCHRONIZATION_SERIES"/"EXP_161_FROZEN_CANDIDATE_LISTWISE_SCORER"/"summary.json"
DEFAULT_OUT=HERE/"EXP_163_EXPLICIT_ENDPOINT_PROTOTYPE_K20";FOLDS=("eth","hotel","univ","zara01","zara02")


def source(fold,seed):return V2_ROOT/fold/"v2"/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"/"best_selected.pth"
def run_dir(out,fold,seed):return out/"results"/fold/"f16_temporal_social_cross"/f"seed_{seed}"/"val_scene_-1"
def prototype(out,fold):return out/"prototypes"/f"{fold}_5x4.npz"


def train_command(a,out,fold,seed):
    return [str(PYTHON),"-u",str(TRAINER),"--project-root",str(PROJECT),"--config-file",str(CONFIG),"--data-path",str(DATA/f"{fold}_data_train.npy"),
            "--num-path",str(DATA/f"{fold}_num_train.npy"),"--test-data-path",str(DATA/f"{fold}_data_test.npy"),"--test-num-path",str(DATA/f"{fold}_num_test.npy"),
            "--variant","f16_temporal_social_cross","--decoder-experiment","p20_explicit_endpoint_path_prototypes","--decoder-module",str(DECODER),
            "--seed",str(seed),"--val-scene-id","-1","--validation-fraction","0.05","--validation-seed","20260830","--validation-split-mode","scene_tail",
            "--epochs",str(a.epochs),"--val-freq",str(a.val_freq),"--batch-size",str(a.batch_size),"--device",a.device,"--max-agents","9","--graph-top-k","4","--graph-radius","6.0",
            "--scheduler","monotonic_cosine","--warmup-epochs","1","--learning-rate","0.00025","--output-root",str(out/"results"/fold),
            "--oracle-coverage-weight","0.20","--oracle-coverage-temperature","0.10","--oracle-coverage-fde-weight","0.25","--checkpoint-criterion","oracle_combo",
            "--ema-decay","0.995","--focal-heading-normalize","--augment-reflection","--augment-speed-min","0.9","--augment-speed-max","1.1","--augment-probability","0.5"]


def baseline_rows():
    k={r["fold"]:r for r in json.loads(K20_SUMMARY.read_text(encoding="utf-8"))["rows"] if r["seed"]==1}
    vraw=json.loads(V2_SUMMARY.read_text(encoding="utf-8"))["per_fold"];v={r["fold"]:r for r in vraw if r["family"]=="v2" and r["seed"]==1};return k,v


def summarize(out,seeds):
    k20,v2=baseline_rows();rows=[]
    for fold in FOLDS:
        for seed in seeds:
            path=run_dir(out,fold,seed)/"metrics.json"
            if not path.is_file():continue
            data=json.loads(path.read_text(encoding="utf-8"));e=data.get("evaluation_test_best_selected")
            if not e:continue
            p=e["oracle"];row={"fold":fold,"seed":seed,"samples":e["samples"],"minADE":p["min_ade_20"],"minFDE":p["min_fde_20"],"same_path_FDE":p["same_path_fde"],
                "top1_ADE":e["original"]["ade"],"top1_FDE":e["original"]["fde"],"parameters":data["locked_backend"]["parameter_counts"]["total"],
                "decoder_parameters":data["locked_backend"]["parameter_counts"]["decoder"]};rows.append(row)
    out.mkdir(parents=True,exist_ok=True)
    if not rows:return
    seed1=[r for r in rows if r["seed"]==1];available_folds=[f for f in FOLDS if any(r["fold"]==f for r in seed1)]
    macro={m:float(np.mean([r[m] for r in seed1])) for m in ("minADE","minFDE","same_path_FDE","top1_ADE","top1_FDE")}
    vmacro={"minADE":float(np.mean([v2[f]["oracle"]["minADE"] for f in available_folds])),"minFDE":float(np.mean([v2[f]["oracle"]["minFDE"] for f in available_folds])),"same_path_FDE":float(np.mean([v2[f]["oracle"]["same_path_FDE"] for f in available_folds]))}
    fold_eval=[]
    for r in seed1:
        v=v2[r["fold"]]["oracle"];fold_eval.append({"fold":r["fold"],"ADE_regression_vs_V2":r["minADE"]/v["minADE"]-1,"FDE_regression_vs_V2":r["minFDE"]/v["minFDE"]-1})
    ade_gain=1-macro["minADE"]/vmacro["minADE"];fde_gain=1-macro["minFDE"]/vmacro["minFDE"];same_gain=1-macro["same_path_FDE"]/vmacro["same_path_FDE"]
    ade_non=sum(x["ADE_regression_vs_V2"]<=0 for x in fold_eval);fde_non=sum(x["FDE_regression_vs_V2"]<=0 for x in fold_eval)
    max_ade=max(x["ADE_regression_vs_V2"] for x in fold_eval);max_fde=max(x["FDE_regression_vs_V2"] for x in fold_eval)
    complete=available_folds==list(FOLDS)
    if complete:
        univ=next(x for x in fold_eval if x["fold"]=="univ")
        passed=(ade_gain>=0 and fde_gain>=0 and max(ade_gain,fde_gain)>=.02 and same_gain>=0 and ade_non>=4 and fde_non>=4 and max_ade<=.05 and max_fde<=.05 and univ["ADE_regression_vs_V2"]<=0 and univ["FDE_regression_vs_V2"]<=0)
        conditional=(ade_gain>=-.01 and fde_gain>=-.01 and max_ade<=.05 and max_fde<=.05)
        decision="PASSED" if passed else "CONDITIONAL" if conditional else "REJECTED"
    else:decision="PARTIAL_SMOKE_ONLY"
    payload={"experiment":"EXP-163 Explicit Endpoint Prototype K20","decision":decision,"macro":macro,"v2_macro":vmacro,
             "relative_gain_vs_V2":{"minADE":ade_gain,"minFDE":fde_gain,"same_path_FDE":same_gain},"fold_stability":{"ADE_non_regression_folds":ade_non,"FDE_non_regression_folds":fde_non,
             "max_ADE_regression":max_ade,"max_FDE_regression":max_fde,"per_fold":fold_eval},"rows":rows,
             "protocol":{"encoder":"frozen F16 from fold-local V2 checkpoint","candidate_K":20,"available_folds":available_folds,"seed1_fivefold":complete,"new_scorer":False,"M2":False}}
    (out/"summary.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    with (out/"per_fold_summary.csv").open("w",newline="",encoding="utf-8-sig") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=(f"# EXP-163 Result Analysis\n\n## Decision: {decision}\n\nPrototype K20宏平均minADE/minFDE/same-path FDE = {macro['minADE']:.4f}/{macro['minFDE']:.4f}/{macro['same_path_FDE']:.4f}。\n\n"
          f"相对V2改善 = {100*ade_gain:.2f}%/{100*fde_gain:.2f}%/{100*same_gain:.2f}%。ADE/FDE不退化fold = {ade_non}/5、{fde_non}/5；最坏fold退化 = {100*max_ade:.2f}%/{100*max_fde:.2f}%。\n")
    (out/"RESULT_ANALYSIS.md").write_text(text,encoding="utf-8")
    denominator=len(available_folds)
    clean_text=(f"# EXP-163 Result Analysis\n\n## Decision: {decision}\n\n"
                f"Prototype K20 宏平均 minADE/minFDE/same-path FDE = {macro['minADE']:.4f}/{macro['minFDE']:.4f}/{macro['same_path_FDE']:.4f}。\n\n"
                f"相对 V2 改善 = {100*ade_gain:.2f}%/{100*fde_gain:.2f}%/{100*same_gain:.2f}%。"
                f"ADE/FDE 不退化 fold = {ade_non}/{denominator}、{fde_non}/{denominator}；"
                f"最坏 fold 退化 = {100*max_ade:.2f}%/{100*max_fde:.2f}%。\n")
    (out/"RESULT_ANALYSIS.md").write_text(clean_text,encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--folds",nargs="+",default=["all"],choices=("all",*FOLDS));p.add_argument("--seeds",nargs="+",type=int,default=[1])
    p.add_argument("--epochs",type=int,default=30);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--val-freq",type=int,default=5);p.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    p.add_argument("--force-prototypes",action="store_true");p.add_argument("--force",action="store_true");p.add_argument("--dry-run",action="store_true");a=p.parse_args();folds=FOLDS if "all" in a.folds else tuple(a.folds)
    required=(PYTHON,TRAINER,BUILDER,DECODER,CONFIG,K20_SUMMARY,V2_SUMMARY)
    missing=[str(x) for x in required if not x.is_file()]
    missing += [str(source(f,s)) for f in folds for s in a.seeds if not source(f,s).is_file()]
    missing += [str(DATA/f"{f}_data_{split}.npy") for f in folds for split in ("train","test") if not (DATA/f"{f}_data_{split}.npy").is_file()]
    if missing:raise FileNotFoundError("Missing inputs:\n"+"\n".join(missing))
    for fold in folds:
        asset=prototype(a.output_dir,fold)
        if not asset.is_file() or a.force_prototypes:
            cmd=[str(PYTHON),str(BUILDER),"--data-path",str(DATA/f"{fold}_data_train.npy"),"--output",str(asset),"--seed",str(163+FOLDS.index(fold))]
            if a.dry_run:print("[dry-run prototype]",json.dumps(cmd))
            else:subprocess.run(cmd,check=True,cwd=WORKSPACE)
        for seed in a.seeds:
            destination=run_dir(a.output_dir,fold,seed);metrics=destination/"metrics.json"
            if metrics.is_file() and not a.force:print(f"[skip] {fold}/seed{seed}");continue
            cmd=train_command(a,a.output_dir,fold,seed);env=os.environ.copy();env["EXP163_PROTOTYPE_PATH"]=str(asset);env["EXP163_SOURCE_CHECKPOINT"]=str(source(fold,seed))
            if a.force:cmd.append("--force")
            if a.dry_run:print(f"[dry-run train] {fold}/seed{seed} "+json.dumps(cmd));continue
            destination.mkdir(parents=True,exist_ok=True);print(f"[run] {fold}/seed{seed}",flush=True)
            with (destination/"console.log").open("w",encoding="utf-8") as log:
                job=subprocess.Popen(cmd,cwd=PROJECT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1);assert job.stdout
                for line in job.stdout:print(f"[{fold}] {line}",end="",flush=True);log.write(line);log.flush()
            if job.wait():raise RuntimeError(f"Failed {fold}: {destination/'console.log'}")
            summarize(a.output_dir,a.seeds)
    if a.dry_run:print("[done] dry-run passed")
    else:summarize(a.output_dir,a.seeds);print(f"[done] {a.output_dir}")
if __name__=="__main__":main()
