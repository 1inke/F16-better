"""One-command sequential driver for EXP-169A and EXP-169B."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ASTRA=Path(r"C:\coding_toobox\envs\astra2\python.exe");ROOT=Path(__file__).resolve().parent.parent
if str(Path(sys.executable).resolve()).casefold()!=str(ASTRA.resolve()).casefold():
    print(f"[bootstrap] relaunching with {ASTRA}",flush=True);raise SystemExit(subprocess.run([str(ASTRA),*sys.argv]).returncode)
A=ROOT/"CROSS_FAMILY_HORIZON_AUDIT"/"run_exp169a_horizon_complementarity.py";B=ROOT/"OFFICIAL_SPARSE_SOCIAL_COUNTERFACTUAL_AUDIT"/"run_exp169b_sparse_social_counterfactual.py"
def main():
    p=argparse.ArgumentParser();p.add_argument("--device",default="cuda:0");p.add_argument("--folds",nargs="+",default=["all"]);p.add_argument("--seed",type=int,default=1);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--max-samples",type=int,default=0);p.add_argument("--dry-run",action="store_true");a=p.parse_args()
    for path in (A,B):
        if not path.is_file():raise FileNotFoundError(path)
    common=["--device",a.device,"--folds",*a.folds,"--seed",str(a.seed),"--batch-size",str(a.batch_size)]
    if a.max_samples:common += ["--max-samples",str(a.max_samples)]
    if a.dry_run:common.append("--dry-run")
    for label,path in (("EXP-169A",A),("EXP-169B",B)):
        command=[str(ASTRA),"-u",str(path),*common];print(f"[run] {label}\n"+json.dumps(command,ensure_ascii=False),flush=True)
        result=subprocess.run(command,cwd=ROOT)
        if result.returncode:raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    print("[done] EXP-169A and EXP-169B completed",flush=True)
if __name__=="__main__":main()
