"""Audit the five processed ETH/UCY folds before official experiments."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=Path(r"C:\Users\linke\Desktop\DRSE\ASTRA_ACCURACY_Q20\datasets\eth_ucy\processed_data_diverse")
OUT=ROOT/"EXP_156A_PROTOCOL_AUDIT"
FOLDS=("eth","hotel","univ","zara01","zara02")

def digest(sample: np.ndarray, valid: int) -> bytes:
    # Hash only valid agents and all 20 frames; padding cannot create false matches.
    h=hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(sample[:valid]).view(np.uint8));h.update(int(valid).to_bytes(2,"little"))
    return h.digest()

def main():
    rows=[];failures=[]
    for fold in FOLDS:
        paths={k:DATA/f"{fold}_{k}.npy" for k in ("data_train","num_train","data_test","num_test")}
        missing=[str(x) for x in paths.values() if not x.is_file()]
        if missing: raise FileNotFoundError("\n".join(missing))
        train=np.load(paths["data_train"],mmap_mode="r");tn=np.load(paths["num_train"],mmap_mode="r")
        test=np.load(paths["data_test"],mmap_mode="r");vn=np.load(paths["num_test"],mmap_mode="r")
        shape_ok=bool(train.ndim==4 and test.ndim==4 and train.shape[1:]==test.shape[1:]==(9,20,6))
        count_ok=bool(len(train)==len(tn) and len(test)==len(vn) and tn.min()>=1 and tn.max()<=9 and vn.min()>=1 and vn.max()<=9)
        train_hashes={digest(train[i],int(tn[i])) for i in range(len(train))}
        overlap=sum(digest(test[i],int(vn[i])) in train_hashes for i in range(len(test)))
        item={"fold":fold,"train_samples":len(train),"test_samples":len(test),"sample_shape":list(train.shape[1:]),
              "train_valid_agents":[int(tn.min()),int(tn.max())],"test_valid_agents":[int(vn.min()),int(vn.max())],
              "exact_train_test_overlap":overlap,"shape_ok":shape_ok,"count_ok":count_ok}
        rows.append(item)
        if not shape_ok or not count_ok or overlap: failures.append(item)
        print(f"[{fold}] train={len(train)} test={len(test)} overlap={overlap} shape_ok={shape_ok} count_ok={count_ok}")
    protocol={"frequency_hz":2.5,"observation_frames":8,"prediction_frames":12,
              "observation_seconds":8/2.5,"prediction_seconds":12/2.5,
              "configured_prediction_seconds":4.8,"ceil_frames":math.ceil(4.8*2.5)}
    passed=not failures and protocol["ceil_frames"]==12
    payload={"experiment":"EXP-156A official protocol audit","passed":passed,"protocol":protocol,"folds":rows,"failures":failures}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"protocol_audit.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(f"[audit-{'passed' if passed else 'failed'}] {OUT/'protocol_audit.json'}")
    if not passed: raise RuntimeError("Official protocol audit failed")
if __name__=="__main__":main()
