from pathlib import Path
import argparse
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    count=len(np.load(a.data,mmap_mode="r")); a.output.parent.mkdir(parents=True,exist_ok=True)
    encoded=-(np.arange(count,dtype=np.int64)+2); np.save(a.output,encoded)
    check=np.load(a.output,mmap_mode="r")
    if len(check)!=count or int(check[0])!=-2 or int(check[-1])!=-(count+1): raise RuntimeError("sidecar verification failed")
    print(f"[ok] index sidecar samples={count} output={a.output}")


if __name__=="__main__": main()

