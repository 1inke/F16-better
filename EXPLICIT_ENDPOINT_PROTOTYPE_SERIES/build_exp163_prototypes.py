"""Build leakage-free fold-local 5x4 future trajectory prototypes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np


def heading_aligned_futures(raw:np.ndarray,obs_len:int=8)->np.ndarray:
    focal=np.asarray(raw[:,0,:,-2:],dtype=np.float32)
    past=focal[:,:obs_len];future=focal[:,obs_len:]
    lag=min(3,obs_len-1);disp=past[:,-1]-past[:,-1-lag];speed=np.linalg.norm(disp,axis=-1)
    heading=np.zeros_like(disp);heading[:,0]=1.;valid=speed>.03;heading[valid]=disp[valid]/speed[valid,None]
    centered=future-past[:,-1,None];c,s=heading[:,0],heading[:,1]
    return np.stack((centered[...,0]*c[:,None]+centered[...,1]*s[:,None],
                     -centered[...,0]*s[:,None]+centered[...,1]*c[:,None]),axis=-1)


def kmeans(x:np.ndarray,k:int,seed:int,iterations:int=100):
    if len(x)<k:raise ValueError(f"Need at least {k} samples, got {len(x)}")
    rng=np.random.default_rng(seed);centers=[x[rng.integers(len(x))]]
    for _ in range(1,k):
        d=np.min(np.sum((x[:,None]-np.stack(centers)[None])**2,axis=-1),axis=1)
        centers.append(x[rng.choice(len(x),p=d/d.sum())] if d.sum()>0 else x[rng.integers(len(x))])
    centers=np.stack(centers).astype(np.float32);labels=np.zeros(len(x),dtype=np.int64)
    for _ in range(iterations):
        new=np.argmin(np.sum((x[:,None]-centers[None])**2,axis=-1),axis=1)
        if np.array_equal(new,labels):break
        labels=new
        for j in range(k):
            if np.any(labels==j):centers[j]=x[labels==j].mean(0)
            else:
                error=np.min(np.sum((x[:,None]-centers[None])**2,axis=-1),axis=1);centers[j]=x[error.argmax()]
    inertia=float(np.sum((x-centers[labels])**2));return centers,labels,inertia


def build(data_path:Path,output:Path,seed:int):
    raw=np.load(data_path,mmap_mode="r");future=heading_aligned_futures(raw);endpoint=future[:,-1]
    endpoint_centers,endpoint_labels,endpoint_inertia=kmeans(endpoint,5,seed)
    # Stable identity: short-to-long endpoint radius, then lateral position.
    endpoint_order=np.lexsort((endpoint_centers[:,1],np.linalg.norm(endpoint_centers,axis=1)))
    remap=np.empty(5,dtype=np.int64);remap[endpoint_order]=np.arange(5);endpoint_labels=remap[endpoint_labels]
    endpoint_centers=endpoint_centers[endpoint_order]
    fractions=np.arange(1,future.shape[1]+1,dtype=np.float32)[:,None]/future.shape[1]
    prototypes=[];counts=[];path_inertia=[]
    for group in range(5):
        members=future[endpoint_labels==group];residual=members-fractions[None]*members[:,-1,None]
        centers,labels,inertia=kmeans(residual.reshape(len(residual),-1),4,seed+101+group)
        centers=centers.reshape(4,future.shape[1],2);curvature=np.abs(np.diff(centers,axis=1)).sum(axis=(1,2));order=np.argsort(curvature)
        centers=centers[order];trajectory=fractions[None]*endpoint_centers[group][None,None]+centers
        trajectory[:,-1]=endpoint_centers[group];prototypes.append(trajectory);counts.append(np.bincount(labels,minlength=4).tolist());path_inertia.append(inertia)
    prototypes=np.concatenate(prototypes).astype(np.float32);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,trajectory_prototypes=prototypes,endpoint_centers=endpoint_centers.astype(np.float32),
                        endpoint_cluster_counts=np.bincount(endpoint_labels,minlength=5),path_cluster_counts=np.asarray(counts),
                        endpoint_inertia=np.asarray(endpoint_inertia),path_inertia=np.asarray(path_inertia),samples=np.asarray(len(future)))
    metadata={"data_path":str(data_path),"samples":len(future),"seed":seed,"endpoint_groups":5,"path_shapes_per_endpoint":4,
              "num_modes":20,"endpoint_cluster_counts":np.bincount(endpoint_labels,minlength=5).tolist(),"path_cluster_counts":counts,
              "endpoint_inertia":endpoint_inertia,"path_inertia":path_inertia}
    output.with_suffix(".json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"output":str(output),"shape":prototypes.shape,"samples":len(future)},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser();p.add_argument("--data-path",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",type=int,default=163)
    a=p.parse_args();build(a.data_path,a.output,a.seed)
if __name__=="__main__":main()
