"""Tri-state evidence; unknown is never an infeasible negative."""
import numpy as np
from fractions import Fraction as F
UNKNOWN,NEGATIVE,POSITIVE=-1,0,1
def endpoint_excluded(p,g):
    dx=F(float(p[0]))-F(float(g[0]));dy=F(float(p[1]))-F(float(g[1]))
    limit=F(float(.20))+F(float(.40))+F("1e-9")
    return dx*dx+dy*dy>limit*limit
def build(native,gt,envelope,safe):
    scale=max(1.,float(abs(native).max()),float(abs(gt).max()));roundoff=64*np.finfo(float).eps*scale*np.sqrt(2)
    assert roundoff<1e-9,("STOP_ENDPOINT_CERTIFICATE_SCALE",scale,roundoff)
    negative=np.array([[endpoint_excluded(p[-1],g[-1]) for p in paths] for paths,g in zip(native,gt)])
    assert not np.any(envelope&negative) and not np.any(safe&negative) and not np.any(safe&~envelope)
    y=np.full(envelope.shape,UNKNOWN,np.int8);y[negative]=NEGATIVE;y[safe]=POSITIVE
    return y,dict(max_coordinate=scale,conservative_float64_roundoff_bound=roundoff,negative_only_margin=1e-9,not_scientific_evaluation_epsilon=True)
def supervision(envelope,labels,tracks,mode):
    assert mode in ("envelope","witness");pos=envelope if mode=="envelope" else labels==POSITIVE
    neg=labels==NEGATIVE;assert np.all(pos.any(1)&neg.any(1)) and not np.any(pos&neg)
    _,inv,ct=np.unique(tracks,return_inverse=True,return_counts=True)
    weight=(.5*pos/pos.sum(1,keepdims=True)+.5*neg/neg.sum(1,keepdims=True))/ct[inv,None];weight*=pos.size/weight.sum();known=pos|neg
    assert np.all(weight[~known]==0) and abs(weight[pos].sum()-weight[neg].sum())<1e-8
    return pos.astype(np.int8),known,weight
def rank4(score):return np.sort(np.argsort(-score,axis=1,kind="stable")[:,:4],axis=1)
def random4(ids):
    import hashlib
    return np.array([sorted(sorted(range(20),key=lambda s:hashlib.sha256(("repair-random-v1|"+str(int(v))+"|"+str(s)).encode()).digest())[:4]) for v in ids],int)
def bottom4(logs):return np.sort(np.argsort(-logs,axis=1,kind="stable")[:,16:],axis=1)
def metrics(slots,y,groups,scene):
    from math import comb
    picked=np.take_along_axis(y,slots,1);lo=(picked==1).any(1);up=(picked!=NEGATIVE).any(1);op=(y==1).any(1)
    chance=np.array([1.-(comb(20-int(k),4) if 20-int(k)>=4 else 0)/comb(20,4) for k in (y==1).sum(1)])
    masks={"overall":np.ones(len(y),bool),"stable":groups==0,"transition":groups==1,"change":groups==2};masks.update({"scene_"+str(s):scene==s for s in np.unique(scene)});report={}
    for name,m in masks.items():
        known=m&op;n=int(known.sum());report[name]=dict(windows=int(m.sum()),known_opportunity=n,hits=int(lo[known].sum()),Hit4=float(lo[known].mean()) if n else None,all_window_discovery=float(lo[m].mean()),unknown_selected_fraction=float((picked[m]==-1).mean()),lower_all=float(lo[m].mean()),upper_all=float(up[m].mean()),random_expected_Hit4=float(chance[known].mean()) if n else None)
    return report,lo,up,op,chance
def bootstrap(track,mask,delta,seed):
    keys,inv=np.unique(track[mask],return_inverse=True);assert len(keys)>0
    count=np.bincount(inv);sums=np.bincount(inv,weights=delta[mask].astype(float));rng=np.random.default_rng(seed);draw=[]
    for _ in range(2000):
        picked=rng.integers(0,len(keys),len(keys));draw.append(sums[picked].sum()/count[picked].sum())
    return dict(estimate=float(delta[mask].mean()),CI95=np.quantile(draw,[.025,.975]),CI98333=np.quantile(draw,[.05/6,1-.05/6]),tracks=len(keys),windows=int(mask.sum()),iterations=2000,rng=seed,paired_tracks=True,not_three_model_seeds=True)
