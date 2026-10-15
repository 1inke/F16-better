"""Independent saved arrays/objective/statistics. No fit or model forward."""
import common as c
import numpy as np,hashlib
SHARED=np.r_[0:96,120:124]
def rebuilt_features(d,fold,a):
 train,valid=c.inputs()[3].split_masks(d,fold);x=d["X"][train].astype(float);mean=x.mean((0,1));std=np.where(x.std((0,1))<1e-6,1.,x.std((0,1)))
 assert np.array_equal(mean,a["mean"]) and np.array_equal(std,a["std"]);rec=[]
 for i,sid in enumerate(d["ids"][train]):
  for slot in range(20):rec.append((hashlib.sha256(("r2-fixed-rbf64-v1|"+str(int(sid))+"|"+str(slot)).encode()).digest(),int(sid),slot,i))
 rec.sort(key=lambda v:v[:3]);keys=np.array([[v[1],v[2]] for v in rec[:64]]);z=((x-mean)/std).reshape(-1,136);centers=z[np.array([v[3]*20+v[2] for v in rec[:64]])]
 assert np.array_equal(keys,a["center_keys"]) and np.array_equal(centers,a["centers"])
 phi=rbf(z,centers);rm=phi.mean(0);rs=np.where(phi.std(0)<1e-6,1.,phi.std(0));assert abs(rm-a["rbf_mean"]).max()<=1e-12 and abs(rs-a["rbf_std"]).max()<=1e-12
 z=((d["X"].astype(float)-a["mean"])/a["std"]).reshape(-1,136);out=np.c_[z,(rbf(z,a["centers"])-a["rbf_mean"])/a["rbf_std"]].reshape(384,20,200)
 return out,train,valid
def rbf(z,centers):
 out=np.empty((len(z),64))
 for j in range(64):
  if j%8==0:c.guard()
  dh=np.zeros(len(z));dq=np.zeros(len(z))
  for k in list(range(96))+list(range(120,124)):dh+=(z[:,k]-centers[j,k])**2
  for k in list(range(96,120))+list(range(124,136)):dq+=(z[:,k]-centers[j,k])**2
  out[:,j]=np.exp(-.5*(dh/100+dq/36))
 return out
def weights(y,track):
 pos=y==1;neg=y==0;_,inv,count=np.unique(track,return_inverse=True,return_counts=True)
 w=(.5*pos/pos.sum(1,keepdims=True)+.5*neg/neg.sum(1,keepdims=True))/count[inv,None];w*=pos.size/w.sum();return w
def pair_arrays(z,y,track):
 sw=weights(y,track);delta=[];weight=[];win=[];pos=[];neg=[]
 for i,row in enumerate(y):
  mass=sw[i].sum();P=np.flatnonzero(row==1);N=np.flatnonzero(row==0)
  for p in P:
   for n in N:delta.append(z[i,p]-z[i,n]);weight.append(mass/(len(P)*len(N)));win.append(i);pos.append(p);neg.append(n)
 return np.array(delta),np.array(weight),np.array(win),np.array(pos),np.array(neg)
def objective(z,y,tracks,arm,w,b):
 sw=weights(y,tracks)
 if arm=="pointwise":
  known=y!=-1;x=z[known];target=(y==1)[known];weight=sw[known];score=x@w+b;q=np.exp(-np.logaddexp(0.,-score));res=weight*(q-target);value=(weight@(np.logaddexp(0.,score)-target*score)+.5*(w@w))/weight.sum();g=np.r_[(x.T@res+w)/weight.sum(),res.sum()/weight.sum()]
 else:
  delta,a,_,_,_=pair_arrays(z,y,tracks);assert b==0;score=delta@w;q=np.exp(-np.logaddexp(0.,score));value=(a@np.logaddexp(0.,-score)+.5*(w@w))/a.sum();g=(-delta.T@(a*q)+w)/a.sum();assert np.count_nonzero(delta[:,SHARED])==0
 return float(value),g
def independent_metrics(slots,y,scores=None):
 picked=y[np.arange(len(y))[:,None],slots];op=(y==1).any(1);low=(picked==1).any(1);up=(picked!=0).any(1);miss=op&~low
 r=dict(windows=len(y),knownP_windows=int(op.sum()),hits=int(low[op].sum()),Hit4=float(low[op].mean()) if op.any() else None,all_window_lower=float(low.mean()),all_window_unknown_upper=float(up.mean()),selected_U_fraction=float((picked==-1).mean()),knownP_miss=int(miss.sum()),miss_all_N=int((miss&(picked==0).all(1)).sum()),miss_contains_U=int((miss&(picked==-1).any(1)).sum()))
 if scores is not None:
  order=np.argsort(-scores,axis=1,kind="stable");ranks=np.argsort(order,axis=1)+1;best=np.where(y==1,ranks,21).min(1)[op];wins=ties=total=0
  for s,row in zip(scores,y):
   for p in s[row==1]:
    for n in s[row==0]:wins+=int(p>n);ties+=int(p==n);total+=1
  r.update(best_P_rank_mean=float(best.mean()) if len(best) else None,best_P_rank_median=float(np.median(best)) if len(best) else None,P_greater_N_fraction=wins/total if total else None,PN_tie_fraction=ties/total if total else None,PN_pairs=total,pair_fraction_is_pooled_descriptive_not_PU_AUC=True,descriptive_HitK={str(k):float((np.take_along_axis(y,order[:,:k],1)==1).any(1)[op].mean()) if op.any() else None for k in (1,2,4,8)})
 return r
def reports(d,slots,scores=None):
 masks={"overall":np.ones(len(d["ids"]),bool),"stable":d["groups"]==0,"transition":d["groups"]==1,"change":d["groups"]==2};masks.update({"scene_"+str(s):d["scene"]==s for s in np.unique(c.inputs()[0]["scene"])});masks.update({"fold_"+str(f):d["fold"]==f for f in range(3)})
 return {name:independent_metrics(slots[mask],d["labels"][mask],None if scores is None else scores[mask]) if mask.any() else c.empty_report(scores is not None) for name,mask in masks.items()}
def rank(score):return np.sort(np.argsort(-score,axis=1,kind="stable")[:,:4],axis=1)
def equal(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple,np.ndarray)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 elif isinstance(a,(float,np.floating)):assert abs(a-b)<=1e-12,(a,b)
 else:assert a==b,(a,b)
def bootstrap(d,pol):
 y=d["labels"];op=(y==1).any(1);hit={k:(np.take_along_axis(y,v,1)==1).any(1) for k,v in pol.items()};keys=np.unique(d["track"][op]);counts=np.array([(d["track"][op]==k).sum() for k in keys]);sums=np.array([[((hit["pairwise"].astype(float)-hit[r])[op][d["track"][op]==k]).sum() for r in c.ev.REFERENCES] for k in keys]);rng=np.random.default_rng(20261013);draw=[]
 for _ in range(2000):
  ix=rng.integers(0,len(keys),len(keys));draw.append(sums[ix].sum(0)/counts[ix].sum())
 return np.array(draw),hit
