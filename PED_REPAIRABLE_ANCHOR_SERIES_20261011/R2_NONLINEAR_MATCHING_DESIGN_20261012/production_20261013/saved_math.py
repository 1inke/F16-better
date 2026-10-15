"""Independent bounded saved arithmetic; no sklearn fit/solver/Torch."""
import common as c
import numpy as np,hashlib
from math import comb
def weights(y,tracks):
 w=np.zeros(y.shape,float)
 for i in range(len(y)):
  count=int((tracks==tracks[i]).sum())
  for j in range(20):
   if y[i,j] in (0,1):w[i,j]=.5/int((y[i]==y[i,j]).sum())/count
 w*=y.size/w.sum();return w
def arrays(d,fold,arm,a):
 tr,va=c.ic.split_masks(d,fold);X=d["X"][tr].astype(float);mean=X.mean((0,1));std=np.where(X.std((0,1))<1e-6,1.,X.std((0,1)))
 assert np.array_equal(mean,a["mean"]) and np.array_equal(std,a["std"]);assert a["coef"].shape==(136 if arm=="linear" else 200,)
 z=((X-mean)/std).reshape(-1,136)
 if arm=="rbf64":
  rec=[]
  for i,sid in enumerate(d["ids"][tr]):
   for j in range(20):rec.append((hashlib.sha256(("r2-fixed-rbf64-v1|"+str(int(sid))+"|"+str(j)).encode()).digest(),int(sid),j,i))
  rec.sort(key=lambda v:v[:3]);keys=np.array([[v[1],v[2]] for v in rec[:64]]);centers=z[np.array([v[3]*20+v[2] for v in rec[:64]])]
  assert np.array_equal(keys,a["center_keys"]) and np.array_equal(centers,a["centers"])
  phi=rbf(z,centers);rm=phi.mean(0);rs=np.where(phi.std(0)<1e-6,1.,phi.std(0))
  assert abs(rm-a["rbf_mean"]).max()<=1e-12 and abs(rs-a["rbf_std"]).max()<=1e-12
 else:assert a["centers"].shape==(0,136)
 w=weights(d["labels"][tr],d["track"][tr]);target=d["labels"][tr]==1;known=d["labels"][tr]!=-1
 assert np.array_equal(w,c.m.witness_supervision(d["labels"][tr],d["track"][tr])[2])
 trainf=features(d["X"][tr],a);x=trainf[known];y=target[known];sw=w[known];score=x@a["coef"]+float(a["intercept"][0]);p=np.exp(-np.logaddexp(0.,-score));g=np.r_[(x.T@(sw*(p-y))+a["coef"])/sw.sum(),(sw*(p-y)).sum()/sw.sum()]
 assert abs(g).max()<=1e-6
 return dict(gradient_inf=float(abs(g).max())),trainf@a["coef"]+a["intercept"][0],features(d["X"][va],a)@a["coef"]+a["intercept"][0]
def rbf(z,centers):
 # Different memory organization: one center at a time, individual axis accumulation.
 out=np.empty((len(z),64))
 for j in range(64):
  if j%8==0:c.guard()
  dh=np.zeros(len(z));dq=np.zeros(len(z))
  for k in list(range(96))+list(range(120,124)):dh+=(z[:,k]-centers[j,k])**2
  for k in list(range(96,120))+list(range(124,136)):dq+=(z[:,k]-centers[j,k])**2
  out[:,j]=np.exp(-.5*(dh/100+dq/36))
 return out
def features(x,a):
 z=((x.astype(float)-a["mean"])/a["std"]).reshape(-1,136)
 if str(a["arm"])=="rbf64":z=np.c_[z,(rbf(z,a["centers"])-a["rbf_mean"])/a["rbf_std"]]
 return z.reshape(len(x),20,-1)
def rank(score):return np.sort(np.argsort(-score,axis=1,kind="stable")[:,:4],axis=1)
def describe(slots,y,score=None):
 n=len(y);picks=y[np.arange(n)[:,None],slots];op=np.any(y==1,axis=1);lo=np.any(picks==1,axis=1);up=np.any(picks!=0,axis=1)
 ch=np.array([1-(comb(20-int(v),4) if 20-int(v)>=4 else 0)/comb(20,4) for v in (y==1).sum(1)])
 z=dict(windows=n,P_windows=int(op.sum()),hits=int(lo[op].sum()),Hit4=float(lo[op].mean()) if op.any() else None,all_window_lower=float(lo.mean()),all_window_upper=float(up.mean()),selected_U_fraction=float((picks==-1).mean()),uniform_random4_expectation=float(ch[op].mean()) if op.any() else None)
 if score is not None:
  ranks=np.argsort(np.argsort(-score,axis=1,kind="stable"),axis=1)+1;best=np.where(y==1,ranks,21).min(1)[op];wins=ties=total=0
  for s,row in zip(score,y):
   for a in s[row==1]:
    for b in s[row==0]:wins+=int(a>b);ties+=int(a==b);total+=1
  z.update(best_P_rank_mean=float(best.mean()) if len(best) else None,best_P_rank_median=float(np.median(best)) if len(best) else None,P_greater_N_pair_fraction=wins/total if total else None,P_N_tie_fraction=ties/total if total else None,P_N_pair_denominator=total,pair_aggregation="pooled_pairs_descriptive_not_PU_AUC")
 return z
def equal(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b),(set(a),set(b))
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple,np.ndarray)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 elif isinstance(a,(float,np.floating)):assert abs(a-b)<=1e-12,(a,b)
 else:assert a==b,(a,b)
def bootstrap(track,mask,delta,seed):
 keys=sorted(set(track[mask].tolist()));count=np.array([int((track[mask]==k).sum()) for k in keys]);sums=np.array([delta[mask][track[mask]==k].sum() for k in keys]);rng=np.random.default_rng(seed);draw=[]
 for _ in range(2000):
  ix=rng.integers(0,len(keys),len(keys));draw.append(sums[ix].sum()/count[ix].sum())
 return np.array(draw)
