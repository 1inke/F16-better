"""Independent arithmetic; no fitting, label solver, or neural model."""
import common as c
import numpy as np,hashlib
def features(d,fold,mapping):
 _,_,_,con,_=c.inputs();tr,va=con.split_masks(d,fold);x=d["X"][tr].astype(float);mean=x.mean((0,1));std=np.where(x.std((0,1))<1e-6,1.,x.std((0,1)))
 assert np.array_equal(mean,mapping.mean) and np.array_equal(std,mapping.std)
 rec=sorted((hashlib.sha256(("r2-fixed-rbf64-v1|"+str(int(sid))+"|"+str(slot)).encode()).digest(),int(sid),slot,i) for i,sid in enumerate(d["ids"][tr]) for slot in range(20))
 keys=np.array([[v[1],v[2]] for v in rec[:64]]);zt=((x-mean)/std).reshape(-1,136);centers=zt[np.array([v[3]*20+v[2] for v in rec[:64]])]
 assert np.array_equal(keys,mapping.center_keys) and np.array_equal(centers,mapping.centers)
 def rbf(z):
  out=np.empty((len(z),64))
  for j in range(64):
   c.guard();dh=np.zeros(len(z));dq=np.zeros(len(z))
   for k in list(range(96))+list(range(120,124)):dh+=(z[:,k]-centers[j,k])**2
   for k in list(range(96,120))+list(range(124,136)):dq+=(z[:,k]-centers[j,k])**2
   out[:,j]=np.exp(-.5*(dh/100+dq/36))
  return out
 phi=rbf(zt);assert np.max(abs(phi.mean(0)-mapping.rbf_mean))<=1e-12
 assert np.max(abs(np.where(phi.std(0)<1e-6,1.,phi.std(0))-mapping.rbf_std))<=1e-12
 z=((d["X"].astype(float)-mean)/std).reshape(-1,136);return np.c_[z,(rbf(z)-mapping.rbf_mean)/mapping.rbf_std].reshape(384,20,200),tr,va
def retrieve(q,tracks,bank):
 # Independent summation and nearest-per-track algorithm instead of global lexsort+seen.
 groups=[np.flatnonzero(bank.tracks==t) for t in np.unique(bank.tracks)];names=np.unique(bank.tracks)
 ix=np.empty((len(q),5),int);dist=np.empty((len(q),5))
 for i,(x,t) in enumerate(zip(q,tracks)):
  c.guard();ds=np.empty(len(bank.z))
  for a in range(0,len(bank.z),256):
   delta=bank.z[a:a+256]-x;ds[a:a+256]=np.sum(delta*delta,axis=1)/200;c.guard()
  representatives=[]
  for name,g in zip(names,groups):
   if name==t:continue
   order=np.lexsort((bank.slots[g],bank.ids[g],ds[g]));j=g[order[0]];representatives.append((float(ds[j]),int(bank.ids[j]),int(bank.slots[j]),int(j)))
  assert len(representatives)>=5
  choices=sorted(representatives)[:5];ix[i]=[v[3] for v in choices];dist[i]=np.sqrt([v[0] for v in choices])
 return ix,dist
def labels(ix,bank,y):
 ans={}
 selected=y.reshape(-1)[ix];P=(selected==1).sum(1);N=(selected==0).sum(1);known=P+N;prior=((y==1).sum(1)/((y>=0).sum(1)))[bank.window[ix]]
 obs=np.array([P[i]/known[i] if known[i] else np.nan for i in range(len(ix))]);exp=np.array([prior[i][selected[i]>=0].mean() if known[i] else np.nan for i in range(len(ix))]);anyexp=np.array([1-np.prod(1-prior[i][selected[i]>=0]) if known[i] else np.nan for i in range(len(ix))])
 return dict(nP=P,nN=N,nU=5-known,known=known,known_sufficient=known>=3,selected_labels=selected,selected_window_prior=prior,observed_P=obs,expected_P=exp,enrichment=obs-exp,observed_anyP=np.where(known>0,(P>0).astype(float),np.nan),expected_anyP=anyexp)
def statistics(p,saved):
 tracks=saved["track_ids"];rng=np.random.default_rng(20261013);counts=[]
 for _ in range(2000):counts.append(np.bincount(rng.integers(len(tracks),size=len(tracks)),minlength=len(tracks)))
 assert np.array_equal(counts,saved["draw_counts"]);values=np.full((2000,6),np.nan)
 masks=[(p["rank"]>=5)&(p["rank"]<=8)&p["supported"],(p["rank"]>=9)&p["supported"]]
 masks +=[m&(p["groups"]==2) for m in masks]
 for b,w in enumerate(counts):
  c.guard();rw=np.array([w[np.searchsorted(tracks,t)] for t in p["track"]]);mu=[np.average(p["enrichment"][m],weights=rw[m]) if rw[m].sum()>0 else np.nan for m in masks];values[b]=[mu[0],mu[1],mu[0]-mu[1],mu[2],mu[3],mu[2]-mu[3]]
 c.eq(values,saved["values"]);return values
