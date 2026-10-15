import common as c
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
def masks(groups):return [("overall",np.ones(len(groups),bool))]+[(name,groups==i) for i,name in enumerate(("stable","transition","change"))]
def errors(p,gt,ind=False):
 e=np.sqrt(((p-gt[:,None].astype(float))**2).sum(-1)) if ind else np.linalg.norm(p-gt[:,None],axis=-1)
 A=e.mean(-1);F=e[:,:,-1];J=np.maximum(A/.2,F/.4);good=(A<=.2)&(F<=.4);j=J.argmin(1);q=(A+.5*F).argmin(1);segments=e[np.arange(len(p)),j].reshape(-1,3,4).mean(-1)
 return A,F,J,good,segments,q
def motion(p,ind=False):
 if ind:v=np.concatenate((p[:,:,:1],p[:,:,1:]-p[:,:,:-1]),2)/.4;a=(v[:,:,1:]-v[:,:,:-1])/.4;j=(a[:,:,1:]-a[:,:,:-1])/.4
 else:v=np.diff(p,axis=2,prepend=np.zeros_like(p[:,:,:1]))/.4;a=np.diff(v,axis=2)/.4;j=np.diff(a,axis=2)/.4
 s,aa,jj=[np.sqrt((x*x).sum(-1)) for x in (v,a,j)]
 return dict(speed_p95=float(np.quantile(s,.95)),accel_mean=float(aa.mean()),accel_p95=float(np.quantile(aa,.95)),jerk_mean=float(jj.mean()),jerk_p95=float(np.quantile(jj,.95)))
def clusters(p,good,ind=False):
 answer=np.empty((len(p),2),int)
 for i,(row,g) in enumerate(zip(p,good)):
  if i%64==0:c.guard()
  if len(row)==1:answer[i]=[1,int(g[0])];continue
  if ind:
   dist=[]
   for j in range(20):
    for k in range(j+1,20):
     d=np.sqrt(((row[j]-row[k])**2).sum(-1));dist.append(max(float(d.mean()/.1),float(d[-1]/.2)))
   condensed=np.array(dist)
  else:
   d=np.linalg.norm(row[:,None]-row[None],axis=-1);D=np.maximum(d.mean(-1)/.1,d[:,:,-1]/.2);condensed=D[np.triu_indices(20,1)]
  label=fcluster(linkage(condensed,method="complete"),1,criterion="distance");answer[i]=[len(np.unique(label)),len(np.unique(label[g]))]
 return answer
def evaluate(p,gt,groups,native,ind=False):
 A,F,J,g,segments,q=errors(p,gt,ind);na,nf,nj,ng,_,_=errors(native,gt,ind);good=g.any(1);base=ng.any(1);cl=clusters(p,g,ind);metrics={}
 for name,mask in masks(groups):
  n=int(mask.sum())
  if not n:metrics[name]=dict(n=0);continue
  mot=motion(p[mask],ind);nm=motion(native[mask],ind)
  metrics[name]=dict(n=n,K=p.shape[1],good=int(good[mask].sum()),coverage=float(good[mask].mean()),endpoint_good=int((F[mask]<=.4).any(1).sum()),endpoint_coverage=float((F[mask]<=.4).any(1).mean()),minADE=float(A[mask].min(1).mean()),minFDE=float(F[mask].min(1).mean()),J=float(J[mask].min(1).mean()),rescued=int((good&~base&mask).sum()),lost=int((~good&base&mask).sum()),retained=int((good&base&mask).sum()),effective_distinct=float(cl[mask,0].mean()),effective_good=float(cl[mask,1].mean()),single_Jclosest_segments=segments[mask].mean(0),sensitivity_03_05_good=int(((A<=.3)&(F<=.5)).any(1)[mask].sum()),winner_counts=np.bincount(q[mask],minlength=p.shape[1]),motion=mot,native_motion=nm,motion_failed=[k for k in mot if mot[k]>1.01*nm[k]])
 aux=dict(ADE=A,FDE=F,J=J,good_candidate=g,clusters=cl,quality_winner=q,vectors=np.stack((good,A.min(1),F.min(1),J.min(1)),1))
 return metrics,aux
def sample(data,ix,p,**extras):return dict(sample_id=data["ids"][ix],groups=data["groups"][ix],track=data["track"][ix],scene=data["scene"][ix],gt=data["gt"][ix],native20=data["native"][ix].astype(float),paths=p,**extras)
