"""R5 GT-free 7 observed velocities x12 immutable native velocities xdot/cross."""
import numpy as np
def velocities(past,native):
 p=np.asarray(past,dtype=np.float64);n=np.asarray(native,dtype=np.float64)
 assert p.ndim==3 and p.shape[1:]==(8,2) and n.shape==(len(p),20,12,2)
 assert np.isfinite(p).all() and np.isfinite(n).all() and not np.count_nonzero(p[:,-1])
 return np.diff(p,axis=1)/.4,np.diff(np.concatenate((np.zeros((len(p),20,1,2)),n),axis=2),axis=2)/.4
def primitive(past,native):
 u,v=velocities(past,native)
 D=u[:,None,:,None,0]*v[:,:,None,:,0]+u[:,None,:,None,1]*v[:,:,None,:,1]
 C=u[:,None,:,None,0]*v[:,:,None,:,1]-u[:,None,:,None,1]*v[:,:,None,:,0]
 return np.stack((D,C),axis=-1).reshape(len(u),20,168)
def build(past,native):return primitive(past,native).astype(np.float32)
def independent(past,native):
 out=np.empty((len(past),20,168),np.float32)
 for i in range(len(past)):
  for j in range(20):
   row=[]
   for t in range(1,8):
    ux=(float(past[i,t,0])-float(past[i,t-1,0]))/.4;uy=(float(past[i,t,1])-float(past[i,t-1,1]))/.4
    for k in range(12):
     vx=(float(native[i,j,k,0])-(float(native[i,j,k-1,0]) if k else 0.))/.4;vy=(float(native[i,j,k,1])-(float(native[i,j,k-1,1]) if k else 0.))/.4
     row.extend((ux*vx+uy*vy,ux*vy-uy*vx))
   out[i,j]=row
 return out
def statistics(T,train):
 a=T[train].reshape(-1,168).astype(float);mean=a.mean(0);raw_std=a.std(0);scale=np.where(raw_std<1e-6,1.,raw_std)
 return dict(temporal_mean=mean,temporal_raw_std=raw_std,temporal_scale=scale,temporal_floor=raw_std<1e-6,column_order=np.array([[t,k,c] for t in range(1,8) for k in range(1,13) for c in (0,1)]))
def transform(base,T,stats,arm):
 assert arm in ("control","temporal");assert base.shape==(len(T),20,200) and T.shape==(len(base),20,168)
 return base if arm=="control" else np.concatenate((base,(T.astype(float)-stats["temporal_mean"])/stats["temporal_scale"]),axis=2)
