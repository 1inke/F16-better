"""Unchanged scientific quality and native motion convention, inherited read-only."""
import common as c
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
v=c.module("m1_readonly_g1_metrics",c.G1/"metrics.py")
masks=v.masks;errors=v.errors;motion=v.motion;sample=v.sample
legacy_clusters=v.clusters
def clusters(p,good,ind=False):
 if not ind:return legacy_clusters(p,good,False)
 # Same 190 pairs, per-pair 12-point reduction and original SciPy complete-link.
 # Only the Python pair loop is vectorized, not clustering or science thresholds.
 answer=np.empty((len(p),2),int)
 if p.shape[1]==1:
  answer[:,0]=1;answer[:,1]=good[:,0];return answer
 assert p.shape[1]==20
 first,second=np.triu_indices(20,1)
 for start in range(0,len(p),128):
  c.guard();part=p[start:start+128]
  difference=part[:,first]-part[:,second]
  distances=np.sqrt((difference*difference).sum(-1))
  condensed=np.maximum(distances.mean(-1)/.1,distances[:,:,-1]/.2)
  for i,row in enumerate(condensed):
   label=fcluster(linkage(row,method='complete'),1,criterion='distance')
   answer[start+i]=[len(np.unique(label)),len(np.unique(label[good[start+i]]))]
 return answer
v.clusters=clusters
evaluate=v.evaluate
