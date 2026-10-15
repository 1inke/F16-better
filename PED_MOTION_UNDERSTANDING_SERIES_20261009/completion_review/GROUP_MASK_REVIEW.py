import sys,json,hashlib
sys.dont_write_bytecode=True
import audit_saved as a
import numpy as np
def main():
 locks=a.read(a.C/'input_manifest.json')['files'];before={p:a.digest(p) for p in locks};assert before==locks
 counts=[]
 for fold in a.read(a.H/'protocol.json')['folds']:
  a.guard();d=a.load(a.H/f'data/{fold}_evaluation.npz');c=a.read(a.H/f'calibration/{fold}.json');o=d['obs'];n=d['valid'];v=np.diff(o.astype(float),axis=2);s=np.linalg.norm(v[:,0],axis=-1);first=s[:,:3].mean(1);last=s[:,-3:].mean(1)
  ratio=np.divide(last,first,out=np.ones_like(last),where=first>0);c1=(ratio<.75)|(ratio>1.33)|((first==0)&(last>0))
  va=v[:,0,:3].mean(1);vb=v[:,0,-3:].mean(1);angle=np.degrees(np.arctan2(va[:,0]*vb[:,1]-va[:,1]*vb[:,0],(va*vb).sum(1)));c2=(np.linalg.norm(va,axis=1)>=.02)&(np.linalg.norm(vb,axis=1)>=.02)&(abs(angle)>15)
  pred=o[:,:,-1,None]+v[:,:,-1,None]*np.arange(1,13)[None,None,:,None];dist=np.linalg.norm(pred[:,1:]-pred[:,:1],axis=-1);mask=np.arange(1,9)[None]<n[:,None];c3=np.where(mask[:,:,None],dist,np.inf).min((1,2))<1;cue=c1|c2|c3
  F=a.errors(a.cv(o),d['gt'])[1];q50,q75,q90=c['cv_fde_quantiles'];non=F>q75;success=F<=q50
  sb=np.linalg.norm(o[:,0,-1]-o[:,0,-2],axis=1);bins=np.searchsorted(c['speed_edges'],sb,side='right')*4+np.searchsorted([1,2,4],n-1,side='right');matched=np.zeros(len(F),bool);deficits={}
  for k in np.unique(bins):
   target=int(((bins==k)&non&cue).sum());pool=sorted(np.flatnonzero((bins==k)&success),key=lambda i:hashlib.sha256(f'20261009:{int(d["ids"][i])}'.encode()).hexdigest());matched[pool[:target]]=True;deficits[str(k)]=max(0,target-len(pool))
  groups=dict(all=np.ones(len(F),bool),nonCV=non,strong_nonCV=F>q90,nonCV_cue_positive=non&cue,nonCV_cue_negative=non&~cue,cue_positive=cue,cue_negative=~cue,CV_success=success,matched_CV_success=matched,stationary=sb<.02)
  for seed in (1,2,3):
   z=a.load(a.H/f'jobs/{fold}_seed{seed}/outputs.npz');s=a.read(a.H/f'jobs/{fold}_seed{seed}/summary.json');assert s['matching_deficits']==deficits and np.array_equal(bins,z['bin'])
   for k,m in groups.items():assert np.array_equal(m,z['mask_'+k])
  counts.append(dict(fold=fold,groups={k:dict(n=int(m.sum()),tracks=len(np.unique(d['track'][m]))) for k,m in groups.items()},matching_deficits=deficits,C1=int(c1.sum()),C2=int(c2.sum()),C3=int(c3.sum())))
 assert {p:a.digest(p) for p in locks}==before
 a.write(a.C/'GROUP_MASK_REVIEW.json',dict(status='PASS',all_masks_bins_matching_independently_reconstructed=True,records=counts,source_input_initial_final_SHA_unchanged=True))
if __name__=='__main__':main()

