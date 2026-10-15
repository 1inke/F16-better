import common as c
import numpy as np
def build(data,st,ix,kind):
 vel=data["vel"][ix];cv=vel[:,None]*(.4*np.arange(1,13))[None,:,None]
 if kind=="CV1":return cv[:,None],None
 if kind=="prototype20":
  goal=4.8*vel[:,None]+st["centers"][None];return (np.arange(1,13)/12.)[None,None,:,None]*goal[:,:,None],None
 assert kind=="retrieval20";tr=data["split"]["FIT"];z=(data["vel"][tr]-st["velocity_mean"])/st["velocity_scale"];residual=data["gt"][tr].astype(float)-data["vel"][tr,None]*(.4*np.arange(1,13))[None,:,None];out=np.empty((len(ix),20,12,2));donors=np.empty((len(ix),20),int)
 for row,(index,v) in enumerate(zip(ix,vel)):
  c.guard();q=(v-st["velocity_mean"])/st["velocity_scale"];dist=((z-q)**2).sum(1);order=np.lexsort((data["ids"][tr],dist));selected=[];seen=set()
  for j in order:
   track=data["track"][tr[j]]
   if track==data["track"][index] or track in seen:continue
   seen.add(track);selected.append(j)
   if len(selected)==20:break
  assert len(selected)==20,"STOP_RETRIEVAL20_TRACK_SUPPORT"
  donors[row]=data["ids"][tr[selected]];out[row]=cv[row,None]+residual[selected]
 return out,donors
