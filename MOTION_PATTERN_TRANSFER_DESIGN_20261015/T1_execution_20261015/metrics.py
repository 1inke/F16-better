"""Read-only original scientific arithmetic, original SciPy complete-link."""
import common as c
G=c.module('t1_readonly_G1_metrics',c.G1/'metrics.py')
M1=c.R/'MOTION_STATE_GOAL_PROCESS_DESIGN_20261014/M1_execution_20261014'
fast=c.module('t1_readonly_independent_cluster_implementation',M1/'metrics.py')
masks=G.masks;errors=G.errors;motion=G.motion;sample=G.sample
clusters=fast.clusters;G.clusters=clusters;evaluate=G.evaluate

def enrich(z,met,aux):
 import numpy as np
 aux['vectors5']=np.column_stack((aux['vectors'],(aux['FDE']<=.4).any(1)))
 a,f,j,g,_,_=errors(z['native20'],z['gt'])
 aux['native_vectors5']=np.stack((g.any(1),a.min(1),f.min(1),j.min(1),(f<=.4).any(1)),1)
 err=np.linalg.norm(z['paths']-z['gt'][:,None].astype(float),axis=-1);j=aux['J'].argmin(1);q=aux['quality_winner']
 aux['J_closest']=j;aux['Q_segments']=err[np.arange(len(q)),q].reshape(-1,3,4).mean(-1);aux['J_segments']=err[np.arange(len(j)),j].reshape(-1,3,4).mean(-1)
 drift=np.linalg.norm(z['correction'].astype(float),axis=-1)
 jump=np.linalg.norm(z['paths'][:,:,0]/.4-z['velocity'][:,None],axis=-1)
 for group,mask in masks(z['groups']):
  if not mask.any():continue
  met[group]['Qwinner_segments']=aux['Q_segments'][mask].mean(0)
  met[group]['mode_mean_correction_m']=drift[mask].mean((0,2))
  met[group]['initial_velocity_jump_diagnostic']={'mean':float(jump[mask].mean()),'p95':float(np.quantile(jump[mask],.95)),'not_original_11acceleration_gate':True}
 return met,aux
