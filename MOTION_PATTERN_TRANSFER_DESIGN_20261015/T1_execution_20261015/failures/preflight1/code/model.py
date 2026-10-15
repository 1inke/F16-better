"""Same full-pattern generator; only loss-side responsibility differs."""
import common as c
import numpy as np,torch,random,hashlib
from torch import nn
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
DEVICE=torch.device('cuda:0');ARMS=('dynamic_winner','fixed_motion_responsibility')
legacy=c.module('t1_readonly_G1_checkpoint_objective',c.G1/'model.py')
seed=legacy.seed;optimizer_state_unused=None
rng_state=legacy.rng_state;restore_rng=legacy.restore_rng;checkpoint=legacy.checkpoint;restore=legacy.restore;tensor_hash=legacy.tensor_hash

class Pattern20(nn.Module):
 def __init__(self,st):
  super().__init__()
  for key in ('mean','std','centers'):self.register_buffer(key,torch.from_numpy(st[key].copy()))
  self.register_buffer('time_seconds',torch.arange(1,13,dtype=torch.float64)*.4)
  self.head=nn.Sequential(nn.Linear(122,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,24))
  nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
 def forward(self,p2,velocity):
  B=len(p2);cent=self.centers.reshape(20,24).to(p2.dtype)
  raw=torch.cat((p2[:,None].expand(-1,20,-1),velocity.to(p2.dtype)[:,None].expand(-1,20,-1),cent[None].expand(B,-1,-1)),2)
  normalized=((raw.double()-self.mean)/self.std).to(p2.dtype)
  correction=self.head(normalized).reshape(B,20,12,2)
  base=self.time_seconds[None,None,:,None]*velocity.double()[:,None,None]+self.centers[None]
  return base+correction.double(),correction

def inputs(data,ix):return tuple(torch.from_numpy(data[k][ix]).to(DEVICE) for k in ('p2','vel'))
def optimizer(net):return torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,foreach=False)

def objective(paths,gt,arm,labels=None):
 assert arm in ARMS
 e=torch.linalg.vector_norm(paths-gt.double()[:,None],dim=-1);Q=e.mean(-1)+.5*e[:,:,-1];qmin,qw=Q.min(1)
 if arm==ARMS[0]:reg=qmin.mean();responsibility=qw
 else:
  assert labels is not None and labels.shape==(len(paths),) and labels.dtype==torch.int64
  responsibility=labels;reg=Q.gather(1,labels[:,None]).mean()
 pair=torch.triu_indices(20,20,1,device=paths.device)
 dist=torch.linalg.vector_norm(paths[:,pair[0]]-paths[:,pair[1]],dim=-1).mean(-1)
 div=torch.relu(.3-dist).square().mean()
 return reg+.05*div,reg,div,qw,responsibility,Q

def objective_numpy(paths,gt,arm,labels=None):
 e=np.sqrt(np.square(paths-gt[:,None].astype(float)).sum(-1));q=e.mean(-1)+.5*e[:,:,-1];qw=q.argmin(1);z=qw if arm==ARMS[0] else np.asarray(labels)
 d=[]
 for j in range(20):
  for k in range(j+1,20):d.append(np.maximum(0,.3-np.sqrt(np.square(paths[:,j]-paths[:,k]).sum(-1)).mean(-1))**2)
 return float(q[np.arange(len(q)),z].mean()+.05*np.mean(d)),qw,z

def step(net,opt,data,ix,arm):
 out,correction=net(*inputs(data,ix));gt=torch.from_numpy(data['gt'][ix]).to(DEVICE);fixed=torch.from_numpy(data['fixed_labels'][ix]).to(DEVICE)
 loss,q,d,qw,z,Q=objective(out,gt,arm,fixed)
 before=torch.cat([p.detach().reshape(-1) for p in net.parameters()]).clone()
 beforec=correction.detach().double().clone()
 diagnostic=dict(n=len(ix),total=loss.item(),quality=q.item(),diversity=d.item(),switch_count=int((qw!=fixed).sum().item()),fixed_minus_minQ_sum=float((Q.gather(1,fixed[:,None]).flatten()-Q.min(1).values).sum().item()),responsibility_counts=np.bincount(z.detach().cpu().numpy(),minlength=20),Q_winner_counts=np.bincount(qw.detach().cpu().numpy(),minlength=20))
 opt.zero_grad(set_to_none=True);loss.backward()
 assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
 opt.step()
 with torch.no_grad():
  afterc=net(*inputs(data,ix))[1].double();norm=torch.linalg.vector_norm((afterc-beforec).reshape(len(ix),20,24),dim=2)
  diagnostic['mode_output_update_norm_sum']=norm.sum(0).cpu().numpy()
  after=torch.cat([p.detach().reshape(-1) for p in net.parameters()]);diagnostic['shared_parameter_update_norm']=float(torch.linalg.vector_norm(after-before).item())
 return diagnostic

def epoch(net,opt,data,indices,g,index,arm,batch=256):
 net.train();order=indices[torch.randperm(len(indices),generator=g).numpy()];rows=[]
 for start in range(0,len(order),batch):c.guard();rows.append(step(net,opt,data,order[start:start+batch],arm))
 n=sum(x['n'] for x in rows)
 return dict(epoch=index,n=n,optimization_steps=len(rows),**{k:sum(x[k]*x['n'] for x in rows)/n for k in ('total','quality','diversity')},
  regression_responsibility_counts=np.sum([x['responsibility_counts'] for x in rows],axis=0).tolist(),Q_winner_counts=np.sum([x['Q_winner_counts'] for x in rows],axis=0).tolist(),
  fixed_Q_switch_rate=sum(x['switch_count'] for x in rows)/n,fixed_minus_minQ=sum(x['fixed_minus_minQ_sum'] for x in rows)/n,
  per_mode_mean_output_update_norm=(np.sum([x['mode_output_update_norm_sum'] for x in rows],axis=0)/n).tolist(),
  mean_shared_parameter_step_norm=float(np.mean([x['shared_parameter_update_norm'] for x in rows])),
  per_mode_parameter_update_not_identifiable_shared_MLP=True,batches=[dict(n=x['n'],Q_winner_counts=x['Q_winner_counts'].tolist(),regression_responsibility_counts=x['responsibility_counts'].tolist(),fixed_Q_switch_count=x['switch_count']) for x in rows],preupdate_population_batchweighted=True)

def generated(net,data,ix):
 net.eval();out=[];delta=[]
 with torch.inference_mode():
  for first in range(0,len(ix),128):
   c.guard();p,d=net(*inputs(data,ix[first:first+128]));out.append(p.cpu().numpy());delta.append(d.cpu().numpy())
 p=np.concatenate(out);d=np.concatenate(delta)
 base=np.arange(1,13)[None,None,:,None]*.4*data['vel'][ix,None,None]+net.centers.detach().cpu().numpy()[None]
 assert np.array_equal(p,base+d.astype(float)) and np.isfinite(p).all() and p.shape==(len(ix),20,12,2)
 return p,d
