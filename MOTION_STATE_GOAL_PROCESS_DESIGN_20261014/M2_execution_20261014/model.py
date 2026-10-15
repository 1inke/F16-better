"""M2 goal-head-only joint training with exact discrete endpoint projection."""
import common as c
import numpy as np,torch
from torch import nn
legacy=c.module("m2_readonly_G1_model",c.G1/"model.py")
Goal20=legacy.Goal20;DEVICE=legacy.DEVICE;seed=legacy.seed
objective=legacy.objective;objective_numpy=legacy.objective_numpy
rng_state=legacy.rng_state;restore_rng=legacy.restore_rng;checkpoint=legacy.checkpoint;restore=legacy.restore;tensor_hash=legacy.tensor_hash
ARMS=("frozen_goal","joint_goal")
FIELDS=("paths","goals","v0","raw_b","projected_a","d","e","minimum_norm","null_term","velocities")
def project(b,g,v0):
 w=torch.arange(12,0,-1,device=b.device,dtype=b.dtype)[None,None,:,None]
 d=(g-12*.4*v0[:,None])/(.4*.4);wb=(w*b).sum(2);e=d-wb
 minimum=w/650*d[:,:,None];null=b-w/650*wb[:,:,None]
 a=b+w/650*e[:,:,None];v=v0[:,None,None]+.4*torch.cumsum(a,2);p=.4*torch.cumsum(v,2)
 return dict(paths=p,goals=g,v0=v0,raw_b=b,projected_a=a,d=d,e=e,minimum_norm=minimum,null_term=null,velocities=v)
def project_numpy(b,g,v0):
 w=np.arange(12,0,-1,dtype=float)[None,None,:,None]
 # Independent explicit temporal loops, not torch cumsum.
 d=(g-4.8*v0[:,None])/.16;wb=np.zeros_like(g)
 for t in range(12):wb+=float(12-t)*b[:,:,t]
 e=d-wb;a=np.empty_like(b);minimum=np.empty_like(b);null=np.empty_like(b)
 p=np.empty_like(b);v=np.empty_like(b);lastp=np.zeros_like(g);lastv=np.broadcast_to(v0[:,None],g.shape).copy()
 for t in range(12):
  minimum[:,:,t]=(12-t)/650*d;null[:,:,t]=b[:,:,t]-(12-t)/650*wb
  a[:,:,t]=b[:,:,t]+(12-t)/650*e
  lastv=lastv+.4*a[:,:,t];lastp=lastp+.4*lastv;v[:,:,t]=lastv;p[:,:,t]=lastp
 return dict(paths=p,goals=g,v0=v0,raw_b=b,projected_a=a,d=d,e=e,minimum_norm=minimum,null_term=null,velocities=v)
class JointProcess(nn.Module):
 def __init__(self,st,state,gs,arm):
  super().__init__();assert arm in ARMS;self.arm=arm
  self.goal=Goal20(st);self.goal.load_state_dict(state);self.goal.requires_grad_(False)
  if arm=="joint_goal":self.goal.head.requires_grad_(True)
  for name in ("goal_mean","goal_std","v0_mean","v0_std"):
   self.register_buffer(name,torch.from_numpy(gs[name].copy()))
  self.process=nn.Sequential(nn.Linear(84,64),nn.GELU(),nn.Linear(64,24))
  nn.init.zeros_(self.process[-1].weight);nn.init.zeros_(self.process[-1].bias)
 def forward(self,p2,velocity):
  with torch.no_grad():h=self.goal.context((p2-self.goal.mean)/self.goal.std);q=self.goal.query[None].expand(len(h),-1,-1)
  x=torch.cat((h[:,None].expand(-1,20,-1),q),2)
  if self.arm=="frozen_goal":
   with torch.no_grad():delta=self.goal.head(x)
  else:delta=self.goal.head(x)
  g=4.8*velocity.double()[:,None]+self.goal.centers[None]+delta.double()
  gz=((g-self.goal_mean)/self.goal_std).to(h.dtype)
  vz=((velocity.double()-self.v0_mean)/self.v0_std).to(h.dtype)[:,None].expand(-1,20,-1)
  b=self.process(torch.cat((x,gz,vz),2)).double().reshape(-1,20,12,2)
  return project(b,g,velocity.double())
def parent_state(s):
 return torch.load(c.G1/f"runs/seed_{s}/checkpoint.pt",map_location="cpu",weights_only=False)["model"]
def fresh(s,arm):
 seed(s);st=c.npz(c.G1/"statistics.npz");state=parent_state(s);gs=c.npz(c.H/f"goal_stats_{s}.npz")
 net=JointProcess(st,state,gs,arm).to(DEVICE)
 assert tensor_hash(net.goal.state_dict())==tensor_hash(state)
 return net
def inputs(data,ix):return tuple(torch.from_numpy(data[k][ix]).to(DEVICE) for k in ("p2","vel"))
def optimizer(net):
 return torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,foreach=False)
def epoch(net,opt,data,indices,g,index,batch=256):
 net.train();order=indices[torch.randperm(len(indices),generator=g).numpy()];sums=np.zeros(3);counts=np.zeros(20,int);n=0
 for first in range(0,len(order),batch):
  c.guard();ix=order[first:first+batch];p=net(*inputs(data,ix))["paths"];loss,q,d,w=objective(p,torch.from_numpy(data["gt"][ix]).to(DEVICE))
  opt.zero_grad(set_to_none=True);loss.backward()
  assert torch.isfinite(loss)
  for name,param in net.named_parameters():
   if param.requires_grad:assert param.grad is not None and torch.isfinite(param.grad).all(),name
   else:assert param.grad is None,name
  opt.step();N=len(ix);sums+=np.array([loss.item(),q.item(),d.item()])*N;n+=N;counts+=np.bincount(w.detach().cpu().numpy(),minlength=20)
 return dict(epoch=index,n=n,total=sums[0]/n,quality=sums[1]/n,diversity=sums[2]/n,winner_counts=counts.tolist(),optimization_steps=int(np.ceil(n/batch)),preupdate_population_batchweighted=True)
def generated(net,data,ix):
 net.eval();out={k:[] for k in FIELDS}
 with torch.inference_mode():
  for first in range(0,len(ix),128):
   c.guard();z=net(*inputs(data,ix[first:first+128]))
   for k in out:out[k].append(z[k].cpu().numpy())
 out={k:np.concatenate(a) for k,a in out.items()}
 assert out["paths"].shape==(len(ix),20,12,2) and all(np.isfinite(x).all() for x in out.values())
 assert np.max(abs(out["paths"][:,:,-1]-out["goals"]))<=2e-5
 return out
def frozen_check(net,initial):
 for k,v in net.state_dict().items():
  if net.arm=="joint_goal" and k.startswith("goal.head."):continue
  if k.startswith("process."):continue
  assert torch.equal(v.detach().cpu(),initial[k]),("FROZEN_CHANGED",k)
 return True
