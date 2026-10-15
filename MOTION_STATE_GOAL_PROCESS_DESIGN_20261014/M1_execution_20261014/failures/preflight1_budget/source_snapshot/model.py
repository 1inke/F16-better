import common as c
import numpy as np,torch,random
from torch import nn
legacy=c.module("m1_readonly_g1_model",c.G1/"model.py")
DEVICE=legacy.DEVICE;seed=legacy.seed;objective=legacy.objective;objective_numpy=legacy.objective_numpy
rng_state=legacy.rng_state;restore_rng=legacy.restore_rng;checkpoint=legacy.checkpoint;restore=legacy.restore;tensor_hash=legacy.tensor_hash
ARMS=("last_velocity","dynamic_history")
class MotionGoal(nn.Module):
 def __init__(self,st,arm):
  super().__init__();assert arm in ARMS;self.arm=arm
  for name in ("mean","std","centers","velocity_mean","velocity_std","acceleration_mean","acceleration_std"):
   self.register_buffer(name,torch.from_numpy(st[name].copy()))
  self.register_buffer("s",torch.arange(1,13,dtype=torch.float64)/12)
  self.motion=nn.GRU(5,32,num_layers=1,bias=True,batch_first=True,dropout=0,bidirectional=False)
  self.context=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,64),nn.GELU())
  self.query=nn.Parameter(torch.randn(20,16)*.02)
  self.head=nn.Sequential(nn.Linear(80,64),nn.GELU(),nn.Linear(64,2))
  nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
 def tokens(self,past):
  vel=((past.double()[:,1:]-past.double()[:,:-1])/.4).float()
  acc=((vel.double()[:,1:]-vel.double()[:,:-1])/.4).float()
  if self.arm=="last_velocity":
   vel=vel[:,-1:, :].expand(-1,7,-1);acc=torch.zeros_like(acc)
  # Inherited common float64 stats; cast normalized tokens back to float32 NN input.
  vn=((vel.double()-self.velocity_mean)/self.velocity_std).to(past.dtype)
  an=((acc.double()-self.acceleration_mean)/self.acceleration_std).to(past.dtype)
  an=torch.cat((torch.zeros_like(vn[:,:1]),an),1)
  mask=torch.ones_like(vn[...,:1]);mask[:,0]=0
  return torch.cat((vn,an,mask),2)
 def forward(self,p2,past,velocity):
  token=self.tokens(past);_,hidden=self.motion(token)
  h=self.context(torch.cat(((p2-self.mean)/self.std,hidden[-1]),1))
  x=torch.cat((h[:,None].expand(-1,20,-1),self.query[None].expand(len(h),-1,-1)),2)
  delta=self.head(x);goals=4.8*velocity.double()[:,None]+self.centers[None]+delta.double()
  return self.s[None,None,:,None]*goals[:,:,None],goals,h,delta

def inputs(data,ix):
 return tuple(torch.from_numpy(data[k][ix]).to(DEVICE) for k in ("p2","past","vel"))
def optimizer(net):return torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,foreach=False)
def epoch(net,opt,data,indices,g,index,batch=256):
 net.train();order=indices[torch.randperm(len(indices),generator=g).numpy()];sums=np.zeros(3);counts=np.zeros(20,int);n=0
 for first in range(0,len(order),batch):
  c.guard();ix=order[first:first+batch];out=net(*inputs(data,ix))[0]
  loss,q,d,w=objective(out,torch.from_numpy(data["gt"][ix]).to(DEVICE))
  opt.zero_grad(set_to_none=True);loss.backward()
  assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
  opt.step();N=len(ix);sums+=np.array([loss.item(),q.item(),d.item()])*N;n+=N;counts+=np.bincount(w.detach().cpu().numpy(),minlength=20)
 return dict(epoch=index,n=n,total=sums[0]/n,quality=sums[1]/n,diversity=sums[2]/n,winner_counts=counts.tolist(),optimization_steps=int(np.ceil(n/batch)),preupdate_population_batchweighted=True)
def generated(net,data,ix):
 net.eval();paths=[];goals=[];deltas=[]
 with torch.inference_mode():
  for start in range(0,len(ix),128):
   c.guard();p,g,_,delta=net(*inputs(data,ix[start:start+128]));paths.append(p.cpu().numpy());goals.append(g.cpu().numpy());deltas.append(delta.cpu().numpy())
 p=np.concatenate(paths);g=np.concatenate(goals);delta=np.concatenate(deltas)
 assert p.shape==(len(ix),20,12,2) and np.isfinite(p).all()
 assert np.array_equal(p,(np.arange(1,13)/12.)[None,None,:,None]*g[:,:,None])
 return p,g,delta
