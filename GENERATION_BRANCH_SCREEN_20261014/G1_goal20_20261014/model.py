import common as c
import numpy as np,torch,random,copy
from torch import nn
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
DEVICE=torch.device("cuda:0")
def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
class Goal20(nn.Module):
 def __init__(self,st):
  super().__init__();self.register_buffer("mean",torch.from_numpy(st["mean"]));self.register_buffer("std",torch.from_numpy(st["std"]));self.register_buffer("centers",torch.from_numpy(st["centers"]));self.register_buffer("s",torch.arange(1,13,dtype=torch.float64)/12.)
  self.context=nn.Sequential(nn.Linear(96,128),nn.GELU(),nn.Linear(128,64),nn.GELU());self.query=nn.Parameter(torch.randn(20,16)*.02);self.head=nn.Sequential(nn.Linear(80,64),nn.GELU(),nn.Linear(64,2));nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
 def forward(self,p2,velocity):
  h=self.context((p2-self.mean)/self.std);x=torch.cat((h[:,None].expand(-1,20,-1),self.query[None].expand(len(h),-1,-1)),2);delta=self.head(x);goal=4.8*velocity.double()[:,None]+self.centers[None]+delta.double();path=self.s[None,None,:,None]*goal[:,:,None];return path,goal,h,delta
def objective(paths,gt):
 errors=torch.linalg.vector_norm(paths-gt.double()[:,None],dim=-1);A=errors.mean(-1);F=errors[:,:,-1];Q=A+.5*F;value,winner=Q.min(1);pairs=torch.triu_indices(20,20,1,device=paths.device);distance=torch.linalg.vector_norm(paths[:,pairs[0]]-paths[:,pairs[1]],dim=-1).mean(-1);div=torch.relu(.3-distance).square().mean()
 return value.mean()+.05*div,value.mean(),div,winner
def objective_numpy(paths,gt):
 e=np.sqrt(((paths-gt[:,None].astype(float))**2).sum(-1));Q=e.mean(-1)+.5*e[:,:,-1];div=[]
 for a in range(20):
  for b in range(a+1,20):div.append(np.maximum(0,.3-np.linalg.norm(paths[:,a]-paths[:,b],axis=-1).mean(-1))**2)
 return float(Q.min(1).mean()+.05*np.mean(div)),Q.argmin(1)
def rng_state(g):return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),loader=g.get_state(),numpy=np.random.get_state(),python=random.getstate())
def restore_rng(v,g):torch.set_rng_state(v["torch"]);torch.cuda.set_rng_state_all(v["cuda"]);g.set_state(v["loader"]);np.random.set_state(v["numpy"]);random.setstate(v["python"])
def checkpoint(path,m,opt,g,history,manifest):
 v=dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=rng_state(g),history=history,manifest=manifest,epoch=len(history));tmp=str(path)+".tmp";torch.save(v,tmp);import os;os.replace(tmp,path)
def restore(path,m,opt,g,manifest):
 v=torch.load(path,map_location="cpu",weights_only=False);assert v["manifest"]==manifest,"RESUME_SOURCE_MISMATCH";m.load_state_dict(v["model"]);opt.load_state_dict(v["optimizer"]);restore_rng(v["rng"],g);return v["history"]
def tensor_hash(state):
 import hashlib
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());a=v.detach().cpu().numpy();h.update(str(a.dtype).encode());h.update(a.tobytes())
 return h.hexdigest()
def epoch(m,opt,data,indices,g,index,batch=256):
 m.train();order=indices[torch.randperm(len(indices),generator=g).numpy()];sums=np.zeros(3);counts=np.zeros(20,int);n=0
 for first in range(0,len(order),batch):
  c.guard();ix=order[first:first+batch];p2=torch.from_numpy(data["p2"][ix]).to(DEVICE);vel=torch.from_numpy(data["vel"][ix]).to(DEVICE);gt=torch.from_numpy(data["gt"][ix]).to(DEVICE);out=m(p2,vel)[0];loss,q,d,w=objective(out,gt);opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(loss) and all(x.grad is None or torch.isfinite(x.grad).all() for x in m.parameters());opt.step();N=len(ix);sums+=np.array([loss.item(),q.item(),d.item()])*N;n+=N;counts+=np.bincount(w.detach().cpu().numpy(),minlength=20)
 return dict(epoch=index,n=n,total=sums[0]/n,quality=sums[1]/n,diversity=sums[2]/n,winner_counts=counts,optimization_steps=int(np.ceil(n/batch)),preupdate_population_batchweighted=True)
def generated(m,data,ix):
 m.eval();paths=[];goals=[];deltas=[]
 with torch.inference_mode():
  for first in range(0,len(ix),128):
   c.guard();ind=ix[first:first+128];out,g,h,delta=m(torch.from_numpy(data["p2"][ind]).to(DEVICE),torch.from_numpy(data["vel"][ind]).to(DEVICE));paths.append(out.cpu().numpy());goals.append(g.cpu().numpy());deltas.append(delta.cpu().numpy())
 p=np.concatenate(paths);g=np.concatenate(goals);delta=np.concatenate(deltas);expected=(np.arange(1,13)/12.)[None,None,:,None]*g[:,:,None];assert np.array_equal(p,expected),"PATH_NUMPY_RECONSTRUCTION"
 assert np.isfinite(p).all() and p.shape==(len(ix),20,12,2);return p,g,delta
