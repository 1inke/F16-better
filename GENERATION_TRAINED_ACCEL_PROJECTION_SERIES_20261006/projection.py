"""Human-approved numerical repair; fixed double precision reservation, no GT."""
import torch
DT=.4
UNIT=torch.finfo(torch.float64).eps/2
GAMMA512=512*UNIT/(1-512*UNIT)

def physical_mean(path):
    v=torch.diff(path,dim=2,prepend=torch.zeros_like(path[:,:,:1]))/DT
    return torch.linalg.vector_norm(torch.diff(v,dim=2)/DT,dim=-1).mean((1,2))

def project(base,increments,limit=None):
    assert base.shape==increments.shape and base.ndim==4 and base.shape[1:]==(20,12,2)
    assert base.dtype==increments.dtype==torch.float32
    assert torch.isfinite(base).all() and torch.isfinite(increments).all()
    b=base.double();u=increments.double()
    mb=physical_mean(b)
    d=torch.linalg.vector_norm(torch.diff(u,dim=2)/DT**2,dim=-1).mean((1,2))
    scale=b.abs().amax((1,2,3))+12*u.abs().amax((1,2,3))
    reserve=8*GAMMA512*scale/DT**2
    nominal=torch.ones_like(mb);alpha=torch.ones_like(mb)
    nz=d!=0
    nominal[nz]=(.01*mb[nz]/d[nz]).clamp(0,1)
    alpha[nz]=((.01*mb[nz]-reserve[nz]).clamp_min(0)/d[nz]).clamp(0,1)
    mean=u.mean(2,keepdim=True)
    zero=(mb==0)&nz
    mean=torch.where(zero[:,None,None,None],mean.float().double(),mean)
    up=mean+alpha[:,None,None,None]*(u-mean)
    up=torch.where((~nz)[:,None,None,None],u,up)
    if limit is not None:
        bound=limit.double();up=torch.maximum(-bound,torch.minimum(bound,up))
    path=b+up.cumsum(2)
    assert torch.isfinite(path).all()
    return path,up,dict(alpha=alpha,alpha_nominal=nominal,native_mean=mb,delta_mean=d,
                        roundoff_reserve=reserve,zero_budget_mean_quantized=zero)
