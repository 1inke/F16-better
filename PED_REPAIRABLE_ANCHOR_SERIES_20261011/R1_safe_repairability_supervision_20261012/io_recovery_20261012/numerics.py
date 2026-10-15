"""Six-dimensional per-case GT-oracle optimization, NOT neural training."""
import numpy as np
from scipy.optimize import minimize
DT=.4
T=np.arange(1,13,dtype=float)
B=np.stack([np.interp(T,[0,4,8,12],np.eye(4)[j+1]) for j in range(3)],1)
RAD=.20*(T/12)**2
NODE=.20*(np.array([4.,8.,12.])/12)**2
def derivatives(p):
    v=np.diff(p,axis=-2,prepend=np.zeros_like(p[...,:1,:]))/DT
    a=np.diff(v,axis=-2)/DT;j=np.diff(a,axis=-2)/DT
    return v,a,j
def motion(p):
    v,a,j=derivatives(np.asarray(p,dtype=float))
    v,a,j=(np.linalg.norm(z,axis=-1) for z in (v,a,j))
    return np.array([np.quantile(v,.95),a.mean(),np.quantile(a,.95),j.mean(),np.quantile(j,.95)])
def quality(p,gt):
    e=np.linalg.norm(p-gt,axis=-1);return float(e.mean()),float(e[-1]),float(max(e.mean()/.2,e[-1]/.4))
def decode(anchor,raw32):
    raw=np.asarray(raw32,dtype=np.float32).astype(float).reshape(3,2)
    u=np.tanh(raw);u/=np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),1.)
    proposal=B@(u*NODE[:,None]);norm=np.linalg.norm(proposal,axis=-1)
    ratio=np.ones(12);np.divide(RAD,norm,out=ratio,where=norm>0)
    lam=min(1.,float(ratio.min()));res=proposal*lam
    return anchor.astype(float)+res,res,u,lam
def proposed_raw(unit_controls):
    # A fixed witness construction, not tolerance relaxation or a solver retry.
    return np.arctanh(np.clip(np.asarray(unit_controls).reshape(3,2),-1+1e-7,1-1e-7)).astype(np.float32)
def validate(native,gt,slot,raw):
    path,res,u,lam=decode(native[slot],raw)
    bank=native.astype(float).copy();bank[slot]=path
    ade,fde,J=quality(path,gt);m=motion(bank);base=motion(native)
    bound=bool(np.all(np.linalg.norm(res,axis=-1)<=RAD))
    good=ade<=.2 and fde<=.4
    safe=bool(np.all(m<=1.01*base))
    inner=all(np.all(np.linalg.norm(a,axis=-1)<=1.01*np.linalg.norm(b,axis=-1)) for a,b in zip(derivatives(path),derivatives(native[slot].astype(float))))
    return dict(ADE=ade,FDE=fde,J=J,bound_literal=bound,quality_good=bool(good),window_allfive_PASS=safe,pointwise_inner_PASS=bool(inner),witness=bool(bound and good and safe),motion=m,native_motion=base,lambda_=lam),path,bank
def solve(native,gt,slot,mode,guard=lambda:None):
    assert mode in ("shape","pointwise")
    p=native[slot].astype(float);gt=gt.astype(float)
    old_norm=[np.linalg.norm(a,axis=-1) for a in derivatives(p)]
    def constraints(x):
        guard();u=x[:6].reshape(3,2);r=B@(u*NODE[:,None]);q=p+r
        e=np.linalg.norm(q-gt,axis=-1)
        cs=[1.-np.linalg.norm(u,axis=1),RAD-np.linalg.norm(r,axis=1),
            np.array([x[6]-e.mean()/.2,x[6]-e[-1]/.4])]
        if mode=="pointwise":
            cs.extend(1.01*b-np.linalg.norm(a,axis=-1) for a,b in zip(derivatives(q),old_norm))
        return np.concatenate(cs)
    x0=np.zeros(7);x0[6]=quality(p,gt)[2]+1e-3
    result=minimize(lambda x:x[6],x0,method="SLSQP",
        jac=lambda x:np.array([0.,0.,0.,0.,0.,0.,1.]),
        bounds=[(-1.,1.)]*6+[(0.,None)],
        constraints=[dict(type="ineq",fun=constraints)],
        options=dict(maxiter=150,ftol=1e-9,disp=False))
    assert np.isfinite(result.x).all()
    raw=proposed_raw(result.x[:6]);audit,path,bank=validate(native,gt,slot,raw)
    return dict(mode=mode,solver_success=bool(result.success),solver_status=int(result.status),
        solver_message=str(result.message),iterations=int(result.nit),objective=float(result.fun),
        min_constraint=float(constraints(result.x).min()),unit_controls=result.x[:6].reshape(3,2),
        raw32=raw,**audit),path,bank
def independent_verify(native,gt,slot,raw):
    # Scalar reconstruction using explicit basis interpolation and old derivative definitions.
    raw=np.asarray(raw,dtype=np.float32).astype(float).reshape(3,2)
    u=[]
    for a in raw:
        z=np.tanh(a);u.append(z/max(1.,float(np.sqrt((z*z).sum()))))
    u=np.array(u);node=u*NODE[:,None]
    r=np.stack([np.interp(T,[0,4,8,12],np.r_[0.,node[:,d]]) for d in range(2)],1)
    length=np.sqrt((r*r).sum(1));lam=min([1.]+[float(RAD[i]/length[i]) for i in range(12) if length[i]>0])
    r*=lam;q=native[slot].astype(float)+r;bank=native.astype(float).copy();bank[slot]=q
    e=np.sqrt(((q-gt.astype(float))**2).sum(1))
    def m(a):
        vs=np.concatenate((a[:,:1],a[:,1:]-a[:,:-1]),axis=1)/.4
        ac=(vs[:,1:]-vs[:,:-1])/.4;jer=(ac[:,1:]-ac[:,:-1])/.4
        s,x,y=[np.sqrt((b*b).sum(-1)) for b in (vs,ac,jer)]
        return np.array([np.quantile(s,.95),x.mean(),np.quantile(x,.95),y.mean(),np.quantile(y,.95)])
    mot=m(bank);base=m(native.astype(float))
    valid=bool(e.mean()<=.2 and e[-1]<=.4 and np.all(np.sqrt((r*r).sum(1))<=RAD) and np.all(mot<=base*1.01))
    return dict(witness=valid,ADE=float(e.mean()),FDE=float(e[-1]),motion=mot),q,bank
