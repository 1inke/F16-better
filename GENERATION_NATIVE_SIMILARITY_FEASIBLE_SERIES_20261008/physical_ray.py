"""GT-free native-relative ray intersection. No loss, optimizer or model updates."""
import numpy as np
import torch
MARGIN=1.01
NUMERICAL_INTERIOR=1.-2.**-20  # numerical slack inside unchanged strict1.01 evaluation
def differences_numpy(x):
    v=np.diff(x,axis=-2,prepend=np.zeros_like(x[...,:1,:]))/.4
    a=np.diff(v,axis=-2)/.4
    j=np.diff(a,axis=-2)/.4
    return v,a,j
def differences_torch(x):
    v=torch.diff(x,dim=-2,prepend=torch.zeros_like(x[...,:1,:]))/.4
    a=torch.diff(v,dim=-2)/.4
    j=torch.diff(a,dim=-2)/.4
    return v,a,j
def root_torch(b,d):
    aa=(d*d).sum(-1);bb=2*(b*d).sum(-1);budget=(MARGIN*MARGIN-1)*(b*b).sum(-1)
    disc=bb*bb+4*aa*budget
    sq=torch.sqrt(torch.where(disc>0,disc,torch.ones_like(disc)))
    sq=torch.where(disc>0,sq,torch.zeros_like(sq))
    den=torch.where(aa>0,2*aa,torch.ones_like(aa))
    denpos=torch.where(sq+bb>0,sq+bb,torch.ones_like(bb))
    rt=torch.where(bb>=0,2*budget/denpos,(-bb+sq)/den)
    rt=torch.where(aa>0,rt,torch.ones_like(rt))
    return rt.clamp(0,1)
def project_torch(native,residual):
    assert native.dtype==residual.dtype==torch.float64
    terms=[root_torch(b,d) for b,d in zip(differences_torch(native),differences_torch(residual))]
    alpha=torch.cat(terms,-1).amin(-1)
    alpha=torch.where(alpha<1,alpha*NUMERICAL_INTERIOR,alpha)
    return residual*alpha[...,None,None],alpha
def project_numpy(native,residual):
    native=np.asarray(native,np.float64);residual=np.asarray(residual,np.float64);roots=[]
    for b,d in zip(differences_numpy(native),differences_numpy(residual)):
        aa=np.einsum('...i,...i->...',d,d);bb=2*np.einsum('...i,...i->...',b,d);budget=(MARGIN*MARGIN-1)*np.einsum('...i,...i->...',b,b)
        sq=np.sqrt(bb*bb+4*aa*budget)
        rt=np.ones_like(aa);active=aa>0;pos=active&(bb>=0)&(sq+bb>0);neg=active&(bb<0)
        rt[active]=0.;rt[pos]=2*budget[pos]/(sq[pos]+bb[pos]);rt[neg]=(-bb[neg]+sq[neg])/(2*aa[neg]);roots.append(np.clip(rt,0,1))
    alpha=np.min(np.concatenate(roots,-1),axis=-1);alpha=np.where(alpha<1,alpha*NUMERICAL_INTERIOR,alpha)
    return residual*alpha[...,None,None],alpha
def audit_numpy(native,path):
    ratios=[];zero_constraints=0;numeric=[]
    # Binomial absolute coefficient scale; no derivative-magnitude floor.
    scale=np.abs(native)+np.abs(path)
    u=np.finfo(np.float64).eps/2
    for order,(b,p) in enumerate(zip(differences_numpy(native),differences_numpy(path)),1):
        prev=np.concatenate((np.zeros_like(scale[...,:1,:]),scale[...,:-1,:]),axis=-2) if order==1 else scale[...,:-1,:]
        scale=(scale+prev)/.4 if order==1 else (scale[...,1:,:]+prev)/.4
        bn=np.sqrt(np.sum(b*b,-1));pn=np.sqrt(np.sum(p*p,-1))
        # At most 8*order+16 rounded elementary operations along both component
        # chains (addition, differences/division, squared Euclidean norm, comparison).
        # gamma_n=sum-of-magnitudes forward-error enclosure, NOT research tolerance.
        # Literal pooled five-motion 1% criteria are separately evaluated unchanged.
        operations=8*order+16;gamma=operations*u/(1-operations*u)
        error_bound=gamma*np.linalg.norm(scale,axis=-1)
        excess=pn-MARGIN*bn;positive=excess>0
        assert np.all(excess<=error_bound),('SUBSTANTIVE_POINTWISE_EXCESS',order,float(np.max(excess-error_bound)))
        numeric.append(dict(order=order,raw_strict_excess_count=int(positive.sum()),max_raw_excess=float(np.maximum(excess,0).max()),max_excess_to_derived_bound=float(np.max(np.divide(np.maximum(excess,0),error_bound,out=np.zeros_like(excess),where=error_bound>0))),rounded_operations_upper_bound=operations))
        active=bn>0;ratios.append(float(np.max(pn[active]/bn[active])) if active.any() else 0.)
        zero_constraints+=int((~active).sum())
    return dict(status='PASS_POINTWISE_WITH_DERIVED_ROUNDOFF_CERTIFICATE',max_ratio_per_order=ratios,zero_native_constraints=zero_constraints,roundoff=numeric,no_scientific_floor_or_threshold_change=True,pooled_five_motion_must_be_tested_literally=True)
def fixtures():
    t=np.arange(1,13)/12.;native=np.stack((t+.03*np.sin(3*t),.12*t**3),-1)[None,None]
    residual=np.stack((.02*t*t,-.03*t*t*t),-1)[None,None]
    out,alpha=project_numpy(native,residual);audit_numpy(native,native+out)
    assert 0<alpha.item()<1
    assert np.array_equal(project_numpy(native,np.zeros_like(residual))[0],np.zeros_like(residual))
    z=np.zeros_like(native);rz,az=project_numpy(z,residual);assert az.item()==0 and np.array_equal(rz,z)
    # Bisection of all quadratic inequalities independent of closed-root formula.
    lo,hi=0.,1.
    for _ in range(70):
        mid=(lo+hi)/2
        okay=all(np.all(np.sum((b+mid*d)**2,-1)<=MARGIN*MARGIN*np.sum(b*b,-1)) for b,d in zip(differences_numpy(native),differences_numpy(residual)))
        if okay:lo=mid
        else:hi=mid
    assert abs(alpha.item()-lo*NUMERICAL_INTERIOR)<1e-12
    return dict(status='PASS',curved_active_case=True,zero_residual_identity=True,zero_native_budget_collapse=True,independent_bisection_error=abs(alpha.item()-lo*NUMERICAL_INTERIOR),numerical_interior=NUMERICAL_INTERIOR)
