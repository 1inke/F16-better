"""GT-free native global similarity geometry. Fixed scientific 1.01 gates."""
import numpy as np
import torch
import physical_ray as pr
MARGIN=1.01
INTERIOR=1.-2.**-20
def project_numpy(native,proposal):
    p=np.asarray(native,np.float64);r=np.asarray(proposal,np.float64)
    assert p.shape==r.shape and p.shape[-2:]==(12,2) and np.isfinite(p).all() and np.isfinite(r).all()
    energy=(p*p).sum((-2,-1));den=np.where(energy>0,energy,1.)
    a=1+(p*r).sum((-2,-1))/den;b=(p[...,0]*r[...,1]-p[...,1]*r[...,0]).sum(-1)/den
    a=np.where(energy>0,a,1.);b=np.where(energy>0,b,0.)
    scale=np.hypot(a,b);factor=np.where(scale>MARGIN,MARGIN*INTERIOR/np.where(scale>0,scale,1.),1.)
    ca=a*factor;cb=b*factor;Jp=np.stack((-p[...,1],p[...,0]),-1)
    fit=(ca-1)[...,None,None]*p+cb[...,None,None]*Jp
    norm=np.linalg.norm(fit,axis=-1);bounds=.20*(np.arange(1,13)/12.)**2
    ratio=np.divide(bounds,norm,out=np.ones_like(norm),where=norm>0)
    lam=np.minimum(1.,ratio.min(-1));lam=np.where(lam<1,lam*INTERIOR,lam)
    residual=fit*lam[...,None,None]
    fa=1+lam*(ca-1);fb=lam*cb
    return residual,dict(fit_a=a,fit_b=b,scale_before=scale,capped_a=ca,capped_b=cb,position_lambda=lam,final_a=fa,final_b=fb,final_scale=np.hypot(fa,fb),zero_energy=energy==0)
def project_torch(p,r):
    assert p.dtype==r.dtype==torch.float64
    energy=(p*p).sum((-2,-1));den=torch.where(energy>0,energy,torch.ones_like(energy))
    a=1+(p*r).sum((-2,-1))/den;b=(p[...,0]*r[...,1]-p[...,1]*r[...,0]).sum(-1)/den
    a=torch.where(energy>0,a,torch.ones_like(a));b=torch.where(energy>0,b,torch.zeros_like(b))
    scale=torch.linalg.vector_norm(torch.stack((a,b),-1),dim=-1)
    factor=torch.where(scale>MARGIN,MARGIN*INTERIOR/torch.where(scale>0,scale,torch.ones_like(scale)),torch.ones_like(scale))
    ca=a*factor;cb=b*factor;Jp=torch.stack((-p[...,1],p[...,0]),-1)
    fit=(ca-1)[...,None,None]*p+cb[...,None,None]*Jp
    norm=torch.linalg.vector_norm(fit,dim=-1);bounds=.20*(torch.arange(1,13,device=p.device,dtype=p.dtype)/12.)**2
    ratio=torch.where(norm>0,bounds/torch.where(norm>0,norm,torch.ones_like(norm)),torch.ones_like(norm))
    lam=torch.minimum(torch.ones_like(energy),ratio.amin(-1));lam=torch.where(lam<1,lam*INTERIOR,lam)
    residual=fit*lam[...,None,None];fa=1+lam*(ca-1);fb=lam*cb
    return residual,dict(fit_a=a,fit_b=b,scale_before=scale,capped_a=ca,capped_b=cb,position_lambda=lam,final_a=fa,final_b=fb,final_scale=torch.linalg.vector_norm(torch.stack((fa,fb),-1),dim=-1),zero_energy=energy==0)
def audit_numpy(p,path,geometry):
    p=np.asarray(p,np.float64);path=np.asarray(path,np.float64)
    fa=geometry['final_a'];fb=geometry['final_b'];Jp=np.stack((-p[...,1],p[...,0]),-1)
    expected=fa[...,None,None]*p+fb[...,None,None]*Jp
    error=float(abs(expected-path).max());assert error<=2e-12,error
    assert np.all(geometry['final_scale']<=MARGIN)
    bound=.20*(np.arange(1,13)/12.)**2
    assert np.all(np.linalg.norm(path-p,axis=-1)<=bound+2e-15)
    physics=pr.audit_numpy(p,path)
    return dict(status='PASS',similarity_reconstruction_max=error,final_scale_max=float(geometry['final_scale'].max()),zero_energy_candidates=int(geometry['zero_energy'].sum()),position_lambda_quantiles=np.quantile(geometry['position_lambda'],[0,.1,.5,.9,1]),pointwise_roundoff_certificate=physics,not_new_turn_stop_shape=True,all_five_pooled_must_be_checked_literally=True)
def fixtures():
    t=np.arange(1,13,dtype=float);p=np.stack((t/8,np.zeros(12)),-1)[None,None]
    theta=.025;rot=np.stack((np.cos(theta)*p[...,0]-np.sin(theta)*p[...,1],np.sin(theta)*p[...,0]+np.cos(theta)*p[...,1]),-1)
    r,geo=project_numpy(p,rot-p);audit_numpy(p,p+r,geo)
    assert np.linalg.norm(r,axis=-1).max()>.001
    zero,zg=project_numpy(p,np.zeros_like(p));assert np.array_equal(zero,np.zeros_like(p)) and np.array_equal(zg['final_scale'],np.ones_like(zg['final_scale']))
    rz,zz=project_numpy(np.zeros_like(p),np.ones_like(p));assert not rz.any() and zz['zero_energy'].all()
    large,gg=project_numpy(p,p*.5);audit_numpy(p,p+large,gg);assert gg['scale_before'].min()>1.01 and gg['final_scale'].max()<=1.01
    short,sg=project_numpy(p,np.ones_like(p)*3);audit_numpy(p,p+short,sg);assert sg['position_lambda'].max()<1
    rejected=False
    try:pr.audit_numpy(np.zeros_like(p),np.ones_like(p)*1e-7)
    except AssertionError:rejected=True
    assert rejected
    return dict(status='PASS',rotation_constant_velocity_nonzero_max=float(np.linalg.norm(r,axis=-1).max()),identity_EXACT=True,zero_energy_identity=True,scale_cap=True,position_cap=True,substantive_zero_derivative_defect_rejected=True)
