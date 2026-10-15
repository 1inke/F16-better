"""Same balanced L2 objective in standalone NumPy; reference solver only preflight."""
import numpy as np,time
def sigmoid(z):
    ans=np.empty_like(z);m=z>=0;ans[m]=1/(1+np.exp(-z[m]));e=np.exp(z[~m]);ans[~m]=e/(1+e);return ans
def weights(y):
    n=len(y);ct=np.bincount(y.astype(int),minlength=2);assert ct.min()>0
    return n/(2.*ct[y.astype(int)])
def pieces(X,y,w,b,C=1.):
    sw=weights(y);S=sw.sum();z=X@w+b;p=sigmoid(z);rr=sw*(p-y)/S
    loss=float(np.dot(sw,np.logaddexp(0,z)-y*z)/S+np.dot(w,w)/(2*C*S))
    g=np.r_[X.T@rr+w/(C*S),rr.sum()]
    return loss,g,p,sw,S
def certificate(X,y,w,b,C=1.):
    f,g,_,_,S=pieces(X,y,w,b,C);assert np.isfinite(f) and np.isfinite(g).all()
    return dict(objective=f,gradient_inf=float(abs(g).max()),weight_sum=float(S),n=len(y),positive=int(y.sum()),intercept_unpenalized=True,L2_C=float(C))
def hessian(X,y,v,C=1.):
    w,b=v[:-1],v[-1];_,_,p,sw,S=pieces(X,y,w,b,C);cur=sw*p*(1-p)/S
    H=np.empty((X.shape[1]+1,X.shape[1]+1));H[:-1,:-1]=X.T@(cur[:,None]*X);H[:-1,:-1]+=np.eye(X.shape[1])/(C*S)
    hb=X.T@cur;H[:-1,-1]=hb;H[-1,:-1]=hb;H[-1,-1]=cur.sum();return H
def reference_audit(d,ix,model):
    import common as c
    from scipy.optimize import minimize
    from threadpoolctl import threadpool_limits
    t=time.perf_counter();X=c.x(d,ix).reshape(-1,120).astype(float);X=(X-model["mean"])/model["std"]
    y=c.label(d,ix)["potential"].reshape(-1).astype(np.int8);v=np.r_[model["coef"],model["intercept"]]
    def fg(v):f,g,*_=pieces(X,y,v[:-1],v[-1]);return f,g
    with threadpool_limits(limits=1):
        # Independently coded analytic Hessian and scipy trust-region, fixed zero init.
        opt=minimize(fg,np.zeros(121),jac=True,hess=lambda z:hessian(X,y,z),method="trust-exact",options=dict(gtol=1e-9,maxiter=500))
        rc=certificate(X,y,opt.x[:-1],float(opt.x[-1]));sc=certificate(X,y,v[:-1],float(v[-1]))
        assert rc["gradient_inf"]<=1e-6 and sc["gradient_inf"]<=1e-6
        gap=abs(rc["objective"]-sc["objective"]);assert gap<=1e-9,("OBJECTIVE_DISAGREEMENT",gap)
        fp=abs(sigmoid(X@v[:-1]+v[-1])-sigmoid(X@opt.x[:-1]+opt.x[-1]))
        assert fp.max()<=1e-3,("REFERENCE_PROBABILITY_DISAGREEMENT",float(fp.max()))
        # Derivative check away from optimum, fixed deterministic parameter/directions.
        probe=np.sin(np.arange(121)*.17)*.03;f,g=fg(probe);H=hessian(X,y,probe);errs=[];hers=[]
        for k in range(3):
            u=np.cos(np.arange(121)*(.11+k*.07));u/=np.linalg.norm(u);ep=1e-5
            fd=(fg(probe+ep*u)[0]-fg(probe-ep*u)[0])/(2*ep)
            hd=(fg(probe+ep*u)[1]-fg(probe-ep*u)[1])/(2*ep)
            errs.append(float(abs(fd-g@u)));hers.append(float(abs(hd-H@u).max()))
        assert max(errs)<=1e-8 and max(hers)<=1e-8
        ev=np.linalg.eigvalsh(hessian(X,y,v));assert ev.min()>0
    c.guard()
    return dict(status="PASS_SAME_OBJECTIVE_INDEPENDENT_STATIONARITY_AND_REFERENCE",solver="scipy_trust_exact_FIXED_NOT_DEPLOYED",reference_success_flag=bool(opt.success),reference_message=str(opt.message),reference_iterations=int(opt.nit),sklearn=sc,reference=rc,objective_difference=gap,probability_difference_max=float(fp.max()),FD_gradient_max=max(errs),FD_hessian_max=max(hers),Hessian_min_eigenvalue=float(ev.min()),Hessian_condition=float(ev.max()/ev.min()),not_H4_or_outcome_selection=True,elapsed_seconds=time.perf_counter()-t)
