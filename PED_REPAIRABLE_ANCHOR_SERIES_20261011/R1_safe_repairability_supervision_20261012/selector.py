"""Matched weighted linear selector; no class-weight double counting."""
import common as c
c.guard()
import tempfile
(c.H/"tmp").mkdir(exist_ok=True);tempfile.tempdir=str(c.H/"tmp")
import numpy as np,warnings,time
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from labels import supervision
from scipy.special import expit
PARAMS=dict(C=1.,penalty="l2",class_weight=None,solver="newton-cholesky",tol=1e-6,max_iter=500)
def objective(X,y,sw,w,b):
    S=sw.sum();z=X@w+b;r=sw*(expit(z)-y)
    f=float(sw@(np.logaddexp(0,z)-y*z)/S+(w@w)/(2*S));g=np.r_[(X.T@r+w)/S,r.sum()/S];return f,g
def fit(X,e,y,tracks,mode,mean,std):
    target,known,weight=supervision(e,y,tracks,mode);z=((X.astype(float)-mean)/std)[known];target=target[known];sw=weight[known];t=time.perf_counter()
    model=LogisticRegression(**PARAMS)
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always");model.fit(z,target,sample_weight=sw)
    assert not any(issubclass(w.category,ConvergenceWarning) for w in caught) and model.n_iter_[0]<500,"STOP_CONVERGENCE"
    f,g=objective(z,target,sw,model.coef_[0],float(model.intercept_[0]));assert abs(g).max()<=1e-6,("STOP_WEIGHTED_STATIONARITY",float(abs(g).max()))
    assert model.classes_.tolist()==[0,1];obj=dict(coef=model.coef_[0].copy(),intercept=model.intercept_.copy(),mean=mean.copy(),std=std.copy(),known=known,weight=weight,target=target)
    report=dict(iterations=int(model.n_iter_[0]),seconds=time.perf_counter()-t,objective=f,gradient_inf=float(abs(g).max()),warnings=[str(w.message) for w in caught],total_weight=float(sw.sum()),positive_weight=float(sw[target==1].sum()),negative_weight=float(sw[target==0].sum()),known_candidates=len(target),train_windows=len(X),train_tracks=len(np.unique(tracks)),parameters=PARAMS)
    return obj,report
def scores(X,m):return ((X.astype(float)-m["mean"])/m["std"])@m["coef"]+float(m["intercept"][0])
def numerical_fixture(X,e,y,tracks,mean,std):
    from scipy.optimize import minimize
    audits=[]
    for mode in ("envelope","witness"):
        m,rep=fit(X,e,y,tracks,mode,mean,std);again,_=fit(X,e,y,tracks,mode,mean,std)
        assert all(np.array_equal(m[k],again[k]) for k in m),"REPEAT_FIT_MISMATCH"
        target,known,weight=supervision(e,y,tracks,mode);z=((X.astype(float)-mean)/std)[known];target=target[known];sw=weight[known]
        def fg(v):return objective(z,target,sw,v[:-1],v[-1])
        def hh(v):
            p=expit(z@v[:-1]+v[-1]);aug=np.c_[z,np.ones(len(z))];h=aug.T@((sw*p*(1-p))[:,None]*aug)/sw.sum();h[:-1,:-1]+=np.eye(136)/sw.sum();return h
        with threadpool_limits(limits=1):rr=minimize(fg,np.zeros(137),jac=True,hess=hh,method="trust-exact",options=dict(gtol=1e-9,maxiter=500))
        loss,gr=fg(rr.x);assert abs(gr).max()<=1e-6 and abs(loss-rep["objective"])<=1e-9
        probe=np.sin(np.arange(137)*.17)*.03;f,g=fg(probe);h=hh(probe);errs=[]
        for k in range(3):
            u=np.cos(np.arange(137)*(.11+k*.07));u/=np.linalg.norm(u);eps=1e-5
            ge=abs((fg(probe+eps*u)[0]-fg(probe-eps*u)[0])/(2*eps)-g@u);he=abs((fg(probe+eps*u)[1]-fg(probe-eps*u)[1])/(2*eps)-h@u).max()
            assert ge<=1e-8 and he<=1e-8;errs.append([float(ge),float(he)])
        audits.append(dict(mode=mode,fit=rep,repeat_exact=True,reference_gradient=float(abs(gr).max()),objective_gap=abs(loss-rep["objective"]),FD=errs,fixture_labels_not_actual_safe_labels=True))
    return audits
