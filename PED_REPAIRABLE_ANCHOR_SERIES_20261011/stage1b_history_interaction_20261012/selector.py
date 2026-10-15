"""Train-only linear envelope-potential probe, no neural/GPU optimizer."""
import common as c
c.guard()
import tempfile
_task_temp=c.H/'tmp';_task_temp.mkdir(exist_ok=True);tempfile.tempdir=str(_task_temp)
import warnings,time,gc
import numpy as np,sklearn
from numerical_audit import certificate
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
PARAMS=dict(C=1.,penalty="l2",class_weight="balanced",solver="newton-cholesky",tol=1e-6,max_iter=500)
def fit(d,ix,arm):
    c.guard();t=time.perf_counter();mu,sd=c.normalizer(d,ix,arm)
    a=c.x(d,ix,arm).reshape(-1,124 if arm=="additive" else 136).astype(np.float64);a=(a-mu)/sd
    y=c.label(d,ix)["potential"].reshape(-1).astype(np.int8);assert len(np.unique(y))==2
    m=LogisticRegression(**PARAMS)
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always");solver_started=time.perf_counter();m.fit(a,y);solver_seconds=time.perf_counter()-solver_started
    report=dict(solver_seconds=solver_seconds,elapsed_seconds=time.perf_counter()-t,n_windows=len(ix),n_candidate_rows=len(y),positive_candidates=int(y.sum()),iterations=int(m.n_iter_[0]),warnings=[str(x.message) for x in ws],sklearn=sklearn.__version__,parameters=PARAMS,threads=1)
    c.guard()
    if any(issubclass(x.category,ConvergenceWarning) for x in ws) or m.n_iter_[0]>=500:
        raise RuntimeError("STOP_SELECTOR_CONVERGENCE_"+str(report))
    cert=certificate(a,y,m.coef_[0],float(m.intercept_[0]),PARAMS["C"])
    assert cert["gradient_inf"]<=PARAMS["tol"],("STOP_INDEPENDENT_STATIONARITY",cert)
    report["independent_objective_certificate"]=cert
    assert m.classes_.tolist()==[0,1] and np.isfinite(m.coef_).all() and np.isfinite(m.intercept_).all()
    # Actual sklearn probabilities against standalone sigmoid on same real features.
    p=m.predict_proba(a[:256])[:,1];t0=a[:256]@m.coef_[0]+m.intercept_[0]
    pp=np.empty_like(t0);pos=t0>=0;pp[pos]=1/(1+np.exp(-t0[pos]));et=np.exp(t0[~pos]);pp[~pos]=et/(1+et)
    assert np.max(abs(p-pp))<=1e-12
    obj=dict(mean=mu,std=sd,coef=m.coef_[0].copy(),intercept=m.intercept_.copy(),classes=m.classes_.copy(),fit_ids=d["ids"][ix].copy())
    del a,m,y;gc.collect();return obj,report
def predict(d,ix,model,arm,history_index=None):return c.infer(d,ix,model,arm,history_index)
def manifest(model,ix,d,source_sha,arm):
    return dict(source_manifest_sha=source_sha,protocol_sha=c.sha(c.H/"protocol.json"),split_registry_sha=c.sha(c.H/"split_registry.npz"),labels_sha=c.sha(c.H/"labels_offline.npz"),fit_ids_sha=c.digest(d["ids"][ix].tolist()),n_fit_windows=len(ix),arm=arm,features=124 if arm=="additive" else 136,parameters=PARAMS,sklearn=sklearn.__version__,threads=1,labels_not_forward_inputs=True,kind="CPU_NEWTON_probe_not_torch_epoch_resume")
