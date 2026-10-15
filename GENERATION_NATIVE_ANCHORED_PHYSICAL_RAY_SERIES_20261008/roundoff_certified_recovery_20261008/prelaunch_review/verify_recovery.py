"""Current-source saved audits + explicit nonzero derivative convergence; no formal training."""
import sys,copy,importlib.util
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import run as r
import numpy as np,torch
def main():
    r.guard();files=r.read(r.H/'source_manifest.json')['files'];r.q.verify(files);records=[]
    for arm in r.ARMS:
        z=r.q.npz(r.H/'prelaunch_review'/f'{arm}_smoke128.npz')
        a,x=r.evaluate(z);b,y=r.evaluate(z,True);assert r.v.compare(a,b)==0 and all(np.array_equal(x[k],y[k]) for k in x)
        records.append(dict(arm=arm,rows=128,saved_full_quality_complete_link_five_motion_error=0,temporal=r.temporal_audit(z)))
    original=r.q.npz(r.H.parent/'numeric_recovery_20261008/original_failure_reproduction.npz')
    certificate=r.pr.audit_numpy(original['reference4'],original['refined4'])
    assert sum(x['raw_strict_excess_count'] for x in certificate['roundoff'])==2
    rejected=False
    try:r.pr.audit_numpy(np.zeros((1,1,12,2)),np.ones((1,1,12,2))*1e-7)
    except AssertionError:rejected=True
    assert rejected
    ns,data,sp=r.data_load();st=r.statistics(data,sp['grouped_train_index']);ix=sp['grouped_train_index'][:256]
    f=torch.from_numpy(r.features(data,ix)).to(r.DEVICE);a=torch.from_numpy(data['anchor'][ix]).to(r.DEVICE);cv=torch.from_numpy(data['CV'][ix]).to(r.DEVICE);gt=torch.from_numpy(data['gt'][ix]).to(r.DEVICE)
    r.q.seed_all(42);m=r.Refiner(st).to(r.DEVICE);o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    for step in range(8):
        out,_=m(f,a,cv,'physical');loss=r.q.objective(out,gt)[0];o.zero_grad(set_to_none=True);loss.backward();o.step()
    out,res,u=m(f,a,cv,'physical',True);winner=r.q.objective(out,gt)[-1]
    proposed=torch.einsum('ti,bkid->bktd',m.time_basis,u)*(.20*(torch.arange(1,13,device=r.DEVICE,dtype=torch.float64)/12)**2)[None,None,:,None]
    spec=importlib.util.spec_from_file_location('preserved_parent_projection',r.H.parent/'physical_ray.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    p,alpha=r.pr.project_torch(a.double(),proposed);op,oa=old.project_torch(a.double(),proposed)
    assert torch.equal(p,op) and torch.equal(alpha,oa)
    aw=alpha.gather(1,winner[:,None]).squeeze(1);eligible=torch.where(aw>0)[0];chosen=eligible[torch.argsort(aw[eligible],descending=True)[:16]]
    c=copy.deepcopy(m).double();before={k:v.detach().clone() for k,v in c.state_dict().items()};param=c.net[-1].bias
    def calc():
        out,_,u=c(f[chosen].double(),a[chosen].double(),cv[chosen].double(),'physical',True)
        proposal=torch.einsum('ti,bkid->bktd',c.time_basis,u)*(.20*(torch.arange(1,13,device=r.DEVICE,dtype=torch.float64)/12)**2)[None,None,:,None]
        faces=torch.cat([r.pr.root_torch(b,d) for b,d in zip(r.pr.differences_torch(a[chosen].double()),r.pr.differences_torch(proposal))],-1).argmin(-1)
        return r.q.objective(out,gt[chosen].double()),faces
    base,faces=calc();grad=torch.autograd.grad(base[0],param)[0];assert grad.norm()>1e-12
    direction=grad/grad.norm();analytic=float((grad*direction).sum());saved=param.detach().clone();fd=[]
    for eps in (1e-4,1e-5,1e-6,1e-7):
        with torch.no_grad():param.copy_(saved+eps*direction)
        up,uf=calc()
        with torch.no_grad():param.copy_(saved-eps*direction)
        down,df=calc()
        value=float(((up[0]-down[0])/(2*eps)).detach());error=abs(value-analytic);switch=int(((up[-1]!=base[-1])|(down[-1]!=base[-1])).sum());face_switch=int(((uf!=faces)|(df!=faces)).sum())
        fd.append(dict(eps=eps,analytic=analytic,finite_difference=value,error=error,winner_switch=switch,ray_constraint_face_switch=face_switch,original_error_rule_pass=error<=1e-7+1e-3*abs(analytic)))
    assert all(x['original_error_rule_pass'] and x['winner_switch']==0 for x in fd[-2:])
    with torch.no_grad():param.copy_(saved)
    assert all(torch.equal(v,before[k]) for k,v in c.state_dict().items())
    r.q.verify(files);r.guard()
    r.write('prelaunch_review/FINAL_SOURCE_REVIEW.json',dict(status='PASS_TECHNICAL_REVIEW_NOT_FORMAL_RESULT',source_sha=r.digest(files),locked_files=len(files),review_script_sha=r.sha(__file__),source_initial_final_unchanged=True,saved_smoke=records,exact_original_two_case_certificate=certificate,substantive_defect_rejected=True,production_projection_bitwise_EXACT_parent=True,nonzero_active_primary_windows=int(len(eligible)),nonzero_gradient_FD_convergence=fd,nominal_larger_steps_not_claimed_nonzero_FD_PASS=True,first16_physical_nominal_FD_zero_not_nonzero_proof=True,all_double_copy_weights_buffers_restored_EXACT=True,resource=r.q.RESOURCE,no_formal_training_or_old_model_access=True,not_external_review=True))
    print('TECHNICAL_REVIEW_PASS',fd,flush=True)
if __name__=='__main__':main()

