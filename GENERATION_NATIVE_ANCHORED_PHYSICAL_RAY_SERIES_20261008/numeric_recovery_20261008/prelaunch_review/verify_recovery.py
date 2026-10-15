"""Fresh current-source review, fixed saved fixtures and active-branch FD. Not formal training."""
import sys,copy
sys.dont_write_bytecode=True
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import run as r,preflight as p
import numpy as np,torch
def main():
    r.guard();files=r.read(r.H/'source_manifest.json')['files'];r.q.verify(files);records=[]
    for arm in r.ARMS:
        z=r.q.npz(r.H/'prelaunch_review'/f'{arm}_smoke128.npz')
        a,x=r.evaluate(z);b,y=r.evaluate(z,True)
        assert r.v.compare(a,b)==0 and all(np.array_equal(x[k],y[k]) for k in x)
        fallback=int(z['projection_fallback'].sum()) if arm=='physical' else 0
        records.append(dict(arm=arm,saved_rows=128,full_quality_complete_link_five_motion_error=0,numerical_fallback_candidates=fallback,candidates=512))
    original=r.q.npz(r.H/'original_failure_reproduction.npz')
    rejected=~r.pr.feasibility_numpy(original['reference4'],original['refined4'])
    assert int(rejected.sum())==2
    repaired=np.where(rejected[...,None,None],original['reference4'],original['refined4'])
    r.pr.audit_numpy(original['reference4'],repaired)
    ns,data,sp=r.data_load();st=r.statistics(data,sp['grouped_train_index']);ix=sp['grouped_train_index'][:256]
    f=torch.from_numpy(r.features(data,ix)).to(r.DEVICE);a=torch.from_numpy(data['anchor'][ix]).to(r.DEVICE);cv=torch.from_numpy(data['CV'][ix]).to(r.DEVICE);gt=torch.from_numpy(data['gt'][ix]).to(r.DEVICE)
    r.q.seed_all(42);m=r.Refiner(st).to(r.DEVICE);o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    for step in range(8):
        out,_=m(f,a,cv,'physical');loss=r.q.objective(out,gt)[0];o.zero_grad(set_to_none=True);loss.backward();o.step()
    out,_=m(f,a,cv,'physical');winner=r.q.objective(out,gt)[-1];alpha=m.last_projection['applied_alpha']
    eligible=torch.where(alpha.gather(1,winner[:,None]).squeeze(1)>0)[0]
    assert len(eligible)>0
    chosen=eligible[torch.argsort(alpha.gather(1,winner[:,None]).squeeze(1)[eligible],descending=True)[:16]];c=copy.deepcopy(m).double();before={k:v.detach().clone() for k,v in c.state_dict().items()};param=c.net[-1].bias
    out,_=c(f[chosen].double(),a[chosen].double(),cv[chosen].double(),'physical');loss=r.q.objective(out,gt[chosen].double())[0]
    grad=torch.autograd.grad(loss,param)[0];norm=float(grad.norm());assert norm>1e-12
    direction=grad/grad.norm();analytic=float((grad*direction).sum());saved=param.detach().clone();basewinner=r.q.objective(out,gt[chosen].double())[-1]
    fd=[]
    for eps in (1e-4,1e-5):
        with torch.no_grad():param.copy_(saved+eps*direction)
        up,_=c(f[chosen].double(),a[chosen].double(),cv[chosen].double(),'physical');lu=r.q.objective(up,gt[chosen].double())
        with torch.no_grad():param.copy_(saved-eps*direction)
        dn,_=c(f[chosen].double(),a[chosen].double(),cv[chosen].double(),'physical');ld=r.q.objective(dn,gt[chosen].double())
        value=float(((lu[0]-ld[0])/(2*eps)).detach());error=abs(value-analytic);switch=int(((lu[-1]!=basewinner)|(ld[-1]!=basewinner)).sum())
        print('FD_DIAGNOSTIC',dict(eps=eps,analytic=analytic,finite_difference=value,error=error,winner_switch=switch,selected_alpha=alpha.gather(1,winner[:,None]).squeeze(1)[chosen].detach().cpu().tolist()),flush=True)
        assert switch==0 and error<=1e-7+1e-3*abs(analytic)
        fd.append(dict(eps=eps,analytic=analytic,finite_difference=value,error=error,winner_switch=switch))
    with torch.no_grad():param.copy_(saved)
    assert all(torch.equal(v,before[k]) for k,v in c.state_dict().items())
    r.q.verify(files);r.guard()
    r.write('prelaunch_review/FINAL_SOURCE_REVIEW.json',dict(status='PASS',locked_files=len(files),source_sha=r.digest(files),review_script_sha=r.sha(__file__),all_locked_current_sources_initial_final_unchanged=True,saved_smoke=records,original_two_failed_candidates_conservatively_repaired_strictly=True,actual_eligible_primary_windows=int(len(eligible)),nonzero_active_branch_FD=fd,all_values_restored_EXACT=True,preflight_initial16_physical_FD_was_zero_not_informative_nonzero_proof=True,no_old_model_training_or_formal_count=True,resource=r.q.RESOURCE,not_external_review=True))
    print('FINAL_SOURCE_REVIEW_PASS',records,'active_primary_windows',len(eligible),'FD',fd,flush=True)
if __name__=='__main__':main()
