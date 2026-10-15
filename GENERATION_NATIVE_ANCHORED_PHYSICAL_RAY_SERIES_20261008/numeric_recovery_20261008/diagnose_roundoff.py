"""Reproduce original tiny failed inequality; no production or formal training."""
import sys
sys.dont_write_bytecode=True
import run as r
import numpy as np,torch
def main():
    r.guard()
    r.q.verify(r.read(r.H.parent/'source_manifest.json')['files'])
    ns,data,sp=r.data_load();st=r.statistics(data,sp['grouped_train_index'])
    ix=sp['grouped_train_index'][:256]
    f=torch.from_numpy(r.features(data,ix)).to(r.DEVICE)
    a=torch.from_numpy(data['anchor'][ix]).to(r.DEVICE)
    c=torch.from_numpy(data['CV'][ix]).to(r.DEVICE)
    gt=torch.from_numpy(data['gt'][ix]).to(r.DEVICE)
    r.q.seed_all(42);m=r.Refiner(st).to(r.DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    for step in range(8):
        out,res=m(f,a,c,'physical');value=r.q.objective(out,gt)[0]
        opt.zero_grad(set_to_none=True);value.backward();opt.step()
    small=sp['outer_eval_index'][:128]
    out,res,u=r.generated(m,data,small,'physical',True)
    z=r.portfolio(data,small,out,res,'physical',u)
    np.savez_compressed(r.H/'original_failure_reproduction.npz',**z)
    report=[]
    eps=np.finfo(np.float64).eps
    native=z['reference4']
    for order,(b,p) in enumerate(zip(r.pr.differences_numpy(native),r.pr.differences_numpy(out)),1):
        bn=np.linalg.norm(b,axis=-1);pn=np.linalg.norm(p,axis=-1)
        excess=pn-1.01*bn;bad=excess>0
        # Absolute forward-error scale: finite-difference coefficients, not derivative magnitude.
        # A third difference of near-linear positions may cancel to ~zero.
        scale=np.abs(native)+np.abs(out)
        for k in range(order):
            scale=(scale[...,1:,:]+scale[...,:-1,:])/.4 if k else (scale+np.concatenate((np.zeros_like(scale[...,:1,:]),scale[...,:-1,:]),axis=-2))/.4
        bound=32*eps*np.linalg.norm(scale,axis=-1)
        indexes=np.argwhere(bad);cases=[]
        for i,j,t in indexes:
            cases.append(dict(sample_id=str(z['sample_id'][i]),window=int(i),selected_candidate=int(j),physical_slot=int(z['slots'][i,j]),difference_step=int(t),native_norm=float(bn[i,j,t]),actual_norm=float(pn[i,j,t]),absolute_excess=float(excess[i,j,t]),forward_error_scale_bound=float(bound[i,j,t]),alpha=float(z['projection_alpha'][i,j])))
        report.append(dict(order=order,violations=int(bad.sum()),max_excess=float(excess.max()),max_excess_to_bound=float(np.max(np.divide(np.maximum(excess,0),bound,out=np.zeros_like(excess),where=bound>0))),cases=cases))
    groups=[]
    for name,mask in r.v.masks(z['groups']):
        if not mask.any():continue
        native_motion=r.v.motion(z['native20'][mask]);refined_motion=r.v.motion(z['refined20'][mask])
        groups.append(dict(group=name,n=int(mask.sum()),native=native_motion,refined=refined_motion,strict_all5_1pct=all(refined_motion[k]<=1.01*native_motion[k] for k in native_motion)))
    r.q.verify(r.read(r.H.parent/'source_manifest.json')['files'])
    r.write('NUMERICAL_REPRODUCTION.json',dict(status='DIAGNOSTIC_ONLY',orders=report,pooled_groups=groups,fixture_sha=r.sha(r.H/'original_failure_reproduction.npz'),source_parent_1151_initial_final_unchanged=True,no_formal_training_or_old_model_access=True,resource=r.q.RESOURCE))
    print(report,flush=True);print(groups,flush=True)
if __name__=='__main__':main()
