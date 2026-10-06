"""Fixed synthetic active-output and actual reject branches; no optimizer."""
import run_head as h
import torch,numpy as np,json,shutil
from types import SimpleNamespace
from pathlib import Path
if __name__=='__main__':
    before=json.loads((h.HERE/'preflight.json').read_text(encoding='utf-8'))
    h.verify(before['source_hashes'])
    h.m.write_json(h.HERE/'preflight_before_active_extension.json',before)
    _,_,original=h.n.prepare_backend(torch.device('cpu'));del original
    cache,cfg,fit,cal,values=h.prepared_train(torch.device('cuda:0'))
    changed={**cache,'sample_id':cache['sample_id'].copy()};changed['sample_id'][fit[0]]+=1
    try:h.physical.prepare(h,changed,fit,values[1])
    except AssertionError:badID=True
    else:raise AssertionError('bad IDs accepted')
    fixture=h.HERE/'prelaunch_checks/badscale';fixture.mkdir(parents=True,exist_ok=True)
    with np.load(h.HERE/'physical_accel_scales.npz',allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
    arr['scales'][0]=float('nan');np.savez(fixture/'physical_accel_scales.npz',**arr)
    meta=json.loads((h.HERE/'physical_accel_scales.json').read_text(encoding='utf-8'))
    meta['npz_SHA']=h.p.sha256(fixture/'physical_accel_scales.npz')
    h.m.write_json(fixture/'physical_accel_scales.json',meta)
    proxy=SimpleNamespace(HERE=fixture,p=h.p,m=h.m)
    try:h.physical.prepare(proxy,cache,fit,values[1])
    except AssertionError:badscale=True
    else:raise AssertionError('bad scaler accepted')
    device=torch.device('cuda:0');idx=fit[:128]
    o,s,gt=[torch.from_numpy(values[j][idx]).to(device) for j in (0,2,3)]
    base=torch.zeros((128,20,12,2),device=device)
    group=torch.tensor(cache['change_group'][idx],device=device)
    pscale=torch.tensor(before['fit_protection_scales'],device=device)
    h.p.seed_everything(1);model=h.TimeResponseHead(940,h.c.settings(1,'mean_std')).to(device).eval()
    with torch.no_grad():model.head[-1].bias.copy_(torch.tensor([0.,0.,.5,-.25,0.,0.],device=device))
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    output,res=model(o,base,s)
    penalty,details=h.physical.penalty(output,base,group,model.physical_accel_scales,True)
    assert penalty>0 and any(details['active'])
    params=[v for v in model.parameters() if v.requires_grad]
    vector=torch.autograd.grad(penalty,params,retain_graph=True,allow_unused=True)
    norm=sum(float(v.square().sum()) for v in vector if v is not None)**.5
    assert norm>0 and np.isfinite(norm)
    original_loss=h.original_objective(output,base,gt,res,s,group,h.c.settings(1,'mean_std'),'balanced_protected',pscale)
    control=h.m.objective(output,base,gt,res,s,group,h.c.settings(1,'zero'),'balanced_protected',pscale)
    treatment=h.m.objective(output,base,gt,res,s,group,h.c.settings(1,'mean_std'),'balanced_protected',pscale)
    origgrad=torch.autograd.grad(original_loss,params,retain_graph=True,allow_unused=True)
    ctrlgrad=torch.autograd.grad(control,params,retain_graph=True,allow_unused=True)
    treatgrad=torch.autograd.grad(treatment,params,retain_graph=True,allow_unused=True)
    maxerr=0.
    for a,b,c,v,param in zip(origgrad,ctrlgrad,treatgrad,vector,params):
        a=torch.zeros_like(param) if a is None else a;b=torch.zeros_like(param) if b is None else b
        c=torch.zeros_like(param) if c is None else c;v=torch.zeros_like(param) if v is None else v
        assert torch.equal(a,b),'control extra grad notzero'
        err=float((c-(a+.1*v)).abs().max());maxerr=max(maxerr,err)
        assert torch.allclose(c,a+.1*v,atol=1e-6,rtol=2e-4)
    h.frozen_audit(model)
    for key,v in model.state_dict().items():assert torch.equal(v.detach().cpu(),state[key]),key
    record=dict(status='PASS',synthetic_only=True,natural_8steps_hinge_activation=before['response'],
        synthetic_base_zero=True,fixed_bias=[0,0,.5,-.25,0,0],actual_GPU_parameter_P_norm=norm,
        control_added_parameter_gradient_zero=True,treatment_gradient_decomposition_max=maxerr,
        penalty=float(penalty.detach()),groups=details,bad_ID_rejected=badID,bad_scaler_rejected=badscale,
        no_optimizer=True,no_saved_model=True,all_old_weights_buffers_unchanged=True)
    h.old_manifest()
    h.m.write_json(h.HERE/'prelaunch_checks/ACTIVE_AND_REJECT.json',record)
    before['active_synthetic_parameter_check']=record
    before['source_hashes']=h.source_hashes()
    h.m.write_json(h.HERE/'preflight.json',before)
    h.guard.save('ACTIVE_OUTPUT_REJECT_PASS');print(json.dumps(record),flush=True)
