"""Actual frozen completion arithmetic smoke, no optimizer."""
import run_head as h
import physical_population_audit as q
import torch,numpy as np,json
if __name__=='__main__':
    before=json.loads((h.HERE/'preflight.json').read_text(encoding='utf-8'));h.verify(before['source_hashes'])
    _,_,original=h.n.prepare_backend(torch.device('cpu'));del original
    device=torch.device('cuda:0');cache,cfg,fit,cal,values=h.prepared_train(device);idx=fit[:128]
    o,b,s,gt=[torch.from_numpy(v[idx]).to(device) for v in values[:4]]
    g=torch.tensor(cache['change_group'][idx],device=device);scale=torch.tensor(before['fit_protection_scales'],device=device)
    reports=[]
    with torch.no_grad():
        for arm in h.c.ARMS:
            h.p.seed_everything(1);cfg=h.c.settings(1,arm);model=h.TimeResponseHead(940,cfg).to(device).eval()
            model.head[-1].weight.copy_(torch.linspace(-.002,.002,model.head[-1].weight.numel(),device=device).reshape_as(model.head[-1].weight))
            out,res=model(o,b,s);parts=q.parts(h,out,b,gt,res,s,g,cfg,scale)
            arrays={k:v.cpu().numpy() for k,v in parts.items()}
            pop=q.scalar(arrays,cfg,model.physical_accel_scales.cpu().numpy(),arm)
            orig=float(h.original_objective(out,b,gt,res,s,g,cfg,'balanced_protected',scale))
            actualP=float(h.physical.penalty(out,b,g,model.physical_accel_scales))
            actual=float(h.m.objective(out,b,gt,res,s,g,cfg,'balanced_protected',scale))
            assert abs(pop['original_total']-orig)<2e-6
            assert abs(pop['P']-actualP)<2e-6 and abs(pop['added_total']-actual)<2e-6
            native_error=np.max(abs(h.physical.numpy_accel(b.cpu().numpy().astype(np.float64))-parts['native_accel'].cpu().numpy()))
            assert native_error<2e-5
            h.frozen_audit(model)
            reports.append(dict(arm=arm,original_scalar_error=abs(pop['original_total']-orig),
                P_error=abs(pop['P']-actualP),added_scalar_error=abs(pop['added_total']-actual),numpy_physical_error=float(native_error)))
            del model
    record=dict(status='PASS',actual128FIT_botharms=True,completion_original_and_added_population_components=reports,no_optimizer=True,no_saved_model=True)
    h.m.write_json(h.HERE/'prelaunch_checks/COMPLETION_MATH.json',record)
    before['completion_math_smoke']=record;before['source_hashes']=h.source_hashes()
    h.m.write_json(h.HERE/'preflight.json',before);h.guard.save('COMPLETION_MATH_PASS');print(json.dumps(record),flush=True)
