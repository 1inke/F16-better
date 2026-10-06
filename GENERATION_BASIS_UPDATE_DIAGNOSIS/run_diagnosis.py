"""Independent completed-trial audit and FIT-only basis conditioning diagnosis."""
from pathlib import Path
import importlib.util,sys,json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;SOURCE=ROOT/'GENERATION_LOCAL_BASIS_SERIES'
spec=importlib.util.spec_from_file_location('basis_completed_source',SOURCE/'run_head.py')
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
def main():
    assert not (HERE/'RESULT.json').exists(),'Already complete'
    audit=json.loads((SOURCE/'preflight.json').read_text())
    for path,digest in audit['source_hashes'].items():assert h.p.sha256(Path(path))==digest,path
    assert json.loads((SOURCE/'results/exit_status.json').read_text())['status']=='SUCCESS'
    old=h.old_manifest();h.prepare_backend(torch.device('cpu'))
    locks=json.loads((SOURCE/'results/model_lock.json').read_text())['models'];assert len(locks)==18
    history=[]
    for path,digest in locks.items():
        assert h.p.sha256(Path(path))==digest,path
        saved=torch.load(path,map_location='cpu',weights_only=False);model=h.LocalHead(940,saved['config']);model.load_state_dict(saved['state_dict']);h.frozen_audit(model)
    train=h.p.load_npz(h.p.TRAIN_CACHE);cfg=h.c.settings(1);fit,cal=h.p.internal_split(len(train['sample_id']),cfg['calibration_fraction'])
    values=h.r.prepared(train,False,fit=fit)
    for arm in h.c.ARMS:
        for seed in (1,2,3):
            folder=SOURCE/f'results/models/{arm}/seed_{seed}/balanced_protected'
            resume=torch.load(folder/'resume.pth',map_location='cpu',weights_only=False);assert resume['epoch']==80
            selected=torch.load(folder/'constrained.pth',map_location='cpu',weights_only=False)
            history.append(dict(arm=arm,seed=seed,epoch_selected=selected['epoch'],trained_feasible=selected['trained_feasible'],history=resume['history'],final_readout_norm=float(resume['state_dict']['head.1.weight'].norm())))
    # Outcome-free allFIT covariance; no CAL/dev input is used to fit basis geometry.
    geometry={};device=torch.device('cuda:0')
    for arm in h.c.ARMS:
        model=h.LocalHead(940,h.c.settings(1,arm)).to(device).eval();total=torch.zeros(128,dtype=torch.float64,device=device);gram=torch.zeros(128,128,dtype=torch.float64,device=device);count=0
        with torch.no_grad():
            for start in range(0,len(fit),256):
                features=model.inputs(torch.from_numpy(values[0][fit[start:start+256]]).to(device));phi=model.head[0](features).reshape(-1,128).double()
                total+=phi.sum(0);gram+=phi.T@phi;count+=len(phi)
        mean=total/count;covariance=gram/count-mean[:,None]*mean[None];eigen=torch.linalg.eigvalsh(covariance).clamp_min(0)
        geometry[arm]=dict(n_candidate_points=count,mean_abs=float(mean.abs().mean()),centered_std_mean=float(covariance.diag().clamp_min(0).sqrt().mean()),eigen_min=float(eigen.min()),eigen_max=float(eigen.max()),eigen_median=float(eigen.median()),rank_above_relative_1e4=int((eigen>eigen.max()*1e-4).sum()),participation_rank=float(eigen.sum().square()/eigen.square().sum().clamp_min(1e-30)),condition_number_positive=float(eigen.max()/eigen[eigen>eigen.max()*1e-12].min()))
        destination=HERE/f'{arm}_fit_geometry.npz'
        arrays=dict(mean=mean.cpu().numpy(),covariance=covariance.cpu().numpy(),eigenvalues=eigen.cpu().numpy(),fit_ids=train['sample_id'][fit])
        if destination.exists():
            with np.load(destination) as prior:
                for key,value in arrays.items():np.testing.assert_allclose(prior[key],value,atol=1e-10,rtol=0)
        else:np.savez_compressed(destination,**arrays)
    # Independently recompute every saved prediction metric/explicit ID on dev, audit only.
    dev=h.p.load_npz(h.p.DEV_CACHE);dv=h.r.prepared(dev,False,mean=values[4],std=values[5]);base,speed,future=dv[1:4]
    paths=list((SOURCE/'results').glob('seed*.npz'));assert len(paths)==18
    for path in paths:
        saved=h.n.RAW_LOAD(path);assert np.array_equal(saved['sample_id'],dev['sample_id'])
        assert saved['trajectory'].shape==(1516,20,12,2) and np.isfinite(saved['trajectory']).all()
        actual=h.r.generation_metrics(saved['trajectory'],base,speed,future)
        for key,value in actual.items():np.testing.assert_allclose(saved[key],value,atol=1e-6,rtol=0,err_msg=str(path)+key)
    assert h.old_manifest()==old
    source={str(path):h.p.sha256(path) for path in [Path(__file__),SOURCE/'run_head.py',SOURCE/'protocol.json',SOURCE/'fit_basis.npz',SOURCE/'results/summary.json']}
    (HERE/'RESULT.json').write_text(json.dumps(dict(status='PASS_AUDIT_AND_DESCRIPTIVE_FIT_GEOMETRY',models_SHA_frozen=18,predictions_ID_finite_K20_allmetrics=18,old_files_unchanged=old,source_hashes=source,FIT_only_geometry=True,optimizer_steps=0,geometry=geometry,training_history=history,limitations='Dev read only for independent audit of already completed trial, never geometry fit. Covariance/effective rank descriptive, not proof conditioning is main cause or whitening benefits; loss/min-winner and weight-decay geometry also matter.'),indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',geometry=geometry,models=18,predictions=18,oldfiles=old)),flush=True)
if __name__=='__main__':main()
