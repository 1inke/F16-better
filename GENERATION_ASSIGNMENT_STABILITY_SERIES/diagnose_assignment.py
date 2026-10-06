"""FIT/CAL-only assignment diagnosis; no optimizer, no DEV/test."""
from pathlib import Path
import importlib.util,sys,json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'GENERATION_SOFT_SET_SERIES'
spec=importlib.util.spec_from_file_location('assignment_soft_source',PARENT/'run_head.py')
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
if __name__=='__main__':
    _,_,original=h.n.prepare_backend(torch.device('cpu'));del original
    lock=json.loads((PARENT/'results/model_lock.json').read_text())['models']
    for path,digest in lock.items():assert h.p.sha256(Path(path))==digest
    assert json.loads((PARENT/'results/independent_audit.json').read_text())['status']=='PASS'
    h.old_manifest()
    cache=h.p.load_npz(h.p.TRAIN_CACHE);fit,cal=h.p.internal_split(len(cache['sample_id']),h.c.settings(1)['calibration_fraction'])
    rng=np.random.default_rng(20261004);selected=dict(fit=rng.choice(fit,1024,replace=False),cal=rng.choice(cal,512,replace=False))
    values=h.r.prepared(cache,False,fit=fit);rows=[]
    for arm in ('zero','mean_std'):
        for seed in (1,2,3):
            path=PARENT/'results/models'/arm/('seed_'+str(seed))/'balanced_protected/constrained.pth'
            saved=torch.load(path,map_location='cpu',weights_only=False);model=h.SetHead(940,saved['config']).to('cuda:0').eval();model.load_state_dict(saved['state_dict']);h.a.frozen_audit(model)
            for split,ids in selected.items():
                out=h.r.g.predict(model,*[v[ids] for v in values[:3]],torch.device('cuda:0'));base=values[1][ids];gt=values[3][ids]
                def quality(x):
                    error=np.linalg.norm(x-gt[:,None],axis=-1);return error.mean(-1)+saved['config']['fde_loss_weight']*error[:,:,-1]
                q=quality(out);qb=quality(base);nativewin=qb.argmin(-1);learnedwin=q.argmin(-1)
                logits=-(q-q.min(-1,keepdims=True))/h.TAU;weights=np.exp(logits);weights/=weights.sum(-1,keepdims=True)
                joint_best=q.min(-1);joint_fixed=q[np.arange(len(ids)),nativewin]
                for group,mask in [('overall',np.ones(len(ids),bool))]+[(name,cache['change_group'][ids]==i) for i,name in enumerate(('stable','transition','change'))]:
                    if mask.any():rows.append(dict(arm=arm,seed=seed,epoch=saved['epoch'],split=split,group=group,n=int(mask.sum()),native_assignment_switch=float((nativewin[mask]!=learnedwin[mask]).mean()),native_assignment_extra_quality=float((joint_fixed[mask]-joint_best[mask]).mean()),soft_effective_count=float((1/(weights[mask]**2).sum(-1)).mean()),soft_weight_on_native_winner=float(weights[np.arange(len(ids)),nativewin][mask].mean())))
    sources={str(Path(__file__).resolve()):h.p.sha256(Path(__file__).resolve()),str(PARENT/'results/summary.json'):h.p.sha256(PARENT/'results/summary.json'),**lock}
    report=dict(status='PASS_FIT_CAL_DESCRIPTIVE_ONLY',no_optimizer=True,no_dev_test=True,subset_seed=20261004,sample_ids={k:cache['sample_id'][v].tolist() for k,v in selected.items()},rows=rows,source_hashes=sources,old_files_unchanged=h.old_manifest(),limitation='Assignment switches are not proof of harmful training switching. Fixednative labels may preserve wrong candidate assignment; no performance guarantee.')
    (HERE/'ASSIGNMENT_DIAGNOSIS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps([r for r in rows if r['split']=='fit' and r['group']=='change'],indent=2))
