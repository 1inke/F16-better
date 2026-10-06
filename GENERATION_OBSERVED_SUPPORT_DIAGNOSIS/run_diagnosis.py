"""Fixed FIT/CAL observed-neighbor future-support diagnosis; no optimizer/DEV/test."""
from pathlib import Path
import importlib.util,sys,json,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'GENERATION_PATH_MIXTURE_SERIES'
spec=importlib.util.spec_from_file_location('support_parent',PARENT/'run_head.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
def main():
    print('[DIAGNOSIS] source/hash/setup; no training',flush=True)
    assert json.loads((PARENT/'CYCLE_8_REVIEW.json').read_text())['status']=='PASS_COMPLETE_CYCLE_REVIEW'
    manifest=json.loads((PARENT/'old_results_manifest.json').read_text())['files']
    for file in PARENT.rglob('*'):
        if file.is_file() and '__pycache__' not in file.parts and file.name not in ('runtime_state.json','independent_worker.lock'):manifest[str(file)]=h.p.sha256(file)
    for path,digest in manifest.items():assert h.p.sha256(Path(path))==digest
    (HERE/'old_results_manifest.json').write_text(json.dumps(dict(files=manifest)),encoding='utf-8')
    _,_,original=h.BACKEND(torch.device('cpu'));del original
    cache=h.p.load_npz(h.p.TRAIN_CACHE);fit,cal=h.p.internal_split(len(cache['sample_id']),h.c.settings(1)['calibration_fraction'])
    assert (len(fit),len(cal))==(25912,2879)
    rng=np.random.default_rng(20261005);shuffled=rng.permutation(fit);bank=shuffled[:4096];queries={'fit':shuffled[4096:5120],'cal':rng.choice(cal,512,replace=False)}
    assert not np.intersect1d(bank,queries['fit']).size and not np.intersect1d(fit,cal).size
    values=h.r.prepared(cache,False,fit=fit)
    model=h.SetHead(940,h.c.settings(1,'zero')).to('cuda:0').eval();h.a.frozen_audit(model)
    print('[DIAGNOSIS] observed-only frozen context extraction',flush=True)
    def features(ids):
        outputs=[]
        with torch.no_grad():
            for begin in range(0,len(ids),512):
                observation=torch.from_numpy(values[0][ids[begin:begin+512]]).to('cuda:0')
                outputs.append(model.inputs(observation)[:,0,:64].cpu().numpy())
        return np.concatenate(outputs)
    fitfeatures=features(fit);mean=fitfeatures.mean(0);std=np.maximum(fitfeatures.std(0),.001)
    assert np.isfinite(fitfeatures).all() and std.shape==(64,)
    featuremap={int(row):i for i,row in enumerate(fit)}
    bankfeatures=(fitfeatures[[featuremap[int(row)] for row in bank]]-mean)/std
    banktensor=torch.from_numpy(bankfeatures).to('cuda:0');rows=[];savedids={}
    for split,ids in queries.items():
        z=(features(ids)-mean)/std
        with torch.no_grad():dist=torch.cdist(torch.from_numpy(z).to('cuda:0'),banktensor).square()/64;nearest=dist.topk(8,largest=False).indices.cpu().numpy();distance=dist.topk(8,largest=False).values.cpu().numpy()
        random=rng.integers(0,len(bank),size=nearest.shape)
        gt=values[3][ids];near_gt=values[3][bank[nearest]];random_gt=values[3][bank[random]]
        near_mean=near_gt.mean(1);random_mean=random_gt.mean(1)
        near_error=np.linalg.norm(near_mean-gt,axis=-1).mean(-1);random_error=np.linalg.norm(random_mean-gt,axis=-1).mean(-1)
        dispersion=np.linalg.norm(near_gt-near_mean[:,None],axis=-1).mean((-1,-2))
        base_error=np.linalg.norm(values[1][ids]-gt[:,None],axis=-1).mean(-1).min(-1)
        for group,mask in [('overall',np.ones(len(ids),bool))]+[(name,cache['change_group'][ids]==i) for i,name in enumerate(('stable','transition','change'))]:
            if mask.any():rows.append(dict(split=split,group=group,n=int(mask.sum()),observed_neighbor_sq_distance=float(distance[mask].mean()),neighbor_future_mean_ADE=float(near_error[mask].mean()),random_neighbor_future_mean_ADE=float(random_error[mask].mean()),neighbor_future_dispersion_ADE=float(dispersion[mask].mean()),native_oracle_ADE=float(base_error[mask].mean())))
        savedids[split]=dict(query_ids=cache['sample_id'][ids].tolist(),nearest_fit_ids=cache['sample_id'][bank[nearest]].tolist())
    h.a.frozen_audit(model)
    for path,digest in manifest.items():assert h.p.sha256(Path(path))==digest
    report=dict(status='PASS_FIT_CAL_DESCRIPTIVE_ONLY',no_optimizer=True,no_dev_test=True,seed=20261005,fit_scaler_n=len(fit),bank_n=len(bank),neighbors=8,source_hashes={str(Path(__file__)):h.p.sha256(Path(__file__)),**json.loads((PARENT/'preflight.json').read_text())['source_hashes']},fit_bank_sample_ids=cache['sample_id'][bank].tolist(),query_neighbor_ids=savedids,rows=rows,old_files_unchanged=len(manifest),limitation='Frozen 64context Euclidean similarity is only a proxy; descriptive selected query neighborhoods do not prove irreducible future ambiguity, causality, decoder limits or deployment benefit. FutureGT is used only posthoc to measure support, never feature extraction/neighbor selection. Neighbor means use FIT future labels as constructed diagnostic, not learned deployable predictions or new baseline.')
    (HERE/'RESULT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('[COMPLETE DIAGNOSIS] '+json.dumps(rows),flush=True)
if __name__=='__main__':main()
