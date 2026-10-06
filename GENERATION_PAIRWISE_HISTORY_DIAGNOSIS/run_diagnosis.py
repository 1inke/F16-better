"""FIT/CAL observed-only schema and branch feasibility; no optimizer/DEV/test."""
from pathlib import Path
import importlib.util,sys,json,time,traceback
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def main():
    assert not(HERE/'DIAGNOSIS.json').exists(),'Already complete'
    review=json.loads((ROOT/'GENERATION_CYCLE_9_REVIEW/REVIEW.json').read_text());assert review['status']=='PASS_MANDATORY_CYCLE_9_REVIEW'
    spec=importlib.util.spec_from_file_location('pair_schema_source',ROOT/'GENERATION_TANGENT_CONDITIONING_SERIES/run_head.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
    manifest=json.loads((ROOT/'GENERATION_CYCLE_9_REVIEW/all_files_manifest.json').read_text())['files']
    for path in (ROOT/'GENERATION_CYCLE_9_REVIEW').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:manifest[str(path)]=h.p.sha256(path)
    for path,sha in manifest.items():assert h.p.sha256(Path(path))==sha,path
    (HERE/'old_results_manifest.json').write_text(json.dumps(dict(files=manifest)),encoding='utf-8')
    cache=h.p.load_npz(h.p.TRAIN_CACHE);rawpath=ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz';raw=h.n.RAW_LOAD(rawpath)
    assert np.array_equal(raw['sample_id'],cache['sample_id'])
    obs=raw['observation'];assert obs.shape==(28791,940) and np.isfinite(obs).all()
    local=obs[:,160:304].reshape(-1,9,8,2);counts=obs[:,304]
    assert np.array_equal(counts,counts.astype(np.int32)) and counts.min()>=1 and counts.max()<=9
    assert np.max(np.abs(local[:,0].reshape(-1,16)-cache['_past'].reshape(-1,16)))<1e-6
    relative=(local[:,1:]-local[:,:1]).reshape(-1,8,16)
    valid=np.arange(1,9)[None,:]<counts[:,None]
    fit,cal=h.p.internal_split(len(obs),.1);assert (len(fit),len(cal))==(25912,2879)
    allfit=relative[fit][valid[fit]].astype(np.float64);assert len(allfit)>0
    mean=allfit.mean(0).astype(np.float32);std=np.maximum(allfit.std(0),.001).astype(np.float32)
    rng=np.random.default_rng(20261007);subsets={'FIT':np.sort(rng.choice(fit,2048,replace=False)),'CAL':np.sort(rng.choice(cal,512,replace=False))}
    np.savez_compressed(HERE/'fit_pairwise_scalers.npz',mean=mean,std=std,fit_ids=cache['sample_id'][fit])
    np.savez_compressed(HERE/'split_ids.npz',**{key:cache['sample_id'][ids] for key,ids in subsets.items()})
    torch.manual_seed(20261007);encoder=torch.nn.Sequential(torch.nn.Linear(16,64),torch.nn.GELU(),torch.nn.Linear(64,32)).cuda().eval()
    def encode(features,mask):
        values=encoder(features);return (values*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
    rows=[]
    with torch.no_grad():
        for split,ids in subsets.items():
            features=torch.from_numpy((relative[ids]-mean)/std).cuda();mask=torch.from_numpy(valid[ids]).cuda()
            original=encode(features,mask);reverse=encode(features.flip(1),mask.flip(1));control=encode(torch.zeros_like(features),mask)
            assert torch.isfinite(original).all() and torch.allclose(original,reverse,atol=2e-6,rtol=2e-6)
            assert original.std(0).mean()>1e-5 and (original-control).abs().mean()>1e-5
            rows.append(dict(split=split,n=len(ids),neighbor_present_fraction=float(valid[ids].any(1).mean()),neighbor_count_mean=float(valid[ids].sum(1).mean()),feature_std_mean=float(features.std(0).mean()),pooled_std_mean=float(original.std(0).mean()),real_vs_zero_mean_abs=float((original-control).abs().mean()),neighbor_reversal_max_error=float((original-reverse).abs().max())))
    for path,sha in manifest.items():assert h.p.sha256(Path(path))==sha,path
    sources={str(path):h.p.sha256(path) for path in (Path(__file__),HERE/'README.md',rawpath,h.p.TRAIN_CACHE,HERE/'fit_pairwise_scalers.npz',HERE/'split_ids.npz')}
    result=dict(status='PASS_OBSERVATION_ONLY_FIT_CAL_FEASIBILITY',rows=rows,source_hashes=sources,all_old_files_unchanged=len(manifest),fit_valid_neighbor_pairs=len(allfit),actual_new_encoder_parameters=sum(v.numel() for v in encoder.parameters()),raw_schema='local160:304=9x8x2; count304; focalpast0:16; native tokens are separate cached features',optimizer_steps=0,GT_used=False,DEV_test_used=False,limitations='Observed numerical reachability only, not learned improvement or independent information gain. Tokens/social are already enriched; prior tap failures matter. Zero control retains SAME masks/neighbor presence and learnable bias constant. Original native20 order stays locked; neighbor permutation check is not native-candidate equivariance.')
    (HERE/'DIAGNOSIS.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (HERE/('failure_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8');raise
