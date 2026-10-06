"""Fixed FIT/CAL GT-only expressivity diagnosis, not a learned predictor/GT forward."""
from pathlib import Path
import importlib.util,sys,json
import numpy as np,torch
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'GENERATION_TEMPORAL_COMPONENT_SERIES'
def main():
    spec=importlib.util.spec_from_file_location('reachability_source',SOURCE/'run_head.py')
    a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
    a.a.prepare_backend(torch.device('cpu'))
    cache=a.p.load_npz(a.p.TRAIN_CACHE);fit,cal=a.p.internal_split(len(cache['sample_id']),.1)
    assert len(fit)==25912 and len(cal)==2879
    values=a.r.prepared(cache,False,fit=fit);base,speed,gt=values[1:4]
    interpolation=a.n.DECODER_STATE['interpolation'].numpy();assert interpolation.shape==(12,3)
    rows=[]
    for split,ids in [('fit',fit),('cal',cal)]:
        metrics=[]
        for start in range(0,len(ids),512):
            idx=ids[start:start+512];b=base[idx];g=gt[idx,None];limit=1.5*(speed[idx,0,None,None,None]+.03)
            native_velocity=np.diff(b,axis=2,prepend=np.zeros_like(b[:,:,:1]))
            gt_velocity=np.diff(g,axis=2,prepend=np.zeros_like(g[:,:,:1]))
            required=gt_velocity-native_velocity
            exact=np.any(np.all(abs(required)<=limit+1e-6,axis=(2,3)),axis=1)
            endpoint_gap=np.maximum(abs(g[:,:,-1]-b[:,:,-1])-12*limit[:,:,0,:],0)
            endpoint_lower=np.linalg.norm(endpoint_gap,axis=-1).min(1)
            projected=b+np.clip(required,-limit,limit).cumsum(2)
            anchor_delta=(g-b)[:,:,[3,7,11]]
            delta=np.einsum('tp,bkpc->bktc',interpolation,anchor_delta)
            velocity=np.diff(delta,axis=2,prepend=np.zeros_like(delta[:,:,:1]))
            anchored=b+np.clip(velocity,-limit,limit).cumsum(2)
            error_native=np.linalg.norm(b-g,axis=-1)
            error_projected=np.linalg.norm(projected-g,axis=-1)
            error_anchor=np.linalg.norm(anchored-g,axis=-1)
            metrics.append(dict(exact_gt_stepbox_reachable=exact.astype(float),endpoint_necessary_lower_bound=endpoint_lower,native_minade=error_native.mean(2).min(1),native_minfde=error_native[:,:,-1].min(1),constructed_stepclip_minade=error_projected.mean(2).min(1),constructed_stepclip_minfde=error_projected[:,:,-1].min(1),constructed_anchor_minade=error_anchor.mean(2).min(1),constructed_anchor_minfde=error_anchor[:,:,-1].min(1),constructed_anchor_clip_fraction=(abs(velocity)>limit).mean((1,2,3))))
        full={k:np.concatenate([x[k] for x in metrics]) for k in metrics[0]}
        groups=cache['change_group'][ids]
        for name,mask in [('overall',np.ones(len(ids),bool)),('stable',groups==0),('transition',groups==1),('change',groups==2)]:
            assert mask.any();rows.append(dict(split=split,group=name,n=int(mask.sum()),**{k:float(v[mask].mean()) for k,v in full.items()}))
    provenance={str(SOURCE/'run_head.py'):a.p.sha256(SOURCE/'run_head.py'),str(a.p.TRAIN_CACHE):a.p.sha256(a.p.TRAIN_CACHE),str(HERE/'run_diagnosis.py'):a.p.sha256(HERE/'run_diagnosis.py')}
    result=dict(status='PASS_DIAGNOSTIC_ONLY',fit_n=len(fit),cal_n=len(cal),GT_usage='Offline train/cal diagnosis only, not future in model forward',dev_access=False,test_access=False,optimizer_steps=0,velocity_residual_bound=1.5,rows=rows,source_hashes=provenance,limits='Stepbox is superset of 3anchor head function class. Endpoint bound necessary only. GT stepclip and GT3anchors are hand constructions, not learned outputs, not optimal ADE/FDE oracle or a performance guarantee. No deployment GT/guard relaxation.')
    (HERE/'REACHABILITY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(HERE/'split_ids.npz',fit_sample_ids=cache['sample_id'][fit],cal_sample_ids=cache['sample_id'][cal])
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
