"""Full saved metrics/groups/CIs and projection arithmetic, not all model replay."""
import csv,json
import numpy as np
import torch
from preflight_impl import independent_physics
def audit(h):
    cache=h.p.load_npz(h.p.DEV_CACHE);values=h.r.prepared(cache,False)
    base,speed,gt=values[1:4];ids=cache['sample_id'];group=cache['change_group']
    assert len(ids)==len(np.unique(ids))==1516
    native=h.r.generation_metrics(base,base,speed,gt);stats={};maximum=0.;output_sha={}
    for arm in h.c.ARMS:
        for seed in (1,2,3):
            directory=h.c.checkpoint(seed,arm,'final').parent
            completed=h.read(directory/'complete.json');assert completed['status']=='COMPLETE' and completed['epochs']==80;h.verify(completed['model_hashes'])
            with (directory/'history.csv').open(encoding='utf-8-sig',newline='') as f:history=list(csv.DictReader(f))
            assert [int(x['epoch']) for x in history]==list(range(1,81))
            for policy in h.c.POLICIES:
                savedmodel=torch.load(h.c.checkpoint(seed,arm,policy),map_location='cpu',weights_only=False)
                for prefix in ('experts.0.decoder.','experts.1.decoder.'):
                    for key,v in h.DECODER_STATE.items():assert torch.equal(savedmodel['state_dict'][prefix+key],v),key
                path=h.m.RESULTS/f'seed{seed}_{arm}_{policy}.npz';saved=h.n.RAW_LOAD(path)
                assert np.array_equal(saved['sample_id'],ids) and saved['trajectory'].shape==(1516,20,12,2) and np.isfinite(saved['trajectory']).all()
                assert saved['trajectory'].dtype==np.float64
                metrics=h.r.generation_metrics(saved['trajectory'],base,speed,gt);stats[seed,arm,policy]=metrics
                for key,value in metrics.items():
                    error=float(np.max(abs(value-saved[key])));maximum=max(maximum,error);assert error<=1e-6,(arm,seed,policy,key,error)
                err=np.linalg.norm(saved['trajectory']-gt[:,None],axis=-1);slot=(err.mean(-1)+.5*err[:,:,-1]).argmin(-1);chosen=err[np.arange(len(ids)),slot]
                temporal=h.n.RAW_LOAD(h.m.RESULTS/f'temporal/seed{seed}_{arm}_{policy}_temporal.npz')
                assert np.array_equal(temporal['sample_id'],ids) and np.array_equal(temporal['selected_samepath_index'],slot)
                for key,sl in [('early',slice(0,4)),('middle',slice(4,8)),('late',slice(8,12))]:assert np.array_equal(temporal[key],chosen[:,sl].mean(-1))
                projpath=h.m.RESULTS/f'projection/seed{seed}_{arm}_{policy}.npz';arrays=h.n.RAW_LOAD(projpath)
                assert np.array_equal(arrays['sample_id'],ids) and np.array_equal(arrays['base'],base)
                for start in range(0,len(ids),128):
                    sl=slice(start,start+128);h.guard.check()
                    tensors={k:torch.from_numpy(v[sl]) for k,v in arrays.items() if k not in ('sample_id',)}
                    info={k:tensors[k] for k in ('alpha','alpha_nominal','native_mean','delta_mean','roundoff_reserve','zero_budget_mean_quantized')}
                    independent_physics(h,tensors['base'],tensors['raw'],tensors['projected'],tensors['up'],tensors['limit'],info)
                    reconstructed=arrays['base'][sl].astype(np.float64)+np.cumsum(arrays['raw'][sl].astype(np.float64),axis=2) if arm=='zero' else arrays['projected'][sl]
                    assert np.allclose(reconstructed,saved['trajectory'][sl],atol=2e-5,rtol=0)
                for outpath in (path,projpath,h.m.RESULTS/f'temporal/seed{seed}_{arm}_{policy}_temporal.npz'):output_sha[str(outpath)]=h.sha(outpath)
                del savedmodel,saved,arrays
    summary=h.read(h.m.RESULTS/'summary.json')
    for name,row in h.m.grouped(native,group).items():
        for key,v in row.items():assert abs(v-summary['native'][name][key])<=1e-6
    for row in summary['aggregate']:
        mask=dict(h.m.masks(group))[row['group']]
        for key in native:assert abs(float(np.stack([stats[seed,row['arm'],row['policy']][key][mask] for seed in (1,2,3)]).mean())-row[key])<=1e-6
    with (h.m.RESULTS/'per_seed_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        mask=dict(h.m.masks(group))[row['group']];val=stats[int(row['seed']),row['arm'],row['policy']]
        for key in native:assert abs(float(val[key][mask].mean())-float(row[key]))<=1e-6
    cfg=h.settings(1)
    for row in summary['bootstrap']:
        mask=dict(h.m.masks(group))[row['group']];key=row['metric'];tag=row['policy']
        delta=np.stack([stats[seed,'mean_std',tag][key][mask]-(native[key][mask] if row['reference']=='native' else stats[seed,'zero',tag][key][mask]) for seed in (1,2,3)])
        actual=h.m.h.hierarchical_delta(delta,cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])
        assert np.allclose(actual,[row['delta'],row['ci_low'],row['ci_high']],atol=1e-6,rtol=0)
    result=dict(status='PASS',models=18,predictions=18,maximum_sample_metric_difference=maximum,all_sample_group_perseed_bootstrap_CI_recomputed=True,
        six_80epoch_history=True,temporal=True,projection_strict_per_window_and_endpoint_reference=True,old_decoder_all_weights_bool_buffers_exact=True,
        output_SHA=output_sha,scope='Saved arithmetic and saved states; not all models/gradient second replay')
    h.write(h.m.RESULTS/'saved_metrics_recompute.json',result);h.write(h.m.RESULTS/'independent_audit.json',result);return result
