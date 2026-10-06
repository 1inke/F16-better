"""Recompute saved sample/group/CI outputs; not another model replay."""
import csv,json
import numpy as np
def audit(h):
    cache=h.p.load_npz(h.p.DEV_CACHE)
    values=h.r.prepared(cache,False,mean=np.zeros((1,940),np.float32),std=np.ones((1,940),np.float32))
    base,speed,future=values[1:4];group=cache['change_group'];ids=cache['sample_id']
    assert len(ids)==1516 and len(np.unique(ids))==1516
    native=h.r.generation_metrics(base,base,speed,future)
    stats={};maximum=0.
    for arm in h.c.ARMS:
        for seed in (1,2,3):
            manifest=json.loads((h.c.checkpoint(seed,arm,'final').parent/'complete.json').read_text())
            assert manifest['epochs']==80 and manifest['status']=='COMPLETE'
            h.verify(manifest['model_hashes'])
            with (h.c.checkpoint(seed,arm,'final').parent/'history.csv').open(encoding='utf-8-sig',newline='') as stream:
                history=list(csv.DictReader(stream))
            assert [int(x['epoch']) for x in history]==list(range(1,81))
            for policy in h.c.POLICIES:
                saved=h.n.RAW_LOAD(h.m.RESULTS/f'seed{seed}_{arm}_{policy}.npz')
                assert np.array_equal(saved['sample_id'],ids)
                pred=saved['trajectory'];assert pred.shape==(1516,20,12,2) and np.isfinite(pred).all()
                metrics=h.r.generation_metrics(pred,base,speed,future);stats[seed,arm,policy]=metrics
                for key,value in metrics.items():
                    error=float(np.max(np.abs(value-saved[key])));maximum=max(maximum,error)
                    assert np.allclose(value,saved[key],atol=1e-6,rtol=0),(seed,arm,policy,key,error)
    summary=json.loads((h.m.RESULTS/'summary.json').read_text())
    for label,metrics in h.m.grouped(native,group).items():
        for key,value in metrics.items():assert abs(value-summary['native'][label][key])<1e-6
    for row in summary['aggregate']:
        mask=dict(h.m.masks(group))[row['group']]
        for key in native:
            value=float(np.stack([stats[seed,row['arm'],row['policy']][key][mask] for seed in (1,2,3)]).mean())
            assert abs(value-row[key])<1e-6,(row['arm'],row['policy'],row['group'],key)
    with (h.m.RESULTS/'per_seed_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        mask=dict(h.m.masks(group))[row['group']];metrics=stats[int(row['seed']),row['arm'],row['policy']]
        for key in native:assert abs(float(metrics[key][mask].mean())-float(row[key]))<1e-6
    cfg=h.c.settings(1)
    for row in summary['bootstrap']:
        mask=dict(h.m.masks(group))[row['group']];metric=row['metric'];policy=row['policy']
        delta=np.stack([stats[seed,'mean_std',policy][metric][mask]-(native[metric][mask] if row['reference']=='native' else stats[seed,'zero',policy][metric][mask]) for seed in (1,2,3)])
        actual=h.m.h.hierarchical_delta(delta,cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])
        assert np.allclose(actual,[row['delta'],row['ci_low'],row['ci_high']],atol=1e-6,rtol=0)
    # Independently reconstruct saved early/mid/late arrays from trajectories.
    temporal_files=list((h.m.RESULTS/'temporal').glob('*.npz'));assert len(temporal_files)==18
    for path in temporal_files:
        saved=h.n.RAW_LOAD(path);source=h.n.RAW_LOAD(h.m.RESULTS/(path.name.replace('_temporal','')))
        error=np.linalg.norm(source['trajectory']-future[:,None],axis=-1)
        index=(error.mean(-1)+cfg['fde_loss_weight']*error[:,:,-1]).argmin(-1)
        assert np.array_equal(saved['sample_id'],ids) and np.array_equal(index,saved['selected_samepath_index'])
        chosen=error[np.arange(len(ids)),index]
        for key,sl in (('early',slice(0,4)),('middle',slice(4,8)),('late',slice(8,12))):
            assert np.allclose(saved[key],chosen[:,sl].mean(-1),atol=1e-6,rtol=0)
    pulse_files=list((h.m.RESULTS/'pulse').glob('seed*.npz'));assert len(pulse_files)==18
    time_grid=(np.arange(12,dtype=np.float32)+.5)/12
    for path in pulse_files:
        saved=h.n.RAW_LOAD(path);assert np.array_equal(saved['sample_id'],ids)
        phi=1./(1.+np.exp(-(time_grid-saved['center'][...,None])/saved['span'][...,None]))
        assert np.allclose(phi,saved['profile'],atol=1e-5,rtol=0)
        q=np.concatenate([h.route.statistic(h.torch.from_numpy(values[0][s:s+128]).to('cuda:0'),h.torch.from_numpy(base[s:s+128]).to('cuda:0')).cpu().numpy() for s in range(0,len(ids),128)])
        branch=(q.astype(np.float64)>saved['route_median']).astype(np.int64)
        assert np.array_equal(branch,saved['branch']) and np.array_equal(q,saved['q'])
        isuniform='_zero_' in path.name
        weights=np.full((*branch.shape,2),.5,np.float32) if isuniform else np.eye(2,dtype=np.float32)[branch]
        assert np.array_equal(weights,saved['weights'])
        mixed=(saved['expert_increments']*weights[...,None,None]).sum(2)
        mixed=np.clip(mixed,-1.5*(speed[:,0,None,None,None]+.03),1.5*(speed[:,0,None,None,None]+.03))
        if bool(saved['native_fallback']):mixed=np.zeros_like(mixed)
        assert np.array_equal(mixed,saved['mixed_increments'])
        source=h.n.RAW_LOAD(h.m.RESULTS/path.name.replace('_pulse',''))
        assert np.allclose(base+np.cumsum(mixed,axis=2),source['trajectory'],atol=2e-5,rtol=0)
        assert np.allclose(mixed.sum(2),saved['drift'],atol=2e-5,rtol=0)
        assert np.all((saved['center']>=.05-1e-7)&(saved['center']<=.95+1e-7))
        assert np.all((saved['span']>=.05-1e-7)&(saved['span']<=.5+1e-7))
    h.m.write_json(h.m.RESULTS/'saved_metrics_recompute.json',dict(status='PASS',models=18,predictions=18,
        sample_metrics=True,native_group_metrics=True,all_policy_group_aggregates=True,all_per_seed_groups=True,
        registered_bootstrap_CIs=True,six_80epoch_histories=True,temporal_arrays_recomputed=True,pulse_shapes_endpoint_decomposition_recomputed=True,maximum_sample_metric_difference=maximum,
        scope='Saved output arithmetic and histories; not model replays or new independent generalization.'))
    return dict(sample_max_error=maximum,group_rows=len(rows),aggregate_rows=len(summary['aggregate']),CI_rows=len(summary['bootstrap']))
