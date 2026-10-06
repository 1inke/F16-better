"""Bounded saved-data complete-path, distinct-good and pooled-motion audit."""
import json
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
from resource_guard import guard

def quantities(path,gt,scores):
    path=path.astype(np.float64);err=np.linalg.norm(path-gt[:,None].astype(np.float64),axis=-1)
    a=err.mean(-1);f=err[:,:,-1];good=(a<=.2)&(f<=.4)
    selected=scores.argmax(-1);oracle=np.maximum(a/.2,f/.4).argmin(-1);ii=np.arange(len(path))
    efficient=np.zeros(len(path),np.int16);distinct=efficient.copy();i,j=np.triu_indices(20,1)
    for row in range(len(path)):
        if row%128==0:guard.check()
        d=np.linalg.norm(path[row,i]-path[row,j],axis=-1)
        labels=fcluster(linkage(np.maximum(d.mean(-1)/.10,d[:,-1]/.20),method='complete'),1.,criterion='distance')
        unique=np.unique(labels);distinct[row]=len(unique)
        efficient[row]=sum(good[row,labels==label].any() for label in unique)
    e=err[ii,oracle]
    result=dict(coverage=good.any(-1),raw_good=good.sum(-1),effective_good=efficient,all_distinct=distinct,
                selected_good=good[ii,selected],selected_slot=selected,oracle_slot=oracle,
                early=e[:,:4].mean(-1),middle=e[:,4:8].mean(-1),late=e[:,8:].mean(-1),
                selected_ADE=a[ii,selected],selected_FDE=f[ii,selected],
                independent_min_both_but_no_samepath=(a.min(-1)<=.2)&(f.min(-1)<=.4)&~good.any(-1),
                loose_coverage=((a<=.3)&(f<=.5)).any(-1))
    v=np.diff(path,axis=2,prepend=np.zeros_like(path[:,:,:1]))/.4
    acc=np.diff(v,axis=2)/.4;jerk=np.diff(acc,axis=2)/.4
    motion={k:np.linalg.norm(value,axis=-1) for k,value in [('speed',v),('accel',acc),('jerk',jerk)]}
    return result,motion

def audit(h):
    cache=h.p.load_npz(h.p.DEV_CACHE)
    values=h.r.prepared(cache,False,mean=np.zeros((1,940),np.float32),std=np.ones((1,940),np.float32))
    base,gt=values[1],values[3];ids=cache['sample_id'];scores=cache['logits'];groups=cache['change_group']
    assert len(ids)==len(np.unique(ids))==1516
    out=h.m.RESULTS/'quality';out.mkdir(exist_ok=True)
    native,nmotion=quantities(base,gt,scores);allvalues={};records=[];motion_rows=[]
    for arm in h.c.ARMS:
        for seed in (1,2,3):
            for policy in h.c.POLICIES:
                saved=h.n.RAW_LOAD(h.m.RESULTS/f'seed{seed}_{arm}_{policy}.npz')
                assert np.array_equal(saved['sample_id'],ids)
                q,motion=quantities(saved['trajectory'],gt,scores);allvalues[arm,seed,policy]=q
                path=out/f'seed{seed}_{arm}_{policy}_quality.npz'
                np.savez_compressed(path,sample_id=ids,**q)
                actual=h.n.RAW_LOAD(path)
                assert all(np.array_equal(v,actual[k]) for k,v in q.items())
                for group,mask in h.m.masks(groups):
                    records.append(dict(arm=arm,seed=seed,policy=policy,group=group,n=int(mask.sum()),
                        native_good=int(native['coverage'][mask].sum()),good=int(q['coverage'][mask].sum()),
                        rescued=int((~native['coverage'][mask]&q['coverage'][mask]).sum()),
                        lost=int((native['coverage'][mask]&~q['coverage'][mask]).sum()),
                        effective_good_total=int(q['effective_good'][mask].sum()),
                        inherited_selected_good=int(q['selected_good'][mask].sum()),
                        selection_miss=int((q['coverage'][mask]&~q['selected_good'][mask]).sum()),
                        early=float(q['early'][mask].mean()),middle=float(q['middle'][mask].mean()),late=float(q['late'][mask].mean())))
                    row=dict(arm=arm,seed=seed,policy=policy,group=group)
                    for key,v in motion.items():
                        nativev=nmotion[key][mask].reshape(-1);newv=v[mask].reshape(-1)
                        for stat in ('mean','p95'):
                            old=float(np.mean(nativev) if stat=='mean' else np.quantile(nativev,.95))
                            new=float(np.mean(newv) if stat=='mean' else np.quantile(newv,.95))
                            row[key+'_'+stat]=new;row['native_'+key+'_'+stat]=old
                    motion_rows.append(row)
    # Supplemental stricter pooled global motion, not window-p95 averages.
    primary=[r for r in motion_rows if r['arm']=='mean_std' and r['policy']=='constrained']
    keys=('speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95')
    pooled_ok=all(r[k]<=r['native_'+k]*1.01 for r in primary for k in keys)
    h.m.write_json(out/'summary.json',dict(status='PASS_SAVED_QUALITY_ARITHMETIC',records=records,global_pooled_motion=motion_rows,
        native_good=int(native['coverage'].sum()),native_effective_good_total=int(native['effective_good'].sum()),
        global_motion_all_constrained_seeds_groups_le1pct_harm=pooled_ok,
        scorer='Inherited native argmax; NOT rescored/deployed newly calibrated probability',
        scope='Posthoc saved1516K20 completepath and GT-free clusters, per-group pooledALLK motion dt.4; no model/optimizer/replay; repeatedDEV'))
    summary=json.loads((h.m.RESULTS/'summary.json').read_text())
    summary['gates']['global_pooled_motion_each_seed_group_1pct']=bool(pooled_ok)
    summary['gates']['native_and_fresh_joint_CI_exclude_zero']=all(row['ci_high']<0 for row in summary['bootstrap'] if row['policy']=='constrained' and row['group']=='change' and row['metric']=='joint_oracle')
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_CORRELATION_AUDIT' if all(summary['gates'].values()) else 'TRAINED_ACCEL_PROJECTION_GATE_FAIL'
    h.m.write_json(h.m.RESULTS/'summary.json',summary)
    (h.m.RESULTS/'REPORT.md').write_text('# Training-in-loop endpoint-preserving acceleration projection\n\n'+json.dumps(summary,indent=2),encoding='utf-8')
    return dict(status='PASS',quality_predictions=18,pooled_motion_pass=pooled_ok)
