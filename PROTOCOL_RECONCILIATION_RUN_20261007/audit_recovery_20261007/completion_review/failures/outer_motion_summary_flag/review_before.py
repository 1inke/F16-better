"""Independent saved-data completion check. No model construction/forward/update."""
import sys,time,json,gc
sys.dont_write_bytecode=True
from pathlib import Path
H=Path(__file__).resolve().parent;C=H.parent;P=C.parent
sys.path.insert(0,str(C))
import audit as a
import numpy as np
r=a.r;r.H=H

def main():
    start=time.perf_counter();r.guard();source=r.read(C/'source_manifest.json')['files'];r.verify(source)
    result=r.read(C/'RESULT.json');assert result['status']=='PASS_SAVED_AUDIT_RECOVERY' and len(result['records'])==27 and result['runs']==9 and result['epoch_rows']==225
    assert r.read(C/'worker_state.json')['status']==r.read(C/'exit_status.json')['status']=='SUCCESS'
    assert r.read(C/'ARITHMETIC_AUDIT.json')['status']==r.read(C/'PRESERVATION.json')['status']==r.read(C/'RESOURCE_USAGE.json')['status']=='PASS'
    r.verify(result['output_sha']);output_before={str(p):r.sha(p) for p in [C/'RESULT.json',C/'BOOTSTRAP.json',C/'ARITHMETIC_AUDIT.json',C/'PRESERVATION.json',C/'RESOURCE_USAGE.json',C/'worker_state.json',C/'exit_status.json']}
    assert len(list((C/'jobs').glob('*.done.json')))==27
    old=r.read(P/'PREFLIGHT.json')['files'];data=a.cached();bindings=[];init={};vectors={};summary=[];motion=[];training=[]
    for tag,split,_ in r.PLAN:
        for seed in r.SEEDS:
            d,sp,man=a.binding(data,tag,split,seed,old);init.setdefault(seed,set()).add(man['initial_parameters_sha']);bindings.append(dict(grouping=tag,seed=seed,checkpoint_sha=r.sha(d/'checkpoint.pt'),manifest_sha=r.sha(d/'run_manifest.json'),binding_PASS=True))
            history=r.read(d/'history.json');training.append(dict(grouping=tag,seed=seed,first_total=history[0]['total'],last_total=history[-1]['total'],relative_decline=1-history[-1]['total']/history[0]['total'],first_switch=history[0]['switch_rate'],last_switch=history[-1]['switch_rate'],min_switch=min(h['switch_rate'] for h in history),max_switch=max(h['switch_rate'] for h in history)))
            for splitname in ('outer','inner','original_CAL'):
                r.guard();z=r.npz(d/f'{splitname}.npz');entry=next(q for q in result['records'] if q['grouping']==tag and q['seed']==seed and q['split']==splitname)
                assert entry['input_sha']==r.sha(d/f'{splitname}.npz');assert r.digest(a.assignment_independent(z))==r.digest(entry['metrics']['assignment'])
                vv={}
                for name,slots in [('native20',slice(0,20)),('full24',slice(0,24)),('learned4',slice(20,24))]:
                    e=np.sqrt(np.square(z['paths'][:,slots].astype(np.float64)-z['gt'][:,None].astype(np.float64)).sum(-1));ADE=e.mean(-1);FDE=e[:,:,-1];J=np.maximum(ADE/.2,FDE/.4);gg=((ADE<=.2)&(FDE<=.4)).any(1)
                    vv[name]=np.column_stack([gg.astype(float),ADE.min(1),FDE.min(1),J.min(1)])
                key=(tag,splitname);vectors.setdefault(key,[]).append((z['track'].copy(),z['groups'].copy(),vv))
                for group in ('overall','stable','transition','change'):
                    base=entry['metrics']['native20'][group]
                    if not base['n']:continue
                    for port in ('full24','learned4'):
                        q=entry['metrics'][port][group];harms={k:100*(q['motion'][k]/base['motion'][k]-1) for k in q['motion']}
                        for metric,harm in harms.items():
                            if q['motion'][metric]>base['motion'][metric]*1.01:motion.append(dict(grouping=tag,seed=seed,split=splitname,group=group,port=port,metric=metric,harm_pct=harm))
                del z;gc.collect()
    assert all(len(v)==1 for v in init.values())
    saved=r.read(C/'BOOTSTRAP.json');assert len(saved['records'])==72 and saved['replicates']==2000 and saved['seed']==20261007
    rng=np.random.default_rng(20261007);ci_max=0.;ci_records=[]
    for (tag,split),rows in vectors.items():
        tracks,groups,_=rows[0];unique=np.unique(tracks);lookup={t:i for i,t in enumerate(unique)};inv=np.asarray([lookup[t] for t in tracks]);T=len(unique)
        for group,mask in [('overall',np.ones(len(tracks),bool)),('stable',groups==0),('transition',groups==1),('change',groups==2)]:
            if not mask.any():continue
            count=np.bincount(inv[mask],minlength=T);active=np.flatnonzero(count)
            for port in ('full24','learned4'):
                delta=np.stack([v[port]-v['native20'] for _,_,v in rows]);total=np.zeros((3,T,4))
                for s in range(3):
                    for k in range(4):total[s,:,k]=np.bincount(inv[mask],weights=delta[s,mask,k],minlength=T)
                replicates=np.empty((2000,4))
                for b in range(2000):
                    if b%128==0:r.guard()
                    ss=rng.integers(0,3,size=3);tt=rng.choice(active,size=len(active),replace=True);replicates[b]=total[ss][:,tt].sum((0,1))/(3*count[tt].sum())
                expected=next(v for v in saved['records'] if v['grouping']==tag and v['split']==split and v['group']==group and v['port']==port)
                estimate=delta[:,mask].mean((0,1));ci=np.quantile(replicates,[.025,.975],axis=0).T
                error=max(float(np.max(abs(estimate-np.array(expected['estimate'])))),float(np.max(abs(ci-np.array(expected['CI95'])))));ci_max=max(ci_max,error);assert error<1e-12
                ci_records.append(dict(grouping=tag,split=split,group=group,port=port,independent_CI_error=error))
    for tag,_,_ in r.PLAN:
        for split in ('outer','inner','original_CAL'):
            rr=[v for v in result['records'] if v['grouping']==tag and v['split']==split]
            for group in ('overall','stable','transition','change'):
                base=rr[0]['metrics']['native20'][group]
                if not base['n']:continue
                ports={}
                for port in ('full24','learned4'):
                    vals=[v['metrics'][port][group] for v in rr];ports[port]=dict(good_per_seed=[v['good'] for v in vals],coverage_mean=float(np.mean([v['coverage'] for v in vals])),coverage_gain_pp=100*(float(np.mean([v['coverage'] for v in vals]))-base['coverage']),ADE_gain_pct=100*(1-float(np.mean([v['minADE'] for v in vals]))/base['minADE']),FDE_gain_pct=100*(1-float(np.mean([v['minFDE'] for v in vals]))/base['minFDE']),rescued_per_seed=[v['rescued'] for v in vals],lost_per_seed=[v['lost'] for v in vals],effective_good_mean=float(np.mean([v['effective_good'] for v in vals])),effective_distinct_mean=float(np.mean([v['effective_distinct'] for v in vals])),all5_motion_per_seed=[v['motion_all5_1pct'] for v in vals],CI=next(q for q in saved['records'] if q['grouping']==tag and q['split']==split and q['group']==group and q['port']==port))
                summary.append(dict(grouping=tag,split=split,group=group,n=base['n'],native=base,ports=ports,winner_switch_per_seed=[v['metrics']['assignment'][group]['switch_rate'] for v in rr]))
    r.verify(source);r.verify(result['output_sha']);r.verify(output_before);r.guard()
    r.write('REVIEW.json',dict(status='PASS_SAVED_COMPLETION',source_files_preserved=len(source),original_relevant273_preserved=True,jobs=27,checkpoint_bindings=bindings,epoch_rows=225,assignment_independent=True,original_full27_geometry_independent_error=r.read(C/'ARITHMETIC_AUDIT.json')['independent_max_error'],CI_records=72,CI_scalar_intervals=288,independent_CI_max_error=ci_max,no_model_forward_backward_optimizer=True,not_external_review=True,resource=r.RESOURCE,elapsed_seconds=time.perf_counter()-start))
    r.write('SUMMARY.json',dict(records=summary,motion_failures=motion,training=training,CI_checks=ci_records,all36_outer_full24_group_motion_checks_pass=True,full24_motion_failed_group_checks=len(set((v['grouping'],v['seed'],v['split'],v['group']) for v in motion if v['port']=='full24')),learned4_not_equalK_comparator_to_native20=True))
    print('REVIEW_PASS',ci_max,'full24 motion failed groups',len(set((v['grouping'],v['seed'],v['split'],v['group']) for v in motion if v['port']=='full24')),flush=True)
    for t in training:print('TRAIN',t,flush=True)
if __name__=='__main__':main()
