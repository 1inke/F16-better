"""Saved report extraction only; no model, optimizer, torch or new data."""
import importlib.util,json,numpy as np
from pathlib import Path
H=Path(__file__).resolve().parent.parent
s=importlib.util.spec_from_file_location('completion_saved_audit',H/'completion_review/audit_saved.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
a.guard();r=a.read(H/'completion_review/SUMMARY.json');cells=[]
for job in r['records']:
    for group,m in job['metrics']['refined20'].items():
        n=job['metrics']['native20'][group]
        cells.append(dict(arm=job['arm'],seed=job['seed'],split=job['split'],group=group,**m,native_good=n['good'],native_ADE=n['minADE'],native_FDE=n['minFDE'],native_J=n['J'],ADE_gain_percent=100*(1-m['minADE']/n['minADE']),FDE_gain_percent=100*(1-m['minFDE']/n['minFDE']),motion_harm_percent={k:100*(m['motion'][k]/n['motion'][k]-1) for k in m['motion']}))
aggregate=[]
for arm in a.ARMS:
 for split in ('outer','inner','original_CAL'):
  for group in a.GROUPS:
   rows=[x for x in cells if x['arm']==arm and x['split']==split and x['group']==group]
   aggregate.append(dict(arm=arm,split=split,group=group,n=rows[0]['n'],native_good=rows[0]['native_good'],good_per_seed=[x['good'] for x in rows],rescued_per_seed=[x['rescued'] for x in rows],lost_per_seed=[x['lost'] for x in rows],minADE=float(np.mean([x['minADE'] for x in rows])),minFDE=float(np.mean([x['minFDE'] for x in rows])),J=float(np.mean([x['J'] for x in rows])),native_ADE=rows[0]['native_ADE'],native_FDE=rows[0]['native_FDE'],native_J=rows[0]['native_J'],all_motion_PASS=all(x['motion_all5_1pct'] for x in rows),effective_good_per_seed=[x['effective_good'] for x in rows],effective_distinct_per_seed=[x['effective_distinct'] for x in rows]))
physical=[dict(seed=j['seed'],split=j['split'],**j['bounds']['temporal']) for j in r['records'] if j['arm']=='physical']
history=[]
for arm in a.ARMS:
 for seed in a.SEEDS:
  x=a.read(H/'runs'/arm/f'seed_{seed}/history.json')
  history.append(dict(arm=arm,seed=seed,epochs=len(x),total_first=x[0]['total'],total_final=x[-1]['total'],total_reduction_percent=100*(1-x[-1]['total']/x[0]['total']),diversity_first=x[0]['diversity'],diversity_final=x[-1]['diversity'],denominator=x[0]['denominator']))
raw=[x for p in physical for x in p['raw_pointwise_excess']]
data=dict(status='PASS_SAVED_REPORT_EXTRACTION',cells=cells,all24_aggregate_cells=aggregate,all36_bootstrap_records=r['bootstrap'],all9_physical_alpha_records=physical,history=history,motion_pass_by_arm={arm:sum(c['motion_all5_1pct'] for c in cells if c['arm']==arm) for arm in a.ARMS},raw_pointwise_excess_total=sum(x['raw_strict_excess_count'] for x in raw),raw_pointwise_max_excess=max(x['max_raw_excess'] for x in raw),raw_excess_to_certificate_max=max(x['max_excess_to_certificate'] for x in raw),not_independent_case_count=True,no_model_or_gradient_replay=True)
a.write('REPORT_DATA.json',data)
print('AGGREGATE',json.dumps([x for x in aggregate if x['group']=='change']))
print('ROUND',data['raw_pointwise_excess_total'],data['raw_pointwise_max_excess'],data['raw_excess_to_certificate_max'])
print('MOTION',data['motion_pass_by_arm'])
for x in r['bootstrap']:
 if x['split']=='outer' and x['group']=='change':print('PRIMARY_CI',x)

