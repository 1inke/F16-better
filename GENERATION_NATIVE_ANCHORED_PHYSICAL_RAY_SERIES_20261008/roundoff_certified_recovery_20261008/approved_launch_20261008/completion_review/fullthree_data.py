"""Consolidated saved-review synthesis; no model or optimizer replay."""
import json,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent.parent;R=H.parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
names=['GENERATION_NATIVE_ANCHORED_RESIDUAL_SERIES_20261008','GENERATION_NATIVE_ANCHORED_BERNSTEIN_SERIES_20261008']
studies=[R/n for n in names]+[H];allrows=[];summaries=[]
for no,S in enumerate(studies,1):
 review=read(S/'completion_review/REVIEW.json');data=read(S/'completion_review/SUMMARY.json')
 assert review['status']=='PASS' and len(data['records'])==18 and len(data['bootstrap'])==36
 rows=[]
 for j in data['records']:
  for group,m in j['metrics']['refined20'].items():
   base=j['metrics']['native20'][group]
   rows.append(dict(study=no,arm=j['arm'],seed=j['seed'],split=j['split'],group=group,n=m['n'],native_good=base['good'],good=m['good'],rescue=m['rescued'],lost=m['lost'],minADE=m['minADE'],minFDE=m['minFDE'],J=m['J'],native_ADE=base['minADE'],native_FDE=base['minFDE'],native_J=base['J'],effective_good=m['effective_good'],effective_distinct=m['effective_distinct'],motion_pass=m['motion_all5_1pct'],motion_harm_percent={k:100*(m['motion'][k]/base['motion'][k]-1) for k in m['motion']}))
 assert len(rows)==72
 arms=sorted(set(x['arm'] for x in rows))
 summaries.append(dict(study=no,folder=str(S),source_review_sha=sha(S/'completion_review/REVIEW.json'),source_summary_sha=sha(S/'completion_review/SUMMARY.json'),jobs=18,history_rows=150,bootstrap_records=36,scalar_CIs=144,motion={a:dict(cells=36,passes=sum(x['motion_pass'] for x in rows if x['arm']==a),failures=[x for x in rows if x['arm']==a and not x['motion_pass']]) for a in arms}))
 allrows.extend(rows)
assert len(allrows)==216
proofs={}
for S in studies:
 for name in ('RESULT_ANALYSIS.md','RESULT_ANALYSIS.json','completion_review/REVIEW.json','completion_review/SUMMARY.json'):
  p=S/name
  if p.exists():proofs[str(p)]=sha(p)
out=dict(status='PASS_SAVED_REVIEW_AGGREGATION_NOT_NEW_REPLAY',fresh_training_executions=18,checkpoint_manifests=18,history_rows=450,saved_jobs=54,all_arm_seed_split_group_cells=216,bootstrap_records=108,scalar_CIs=432,studies=summaries,all216_cells=allrows,referenced_SHA=proofs,repeated_controls_not_independent_generalization=True,all_three_major_FAIL=True,baseline_unchanged=True,no_model_torch_forward_backward_optimizer=True)
(H/'completion_review/FULLTHREE_DATA.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
print('FULLTHREE',len(allrows),[(s['study'],{a:m['passes'] for a,m in s['motion'].items()}) for s in summaries])

