"""Final new-report QA: metadata, numerical conclusions, source citations. No model replay."""
import json,hashlib,re,sys
from pathlib import Path
H=Path(__file__).resolve().parent.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
r=read(H/'completion_review/REPORT_DATA.json');analysis=read(H/'RESULT_ANALYSIS.json')
assert len(r['cells'])==72 and len(r['all36_bootstrap_records'])==36
physical=[c for c in r['cells'] if c['arm']=='physical']
assert len(physical)==36 and all(c['motion_all5_1pct'] and c['rescued']==c['lost']==0 and c['good']==c['native_good'] for c in physical)
assert all(p['alpha_quantiles'][2]==0 for p in r['all9_physical_alpha_records'])
assert r['raw_pointwise_excess_total']==11 and r['raw_pointwise_max_excess']==2.631931213955104e-15
assert analysis['status']=='ANALYZED' and not analysis['major_success'] and analysis['new_formal_study_completed']==3
full=read(H/'completion_review/FULLTHREE_DATA.json')
assert len(full['all216_cells'])==216 and full['history_rows']==450 and full['saved_jobs']==54
for p,v in full['referenced_SHA'].items():assert sha(Path(p))==v,p
recent=read(H/'RECENT_EXPERIMENT_SYNTHESIS.json');refs={p:sha(Path(p)) for p in recent['sources'].values()}
checks=[]
for f in ('RESULT_ANALYSIS.md','NATIVE_ANCHOR_STUDY_REVIEW.md','RECENT_EXPERIMENT_SYNTHESIS.md'):
 p=H/f;t=p.read_text(encoding='utf-8')
 links=re.findall(r'\]\((C:/[^)]+)\)',t)
 assert all(Path(x).exists() for x in links),(f,links)
 assert 'major' in t.lower() and ('非外' in t or '不是外部' in t)
 checks.append(dict(file=f,SHA=sha(p),absolute_existing_links=len(links)))
control=read(H/'runtime_state.json');assert control['status']=='ANALYZED' and control['formal_study_completed']==3 and control['first_progress_query_count']==1
ledger=read(H.parents[2]/'AUTONOMOUS_GENERATION_CYCLE.json');assert ledger['cycle']==11 and ledger['native_anchored_physical_ray_approved_launch_20261008']['formal_study_completed']==3
out=dict(status='PASS_NEW_REPORT_QA',reports=checks,cited_static_sources_SHA=refs,all_group_numerical_conclusions_verified=True,no_source_model_result_mutation=True,no_PID_task_log_or_resource_queries=True,no_torch_model_optimizer_gradient_replay=True,not_all_historical_model_reaudit=True,not_external_review=True,goal_not_achieved=True,all_three_major_FAIL=True)
(H/'completion_review/FINAL_REPORT_REVIEW.json').write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
print('FINAL_REPORT_QA_PASS',len(checks),len(refs))

