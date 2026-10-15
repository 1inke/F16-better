"""Final source-transition, links, manifests and control-state verification."""
import sys,json,re,importlib.util
sys.dont_write_bytecode=True
from pathlib import Path
H=Path(__file__).resolve().parent;C=H/'numeric_fixture_recovery';P=H.parent
sp=importlib.util.spec_from_file_location('final_v2_math',C/'analyze_v2.py');v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
a=v.a;read=a.read
def source(p):return p.read_text(encoding='utf-8-sig').strip().replace('\r\n','\n')
parent=source(H/'analyze_v2.py');child=source(C/'analyze_v2.py');expected=parent.replace('P=H.parent;V=','P=H.parent.parent;V=').replace('assert np.mean(allr)==.5','assert abs(np.mean(allr)-.5)<=1e-15');assert expected==child
E=H/'efficiency_existing_model';eparent=source(E/'benchmark.py');echild=source(E/'source_introspection_recovery/benchmark.py')
expected=eparent.replace('P=H.parent.parent\n','P=H.parent.parent.parent\n').replace('[loader,regime,front,trainer,adapter]','[loader,regime,front,trainer,adapter,adapter.CONTROLLED]').replace('            path=inspect.getsourcefile(type(mod))','            try:path=inspect.getsourcefile(type(mod))\n            except TypeError:path=None');assert expected==echild
manifest=read(C/'input_manifest.json')['files'];result=read(C/'RESULT.json')
for p,d in manifest.items():a.guard();assert a.sha(p)==d,p
for p,d in result['output_sha'].items():assert a.sha(p)==d,p
for n,d in result['summary_sha'].items():assert a.sha(C/(n+'.json'))==d,n
for name in ['runtime_state.json','protocol.json']:
    read(H/name)
rt=read(P/'runtime_state.json');ledger=read(P.parent/'AUTONOMOUS_GENERATION_CYCLE.json')
assert rt['paused_processes']==[35588,14516] and rt['automation_status']=='PAUSED' and not rt['whole_series_complete']
assert rt['current_report']=='claim_evidence_v2_20261009/REPORT.md'
assert ledger['ped_motion_understanding_E0_E6_completion_20261009']['long_training_processes_remain_suspended']
links=re.findall(r'\]\(([^)]+)\)',source(H/'REPORT.md'));assert all((H/p).is_file() for p in links)
review=read(C/'REVIEW.json');prof=read(C/'PROFILE_REVIEW.json');assert review['checks']+prof['checks']==4017
counts={n:len(read(C/(n+'.json'))['records']) for n in ['A1','A2','A3','A4','A5','A6']};assert counts==dict(A1=30,A2=150,A3=90,A4=57,A5=180,A6=36)
fail=read(E/'source_introspection_recovery/FAILURE.json');assert 'CACHE_ALIGN' in fail['traceback'] and '0.0002770423889160156' in fail['traceback']
codeSHA={str(p):a.sha(p) for p in [Path(__file__),C/'audit_v2.py',C/'audit_profiles.py',H/'REPORT.md',H/'runtime_state.json',H/'analyze_v2.py',H/'FAILURE.json',E/'benchmark.py',E/'FAILURE.json',E/'source_introspection_recovery/benchmark.py',E/'source_introspection_recovery/FAILURE.json']}
a.H=H;a.write('FINAL_REVIEW.json',dict(status='PASS_A1_A6_DELIVERY_E8_FAILURE_DISCLOSED',actual_input_SHA_files=len(manifest),scientific_output_SHA_files=len(result['output_sha']),saved_arithmetic_checks=4017,record_counts=counts,report_links_exist=len(links),original_source_transitions_exact_only_infra=True,whole_series_complete=False,E8_complete=False,E8_alignment_gate_not_relaxed=True,E3_and_automation_still_paused=True,no_new_training=True,code_and_report_SHA_after_execution=codeSHA,not_model_or_external_review=True,resource=a.RES))
print('FINAL QA PASS',counts,len(manifest),len(result['output_sha']),flush=True)
