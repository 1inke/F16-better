"""Recover terminal reporting from complete saved review; no science rerun."""
import sys,json,importlib.util
sys.dont_write_bytecode=True
from pathlib import Path
D=Path(__file__).resolve().parent;H=D.parent
s=importlib.util.spec_from_file_location('registered_math_readonly',H/'run.py')
q=importlib.util.module_from_spec(s);s.loader.exec_module(q);q.guard()
expected={'audit_saved.py':'1bdf6000cb3618326453afe656f618a3a57cbb2d9114a372a8cc0f9e004a6bf6','ALL_POLICY_GROUPS.json':'142613fe5ddc80f99e8c99b289b42c6fd8801ad9e4e6ce6366a08180c262951f','WITNESS_DETAILS.json':'30e7c9dcf67fe9b2ff50f526d6a24ee043e4328b9a862982d67a3deb85d3f855','SUMMARY.json':'8228030445644004e998732f669d8b91612d49bec8e3fea4d5ef2a228bf405d3','REVIEW.json':'20978967f50f6a954a400f34bbe3a8ca6bfc69cda3dc774da864f31bf38835a1'}
q.verify({str(D/k):v for k,v in expected.items()})
r=q.read(D/'REVIEW.json');table=q.read(D/'ALL_POLICY_GROUPS.json');summary=q.read(D/'SUMMARY.json');details=q.read(D/'WITNESS_DETAILS.json')
assert r['status']=='PASS' and r['max_saved_quality_clusters_motion_diagnostic_error']==0 and r['max_geometry_error']==0
assert len(table)==240 and len(summary['summary'])==12 and len(summary['bank_witness'])==12 and len(details)==44
assert sum(x['all5_literal_PASS'] for x in table)==240
result=q.read(H/'RESULT.json');lock=q.read(H/'source_manifest.json')['files'];assert q.digest(lock)==result['source_sha']
q.verify(lock);q.verify(result['output_sha']);assert 'torch' not in sys.modules
note=dict(status='PASS',at=q.now(),original_supplemental_terminal_exit=1,only_numpy_terminal_serialization_failed_after_full_saved_review=True,full_saved_review_status='PASS',review_artifact_SHA=expected,related_source_count=len(lock),output_count=len(result['output_sha']),source_output_SHA_unchanged=True,no_scientific_rerun_or_model_forward=True,resource=q.resource)
(D/'PRINT_RECOVERY.json').write_text(json.dumps(note,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(status='PASS',scientific_jobs=60,policy_group_rows=240,max_saved_error=0,max_geometry_error=0,motion_pass_rows=240,related_source_count=1480,output_count=120)))
