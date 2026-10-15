"""Independent CPU prelaunch saved arithmetic and current-source check. No torch."""
import sys,importlib.util,json,time
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent.parent
s=importlib.util.spec_from_file_location('cpu_saved_audit',H/'completion_review/audit_saved.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def main():
    a.guard();source=a.read(H/'source_manifest.json')['files'];a.verify(source);checks=[]
    for arm in a.ARMS:
        z=a.npz(H/'prelaunch_review'/f'{arm}_smoke128.npz')
        met,ass,vectors,clusters,bounds=a.audit_job(z,arm)
        motion={k:value['motion_all5_1pct'] for k,value in met['refined20'].items()}
        assert all(motion.values()),('ACTUAL_SAVED_SMOKE_MOTION_FAIL',arm,motion)
        checks.append(dict(arm=arm,rows=len(z['sample_id']),full_quality_original_SciPy_complete_link=True,all_groups_five_pooled_literal_1pct=motion,bounds_geometry=bounds))
    a.verify(source);a.guard()
    out=dict(status='PASS',locked_source_count=len(source),source_sha=a.digest(source),source_initial_final_PASS=True,all_saved_smoke=checks,no_torch_forward_optimizer=True,review_script_sha=a.sha(__file__),audit_script_sha=a.sha(H/'completion_review/audit_saved.py'),not_external_review=True,resource=a.resource)
    (H/'prelaunch_review/FINAL_SOURCE_REVIEW.json').write_text(json.dumps(out,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x),encoding='utf-8')
    print('CPU_CURRENT_SOURCE_SMOKE_PASS',len(source),flush=True)
if __name__=='__main__':main()
