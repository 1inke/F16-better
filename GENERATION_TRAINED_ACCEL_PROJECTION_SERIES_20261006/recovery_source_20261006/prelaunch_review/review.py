"""CPU final-source/metadata review only, no forward, optimizer or training."""
from pathlib import Path
import sys,json,hashlib,time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent.parent
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
if __name__=='__main__':
    audit=read(HERE/'preflight.json');resume=read(HERE/'resume_check.json')
    bench=read(HERE/'benchmark.json');workflow=read(HERE/'gate_workflow.json')
    resource=read(HERE/'RESOURCE_USAGE.json')
    assert audit['status']==resume['status']==workflow['status']=='PASS'
    assert resource['status']=='ALL_GATES_PASS'
    assert bench['status']=='PASS_THROUGHPUT_ONLY' and not bench['saved_model']
    for path,digest in audit['source_hashes'].items():assert sha(path)==digest,path
    active=audit['active_actual_head']
    assert active['status']=='PASS' and active['active_windows']==128
    assert active['alpha_max']<1 and active['independent_physics']['strict_per_window']
    assert all(v>0 for k,v in active['gradients'].items() if not k.endswith('key.bias'))
    outputs={}
    for arm in ('zero','mean_std'):
        report=next(r for r in resume['reports'] if r['arm']==arm)
        assert report['params_buffers_optimizer_allRNG_history_calibration_exact']
        for variant in ('continuous','interrupted'):
            complete=read(HERE/f'resume_smoke/{arm}/{variant}/seed_7/balanced_protected/complete.json')
            for path,digest in complete['model_hashes'].items():
                assert sha(path)==digest,path
                outputs[path]=digest
    assert resource['peak_private_GiB']<6 and resource['min_physical_GiB']>=6 and resource['min_commit_GiB']>=6
    result=dict(status='PASS_FINAL_CURRENT_SOURCE_AND_COMPLETED_SMOKE_SHA',
        at=time.time(),source_count=len(audit['source_hashes']),old_files=audit['old_files'],
        full_old_final_SHA_from_exclusive_workflow=True,not_full_old_SHA_repeated_here=True,
        active_actual_head=active,completed_smoke_models_SHA=outputs,
        two_real_FIT_CAL_subset_epochs_not_full25912=True,benchmark=bench,memory=resource,
        review_code_SHA=sha(__file__),no_model_forward=True,no_optimizer=True,
        scope='Same-assistant CPU read-only prelaunch audit, not external review or a production result')
    (HERE/'prelaunch_review/REVIEW.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],source_count=result['source_count'],old_files=result['old_files'],active_windows=active['active_windows'],memory=resource)))
