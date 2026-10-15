"""Reject incompatible source/output without modifying completed fixtures."""
import json
from pathlib import Path
from types import SimpleNamespace
from epoch_recovery import EpochRecovery, atomic_json

H=Path(__file__).resolve().parent

def main():
    actual=json.loads((H/'RESUME_PREFLIGHT.json').read_text(encoding='utf-8'))
    assert actual['status']=='PASS_EPOCH_RECOVERY_FIXTURES_ONLY' and len(actual['records'])==6
    out=H/'negative_recovery_fixtures'
    if out.exists():raise RuntimeError('Existing negative fixtures preserved; no automatic rewrite')
    out.mkdir()
    b=H/'source_input_binding.json';data=json.loads(b.read_text(encoding='utf-8'))
    bad=json.loads(json.dumps(data));first=next(iter(bad['files']));bad['files'][first]='0'*64
    atomic_json(bad,out/'bad_source_binding.json')
    rejected=False
    try:EpochRecovery(H,'full',out/'bad_source_binding.json')
    except RuntimeError as e:
        rejected='STOP_SOURCE' in str(e)
    assert rejected
    src=H/'preflight_runs/full/h3/continuous/f16_temporal_social_cross/seed_1/val_scene_-1'
    m=json.loads((src/'run_manifest.json').read_text(encoding='utf-8'));d=json.loads((src/'done.json').read_text(encoding='utf-8'))
    fake=out/'bad_output';fake.mkdir()
    atomic_json(m,fake/'run_manifest.json');first=next(iter(d['outputs']));d['outputs'][first]='0'*64
    atomic_json(d,fake/'done.json')
    recovery=EpochRecovery(H,'full',b,fixture=True)
    rejected=False
    try:recovery.skip_compatible({'run_dir':fake,'args':SimpleNamespace(**m['arguments'],output_root=fake,force=False)})
    except RuntimeError as e:rejected='STOP_COMPLETED_OUTPUT_SHA' in str(e)
    assert rejected
    changed=dict(m['arguments']);changed['epochs']+=1
    # Protocol is checked before output SHA, so the incompatible budget stops.
    rejected=False
    try:recovery.skip_compatible({'run_dir':fake,'args':SimpleNamespace(**changed,output_root=fake,force=False)})
    except RuntimeError as e:rejected='STOP_COMPLETED_PROTOCOL' in str(e)
    assert rejected
    atomic_json({'status':'PASS_SOURCE_OUTPUT_PROTOCOL_REJECTION_ONLY','old_files_untouched':True,
        'bad_source_rejected':True,'bad_saved_output_rejected':True,'bad_epoch_budget_rejected':True,
        'no_training_or_forward':True,'not_full_production_preflight':True},H/'NEGATIVE_RECOVERY_FIXTURES.json')
    print('PASS_SOURCE_OUTPUT_PROTOCOL_REJECTION_ONLY')

if __name__=='__main__':main()
