"""Final source, math, both-resume, budget and protocol gate; no GPU/model forward."""
from pathlib import Path
import json,hashlib,sys,psutil
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
if __name__=='__main__':
    pre=read(HERE/'preflight.json');resume=read(HERE/'resume_check.json')
    bench=read(HERE/'benchmark.json');protocol=read(HERE/'protocol.json')
    assert pre['status']==resume['status']=='PASS'
    assert pre['parameters_trainable']==dict(zero=45798,mean_std=45798)
    assert pre['reference_math']['centered_dimensions']==30
    assert pre['raw_previous_actual_alignment']['status']=='PASS_SAMEWEIGHTS_RAW_PREVIOUS_ACTUAL_EXACT'
    assert max(pre['raw_previous_actual_alignment']['errors'].values())==0
    assert pre['fit_discrepancy_numeric']['points']==518240
    assert pre['fit_discrepancy_numeric']['CAL_DEV_used']==False
    assert pre['paired_weights_buffers_exact'] and pre['native_identity_exact']
    assert pre['all_original_F16_decoder_scorer_frozen']
    for arm in ('zero','mean_std'):
        assert pre['response'][arm]['state_first_weight_gradient']>0
        assert pre['response'][arm]['state_input_sensitivity']>0
    assert len(resume['reports'])==2 and all(v['status']=='PASS' and v['params_buffers_optimizer_allRNG_history_calibration_exact'] and v['completed_seed_SHA_skip_verified'] for v in resume['reports'])
    assert bench['status']=='PASS_THROUGHPUT_ONLY' and bench['saved_model']==False
    assert all(v['measured_steps']==32 and v['warmup_steps']==4 for v in bench['records'])
    assert protocol['centered_dimensions']==30 and protocol['last2_accel_heading_unchanged']
    for path,digest in pre['source_hashes'].items():assert sha(Path(path))==digest,path
    for path,digest in read(HERE/'old_results_manifest.json')['files'].items():assert sha(Path(path))==digest,path
    # Physical reserve read-only; full physical+commit check performed by actual guard.
    assert psutil.virtual_memory().available>=6*2**30
    report=dict(status='PASS_ACTUAL_GATES_CURRENT_SOURCE_LOCK',source_count=len(pre['source_hashes']),
                old_files=len(read(HERE/'old_results_manifest.json')['files']),params_both=45798,
                both_active_state=True,raw_previous_actual_sameweights_forward_exact=True,
                centered_coordinates=30,common_rawFIT_scalers=True,last2_unchanged=True,
                actual_GPU8_FIT_passed=True,actual_both2epoch_resume_exact=True,
                benchmark=bench,memory_preflight=pre['memory'],memory_latest=read(HERE/'RESOURCE_USAGE.json'),
                scientific_scope='raw_vs_conditionally_centered_state32 only; wholeoldF16decoderScorer fixed',
                independent_external_review=False,no_model_forward_this_review=True,
                design_only_status_is_preregistration_snapshot=True)
    out=HERE/'prelaunch_review';out.mkdir(exist_ok=True)
    (out/'REVIEW.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
