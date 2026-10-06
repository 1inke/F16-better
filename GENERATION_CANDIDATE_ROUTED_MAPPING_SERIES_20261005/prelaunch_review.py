"""Read-only scientific review after actual preflight/resume/benchmark."""
import run_head as h
import json
from datetime import datetime
import torch
if __name__=='__main__':
    pre=json.loads((h.HERE/'preflight.json').read_text(encoding='utf-8'))
    resume=json.loads((h.HERE/'resume_check.json').read_text(encoding='utf-8'))
    bench=json.loads((h.HERE/'benchmark.json').read_text(encoding='utf-8'))
    assert pre['status']=='PASS' and resume['status']=='PASS' and bench['status']=='PASS_THROUGHPUT_ONLY'
    assert len(resume['reports'])==2 and all(x['params_buffers_optimizer_allRNG_history_calibration_exact'] for x in resume['reports'])
    sourcecount=h.verify(pre['source_hashes']);old=h.old_manifest()
    assert not bench['saved_model']
    scalers=h.physical.load(h.HERE)
    for arm in h.c.ARMS:
        path=h.HERE/'resume_smoke'/arm/'continuous/seed_7/balanced_protected/final.pth'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert torch.equal(saved['state_dict']['physical_accel_scales'],torch.from_numpy(scalers))
        assert saved['state_dict']['route_median'].dtype==torch.float64
        assert float(saved['state_dict']['route_median'])==pre['route_FIT']['median']
    memory=h.guard.check()
    out=h.HERE/'prelaunch_review';out.mkdir(exist_ok=True)
    record=dict(status='PASS_READY_FOR_SOLE_PRODUCTION_LAUNCH',checked_at=datetime.now().astimezone().isoformat(),
        reviewer='Same assistant inline review, not external independent reviewer',
        current_source_count=sourcecount,all_prior_files_locked=old,
        preflight_SHA=h.p.sha256(h.HERE/'preflight.json'),resume_SHA=h.p.sha256(h.HERE/'resume_check.json'),benchmark_SHA=h.p.sha256(h.HERE/'benchmark.json'),
        actual_GPU8FIT=True,both_trainable_parameters=91596,fullpaired_weights_buffers=True,
        both_originaltotal_plus_P_scalar_gradient_exact=True,both_raw_expert_parent_nonzero_outputs_exact=True,
        actual_physical_gradient=pre['response'],physical_math=pre['physical_math'],
        both_real_FITCAL_subset_2epoch_allparams_boolscalerbuffers_optimizer_allRNG_history_calibration='EXACT',
        no_fullFIT_twoepoch_claim=True,completed_seed_SHA_skip_verified=True,
        source_badoutput_reject=True,frozenwholeold_F16_decoder_scorer=True,
        process_budget_GiB=6,system_reserve_each_GiB=6,resource=memory,
        preflight_resource=pre['memory'],benchmark_compute_minutes=bench['compute_minutes'],
        estimated_inputs_MiB=162,physical_reference_batch128_MiB=.22,
        eighteen_uncompressed_DEVpaths_MiB=18*1516*20*12*2*4/2**20,all18_not_resident=True,
        one_model_batch128=True,benchmark_savedmodel=False,benchmark_not_gpu_isolation_or_production_guarantee=True,
        production_started=False,formal_counter_not_reset=True,
        limits=['Batch class hinge not population physical gate','Only accelmean added; jerk/p95 may fail',
          'May suppress legitimate changes; coverage-first, no native-only success',
          'FIT/CAL transfer and repeatedDEV limitations; local gradients not actual AdamW effect'])
    h.m.write_json(out/'REVIEW.json',record)
    (out/'FIELD_INTERPRETATION.md').write_text(
      '# Field interpretation\n\nzero = uniform mean of two bounded expert increments; mean_std = fixed FIT median route selecting one expert per native candidate. BOTH originaltotal+.1 physical feedback, rawstate32 and both clocks active. Two experts expected91596 nominal, not equal effectivecapacity/optimization. q is deterministic nativeprediction deviation from lastobservedvelocity, not GT event.\n\n'
      'Objective_history original/P/added/total are batch-n-weighted, while legacy history train_loss averages minibatches. These denominators differ and are not errors. coordinate_P_gradient_norm is an output-coordinate norm, not parameter gradient. Actual parameter gradients are preflight records.\n\n'
      'profile_sum_error legacy is nonzero sigmoid profile sum, not an error; drift contains total endpoint including pulse. Inherited native argmax is a slot index, not new calibrated path probability.\n\n'
      'Physical_population records originaltotal and P with full-split group denominator, not minibatch AdamW direction. No new CAL selection rule. Five group physical gates all remain1%.',encoding='utf-8')
    print(json.dumps(dict(status=record['status'],sourcecount=sourcecount,old=old,memory=memory)),flush=True)
