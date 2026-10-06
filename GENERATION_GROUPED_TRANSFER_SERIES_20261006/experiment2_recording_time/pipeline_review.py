"""Actual frozen saved-output pipeline smoke; no optimizer or new scientific fit."""
import adapter as h
import evaluate as ev
import numpy as np,torch,time,traceback
def main():
    start=time.perf_counter();audit=h.read(h.HERE/'preflight.json');assert audit['status']=='PASS'
    h.verify(audit['source_hashes']);cache,values,splits=h.load_data();outer=splits['outer_eval_index'];chosen=np.concatenate([outer[cache['change_group'][outer]==g][:1] for g in range(3)]);idx=np.r_[chosen,[i for i in outer if i not in set(chosen)][:125]];assert len(idx)==128 and len(np.unique(idx))==128
    cc={k:v[idx] for k,v in cache.items()};vv=[v[idx] for v in values[:4]];Class=h.initialize_classes();device=torch.device('cuda:0')
    saved=torch.load(h.HERE/'resume_smoke/random/continuous/seed_7/balanced_protected/final.pth',map_location='cpu',weights_only=False)
    model=Class(940,saved['config']).to(device);model.load_state_dict(saved['state_dict']);h.frozen_audit(model)
    native,nstamp=ev.saved_job('preflight_smoke_native128',None,cc,vv,device)
    output,stamp=ev.saved_job('preflight_smoke_projected128',model,cc,vv,device)
    q,motion=ev.quantities(output['trajectory'],vv[3],cc['logits']);nq,nmotion=ev.quantities(native['trajectory'],vv[3],cc['logits'])
    rows=ev.summary(q,motion,cc['change_group'],nq,nmotion,'smoke',7,'twoepoch_NOT_PRODUCTION','outer128')
    # Smoke originalCAL loader only alignment/IDs, not selecting or training on CAL.
    cal,cvalues=ev.original_cal_data();assert len(cal['sample_id'])==2879 and not set(cal['sample_id'])&set(cache['sample_id'])
    for x in (*nstamp['outputs'].items(),*stamp['outputs'].items()):h.verify(dict([x]))
    h.frozen_audit(model);h.verify(audit['source_hashes'])
    report=dict(status='PASS',actual_outer128_saved_quality_complete_link_motion_EXACT=True,allparams_bool_scalers_frozen=True,projection_saved_physics=stamp['physical'],source_hashes=audit['source_hashes'],original_CAL_ids=2879,CAL_not_used_for_selection=True,not_new_training=True,resource=h.guard.check(),elapsed_seconds=time.perf_counter()-start,rows=rows)
    h.write(h.HERE/'pipeline_review.json',report);print('PIPELINE_REVIEW_PASS',report['elapsed_seconds'],flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        d=h.HERE/'failures/pipeline_review1';d.mkdir(parents=True,exist_ok=True);(d/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
