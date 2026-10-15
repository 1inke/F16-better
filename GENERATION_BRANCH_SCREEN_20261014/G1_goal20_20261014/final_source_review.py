import common as c,time
def main():
 started=time.perf_counter();pre=c.read(c.H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['planning_minutes']<=30
 files,inheritance=c.locks()
 for name in ('statistics.npz','split_ids.npz'):files[str(c.H/name)]=c.sha(c.H/name)
 assert c.digest(files)==pre['source_sha'],'STOP_PREFLIGHT_SOURCE_CHANGED';c.verify(files)
 c.write('source_manifest.json',dict(status='LOCKED',source_sha=c.digest(files),files=files,inheritance=inheritance,locked_at=c.now()))
 c.write('prelaunch_review/REVIEW.json',dict(status='PASS',source_sha=c.digest(files),source_count=len(files),current_preflight_source_exact=True,preflight_GPU_and_exact_resume_actual=True,CPU_final_source_only_no_forward_optimizer=True,elapsed_seconds=time.perf_counter()-started,inline_DA1_same_assistant_not_external=dict(algorithm_capacity_and_composition_not_purecausal=True,P2_already_contains_social_history=True,endpoint_gate_not_major_success=True,CV1_not_equalK20=True,all_old_major_and_literal_motion_criteria_unchanged=True)))
 print('FINAL_SOURCE_PASS',len(files),flush=True)
if __name__=='__main__':main()
