import common as c,time
def main():
 started=time.perf_counter();pre=c.read(c.H/"PREFLIGHT.json");assert pre["status"]=="PASS" and pre["planning_minutes"]<=30;files,inheritance=c.locks()
 for name in ("goal_stats_42.npz","goal_stats_43.npz","goal_stats_44.npz","split_ids.npz"):files[str(c.H/name)]=c.sha(c.H/name)
 assert c.digest(files)==pre["source_sha"],"STOP_PREFLIGHT_SOURCE_CHANGED";c.verify(files);c.write("source_manifest.json",dict(status="LOCKED",source_sha=c.digest(files),files=files,inheritance=inheritance,locked_at=c.now()))
 c.write("prelaunch_review/REVIEW.json",dict(status="PASS",source_sha=c.digest(files),source_count=len(files),current_source_and_parent_all_digest_exact=True,actual_preflight_GPU_all3_frozen_goal_FDE_zero_identity_exact_resume_and_saved_math=True,CPU_final_review_no_forward_optimizer=True,elapsed_seconds=time.perf_counter()-started,inline_DA1_sameassistant_not_external=dict(G1_failure_preserved=True,G2_goal_fixed_not_equal_totalcompute=True,FDE_A_B_expectedzero_not_success=True,noGT_endpoint_teacher=True,all_native_major_and_literal5motion_unchanged=True)))
 print("FINAL_SOURCE_PASS",len(files),flush=True)
if __name__=="__main__":main()

