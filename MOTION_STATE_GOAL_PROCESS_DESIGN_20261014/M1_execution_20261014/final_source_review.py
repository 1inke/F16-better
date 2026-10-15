import common as c,ast,time
def main():
 started=time.perf_counter();c.guard();pre=c.read(c.H/"PREFLIGHT.json");approval=c.read(c.H/"BUDGET_EXCEPTION.json")
 previous=c.read(c.H/"budget_review/SOURCE_SNAPSHOT.json");assert previous["status"]=="PRESERVED_NOT_PRODUCTION_SOURCE_LOCK"
 assert pre["status"]=="STOP_BUDGET" and pre["planning_minutes"]<=approval["launch_budget_limit_minutes"]
 assert approval["status"]=="HUMAN_APPROVED_SINGLE_ITEM_BUDGET_EXCEPTION" and approval["scope"]=="M1 execution only"
 assert pre["paired_all_initial_params_buffers_EXACT"] and pre["raw_vlast_G1_EXACT"] and pre["physics_count_12_11_10_EXACT"] and pre["stats_chunk_error"]<1e-11
 assert previous["preflight_sha"]==c.sha(c.H/"PREFLIGHT.json") and previous["benchmark_sha"]==c.sha(c.H/"BENCHMARK.json")
 oldfiles=previous["files"];changed=[]
 allowed={str(c.H/n):n for n in ("final_source_review.py","start_independent.ps1")}
 for path,expected in oldfiles.items():
  if path in allowed:
   preserved=c.H/"budget_release_provenance"/("original_"+allowed[path]);assert c.sha(preserved)==expected
   current=c.sha(path);changed.append(dict(path=path,old_sha=expected,new_sha=current,reason="human-approved M1 single-item launch budget gate only"))
  else:assert c.sha(path)==expected,("STOP_SCIENTIFIC_SOURCE_CHANGED",path)
 files,meta=c.locks()
 for name in ("statistics.npz","split_ids.npz","fixed_feature_fixture.npz"):files[str(c.H/name)]=c.sha(c.H/name)
 assert set(oldfiles)==set(files) and len(files)==3738
 for path,expected in oldfiles.items():
  assert path in allowed or files[path]==expected,("UNAUTHORIZED_SOURCE_TRANSITION",path)
 for arm in ("last_velocity","dynamic_history"):
  a=pre["arms"][arm];assert a["parameters"]==34146 and all(x>0 for x in a["actual_GPU8FIT256_all_legal_gradients_positive_finite"].values())
  assert a["tokens_physical_zero_normalization_mask_EXACT"] and a["saved128_full_quality_SciPyclusters_motion_frozen_EXACT"]
  assert all(x["error"]<=1e-8 and x["no_active_face_switch"] for x in a["double_parameter_FD"])
  assert a["resume"]["actualFIT513_inner12_SUBSET_twoepoch_ALLparams_buffers_optimizer_TorchCPU_CUDA_loader_numpy_pythonRNG_history_EXACT"]
  assert a["done_bad_source_output_rejected"]
 for path in c.H.glob("*.py"):ast.parse(path.read_text(encoding="utf8"))
 c.write("budget_release_provenance/PARENT_SOURCE_TRANSITION.json",dict(status="PASS",original_sources=len(oldfiles),unchanged_scientific_sources=len(oldfiles)-2,authorized_control_differences=changed,
  original_source_table_sha=c.sha(c.H/"budget_review/SOURCE_SNAPSHOT.json"),no_model_loss_data_statistics_or_training_change=True))
 extra=[c.H/"BUDGET_EXCEPTION.json",c.H/"BUDGET_RELEASE.md",c.H/"PREFLIGHT.json",c.H/"BENCHMARK.json",c.H/"budget_review/SOURCE_SNAPSHOT.json",
  c.H/"budget_release_provenance/PARENT_SOURCE_TRANSITION.json",c.H/"budget_release_provenance/original_final_source_review.py",c.H/"budget_release_provenance/original_start_independent.ps1"]
 for path in extra:files[str(path)]=c.sha(path)
 c.verify(files);c.guard();source=c.digest(files)
 c.write("source_manifest.json",dict(status="LOCKED",source_sha=source,files=files,inheritance=meta,parent_source_transition="budget_release_provenance/PARENT_SOURCE_TRANSITION.json",locked_at=c.now()))
 c.write("LAUNCH_GATE.json",dict(status="PASS",numeric_preflight_original_sha=c.sha(c.H/"PREFLIGHT.json"),numeric_preflight_reused_scientific_sources_EXACT=True,
  original_budget_STOP_preserved=True,single_item_human_budget_exception=True,planning_minutes=pre["planning_minutes"],source_sha=source,source_count=len(files)))
 c.write("prelaunch_review/REVIEW.json",dict(status="PASS",source_sha=source,source_count=len(files),scientific_source_unchanged_count=len(oldfiles)-2,
  authorized_control_changes=changed,current_resource=c.RESOURCE,CPU_source_AST_resource_only_no_new_GPU_forward_optimizer_or_resume=True,
  reused_actual_GPU_preflight_on_EXACT_scientific_sources=True,inline_ARS_DA1_sameassistant_not_external=dict(P2_already_may_contain_history=True,nominal_not_equal_effective_capacity=True,
  only_budget_exception_science_gates_unchanged=True,old_G2_bridge_and_old_paused_tasks_not_restarted=True),elapsed_seconds=time.perf_counter()-started))
 print("PRELAUNCH_PASS",len(files),source,flush=True)
if __name__=="__main__":main()
