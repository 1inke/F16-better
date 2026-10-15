"""CPU-only budget release; preserve old snapshot, explicitly reconcile exactly two controls."""
import common as c
def main():
 c.guard();pre=c.read(c.H/"PREFLIGHT.json");approval=c.read(c.H/"BUDGET_EXCEPTION.json");old=c.read(c.H/"preflight_source_snapshot.json")
 assert pre["status"]=="STOP_BUDGET" and pre["planning_minutes"]==approval["approved_planning_minutes"]
 assert approval["status"]=="HUMAN_APPROVED_SINGLE_ITEM_BUDGET_EXCEPTION"
 assert pre["planning_minutes"]<=approval["launch_budget_limit_minutes"] and len(old["files"])==3930
 assert old["source_sha"]==c.digest(old["files"])
 allowed={str(c.H/name) for name in ("final_source_review.py","start_independent.ps1")};transition=[];files=old["files"].copy()
 for i,(path,expected) in enumerate(old["files"].items()):
  if i%32==0:c.guard()
  actual=c.sha(path)
  if path in allowed:
   original=c.H/"budget_release_provenance"/(c.H.joinpath(path).name+".original")
   assert c.sha(original)==expected,("ORIGINAL_CONTROL_COPY_NOT_EXACT",path)
   assert actual!=expected,("EXPECTED_AUTHORIZED_CONTROL_CHANGE_MISSING",path)
   transition.append(dict(path=path,old_sha=expected,new_sha=actual,original_copy=str(original),scope="budget/source launch control only"))
   files[path]=actual
  else:assert actual==expected,("UNAUTHORIZED_SOURCE_CHANGE",path)
 assert len(transition)==2
 c.write("budget_release_provenance/PARENT_SOURCE_TRANSITION.json",dict(status="PASS",original_source_count=3930,original_snapshot_sha=old["source_sha"],authorized_control_changes=transition,other3928_sources_EXACT=True,science_model_loss_data_gates_unchanged=True))
 exclude={"runtime_state.json","source_manifest.json","LAUNCH_GATE.json","progress.json","worker_state.json","exit_status.json","worker.lock","worker_stdout.log","worker_stderr.log"}
 additions=[p for p in c.H.iterdir() if p.is_file() and p.name not in exclude]
 for sub in ("budget_release_provenance","failures"):additions+=list((c.H/sub).rglob("*"))
 for p in additions:
  if not p.is_file():continue
  key=str(p);value=c.sha(p)
  if key in files:assert files[key]==value,("NEW_LOCK_CONFLICT",key)
  else:files[key]=value
 c.verify(files);c.guard();source_sha=c.digest(files)
 assert pre["paired_all_initial_state_EXACT"] and pre["geometry"]["S"]==650
 assert set(pre["arms"])=={"frozen_goal","joint_goal"} and pre["stats_chunk_error"]<1e-11
 lock=dict(source_sha=source_sha,files=files,inheritance=dict(original3930_digests_preserved_in_preflight_snapshot=True,authorized_launch_control_changes=2,other3928_EXACT=True))
 c.write("source_manifest.json",lock)
 c.write("LAUNCH_GATE.json",dict(status="PASS",source_sha=source_sha,source_count=len(files),planning_minutes=pre["planning_minutes"],approved_range=approval["approved_duration_range_minutes"],original_preflight_STOP_preserved=True,human_single_item_exception=True))
 c.write("prelaunch_review/REVIEW.json",dict(status="PASS",source_sha=source_sha,source_count=len(files),planning_minutes=pre["planning_minutes"],actual_GPU_resume_evidence_reused_on_EXACT_science_not_replayed=True,CPU_only_final_review=True,resource=c.RESOURCE,authorized_control_changes=transition,original_budgetSTOP_review_preserved=True))
 print("FINAL_SOURCE_BUDGET_RELEASE_PASS",len(files),source_sha,flush=True)
if __name__=="__main__":main()
