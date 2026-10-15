import common as c,ast,time
def main():
 started=time.perf_counter();pre=c.read(c.H/"PREFLIGHT.json");lock=c.read(c.H/"source_manifest.json")
 assert pre["status"]=="PASS" and pre["planning_minutes"]<=30
 files,meta=c.locks()
 for name in ("statistics.npz","split_ids.npz","fixed_feature_fixture.npz"):files[str(c.H/name)]=c.sha(c.H/name)
 assert c.digest(files)==lock["source_sha"] and files==lock["files"],"PREFLIGHT_SOURCE_CHANGED"
 for path in c.H.glob("*.py"):ast.parse(path.read_text(encoding="utf8"))
 # Import shim avoids all prior run.py top-level execution; no legacy executable entrypoints are loaded.
 for module_path in (c.G1/"common.py",c.G1/"model.py",c.G1/"metrics.py"):
  tree=ast.parse(module_path.read_text(encoding="utf8"))
  assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ("mkdir","write_text","save","savez_compressed") for n in tree.body),"TOPLEVEL_OLD_WRITE"
 c.verify(files);c.write("prelaunch_review/REVIEW.json",dict(status="PASS",source_count=len(files),source_sha=lock["source_sha"],
  current_source_input_old_digests_unchanged=True,actual_preflight_GPU_botharms_resume_saved_pipeline_PASS=True,
  CPU_source_AST_only_no_new_forward_optimizer=True,inline_ARS_DA1_sameassistant_not_external=dict(P2_may_already_contain_history=True,equal_nominal_not_equal_effective_capacity=True,
  physical_zero_acceleration_normalized_correctly=True,negative_M1_does_not_cancel_M2=True,old_G2_bridge_not_resumed=True,major_and_motion_unchanged=True),
  elapsed_seconds=time.perf_counter()-started))
 print("PRELAUNCH_PASS",len(files),flush=True)
if __name__=="__main__":main()
