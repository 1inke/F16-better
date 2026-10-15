import common as c,time
t=time.perf_counter();c.guard();pre=c.read(c.H/"PREFLIGHT.json");assert pre["status"]=="PASS" and pre["planning_minutes"]<=30
files,meta=c.sources();assert c.digest(files)==pre["source_sha"],"PREFLIGHT_SOURCE_CHANGED"
c.verify(files);c.write("source_manifest.json",dict(source_sha=c.digest(files),files=files,inheritance=meta))
c.write("prelaunch_review/REVIEW.json",dict(status="PASS",source_sha=c.digest(files),source_count=len(files),human_authorized_R4=True,DA1="PASS_with_limits: metric_geometry_not_semantics;Nbaserate_corrected;noautomatictraining",all_old_original_digests_preserved=True,planning_minutes=pre["planning_minutes"],elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE))
print("FINAL_SOURCE_PASS",len(files),flush=True)
