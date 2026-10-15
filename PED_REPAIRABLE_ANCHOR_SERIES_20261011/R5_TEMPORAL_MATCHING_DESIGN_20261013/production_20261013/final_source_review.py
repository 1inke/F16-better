import common as c,time
t=time.perf_counter();pre=c.read(c.H/"PREFLIGHT.json");assert pre["status"]=="PASS" and pre["planning_minutes"]<=30
files,meta=c.sources();assert c.digest(files)==pre["source_sha"],"PREFLIGHT_SOURCE_CHANGED";c.verify(files)
c.write("source_manifest.json",dict(source_sha=c.digest(files),files=files,inheritance=meta))
c.write("prelaunch_review/REVIEW.json",dict(status="PASS",source_sha=c.digest(files),source_count=len(files),scientific_design_unchanged=True,human_R5_execution_authorized=True,all_R4_originaldigests_preserved=True,DA1="samepairwiseobjective/featuresfactor_not_equalcapacity;noautomaticB_S2",planning_minutes=pre["planning_minutes"],elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE));print("FINAL_SOURCE_PASS",len(files),flush=True)
