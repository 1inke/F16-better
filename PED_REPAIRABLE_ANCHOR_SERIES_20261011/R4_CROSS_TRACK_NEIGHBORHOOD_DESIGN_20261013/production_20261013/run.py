import common as c,independent_math as im
import numpy as np,time
def main():
 t=time.perf_counter();lock=c.read(c.H/"source_manifest.json");c.stage("INITIAL_SOURCE_OLD_SHA",done=0,total=6);c.verify(lock["files"]);d,m,core,con,fn=c.inputs();geometries=[];out=[]
 for fold in range(3):
  ztr,zva,tr,va=c.fold(fold);bank=c.k.make_bank(ztr,d["ids"][tr],d["track"][tr]);query=c.k.make_bank(zva,d["ids"][va],d["track"][va]);g=dict(train_mask=tr,valid_mask=va,train_z=ztr,valid_z=zva)
  for label,q in (("train_LOTO",bank),("validation",query)):
   base=c.H/f"fold_{fold}/{label}";done=base/"done.json";c.stage("LABEL_BLIND_"+label.upper(),fold=fold,done=fold*2+(0 if label=="train_LOTO" else 1),total=6,queries=len(q.z))
   if done.exists():
    saved=c.read(done);c.done_valid(saved,lock["source_sha"]);a=c.npz(base/"geometry.npz");assert np.array_equal(a["features"],(ztr if label=="train_LOTO" else zva)) and np.array_equal(a["sample_id"],d["ids"][tr if label=="train_LOTO" else va])
   else:
    ix,dist=c.k.retrieve(q.z,q.tracks,bank,guard=c.guard);a=c.unit_payload(ztr if label=="train_LOTO" else zva,d["ids"][tr if label=="train_LOTO" else va],d["track"][tr if label=="train_LOTO" else va],bank,ix,dist)
    c.save(f"fold_{fold}/{label}/geometry.npz",**a);c.write(done,dict(status="COMPLETE",source_sha=lock["source_sha"],files={str(base/"geometry.npz"):c.sha(base/"geometry.npz")},query_count=len(q.z),no_labels_in_retrieval=True))
   out.extend([base/"geometry.npz",done]);completed=(fold*2)+(1 if label=="train_LOTO" else 2)
   c.stage("LABEL_BLIND_UNIT_COMPLETE",fold=fold,unit=label,done=completed,total=6)
   if label=="train_LOTO":
    weights=c.k.calibration_weights(d["track"][tr]);radius=c.k.weighted_q95(a["distances"][:,-1],weights);g.update(train_indices=a["indices"],train_distances=a["distances"],train_weights=weights,radius=radius)
   else:g.update(indices=a["indices"],distances=a["distances"])
  c.save(f"fold_{fold}/radius.npz",weights=g["train_weights"],radius=np.array(g["radius"]),train_id=d["ids"][tr],train_track=d["track"][tr]);out.append(c.H/f"fold_{fold}/radius.npz");geometries.append(g)
 c.stage("ALL_SIX_GEOMETRY_COMPLETE_POSTHOC_PNU",done=6,total=6)
 fields,p=c.posthoc(geometries,d)
 c.save("all_candidate_posthoc.npz",sample_id=d["ids"],query_labels=d["labels"],**fields);c.save("primary.npz",**p)
 stats=c.s.six_statistics(p["enrichment"],p["supported"],p["rank"],p["groups"],p["track"],guard=c.guard)
 c.save("bootstrap.npz",track_ids=stats["track_ids"],draw_counts=stats["draw_counts"],values=stats["values"]);c.write("STATISTICS.json",dict(cells=stats["cells"],statistics=stats["statistics"]))
 desc=c.descriptions(d,fields,p);c.write("DESCRIPTIVE.json",desc)
 # In-production saved PNU and component checks; full independent nearest-track audit is separate.
 for g in geometries:
  tr=g["train_mask"];bank=c.k.make_bank(g["train_z"],d["ids"][tr],d["track"][tr]);c.eq(c.k.label_support(g["indices"],bank,d["labels"][tr]),im.labels(g["indices"],bank,d["labels"][tr]))
 for path in ("all_candidate_posthoc.npz","primary.npz","bootstrap.npz","STATISTICS.json","DESCRIPTIVE.json"):out.append(c.H/path)
 c.stage("FINAL_SOURCE_OUTPUT_SHA",done=6,total=6);c.verify(lock["files"]);c.guard();sha={str(path):c.sha(path) for path in out};c.verify(sha)
 c.write("RESULT.json",dict(status="PASS_R4_DIAGNOSTIC_EXECUTION",source_sha=lock["source_sha"],source_count=len(lock["files"]),output_sha=sha,geometry_units=6,validation_queries=7680,training_LOTO_queries=sum(len(g["train_indices"]) for g in geometries),radii=[g["radius"] for g in geometries],statistics=stats["statistics"],cells=stats["cells"],primary=desc["primary"],all_P_secondary_count=329,model_fits=0,neural_forward=False,solver_labels=False,B_started=False,S2_started=False,formal_increment=0,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-t,full_independent_audit_still_required=True))
 c.stage("SUCCESS",done=6,total=6)
if __name__=="__main__":main()
