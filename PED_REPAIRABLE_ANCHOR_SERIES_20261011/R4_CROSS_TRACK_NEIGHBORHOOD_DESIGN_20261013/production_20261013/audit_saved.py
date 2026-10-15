import common as c,independent_math as im
import numpy as np,time,traceback
def main():
 t=time.perf_counter();lock=c.read(c.H/"source_manifest.json");r=c.read(c.H/"RESULT.json");c.verify(lock["files"]);c.verify(r["output_sha"])
 assert c.read(c.H/"worker_state.json")["status"]==c.read(c.H/"exit_status.json")["status"]=="SUCCESS"
 d,m,core,con,fn=c.inputs();gs=[];cert=[];maxerr=0.
 for fold in range(3):
  z,tr,va=im.features(d,fold,m[fold]);bank=c.k.make_bank(z[tr],d["ids"][tr],d["track"][tr]);g=dict(train_mask=tr,valid_mask=va,train_z=z[tr],valid_z=z[va])
  for label,mask in (("train_LOTO",tr),("validation",va)):
   saved=c.npz(c.H/f"fold_{fold}/{label}/geometry.npz");assert np.array_equal(saved["sample_id"],d["ids"][mask]) and np.array_equal(saved["track"],d["track"][mask]);c.eq(z[mask],saved["features"])
   ix,dist=im.retrieve(saved["features"].reshape(-1,200),saved["query_track"],bank)
   assert np.array_equal(ix,saved["indices"]),("INDEPENDENT_NEIGHBOR_KEYS",fold,label)
   maxerr=max(maxerr,float(np.max(abs(dist-saved["distances"]))));c.eq(dist,saved["distances"])
   for key,expect in (("neighbor_id",bank.ids[ix]),("neighbor_slot",bank.slots[ix]),("neighbor_track",bank.tracks[ix])):assert np.array_equal(saved[key],expect)
   delta=saved["features"].reshape(-1,200)[:,None,:]-bank.z[ix];components=np.stack([(delta[:,:,:136]**2).sum(-1)/200,(delta[:,:,136:]**2).sum(-1)/200],-1);c.eq(components,saved["components"]);c.eq(components.sum(-1),dist**2)
   cert.append(dict(fold=fold,unit=label,queries=len(ix),independent_nearest_distinct_tracks=True))
   if label=="train_LOTO":g.update(train_indices=ix,train_distances=saved["distances"])
   else:g.update(indices=ix,distances=saved["distances"])
  rr=c.npz(c.H/f"fold_{fold}/radius.npz");tracks=d["track"][tr];weight=np.repeat(np.array([1/(len(np.unique(tracks))*np.sum(tracks==v)*20) for v in tracks]),20)
  c.eq(weight,rr["weights"]);pairs=sorted(zip(g["train_distances"][:,-1],weight),key=lambda v:v[0]);cum=0.;radius=None
  for value,w in pairs:
   cum+=w
   if cum>=.95*sum(weight):radius=value;break
  assert radius==float(rr["radius"]),(radius,float(rr["radius"]));g.update(radius=radius);gs.append(g)
 fields,p=c.posthoc(gs,d);saved=c.npz(c.H/"all_candidate_posthoc.npz");assert np.array_equal(saved.pop("sample_id"),d["ids"]) and np.array_equal(saved.pop("query_labels"),d["labels"]);c.eq(fields,saved)
 for g in gs:
  tr=g["train_mask"];va=g["valid_mask"];bank=c.k.make_bank(g["train_z"],d["ids"][tr],d["track"][tr]);v=im.labels(g["indices"],bank,d["labels"][tr])
  for key,expected in v.items():c.eq(expected.reshape((va.sum(),20)+expected.shape[1:]),fields[key][va])
 prim=c.npz(c.H/"primary.npz");c.eq(p,prim)
 # Independent best-P identity selection, without reading any neighbor score.
 ranks=np.argsort(np.argsort(-d["pairwise_scores"],axis=1,kind="stable"),axis=1)+1;best=np.where(d["labels"]==1,ranks,21).argmin(1);rows=np.flatnonzero((d["labels"]==1).any(1));assert np.array_equal(rows,p["window_index"]) and np.array_equal(best[rows],p["physical_slot"])
 bs=c.npz(c.H/"bootstrap.npz");values=im.statistics(p,bs);record=c.read(c.H/"STATISTICS.json")
 for j,a in enumerate(record["statistics"]):
  finite=np.isfinite(values[:,j]);assert int(finite.sum())==a["valid_draws"] and int((~finite).sum())==a["invalid_draws"]
  if a["interval"] is not None:c.eq(np.quantile(values[finite,j],[.05/12,1-.05/12]),np.array(a["interval"]))
 calc=c.s.six_statistics(p["enrichment"],p["supported"],p["rank"],p["groups"],p["track"],guard=c.guard);c.eq(record,c.s.json_safe(dict(cells=calc["cells"],statistics=calc["statistics"])))
 c.eq(c.read(c.H/"DESCRIPTIVE.json"),c.s.json_safe(c.descriptions(d,fields,p)))
 # Independently reconstruct window-class macro, not candidate-weighted.
 desc=c.read(c.H/"DESCRIPTIVE.json")["all_P_N_U_window_macro"];pop={"overall":np.ones(384,bool),**{"group_"+str(v):d["groups"]==v for v in range(3)},**{"fold_"+str(v):d["fold"]==v for v in range(3)},**{"scene_"+str(v):d["scene"]==v for v in np.unique(d["scene"])}}
 for name,mask in pop.items():
  for key,expected in desc[name].items():
   vals=fields[key][mask].astype(float);y=d["labels"][mask]
   for code,label in ((1,"P"),(0,"N"),(-1,"U")):
    use=y==code;finite=use&np.isfinite(vals);means=[float(v[m].mean()) for v,m in zip(vals,finite) if m.any()]
    rec=dict(eligible_candidates=int(use.sum()),finite_candidates=int(finite.sum()),eligible_windows=int(use.any(1).sum()),finite_windows=len(means),mean=float(np.mean(means)) if means else None);c.eq(rec,expected[label])
 c.verify(lock["files"]);c.verify(r["output_sha"]);c.guard()
 c.write("completion_review/REVIEW.json",dict(status="PASS",source_sha=lock["source_sha"],source_count=len(lock["files"]),output_count=len(r["output_sha"]),all_six_full_nearest_track_geometry_radius_PNU_primary_macro_sixCI_SHA_PASS=True,geometry_max_error=maxerr,certificates=cert,zero_model_fits=True,no_neural_forward_or_labelsolver=True,not_external_review=True,elapsed_seconds=time.perf_counter()-t,resource=c.RESOURCE))
 print("AUDIT_PASS",flush=True)
if __name__=="__main__":
 try:main()
 except BaseException as e:traceback.print_exc();c.write("completion_review/FAILURE.json",dict(error=repr(e),traceback=traceback.format_exc(),resource=c.RESOURCE));raise
