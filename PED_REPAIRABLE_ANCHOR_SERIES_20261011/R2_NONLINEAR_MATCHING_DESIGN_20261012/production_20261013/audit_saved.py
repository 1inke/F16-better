import common as c,saved_math as sm
import numpy as np,time
def main():
 t=time.perf_counter();c.guard();lock=c.load_locked();r=c.read(c.H/"RESULT.json");assert r["status"]=="PASS_R2_EXECUTION"
 assert c.read(c.H/"worker_state.json")["status"]==c.read(c.H/"exit_status.json")["status"]=="SUCCESS";c.verify(r["output_sha"]);d=c.inputs();pol=c.npz(c.H/"policy_slots.npz");assert np.array_equal(pol.pop("sample_id"),d["ids"])
 maxerr=0.;certs=[]
 for arm in ("linear","rbf64"):
  oof=c.npz(c.H/arm/"OOF.npz");assert np.array_equal(oof["sample_id"],d["ids"]);rebuilt=np.full((384,20),np.nan)
  for fold in range(3):
   c.guard();tr,va=c.ic.split_masks(d,fold);p=c.H/arm/("fold_"+str(fold));a=c.npz(p/"model.npz");man=c.read(p/"run_manifest.json");done=c.read(p/"done.json");c.done_valid(done,lock["source_sha"],{k:v for k,v in r["output_sha"].items() if str(p) in k and not k.endswith("done.json")})
   assert man["source_sha"]==lock["source_sha"] and man["model_sha"]==c.sha(p/"model.npz");assert np.array_equal(man["train_ids"],d["ids"][tr]);assert np.array_equal(man["validation_ids"],d["ids"][va])
   certificate,ts,vs=sm.arrays(d,fold,arm,a);pred=c.npz(p/"predictions.npz");fc=c.npz(p/"fit_contract.npz")
   assert np.array_equal(pred["train_ids"],d["ids"][tr]) and np.array_equal(pred["validation_ids"],d["ids"][va])
   assert np.array_equal(fc["weight"],sm.weights(d["labels"][tr],d["track"][tr]));assert np.array_equal(fc["known"],d["labels"][tr]!=-1) and np.array_equal(fc["target"],d["labels"][tr]==1)
   err=max(float(abs(ts-pred["train_scores"]).max()),float(abs(vs-pred["validation_scores"]).max()));maxerr=max(maxerr,err);assert err<=1e-10
   assert np.array_equal(sm.rank(ts),pred["train_slots"]) and np.array_equal(sm.rank(vs),pred["validation_slots"]);rebuilt[va]=vs;certs.append(dict(arm=arm,fold=fold,score_error=err,**certificate))
   expected=c.read(c.H/"train_diagnostics.json");entry=next(x for x in expected if x["arm"]==arm and x["fold"]==fold)
   for key,row in entry["metrics"].items():
    if key=="overall":mask=np.ones(int(tr.sum()),bool)
    elif key in ("stable","transition","change"):mask=d["groups"][tr]==("stable","transition","change").index(key)
    elif key.startswith("scene_"):mask=d["scene"][tr]==key[6:]
    else:mask=d["fold"][tr]==int(key[5:])
    sm.equal(sm.describe(pred["train_slots"][mask],d["labels"][tr][mask],pred["train_scores"][mask]),row)
  assert abs(rebuilt-oof["scores"]).max()<=1e-10 and np.array_equal(sm.rank(rebuilt),pol[arm])
  assert np.array_equal(oof["slots"],pol[arm])
  if arm=="linear":assert abs(oof["scores"]-d["old_witness_scores"]).max()<=1e-10 and np.array_equal(pol[arm],d["reference_slots"]["old_witness"])
 for arm,slots in pol.items():
  score=c.npz(c.H/arm/"OOF.npz")["scores"] if arm in ("linear","rbf64") else None
  if arm not in ("linear","rbf64"):assert np.array_equal(slots,d["reference_slots"][arm])
  for key,row in r["summary"][arm].items():
   if key=="overall":mask=np.ones(384,bool)
   elif key in ("stable","transition","change"):mask=d["groups"]==("stable","transition","change").index(key)
   elif key.startswith("scene_"):mask=d["scene"]==key[6:]
   else:mask=d["fold"]==int(key[5:])
   sm.equal(sm.describe(slots[mask],d["labels"][mask],None if score is None else score[mask]),row)
 op=(d["labels"]==1).any(1);hit={k:(np.take_along_axis(d["labels"],v,axis=1)==1).any(1) for k,v in pol.items()};saved=c.npz(c.H/"bootstrap_draws.npz");gain=True
 for j,ref in enumerate(("linear","envelope","fixed_random","bottom")):
  delta=hit["rbf64"].astype(float)-hit[ref];draw=sm.bootstrap(d["track"],op,delta,20261012+j);assert np.array_equal(draw,saved[ref]);ci=np.quantile(draw,[.00625,.99375]);z=r["screen"]["comparisons"][ref];assert np.array_equal(ci,z["CI9875"]) and z["estimate"]==delta[op].mean();gain &= z["estimate"]>=.05 and ci[0]>0
 gate=bool(gain)
 for fold in range(3):
  va=(d["fold"]==fold)&op;change=va&(d["groups"]==2);counts={k:int(v[change].sum()) for k,v in hit.items()};expected=r["screen"]["folds"][fold];assert counts==expected["change_hits"];support=int(va.sum())>=20 and len(np.unique(d["track"][va]))>=10;ok=bool(change.any() and all(counts["rbf64"]>=counts[ref] for ref in ("linear","envelope","fixed_random","bottom")));assert support==expected["support_gate"] and ok==expected["change_gate"];gate&=support and ok
 assert bool(gate)==r["screen"]["statistical_screen_only"] and bool(gain)==r["screen"]["gain_gate"]
 c.verify(lock["files"]);c.verify(r["output_sha"]);c.guard()
 c.write("completion_review/REVIEW.json",dict(status="PASS",source_sha=lock["source_sha"],source_count=len(lock["files"]),output_count=len(r["output_sha"]),all6_manifests_contracts_models_scores_train_OOF_groups_Prank_CI_gate=True,score_max_error=maxerr,certificates=certs,linear_R1_EXACT_slots=True,no_fit_solver_torch_neural_forward=True,not_external_review=True,resource=c.RESOURCE,elapsed_seconds=time.perf_counter()-t));print("AUDIT_PASS",flush=True)
if __name__=="__main__":main()
