"""CPU saved-data/checkpoint audit; no optimizer step, network forward or old-model access."""
import common as c,metrics as v,baselines as b
import numpy as np,time
def retrieval_certificate(data,st,ix,donors,paths):
 tr=data["split"]["FIT"];track_keys,inv=np.unique(data["track"][tr],return_inverse=True);trainids=data["ids"][tr];velz=(data["vel"][tr]-st["velocity_mean"])/st["velocity_scale"];T=len(track_keys);lookup={int(sid):int(i) for i,sid in enumerate(data["ids"])}
 for row,index in enumerate(ix):
  c.guard();query=(data["vel"][index]-st["velocity_mean"])/st["velocity_scale"];dist=np.sum((velz-query)**2,axis=1);minimum=np.full(T,np.inf);np.minimum.at(minimum,inv,dist);idmin=np.full(T,np.iinfo(np.int64).max,np.int64);np.minimum.at(idmin,inv,np.where(dist==minimum[inv],trainids,np.iinfo(np.int64).max));valid=track_keys!=data["track"][index];kk=np.flatnonzero(valid);order=np.lexsort((idmin[kk],minimum[kk]));expected=idmin[kk[order[:20]]];assert np.array_equal(expected,donors[row]),"FULL_RETRIEVAL_KEYS"
  donorix=np.array([lookup[int(sid)] for sid in expected]);res=data["gt"][donorix].astype(float)-data["vel"][donorix,None]*(.4*np.arange(1,13))[None,:,None];rebuilt=data["vel"][index,None,None]*(.4*np.arange(1,13))[None,:,None]+res;assert np.array_equal(rebuilt,paths[row])
def one(path,data,indices,expected_metrics,st,kind):
 saved=c.npz(path);assert np.array_equal(saved["sample_id"],data["ids"][indices]) and len(np.unique(saved["sample_id"]))==len(indices)
 for key,expected in (("groups",data["groups"][indices]),("track",data["track"][indices]),("scene",data["scene"][indices]),("gt",data["gt"][indices]),("native20",data["native"][indices].astype(float))):assert np.array_equal(saved[key],expected)
 paths=saved["paths"]
 if kind=="learned":
  goals=4.8*data["vel"][indices,None]+st["centers"][None]+saved["goal_delta"].astype(float);assert np.array_equal(goals,saved["goals"]);expected=(np.arange(1,13)/12.)[None,None,:,None]*goals[:,:,None];assert np.array_equal(expected,paths)
 elif kind=="retrieval20":retrieval_certificate(data,st,indices,saved["donor_sample_id"],paths)
 elif kind=="native20":assert np.array_equal(paths,saved["native20"])
 else:assert np.array_equal(paths,b.build(data,st,indices,kind)[0])
 met,aux=v.evaluate(paths,saved["gt"],saved["groups"],saved["native20"],True);c.eq(c.safe(met),expected_metrics,2e-11);stored=c.npz(path.with_name(path.stem+"_metrics.npz"));assert np.array_equal(stored.pop("sample_id"),saved["sample_id"]) and np.array_equal(stored.pop("groups"),saved["groups"]) and np.array_equal(stored.pop("track"),saved["track"])
 for key,val in aux.items():
  if val.dtype.kind in "biu":assert np.array_equal(val,stored[key])
  else:assert np.max(abs(val-stored[key]))<=2e-11
 return dict(job=str(path),population=len(indices),K=paths.shape[1],full_clusters_motion_arithmetic_PASS=True)
def independent_bootstrap(data):
 rng=np.random.default_rng(20261014);records=[];seeds=(42,43,44)
 for split,ix in data["split"].items():
  learned=[c.npz(c.H/f"runs/seed_{seed}/{split}_metrics.npz")["vectors"] for seed in seeds]
  for group,mask in v.masks(data["groups"][ix]):
   tracks=data["track"][ix][mask];units=np.unique(tracks);members=[np.flatnonzero(tracks==t) for t in units];counts=np.array([len(a) for a in members]);T=len(units)
   for ref in ("native20","prototype20","retrieval20"):
    reference=c.npz(c.H/f"baseline/{split}_{ref}_metrics.npz")["vectors"][mask];diff=np.stack([a[mask]-reference for a in learned]);sums=np.stack([[diff[s,j].sum(0) for j in members] for s in range(3)]);draws=[]
    for _ in range(2000):
     c.guard();ss=rng.integers(0,3,3);tt=rng.integers(0,T,T);draws.append(sums[ss][:,tt].sum((0,1))/(3*counts[tt].sum()))
    records.append(dict(split=split,group=group,reference=ref,n=int(mask.sum()),tracks=T,metrics=("coverage","ADE","FDE","J"),estimate=diff.mean((0,1)),CI_family48=np.quantile(draws,[.05/96,1-.05/96],axis=0).T,perseed_estimate=diff.mean(1),exploratory_fixeddata_seedtrack=True))
 return records

def audit():
 import torch
 started=time.perf_counter();lock=c.read(c.H/"source_manifest.json");r=c.read(c.H/"RESULT.json");assert r["status"]=="PASS_G1_GOAL20_RUN";assert c.read(c.H/"worker_state.json")["status"]==c.read(c.H/"exit_status.json")["status"]=="SUCCESS";c.verify(lock["files"]);c.verify(r["output_sha"]);data=c.load();st=c.npz(c.H/"statistics.npz");tr=data["split"]["FIT"];assert np.array_equal(st["training_ids"],data["ids"][tr]);x=data["p2"][tr].astype(float);assert np.array_equal(st["mean"],x.mean(0).astype(np.float32)) and np.array_equal(st["std"],np.where(x.std(0)<1e-6,1.,x.std(0)).astype(np.float32));cert=[]
 for split,ix in data["split"].items():
  for kind in ("native20","CV1","prototype20","retrieval20"):cert.append(one(c.H/f"baseline/{split}_{kind}.npz",data,ix,r["baseline_metrics"][split][kind],st,kind))
 for record in r["records"]:
  seed=record["seed"];base=c.H/f"runs/seed_{seed}";man=c.read(base/"run_manifest.json");assert man["source_sha"]==lock["source_sha"] and man["statistics_sha"]==c.sha(c.H/"statistics.npz");c.done_valid(c.read(base/"done.json"),lock["source_sha"]);ck=torch.load(base/"checkpoint.pt",map_location="cpu",weights_only=False);assert ck["manifest"]==man and ck["epoch"]==25 and c.safe(ck["history"])==c.read(base/"history.json");assert len(ck["history"])==25 and all(x["n"]==18909 for x in ck["history"]);assert record["optimization_steps"]==1875
  for key in ("mean","std","centers"):assert np.array_equal(ck["model"][key].numpy(),st[key])
  assert ck["optimizer"]["param_groups"][0]["lr"]==.001 and ck["optimizer"]["param_groups"][0]["weight_decay"]==.0001;assert set(ck["rng"])=={"torch","cuda","loader","numpy","python"}
  for split,ix in data["split"].items():cert.append(one(base/f"{split}.npz",data,ix,record["splits"][split],st,"learned"))
 # Deterministic registered 48-record bootstrap arithmetic; no models/gradients replayed.
 rebuilt=independent_bootstrap(data);c.eq(c.read(c.H/"BOOTSTRAP.json")["records"],c.safe(rebuilt),2e-11)
 groupmask=data["groups"][data["split"]["outer"]]==2;n=int(groupmask.sum());native=r["baseline_metrics"]["outer"]["native20"]["change"]["good"];counts=[q["splits"]["outer"]["change"]["endpoint_good"] for q in r["records"]];assert r["G2_upper_screen"]["pass_all3"]==all((x-native)/n>=.05 for x in counts)
 c.verify(lock["files"]);c.verify(r["output_sha"]);c.write("completion_review/REVIEW.json",dict(status="PASS",all28_fullsaved_quality_clusters_motion_baselines_modelcontracts_GOALgeometry_CI_SHA=True,full_retrieval20_independent_track_certificate=True,source_count=len(lock["files"]),output_count=len(r["output_sha"]),checkpoint_CPU_load_only_no_neuralforward_optimizer_or_gradient_replay=True,not_external_review=True,certificates=cert,elapsed_seconds=time.perf_counter()-started,resource=c.RESOURCE));print("AUDIT_PASS",flush=True)
if __name__=="__main__":audit()
