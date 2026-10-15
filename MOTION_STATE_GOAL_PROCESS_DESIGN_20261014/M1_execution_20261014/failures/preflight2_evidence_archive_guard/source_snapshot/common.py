"""New execution context; read-only audited G1 infrastructure, never old top-level runners."""
import os,sys,importlib.util,hashlib,json,time
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;S=H.parent;R=S.parent
G1=R/"GENERATION_BRANCH_SCREEN_20261014/G1_goal20_20261014"
G2=R/"GENERATION_BRANCH_SCREEN_20261014/G2_frozen_goal_process_20261014"
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=module("m1_readonly_g1_common",G1/"common.py")
# Only in-memory output root redirection; all inherited source files remain immutable.
base.H=H
guard=base.guard;now=base.now;read=base.read;write=base.write;sha=base.sha;digest=base.digest
npz=base.npz;save=base.save;verify=base.verify;eq=base.eq;safe=base.safe;stage=base.stage
done_valid=base.done_valid;RESOURCE=base.RESOURCE;INPUTS=base.INPUTS
import numpy as np
DATA=None
def load():
 global DATA
 if DATA is None:
  DATA=base.load()
  # Stored focal float32 positions -> float64 physical differences -> float32 features.
  vel=np.diff(DATA["past"].astype(np.float64),axis=1)/.4
  DATA["raw_velocity"]=vel.astype(np.float32)
  DATA["raw_acceleration"]=(np.diff(DATA["raw_velocity"].astype(np.float64),axis=1)/.4).astype(np.float32)
  assert np.array_equal(DATA["vel"],vel[:,-1])
  assert np.array_equal(DATA["vel"].astype(np.float32),DATA["raw_velocity"][:,-1])
 return DATA
def stats(data):
 old=npz(G1/"statistics.npz");tr=data["split"]["FIT"]
 assert np.array_equal(old["training_ids"],data["ids"][tr])
 assert np.array_equal(old["mean"],data["p2"][tr].astype(float).mean(0).astype(np.float32))
 sd=data["p2"][tr].astype(float).std(0)
 assert np.array_equal(old["std"],np.where(sd<1e-6,1.,sd).astype(np.float32))
 v=data["raw_velocity"][tr].astype(float);a=data["raw_acceleration"][tr].astype(float)
 return dict(mean=old["mean"],std=old["std"],centers=old["centers"],training_ids=old["training_ids"],
  velocity_mean=v.mean((0,1)),velocity_std=np.where(v.std((0,1))<1e-6,1.,v.std((0,1))),
  acceleration_mean=a.mean((0,1)),acceleration_std=np.where(a.std((0,1))<1e-6,1.,a.std((0,1))),
  velocity_elements=np.array(len(tr)*7),acceleration_elements=np.array(len(tr)*6))
def locks():
 old=read(G2/"source_manifest.json");files=old["files"].copy();verify(files)
 out=read(G2/"RESULT.json")["output_sha"];verify(out)
 assert read(G2/"completion_review/REVIEW.json")["status"]=="PASS"
 add=[p for p in G2.iterdir() if p.is_file() and p.suffix in (".py",".md")]+list((G2/"completion_review").glob("*.json"))+list((G2/"completion_figures").glob("*"))
 add+=list(S.glob("*.md"))+list(H.glob("*.py"))+list(H.glob("*.ps1"))+[H/"protocol.json",H/"README.md",H/"DESIGN.md"]
 for p,v in out.items():
  if p in files:assert files[p]==v,("INHERITED_DIGEST_CONFLICT",p)
  else:files[p]=v
 for p in add:
  if not p.is_file():continue
  k=str(p);v=sha(p)
  if k in files:assert files[k]==v,("OLD_DIGEST_CHANGED",k)
  else:files[k]=v
 return files,dict(G2_original_digests=len(old["files"]),G2_original_outputs=len(out),not_all_historical34k=True)
def stat_chunk_error(data,st):
 tr=data["split"]["FIT"];errors=[]
 for name,key in (("velocity","raw_velocity"),("acceleration","raw_acceleration")):
  for chunk in (31,256):
   count=0;sm=np.zeros(2);sq=np.zeros(2)
   for first in range(0,len(tr),chunk):
    a=data[key][tr[first:first+chunk]].astype(float).reshape(-1,2);count+=len(a);sm+=a.sum(0);sq+=(a*a).sum(0)
   mu=sm/count;sd=np.sqrt(np.maximum(0,sq/count-mu*mu));sd=np.where(sd<1e-6,1.,sd)
   errors.extend([float(np.max(abs(mu-st[name+"_mean"]))),float(np.max(abs(sd-st[name+"_std"])))])
 assert max(errors)<1e-11,("STAT_CHUNK",errors)
 return max(errors)
