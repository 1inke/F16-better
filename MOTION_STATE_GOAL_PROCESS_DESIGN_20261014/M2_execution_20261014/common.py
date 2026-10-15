"""M2 isolated output context; read-only G1 cache infrastructure; no old runner import."""
import os,sys,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;S=H.parent;R=S.parent
G1=R/"GENERATION_BRANCH_SCREEN_20261014/G1_goal20_20261014"
M1=S/"M1_execution_20261014"
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=module("m2_readonly_G1_common",G1/"common.py");base.H=H
guard=base.guard;now=base.now;read=base.read;write=base.write;sha=base.sha;digest=base.digest
npz=base.npz;save=base.save;verify=base.verify;eq=base.eq;safe=base.safe;stage=base.stage
done_valid=base.done_valid;RESOURCE=base.RESOURCE;INPUTS=base.INPUTS;load=base.load
def locks():
 old=read(M1/"source_manifest.json");files=old["files"].copy();verify(files)
 out=read(M1/"RESULT.json")["output_sha"];verify(out)
 assert read(M1/"completion_review/REVIEW.json")["status"]=="PASS"
 assert read(M1/"runtime_state.json")["M1_analyzed"] and not read(M1/"RESULT.json")["decision"]["S1"]
 exclude={"runtime_state.json","progress.json","worker_state.json","worker.lock","worker_stdout.log","worker_stderr.log","exit_status.json"}
 add=[p for p in M1.iterdir() if p.is_file() and p.name not in exclude]
 for sub in ("completion_review","failures","budget_release_provenance","budget_review"):
  add+=list((M1/sub).rglob("*"))
 add+=list(S.glob("*.md"))+list(H.glob("*.py"))+list(H.glob("*.ps1"))+[H/"README.md",H/"DESIGN.md",H/"protocol.json"]
 for p,v in out.items():
  if p in files:assert files[p]==v,("INHERITED_DIGEST_CONFLICT",p)
  else:files[p]=v
 for p in add:
  if not p.is_file():continue
  key=str(p);v=sha(p)
  if key in files:assert files[key]==v,("OLD_DIGEST_CHANGED",key)
  else:files[key]=v
 return files,dict(inherited_M1_sources=len(old["files"]),inherited_M1_outputs=len(out),not_all_historical34k=True)
