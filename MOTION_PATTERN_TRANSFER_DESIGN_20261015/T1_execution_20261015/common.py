"""Read-only cache infrastructure; new folder owns every new output."""
import sys, importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent; S=H.parent; R=S.parent
G1=R/'GENERATION_BRANCH_SCREEN_20261014/G1_goal20_20261014'
M2=R/'MOTION_STATE_GOAL_PROCESS_DESIGN_20261014/M2_execution_20261014'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
base=module('t1_readonly_cache_infrastructure',G1/'common.py');base.H=H
guard=base.guard;now=base.now;read=base.read;write=base.write;sha=base.sha;digest=base.digest
npz=base.npz;save=base.save;verify=base.verify;eq=base.eq;safe=base.safe;RESOURCE=base.RESOURCE;load=base.load
def locks():
 old=read(M2/'source_manifest.json');files=old['files'].copy();verify(files)
 out=read(M2/'RESULT.json')['output_sha'];verify(out)
 assert read(M2/'completion_review/REVIEW.json')['status']=='PASS'
 exclude={'runtime_state.json','progress.json','worker_state.json','worker.lock','worker_stdout.log','worker_stderr.log','exit_status.json'}
 add=[p for p in M2.iterdir() if p.is_file() and p.name not in exclude]
 add+=list((M2/'completion_review').rglob('*'))+list((M2/'failures').rglob('*'))+list((M2/'budget_release_provenance').rglob('*'))
 add+=list(S.glob('*.md'))+list(H.glob('*.py'))+[H/'README.md',H/'protocol.json']
 for k,v in out.items():
  if k in files:assert files[k]==v,('INHERITED_DIGEST_CONFLICT',k)
  else:files[k]=v
 for p in add:
  if not p.is_file():continue
  k=str(p);v=sha(p)
  if k in files:assert files[k]==v,('OLD_DIGEST_CHANGED',k)
  else:files[k]=v
 return files,dict(inherited_M2_sources=len(old['files']),inherited_M2_outputs=len(out),not_all_historical34k=True)
