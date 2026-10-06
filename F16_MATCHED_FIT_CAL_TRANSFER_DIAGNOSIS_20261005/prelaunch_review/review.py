"""No model forward: current-source and actual production resume branch fixtures."""
from pathlib import Path
import ast, hashlib, importlib.util,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;SERIES=HERE.parent
sys.path.insert(0,str(SERIES));sys.dont_write_bytecode=True
import diagnosis as d
assert d.read(SERIES/'preflight_state.json')['status']=='SUCCESS'
pre=d.read(SERIES/'preflight.json');assert pre['status']=='PASS'
d.verify(pre['source_hashes'],'PRELAUNCH_CURRENT_SOURCE')
folder=HERE/'fixtures';folder.mkdir(exist_ok=True)
dst=folder/'analysis_output.npz';marker=folder/'done.json'
np.savez_compressed(dst,sample_id=np.array([70001],np.int64),fixture=np.arange(8,dtype=np.float32))
record=dict(status='SUCCESS',source_signature=d.sha(SERIES/'preflight.json'),outputs={str(dst):d.sha(dst)},columns=['sample_id','fixture'])
d.write(marker,record)
tree=ast.parse((SERIES/'diagnosis.py').read_text(encoding='utf-8'))
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
branch=next(n for n in ast.walk(run) if isinstance(n,ast.If) and ast.unparse(n.test)=='marker.exists()')
code=compile(ast.Module(body=branch.body,type_ignores=[]),str(SERIES/'diagnosis.py'),'exec')
ns=dict(marker=marker,dst=dst,HERE=SERIES,read=d.read,sha=d.sha,load_npz=d.load_npz,np=np,data={'ids':np.array([70001],np.int64)},a=0,b=1)
exec(code,ns);assert np.array_equal(ns['z']['fixture'],np.arange(8,dtype=np.float32))
bad={**record,'source_signature':'0'*64};d.write(marker,bad)
try:exec(code,ns)
except AssertionError:source_reject=True
else:raise AssertionError('BAD SOURCE ACCEPTED')
bad={**record,'outputs':{str(dst):'0'*64}};d.write(marker,bad)
try:exec(code,ns)
except AssertionError:output_reject=True
else:raise AssertionError('BAD OUTPUT ACCEPTED')
d.write(marker,record);exec(code,ns)
report=dict(status='PASS_PRELAUNCH_READONLY_ANALYSIS_RESUME',source_files=len(pre['source_hashes']),current_sources_exact=True,
 actual_production_branch_AST=True,completed_analysis_SHA_skip_fixture=True,bad_source_reject=source_reject,bad_output_reject=output_reject,
 no_optimizer_or_training_resume_claim=True,no_model_forward=True,preflight_SHA=d.sha(SERIES/'preflight.json'),
 script_SHA=d.sha(Path(__file__)),fixture_SHA={str(p):d.sha(p) for p in (dst,marker)},memory=d.guard.check())
d.write(HERE/'REVIEW.json',report);print(json.dumps(report),flush=True)

