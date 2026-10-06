"""Saved/native-only quality implementation checks; no model forward or optimizer."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
import run_head as h
from quality_audit import quantities

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
files=[ROOT/'quality_audit.py',Path(__file__),h.p.DEV_CACHE]
before={str(p):sha(p) for p in files}
cache=h.p.load_npz(h.p.DEV_CACHE)
v=h.r.prepared(cache,False,mean=np.zeros((1,940),np.float32),std=np.ones((1,940),np.float32))
q,m=quantities(v[1],v[3],cache['logits'])
assert int(q['coverage'].sum())==733
for row in (0,16,129,1515):
    ade=[];fde=[]
    for k in range(20):
        e=[float(np.sqrt(sum((float(v[1][row,k,t,d])-float(v[3][row,t,d]))**2 for d in (0,1)))) for t in range(12)]
        ade.append(sum(e)/12);fde.append(e[-1])
    good=np.array([(a<=.2 and f<=.4) for a,f in zip(ade,fde)])
    assert good.sum()==q['raw_good'][row]
    assert good[cache['logits'][row].argmax()]==q['selected_good'][row]
    assert np.argmax(-np.maximum(np.array(ade)/.2,np.array(fde)/.4))==q['oracle_slot'][row]
gt=np.zeros((1,12,2));s=np.arange(20)[None]
paths=np.zeros((1,20,12,2))
z,_=quantities(paths,gt,s)
assert z['raw_good'][0]==20 and z['effective_good'][0]==1 and z['all_distinct'][0]==1
paths[:]=2.;paths[0,0]=0.;paths[0,0,-1,0]=.5
paths[0,1]=0.;paths[0,1,:-1,0]=.5
z,_=quantities(paths,gt,s)
assert not z['coverage'][0] and z['independent_min_both_but_no_samepath'][0]
assert before=={str(p):sha(p) for p in files}
report=dict(status='PASS_NATIVE_SAVED_AND_SYNTHETIC_QUALITY_SMOKE',native_windows=1516,native_good=733,
            selected_scoring='Original native argmax only',duplicate_twenty_is_one_effective_good=True,
            independent_minima_not_samepath_rejected=True,scalar_quality_rows=4,
            direct_files_unchanged=before,no_model_forward=True,no_optimizer=True,memory=h.guard.check())
(HERE/'QUALITY_SMOKE.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
