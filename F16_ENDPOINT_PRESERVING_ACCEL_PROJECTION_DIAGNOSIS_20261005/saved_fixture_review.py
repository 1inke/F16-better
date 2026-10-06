"""Independent scalar review of immutable saved test coordinates; no torch/model."""
from pathlib import Path
import hashlib, json, math
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parent
CASE=ROOT/'failures'/'numeric_gate_1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def metric(p):
    values=[]
    for k in range(20):
        previous=[0.,0.]; previous_velocity=None
        for t in range(12):
            position=[float(p[0,k,t,c]) for c in range(2)]
            velocity=[(position[c]-previous[c])/.4 for c in range(2)]
            if previous_velocity is not None:
                values.append(math.hypot(*[(velocity[c]-previous_velocity[c])/.4 for c in range(2)]))
            previous=position;previous_velocity=velocity
    return math.fsum(values)/len(values)
def main():
    original=json.loads((CASE/'REPORT.json').read_text(encoding='utf-8'))
    assert sha(CASE/'fixtures.npz')==original['fixture_output_sha']
    assert original['source_before']==original['source_after']
    for name,value in original['source_after'].items():assert sha(Path(name))==value
    differences=[]; rows=[]
    with np.load(CASE/'fixtures.npz',allow_pickle=False) as z:
        for r in original['rows']:
            key=r['device'].replace(':','_')+'_'+r['fixture']
            b=z[key+'_base'];u=z[key+'_u'];p=z[key+'_path'];up=z[key+'_projected_u']
            mb=metric(b);mp=metric(p)
            ee=max(abs(float(p[0,k,-1,c])-float(b[0,k,-1,c])-math.fsum(float(x) for x in u[0,k,:,c])) for k in range(20) for c in range(2))
            strict=mp<=1.01*mb
            differences.extend([abs(mb-r['native_saved_acceleration_mean'][0]),abs(mp-r['projected_saved_acceleration_mean'][0]),abs(ee-r['endpoint_error_m'])])
            assert strict==r['strict_one_percent_bound_pass']
            rows.append(dict(device=r['device'],fixture=r['fixture'],native_acceleration_mean=mb,projected_acceleration_mean=mp,strict_one_percent_bound_pass=strict,endpoint_error_m=ee))
    assert max(differences)<=1e-15
    assert sha(CASE/'fixtures.npz')==original['fixture_output_sha']
    for name,value in original['source_after'].items():assert sha(Path(name))==value
    out=CASE/'saved_review';out.mkdir(exist_ok=False)
    memory=psutil.Process().memory_info()
    result=dict(status='PASS_SAVED_FIXTURE_RECOMPUTE_CONFIRMS_NUMERIC_STOP',rows=rows,
        strict_failed_count=sum(not r['strict_one_percent_bound_pass'] for r in rows),
        maximum_summary_error=max(differences),saved_output_SHA_before_after_unchanged=True,
        four_original_source_SHA_before_after_unchanged=True,no_torch_model_optimizer=True,
        all_prior_scientific_SHA_revalidation=False,
        resource=dict(peak_private_GiB=memory.peak_pagefile/2**30,peak_RSS_GiB=memory.peak_wset/2**30))
    (out/'REVIEW.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
