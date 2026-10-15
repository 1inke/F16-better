"""Saved-only summary/pairing verification; not external review or full model replay."""
import sys
sys.dont_write_bytecode=True
import json, hashlib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
S=H.parent/'stage2_raw_interventions'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
r=read(H/'RESULT.json'); manifest=read(H/'input_manifest.json')
for p,d in manifest.items():assert sha(p)==d
checks=0; rows=0
for record in r['records']:
    dest=H/record['output'];assert sha(dest)==record['output_sha']
    path=S/f"jobs/{record['fold']}_seed{record['seed']}/{record['intervention']}/outputs.npz"
    with np.load(dest,allow_pickle=False) as z, np.load(path,allow_pickle=False) as original:
        v=z['measurements'];perm=z['assignment'];n=len(v);rows+=n
        assert np.array_equal(z['sample_id'],original['sample_id'])
        assert np.array_equal(np.sort(perm,axis=1),np.tile(np.arange(20),(n,1)))
        assert np.all(v[:,:4]>=0) and np.all((v[:,4]>=0)&(v[:,4]<=1))
        assert np.all(v[:,1]<=v[:,0]+1e-12) and np.all(v[:,2]<=v[:,1]+1e-12)
        assert np.array_equal(v[:,4],(perm!=np.arange(20)).mean(1))
        assert np.allclose(np.quantile(v,.95,axis=0),record['all_p95'],rtol=0,atol=0)
        for group,saved in record['groups'].items():
            mask=original['mask_'+group].astype(bool)
            assert saved['n']==int(mask.sum())
            if mask.any():assert np.array_equal(v[mask].mean(0),saved['means'])
            else:assert saved['means'] is None
            checks+=1
for p,d in manifest.items():assert sha(p)==d
out=dict(status='PASS_SUMMARIES_PERMUTATIONS_AND_SHA',jobs=len(r['records']),rows=rows,
         group_checks=checks,input_files=len(manifest),initial_final_SHA_PASS=True,
         no_torch_forward_optimizer=True,not_full_second_cost_rebuild=True,not_external_review=True)
(H/'REVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out))
