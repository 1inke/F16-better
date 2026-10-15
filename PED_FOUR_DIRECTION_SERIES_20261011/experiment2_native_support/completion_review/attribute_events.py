"""Additional saved-event attribution only; no policy changes or model imports."""
import sys,json,time
from pathlib import Path
from collections import Counter
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import run as R
import numpy as np
A=R.A;H=R.H
def main():
 start=time.perf_counter();A.guard()
 result=A.read(H/'RESULT.json');source=A.read(H/'source_manifest.json')['files']
 A.verify(source);A.verify(result['output_sha'])
 assert result['status']=='PASS_NATIVE_SUPPORT_INTEGRATION_DIAGNOSTIC'
 totals=Counter();failed=Counter();rows=[];records=[]
 for arm in R.ARMS:
  for seed in R.SEEDS:
   for split in R.SPLITS:
    folder=H/'jobs'/arm/f'seed_{seed}'/split
    z=A.npz(folder/'paths.npz');ev=A.read(folder/'events.json')
    assert np.array_equal(z['policy20'],z['native20'])
    ng=A.error(z['native20'],z['gt'])[3].any(1)
    lg=A.error(z['refined4'],z['gt'])[3]
    bank=ng|lg.any(1)
    for group,mask in A.masks(z['groups']):
     why=Counter();bad=Counter();rescue=[]
     for i in np.flatnonzero(mask):
      for e in ev[i]:
       why[e['reason']]+=1;bad.update(e.get('failed_motion',[]))
      if bank[i] and not ng[i]:
       rescue.append(dict(sample_id=str(z['sample_id'][i]),good_learned_slots=np.flatnonzero(lg[i]).tolist(),events=ev[i]))
     records.append(dict(arm=arm,seed=seed,split=split,group=group,n=int(mask.sum()),no_removable_windows=sum(all(e['reason']=='no_removable' for e in ev[i]) for i in np.flatnonzero(mask)),attempt_reasons=dict(why),failed_motion_multiple_count=dict(bad),bank_rescue_count=len(rescue),bank_rescue_cases=rescue))
    why=Counter(e['reason'] for row in ev for e in row)
    bad=Counter(k for row in ev for e in row for k in e.get('failed_motion',[]))
    totals.update(why);failed.update(bad)
    rows.append(dict(arm=arm,seed=seed,split=split,n=len(ev),attempts=4*len(ev),reasons=dict(why),failed_motion_multiple_count=dict(bad),no_removable_windows=sum(all(e['reason']=='no_removable' for e in row) for row in ev),native_distinct_mean=float(np.mean([len(np.unique(r)) for r in z['representative']]))))
    A.guard()
 assert totals['accepted']==0 and sum(totals.values())==138480
 A.verify(source);A.verify(result['output_sha']);A.guard()
 R.write('completion_review/ATTRIBUTION.json',dict(status='PASS_SAVED_EVENT_ATTRIBUTION',no_torch_forward_optimizer=True,source_count=len(source),output_count=len(result['output_sha']),initial_final_SHA_PASS=True,total_rows=34620,total_attempts=138480,attempt_reasons=dict(totals),failed_motion_multiple_count=dict(failed),jobs=rows,groups=records,elapsed_seconds=time.perf_counter()-start,resource=A.resource))
 print(json.dumps(dict(status='PASS',attempt_reasons=dict(totals),failed_motion_multiple_count=dict(failed)),ensure_ascii=False))
if __name__=='__main__':main()

