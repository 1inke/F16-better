"""CPU operational/scientific prelaunch review; no model."""
import os,sys,time,json,importlib.util
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent.parent
sp=importlib.util.spec_from_file_location('e6_current_runner',H/'knn.py');k=importlib.util.module_from_spec(sp);sp.loader.exec_module(k)
def main():
 t=time.perf_counter();k.guard();pre=k.read(H/'PREFLIGHT.json');assert pre['status']=='PASS' and pre['jobs']==14
 locks=k.read(H/'source_manifest.json')['files'];k.verify(locks)
 bank=k.a.load(H/'data/eth_bank.npz');test=k.a.load(k.P/'data/eth_evaluation.npz');test['fold']='eth'
 start=time.perf_counter();p=k.predict(bank,test,'position_velocity','all_train');predict=time.perf_counter()-start
 start=time.perf_counter();metrics=k.summary(p['pred'],test,'eth');seconds=time.perf_counter()-start
 assert len(metrics)>0 and np.isfinite(p['pred']).all()
 k.save('prelaunch_review/smoke_outputs.npz',**p)
 z=k.a.load(H/'prelaunch_review/smoke_outputs.npz');assert np.array_equal(z['pred'],p['pred'])
 assert k.summary(z['pred'],test,'eth')==metrics
 k.verify(locks)
 rec=dict(status='PASS',source_count=len(locks),current_source_SHA_unchanged=True,actual_ETH364_query_metrics_saved_recompute_EXACT=True,query_seconds=predict,full364_metrics_seconds=seconds,nominal_all14_query_seconds=pre['nominal_query_seconds'],nominal_metrics_rows_scaled_seconds=seconds/364*(2*34161+2*2356+2*5910),no_model=True,no_torch=True,no_optimizer=True,resource=k.a.RES,elapsed_seconds=time.perf_counter()-t,limitations='Per-subset track CI cost not linear in rows; output IO and finalSHA extra. Not formal production results. no_sibling preflight composition is ORIGINAL pre-exclusion bank; train_n/train_ids encode actual removal; completion must report retained composition separately.')
 k.write('prelaunch_review/REVIEW.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':main()

