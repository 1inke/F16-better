"""CPU saved-output review; no torch, model, optimizer or source mutation."""
import sys,importlib.util,json,time
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parents[1];D=H/'completion_review'
s=importlib.util.spec_from_file_location('protected_saved',H/'run.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
q=r.q;q.H=D
def main():
    t=time.perf_counter();q.guard();result=q.read(H/'RESULT.json');locks=q.read(H/'source_manifest.json')['files'];q.verify(locks);q.verify(result['output_sha']);data=q.cache();items={};errs=[];admission=[0,0,0,0];rows=0;cells=0;protected_rescue=protected_lost=0
    for tag,seed,split,path,d,man,done in q.jobs():
        z=q.npz(path);logits=q.identity(z,data,d,split);rec=q.read(H/'jobs'/f'{tag}_{seed}_{split}.json');v=q.npz(H/'jobs'/f'{tag}_{seed}_{split}.npz')
        m,iv=r.evaluate(z,logits,True);error=q.compare(m,dict(metrics=rec['metrics'],admission=rec['admission']));assert error<=1e-12
        assert all(q.np.array_equal(iv[k],v[k]) for k in iv)
        assert q.np.array_equal(v['protected20'],z['paths'][:,:20]);assert q.np.array_equal(v['source_slots'],q.np.tile(q.np.arange(20),(len(z['paths']),1)))
        for g,x in rec['metrics']['protected20'].items():
            assert q.compare(x,rec['metrics']['native20'][g])==0 and x['motion_all5_1pct'];cells+=1
        a=rec['admission']['overall'];admission=[x+y for x,y in zip(admission,a['decision_counts'])];rows+=a['n'];protected_rescue+=rec['metrics']['protected20']['overall']['rescued'];protected_lost+=rec['metrics']['protected20']['overall']['lost'];errs.append(error)
        items.setdefault((tag,split),[]).append((seed,{k:v[k] for k in ('sample_id','groups','track','vectors')}))
        print('REVIEW_SAVED',tag,seed,split,'error',error,flush=True)
        del z,iv,v
    ci=r.bootstrap(items);assert q.compare(ci,q.read(H/'BOOTSTRAP.json')['records'])==0
    q.verify(locks);q.verify(result['output_sha']);q.guard()
    out=dict(status='PASS',jobs=27,rows=rows,output_SHA_count=len(result['output_sha']),locked_files=len(locks),independent_max_error=max(errs),full_saved_selector_quality_complete_link_five_motion=True,protected_paths_bitwise_native_all27=True,protected_metrics_exact_native_all108=True,protected_motion_PASS_cells=cells,accepted_paths=admission[3],decision_counts=admission,decision_denominator=sum(admission),rescue=protected_rescue,lost=protected_lost,bootstrap_records=len(ci),scalar_CIs=len(ci)*4,bootstrap_reconstruction_max_error=0,source_output_SHA_initial_final_PASS=True,not_all_historical_files=True,no_torch_model_forward_backward_optimizer=True,not_external_review=True,elapsed_seconds=time.perf_counter()-t,resource=q.RESOURCE)
    q.write('REVIEW.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        q.write('FAILURE.json',dict(traceback=traceback.format_exc(),resource=q.RESOURCE));raise
