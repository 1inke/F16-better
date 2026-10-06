"""Read-only saved-output completion review. No torch/model/solver/optimizer."""
from pathlib import Path
from datetime import datetime
import hashlib, json, time, sys
sys.dont_write_bytecode = True
import numpy as np
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent.parent
PREV = ROOT/'F16_CUE_RESPONSE_CONCORDANCE_20261004/recovery_memory_20261004'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()
def write(p,v):
    assert HERE in p.parents
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def arrays(p,keys=None):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in (keys or z.files)}
def metric(p,y):
    delta=p.astype(np.float64)-y[:,None].astype(np.float64)
    distance=np.sqrt(np.sum(delta*delta,axis=-1))
    return distance.mean(-1),distance[:,:,-1]
def population_masks(g):
    clear=g['GT_event'].astype(bool)&~g['GT_uncertain'].astype(bool)
    masks={'all':np.ones(len(clear),bool),'clear':clear,
        'no_event':~g['GT_event'].astype(bool)&~g['GT_uncertain'].astype(bool),
        'boundary':g['GT_uncertain'].astype(bool),
        'specific_first_compatible':clear&g['cue_first_compatible'].astype(bool),
        'specific_coarse_path_failure':clear&g['cue_first_compatible'].astype(bool)&(g['class_0']==7)}
    masks.update({'scene_'+str(s):g['scene_family']==s for s in np.unique(g['scene_family'])})
    masks.update({'compatible_kind'+str(k):clear&g['cue_first_compatible'].astype(bool)&(g['GT_kind']==k) for k in (1,2,3,4)})
    return masks

def main():
    start=time.perf_counter()
    result=read(STUDY/'RESULT.json'); arithmetic=read(STUDY/'ARITHMETIC_AUDIT.json')
    frozen=read(STUDY/'FROZEN_REPLAY_AUDIT.json'); provenance=read(STUDY/'RECOVERY_PROVENANCE.json')
    worker=read(STUDY/'independent_worker_state.json'); exitv=read(STUDY/'exit_status.json')
    assert worker['status']==exitv['status']=='SUCCESS'
    assert result['status']=='PASS_DIAGNOSTIC_ONLY_CURVE_REACHABILITY'
    assert arithmetic['status']==frozen['status']=='PASS' and provenance['status']=='SUCCESS'
    assert arithmetic['blocks']==result['blocks']==450
    assert arithmetic['old_files_unchanged']==result['old_files_SHA_unchanged']==22398
    assert arithmetic['source_files_unchanged']==result['source_files_SHA_unchanged']==304
    assert frozen['parameters_buffers_exact'] and not frozen['GT_forward']
    assert provenance['imported_done_blocks']==171 and provenance['newly_fitted_blocks']==279
    locks=read(STUDY/'preflight.json')['source_hashes']; assert len(locks)==304
    for path,expected in locks.items(): assert sha(path)==expected,('SOURCE_MISMATCH',path)
    direct=[STUDY/x for x in ('RESULT.json','ARITHMETIC_AUDIT.json','FROZEN_REPLAY_AUDIT.json',
        'RESOURCE_USAGE.json','RECOVERY_PROVENANCE.json','independent_worker_state.json','exit_status.json')]
    direct += [STUDY/'results'/f'{pop}_classification.npz' for pop in ('FIT','CAL')]
    direct += [PREV/'jobs'/f'{pop}_native.npz' for pop in ('FIT','CAL')]
    labelpath=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/labels/train.npz'
    cachepath=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
    direct += [labelpath,cachepath]
    before={str(p):sha(p) for p in direct}
    labels=arrays(labelpath,['sample_id','future'])
    cache=arrays(cachepath,['sample_id','candidate'])
    assert np.array_equal(labels['sample_id'],cache['sample_id'])
    assert len(np.unique(labels['sample_id']))==28791
    native=cache['candidate'][...,:24].reshape(-1,20,12,2)
    completed_outputs={}; imported=0; block_count=0; metricmax=0.; boundmax=0.
    maxprofile=0.; maxprod=0.; tables={}
    for pop,lo,hi in [('FIT',0,25912),('CAL',25912,28791)]:
        aggregate=arrays(STUDY/'results'/f'{pop}_classification.npz')
        g=arrays(PREV/'jobs'/f'{pop}_native.npz', ['sample_id','scene_family','GT_event','GT_uncertain','GT_kind','cue_first_compatible','class_0'])
        assert np.array_equal(aggregate['sample_id'],g['sample_id'])
        assert np.array_equal(g['sample_id'],labels['sample_id'][lo:hi])
        for left in range(lo,hi,64):
            right=min(hi,left+64); name=f'{pop}_{left-lo:05d}_{right-lo:05d}'
            done=read(STUDY/'chunks'/f'{name}.done.json')
            assert done['status']=='SUCCESS' and done['source_hashes']==locks
            for p,expected in done['outputs'].items():
                assert sha(p)==expected,('OUTPUT_MISMATCH',p); completed_outputs[p]=expected
            imported+=int(bool(done.get('imported_parent_npz')))
            p=Path(done.get('imported_parent_npz',str(STUDY/'chunks'/f'{name}.npz')))
            z=arrays(p); sl=slice(left-lo,right-lo)
            assert np.array_equal(z['sample_id'],labels['sample_id'][left:right])
            y=labels['future'][left:right]; b=native[left:right]
            collections={'native':b,'pulse':z['pulse_paths'],'superset':z['superset_paths']}
            collections.update({f'learned{s+1}':z['learned_paths'][s] for s in range(3)})
            for method,paths in collections.items():
                a,f=metric(paths,y)
                if method in ('native','pulse','superset'):
                    metricmax=max(metricmax,float(np.max(np.abs(a-z[method+'_ade']))),float(np.max(np.abs(f-z[method+'_fde']))))
                raw=np.stack([((a<=at)&(f<=ft)).sum(1) for at,ft in ((.2,.4),(.3,.5))],axis=1)
                assert np.array_equal(raw,z[method+'_raw_good'])
                assert np.array_equal(raw,aggregate[method+'_raw_good'][sl])
                eff=z[method+'_effective_good']
                assert np.array_equal(eff,aggregate[method+'_effective_good'][sl])
                assert np.all((eff>=0)&(eff<=raw[:,:,None]))
            pw=(z['pulse_ade']<=.2-2e-5)&(z['pulse_fde']<=.4-2e-5)
            sw=(z['superset_ade']<=.2-2e-5)&(z['superset_fde']<=.4-2e-5)
            delta=y[:,None].astype(float)-b.astype(float)
            box=z['limit'][:,None,None,None]*np.arange(1,13)[None,None,:,None]
            residual=np.maximum(np.abs(delta)-box,0)
            lower=np.sqrt((residual*residual).sum(-1))
            boundmax=max(boundmax,float(np.max(np.abs(lower.mean(-1)-z['bound_ade_lower']))),float(np.max(np.abs(lower[:,:,-1]-z['bound_fde_lower']))))
            cert=((lower.mean(-1)>.2+2e-5)|(lower[:,:,-1]>.4+2e-5)).all(1)
            cls=np.full(len(y),3,np.int8);cls[cert]=2;cls[~pw.any(1)&sw.any(1)]=1;cls[pw.any(1)]=0
            assert np.array_equal(cls,z['classification']) and np.array_equal(cls,aggregate['classification'][sl])
            assert np.array_equal(pw,aggregate['pulse_feasible_witness'][sl])
            assert np.array_equal(sw,aggregate['superset_feasible_witness'][sl])
            audit=read(STUDY/'chunks'/f'{name}.audit.json')
            assert audit['status']=='PASS' and audit['numeric_profile']['production_profile_bitwise_equal']
            assert audit['original_components']['primary_same_candidate_adequacy_exact']
            assert audit['original_components']['robust_window_witness_exact']
            maxprofile=max(maxprofile,audit['numeric_profile']['zero_sum_actual_max'])
            maxprod=max(maxprod,audit['original_components']['production_path_norm_error_max_m'])
            block_count+=1
        records=[]
        for record in result['all_summaries'][pop]:
            mask=population_masks(g)[record['group']]; n=int(mask.sum()); assert n==record['n']
            if n==0: records.append({'group':record['group'],'n':0});continue
            cls=np.bincount(aggregate['classification'][mask],minlength=4).tolist()
            assert cls==record['classification_counts'] and sum(cls)==n
            pg=aggregate['pulse_feasible_witness'][mask].any(1)
            sg=aggregate['superset_feasible_witness'][mask].any(1)
            assert int(pg.sum())==record['pulse_witness_coverage'] and int(sg.sum())==record['superset_witness_coverage']
            methods={}; ng=aggregate['native_raw_good'][mask,0]>0; gaps=[]
            for method,stored in record['methods'].items():
                raw=aggregate[method+'_raw_good'][mask];eff=aggregate[method+'_effective_good'][mask]
                good=raw[:,0]>0
                derived={'coverage':int(good.sum()),'rescued':int((good&~ng).sum()),'lost':int((~good&ng).sum()),
                    'raw_distribution':np.bincount(raw[:,0],minlength=21).tolist(),
                    'effective_distribution':np.bincount(eff[:,0,0],minlength=21).tolist(),
                    'sensitivity_coverage':int((raw[:,1]>0).sum()),
                    'effective_sensitivity_distributions':[np.bincount(eff[:,0,i],minlength=21).tolist() for i in range(3)]}
                assert derived==stored,(pop,record['group'],method)
                methods[method]={'coverage':int(good.sum()),'coverage_pct':100*float(good.mean()),
                    'rescued':derived['rescued'],'lost':derived['lost'],
                    'one_effective_good':int((eff[:,0,0]==1).sum()),
                    'two_or_more_effective_good':int((eff[:,0,0]>=2).sum()),
                    'mean_raw_good':float(raw[:,0].mean()),'mean_effective_good':float(eff[:,0,0].mean())}
            for s in (1,2,3): gaps.append(int((pg&(aggregate[f'learned{s}_raw_good'][mask,0]==0)).sum()))
            assert gaps==record['learned_gap_to_pulse']
            records.append({'group':record['group'],'n':n,'classification_counts':cls,
                'classification_pct':[100*x/n for x in cls],'native_coverage':int(ng.sum()),
                'pulse_witness':int(pg.sum()),'superset_witness':int(sg.sum()),
                'learned_coverage':[methods[f'learned{s}']['coverage'] for s in (1,2,3)],
                'expressible_but_learned_missing':gaps,'methods':methods})
        tables[pop]=records
    assert block_count==450 and imported==171 and metricmax<=1e-12 and boundmax<=1e-12
    for path,expected in locks.items(): assert sha(path)==expected
    after={str(p):sha(p) for p in direct}; assert before==after
    write(HERE/'REVIEW.json',{'status':'PASS_SAVED_OUTPUT_COMPLETION_REVIEW','at':datetime.now().astimezone().isoformat(),
        'worker_exit_success':True,'blocks':block_count,'imported_blocks':imported,'new_blocks':279,
        'done_output_files_SHA_reverified':len(completed_outputs),'source_SHA_reverified':len(locks),
        'direct_input_before_after_SHA':before,'all_saved_sample_quality_raw_counts_classification_group_summaries_exact':True,
        'saved_effective_counts_and_group_histograms_exact':True,'supplement_reclustered_all_paths':False,
        'full_effective_reclustering_in_completed_worker':True,'sample_metric_max_error_m':metricmax,
        'geometric_lowerbound_max_error_m':boundmax,'all450_original_component_audit_reports_checked':True,
        'production_path_norm_error_max_m':maxprod,'profile_float_sum_max':maxprofile,
        'model_forward':False,'numeric_fitting':False,'optimizer':False,'DEV_access':False,
        'old22398_SHA_basis':'Full final worker verification, not rerun by this supplement.',
        'resource_basis':'RESOURCE_USAGE completed worker; supplement works one64row block.',
        'seconds':time.perf_counter()-start})
    write(HERE/'GROUP_TABLES.json',{'status':'PASS_DESCRIPTIVE_ONLY','groups':tables,
        'original_quality_m':[.2,.4],'sensitivity_m':[.3,.5],'effective_dedup_m':[.1,.2],
        'motion_or_all_formal_gates_of_oracle':'NOT_ASSESSED_NO_DEPLOYABLE_ORACLE_CLAIM'})
    print(json.dumps({'status':'PASS','blocks':block_count,'sources':len(locks),'outputs':len(completed_outputs),
        'metric_error':metricmax,'lowerbound_error':boundmax,'seconds':time.perf_counter()-start},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
