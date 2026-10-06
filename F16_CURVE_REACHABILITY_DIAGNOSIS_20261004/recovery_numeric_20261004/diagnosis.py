"""Additive numerical-audit recovery; all scientific kernels stay immutable."""
from pathlib import Path
from datetime import datetime
import argparse, ast, gc, importlib.util, json, math, os, sys, time, traceback, shutil
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
spec = importlib.util.spec_from_file_location('immutable_curve_reachability_parent', PARENT/'diagnosis.py')
d = importlib.util.module_from_spec(spec);sys.modules[spec.name] = d;spec.loader.exec_module(d)
PARENT_LOCKS = d.read(PARENT/'preflight.json')['source_hashes']
ORIGINAL_NPZ = d.npz
ORIGINAL_SNAPSHOT = d.snapshot
ORIGINAL_SETUP = d.setup_replay
ORIGINAL_PREFLIGHT = d.preflight
d.HERE = HERE;d.old.HERE = HERE
read,write,digest,save = d.read,d.write,d.digest,d.save
MODEL_METADATA = []
PRODUCTION_MODEL = None
PRODUCTION_DATA = None
ID_INDEX = {}
CURRENT_INDICES = None
ENABLE_COMPONENT_AUDIT = True

def sources():
    locks = dict(PARENT_LOCKS)
    for p in list(HERE.glob('*.py'))+list(HERE.glob('*.ps1')):
        locks[str(p)] = digest(p)
    for name in ('README.md','REPAIR_DESIGN.md','old_results_manifest.json'):
        locks[str(HERE/name)] = digest(HERE/name)
    locks[str(PARENT/'preflight.json')] = digest(PARENT/'preflight.json')
    locks[str(PARENT/'failure_review/REVIEW.json')] = digest(PARENT/'failure_review/REVIEW.json')
    return locks

def snapshot():
    target = HERE/'old_results_manifest.json'
    if target.exists():return read(target)['files']
    files = dict(read(PARENT/'old_results_manifest.json')['files'])
    for p in PARENT.rglob('*'):
        if HERE in p.parents or not p.is_file():continue
        if '__pycache__' in p.parts or p.suffix in ('.lock','.pending','.tmp') or p.name=='runtime_state.json':continue
        files[str(p)] = digest(p)
    write(target,dict(files=files,inherited_count=21852,
        includes='Parent failure source/logs/exit and all171 completed scientific blocks; no overwrite or deletion.',
        excludes=['.memory','AUTONOMOUS_GENERATION_CYCLE.json','runtime_state.json','locks/temp/bytecode']))
    return files

def profile_audit(x,actual):
    """Condition-aware forward-error bound, not a replacement arbitrary cutoff.

    Reference centers the stored float32 Gaussian using math.fsum in float64.
    Each subtraction/division error is bounded using u/(1-u); the measured
    mean-rounding and normalization-denominator differences are explicit.
    The profile supplied to this audit must be bitwise the production formula.
    Existing absolute path/endpoint2e-5m checks remain separately mandatory.
    """
    import torch
    with torch.inference_mode():
        code = torch.tensor(d.codes(x).astype(np.float32))
        center = .05+.9*code[:,4].sigmoid();span = .05+.45*code[:,5].sigmoid()
        tm = (torch.arange(12,dtype=torch.float32)+.5)/12
        gaussian = torch.exp(-.5*((tm-center[:,None])/span[:,None]).square())
        mean = gaussian.mean(-1,keepdim=True)
        centered = gaussian-mean
        denom = centered.abs().amax(-1,keepdim=True).clamp_min(1e-6)
        expected = (centered/denom).numpy()
    assert np.array_equal(expected,actual),'Production profile altered, stop'
    g = gaussian.numpy().astype(np.float64);m32 = mean.numpy().astype(np.float64)
    centered32 = centered.numpy().astype(np.float64);denom32 = denom.numpy().astype(np.float64)
    m64 = np.array([math.fsum(row.tolist())/12 for row in g])[:,None]
    centered64 = g-m64
    denom64 = np.maximum(np.max(np.abs(centered64),axis=1,keepdims=True),1e-6)
    reference = centered64/denom64
    u = np.finfo(np.float32).eps/2
    eta = float(np.nextafter(np.float32(0),np.float32(1)))/2
    subtract_bound = np.abs(m32-m64)+(u/(1-u))*np.abs(g-m32)+eta
    denominator_difference = np.abs(denom32-denom64)
    divide_bound = (u/(1-u))*np.abs(centered32/denom32)+eta
    bound = subtract_bound/denom32+np.abs(centered64)*denominator_difference/(denom32*denom64)+divide_bound
    # Float64 reference arithmetic itself is included explicitly.
    bound += np.finfo(np.float64).eps*16*(1+np.abs(reference))
    difference = np.abs(actual.astype(np.float64)-reference)
    assert np.all(difference<=bound),'Profile error exceeds derived elementwise roundoff envelope'
    zero_reference = np.array([math.fsum(row.tolist()) for row in reference])
    zero_actual = np.array([math.fsum(row.tolist()) for row in actual.astype(np.float64)])
    sum_bound = bound.sum(1)+np.abs(zero_reference)
    assert np.all(np.abs(zero_actual)<=sum_bound),'Profile violates derived zero-sum roundoff envelope'
    assert np.max(np.abs(zero_reference))<1e-11,'High precision reference is not centered'
    return dict(status='PASS_DERIVED_ROUNDOFF_AND_HIGH_PRECISION',
        zero_sum_actual_max=float(np.max(np.abs(zero_actual))),
        zero_sum_derived_bound_max=float(np.max(sum_bound)),
        high_precision_zero_sum_max=float(np.max(np.abs(zero_reference))),
        profile_high_precision_error_max=float(np.max(difference)),
        profile_elementwise_bound_max=float(np.max(bound)),
        production_profile_bitwise_equal=True,float32_unit_roundoff=float(u))

# Compile only the original audit function with its dimensionless fixed
# cutoff replaced by the independently derived profile audit above.
parent_text = (PARENT/'diagnosis.py').read_text(encoding='utf-8-sig')
node = next(n for n in ast.parse(parent_text).body if isinstance(n,ast.FunctionDef) and n.name=='audit')
original_audit_text = ast.get_source_segment(parent_text,node)
old_assert = 'assert float_error<2e-5 and np.max(np.abs(pr.sum(-1)))<6e-6'
assert original_audit_text.count(old_assert)==1
new_audit_text = original_audit_text.replace(old_assert,'assert float_error<2e-5')
audit_namespace = dict(d.__dict__)
exec(compile(new_audit_text,str(HERE/'audit_immutable_except_profile.py'),'exec'),audit_namespace)
unchanged_remaining_audit = audit_namespace['audit']

def original_components_audit(base,gt,L,z):
    if PRODUCTION_MODEL is None:return dict(status='NOT_RUN_SYNTHETIC_FIXTURE')
    import torch
    # Recover exact observed speed by the original ID-aligned batch, supplied
    # by the current chunk. No future/GT is passed to predictor forward.
    x = z['pulse_params'];codes = torch.tensor(d.codes(x.reshape(-1,6)).reshape(*x.shape[:2],6).astype(np.float32))
    speed = CURRENT_SPEED
    assert len(speed)==len(base)
    with torch.inference_mode():
        residual,parts = PRODUCTION_MODEL.components(codes,torch.from_numpy(speed))
        paths = base+residual.cumsum(2).numpy()
    norm_error = np.linalg.norm(paths.astype(float)-z['pulse_paths'].astype(float),axis=-1)
    assert np.max(norm_error)<2e-5,'Original production component path diverges'
    a,f = d.quality(paths,gt);pa,pf = d.quality(z['pulse_paths'],gt)
    primary_actual = (a<=.2)&(f<=.4);primary_saved = (pa<=.2)&(pf<=.4)
    # Same-candidate adequacy/counts, not merely minADE/FDE or averaged counts.
    assert np.array_equal(primary_actual,primary_saved),'Primary candidate classification changed under actual production operations'
    robust_actual = (a<=.2-2e-5)&(f<=.4-2e-5)
    robust_saved = (pa<=.2-2e-5)&(pf<=.4-2e-5)
    assert np.array_equal(robust_actual.any(1),robust_saved.any(1)),'Window witness classification changed'
    return dict(status='PASS_ORIGINAL_COMPONENTS_CLASSIFICATION',
        production_path_norm_error_max_m=float(np.max(norm_error)),
        primary_same_candidate_adequacy_exact=True,robust_window_witness_exact=True)

CURRENT_SPEED = None
def audit(base,gt,L,z):
    result = unchanged_remaining_audit(base,gt,L,z)
    x=z['pulse_params'].reshape(-1,6);ell=np.repeat(L,20)
    _,_,profile = d.production_decode(x,ell)
    result['numeric_profile'] = profile_audit(x,profile)
    # All original source checks other than the flawed fixed profile cutoff
    # remain mandatory, including bitwise saved paths,2e-5 endpoint/path,
    # clip/bounds/finite quality and scalar independent lower bounds.
    if ENABLE_COMPONENT_AUDIT and CURRENT_SPEED is not None:
        result['original_components'] = original_components_audit(base,gt,L,z)
    return result

def setup_replay():
    global MODEL_METADATA,PRODUCTION_MODEL
    h,models = ORIGINAL_SETUP()
    MODEL_METADATA = [meta for _,_,meta in models]
    PRODUCTION_MODEL = models[0][0]
    return h,models

def npz(path,keys=None):
    path=Path(path)
    if path.parent==HERE/'chunks' and not path.exists():
        link=HERE/'chunks'/(path.stem+'.done.json')
        if link.exists():
            meta=read(link)
            if meta.get('imported_parent_npz'):path=Path(meta['imported_parent_npz'])
    return ORIGINAL_NPZ(path,keys)

def chunk_done(name,locks):
    p=HERE/'chunks'/(name+'.done.json')
    if not p.exists():return False
    meta=read(p);assert meta['source_hashes']==locks,'Recovery source mismatch'
    for path,sha in meta['outputs'].items():assert digest(path)==sha,'Recovery completed output SHA mismatch'
    return True

d.snapshot=snapshot;d.sources=sources;d.audit=audit;d.setup_replay=setup_replay;d.npz=npz;d.chunk_done=chunk_done
# Use explicit IDs for saved blocks and the original replay's indices for new
# fits. Never guess the current block from coincidentally similar coordinates.
ORIGINAL_AUDIT = d.audit
def audit_with_speed(base,gt,L,z):
    global CURRENT_SPEED
    if ENABLE_COMPONENT_AUDIT and PRODUCTION_DATA is not None:
        indices = np.array([ID_INDEX[int(v)] for v in z['sample_id']]) if 'sample_id' in z else CURRENT_INDICES
        assert indices is not None and len(indices)==len(base)
        assert np.array_equal(PRODUCTION_DATA['base'][indices],base)
        assert np.array_equal(PRODUCTION_DATA['future'][indices],gt)
        CURRENT_SPEED = PRODUCTION_DATA['speed'][indices]
    else:CURRENT_SPEED=None
    return ORIGINAL_AUDIT(base,gt,L,z)
d.audit=audit_with_speed
ORIGINAL_LOAD = d.load_data
def load_data():
    global PRODUCTION_DATA,ID_INDEX
    value=ORIGINAL_LOAD();PRODUCTION_DATA=value
    ID_INDEX={int(v):i for i,v in enumerate(value['ids'])}
    return value
d.load_data=load_data
ORIGINAL_REPLAY = d.replay
def replay(h,models,data,indices):
    global CURRENT_INDICES,CURRENT_SPEED
    out=ORIGINAL_REPLAY(h,models,data,indices)
    CURRENT_INDICES=indices.copy();CURRENT_SPEED=data['speed'][indices]
    return out
d.replay=replay

def preflight():
    global PRODUCTION_DATA,CURRENT_SPEED,PRODUCTION_MODEL,ENABLE_COMPONENT_AUDIT
    snapshot()
    # Original full-input/model/frozen/Jacobian/64row science preflight, with
    # all kernels and budget unchanged. It also tests source reject/SHA skip.
    # Disable coordinate lookup for its non-contiguous256 FIT workload.
    ENABLE_COMPONENT_AUDIT=False
    ORIGINAL_PREFLIGHT()
    ENABLE_COMPONENT_AUDIT=True
    pre=read(HERE/'preflight.json')
    data=d.load_data();h,models=d.setup_replay();d.replay(h,models,data,np.arange(1))
    L=(1.5*(data['speed'][:,0]+.03)).astype(np.float32).astype(float)
    locks=sources();imported=[];reports=[];start=time.perf_counter()
    # Validate all completed original blocks; do not recompute their fits or
    # overwrite original NPZ/audit/done files. Save additive revalidation links.
    for p in sorted((PARENT/'chunks').glob('FIT_*.done.json')):
        d.monitor.check();meta=read(p);assert meta['source_hashes']==PARENT_LOCKS
        assert meta['status']=='SUCCESS'
        for path,sha in meta['outputs'].items():assert digest(path)==sha
        name=p.name[:-10];left,right=map(int,name.split('_')[1:]);idx=np.arange(left,right)
        z=ORIGINAL_NPZ(PARENT/'chunks'/(name+'.npz'))
        assert np.array_equal(z['sample_id'],data['ids'][idx]) and np.array_equal(z['limit'],L[idx])
        CURRENT_SPEED=data['speed'][idx]
        a=audit(data['base'][idx],data['future'][idx],L[idx],z)
        fresh=d.summarize({**z,'native_paths':data['base'][idx]},z['learned_paths'],data['future'][idx],None)
        assert all(np.array_equal(z[k],v) for k,v in fresh.items()),'Imported saved classification arithmetic changed'
        ap=HERE/'chunks'/(name+'.audit.json');write(ap,a)
        parent_npz=PARENT/'chunks'/(name+'.npz')
        write(HERE/'chunks'/(name+'.done.json'),dict(status='SUCCESS',source_hashes=locks,
            outputs={str(parent_npz):digest(parent_npz),str(ap):digest(ap)},rows=len(idx),
            imported_parent_npz=str(parent_npz),parent_done_SHA=digest(p),
            original_source_hashes=PARENT_LOCKS,model_training=False,refitted=False))
        reports.append(a);imported.append(name)
    assert len(imported)==171
    # Exact failed block is recomputed only as a discarded validation fixture.
    idx=np.arange(10944,11008);CURRENT_SPEED=data['speed'][idx]
    failed=d.compute(data['base'][idx],data['future'][idx],L[idx])
    failed_review=audit(data['base'][idx],data['future'][idx],L[idx],failed)
    x=failed['pulse_params'].reshape(-1,6);_,_,pr=d.production_decode(x,np.repeat(L[idx],20))
    corrupt=pr.copy();corrupt[0,0]+=.01
    rejected=False
    try:profile_audit(x,corrupt)
    except AssertionError:rejected=True
    assert rejected,'Corrupted profile accepted'
    bad=dict(locks);bad[str(HERE/'diagnosis.py')]='0'*64
    rejected_source=False
    try:chunk_done(imported[0],bad)
    except AssertionError:rejected_source=True
    assert rejected_source
    # Fixed FIT-only clock grid, includes bothrange edges; no outcome tuning.
    clocks=np.array([[0,0,.999999,-.999999,c,s] for c in (.050001,.1,.3,.5,.9,.949999) for s in (.050001,.08,.275,.499999)])
    _,_,gridprofile=d.production_decode(clocks,np.full(len(clocks),.1))
    gridreview=profile_audit(clocks,gridprofile)
    for model,initial,meta in models:
        h.frozen_audit(model)
        import torch
        assert all(torch.equal(initial[k],v) for k,v in model.state_dict().items())
    resource=d.monitor.finish()
    write(HERE/'NUMERIC_REPAIR_PREFLIGHT.json',dict(status='PASS',imported_done_blocks=len(imported),
        imported_output_SHA_verified=True,imported_classification_and_effective_counts_exact=True,
        actual_production_candidate_adequacy_exact=True,actual_production_window_witness_exact=True,
        original_solver_source_unchanged=True,original_protocol_unchanged=True,
        failed_block=failed_review,clock_grid=gridreview,corrupted_profile_rejected=rejected,
        source_mismatch_rejected=rejected_source,fixture_fit_discarded=True,
        validation_seconds=time.perf_counter()-start,resource=resource,models=MODEL_METADATA))
    pre.update(learned_replay_models=MODEL_METADATA,imported_done_blocks=171,remaining_new_blocks=279,
        numeric_repair_preflight='PASS',numeric_repair_preflight_SHA=digest(HERE/'NUMERIC_REPAIR_PREFLIGHT.json'),science_kernels_original_SHA_unchanged=True,
        all_saved_imported_metrics_classification_dedup_exact=True,source_hashes=locks,
        original_unrepaired_failure_preserved=True)
    # ETA accounts for remaining fit/replay, final450 saved audits, IO/hashes.
    remaining_fit=pre['benchmark_seconds']/256*(28791-10944)
    revalidation=time.perf_counter()-start
    estimate=remaining_fit+revalidation*450/171+2*pre['measured_single_old_hash_seconds']+90
    planning=max(8,int(np.ceil(estimate/60)))
    pre.update(planning_minutes=planning,estimate_minutes=planning,
        eta_minutes=[max(6,int(planning*.75)),int(np.ceil(planning*1.5))])
    write(HERE/'preflight.json',pre)
    d.verify({**read(HERE/'old_results_manifest.json')['files'],**locks},'RECOVERY_PREFLIGHT_FINAL_SHA')
    del models,data,failed,z;PRODUCTION_DATA=None;PRODUCTION_MODEL=None;CURRENT_SPEED=None;gc.collect()
    print('RECOVERY_PREFLIGHT_PASS '+json.dumps({k:pre[k] for k in ('old_files','registered_sources','benchmark_seconds','planning_minutes','eta_minutes','imported_done_blocks','remaining_new_blocks')}),flush=True)

def run():
    pre=read(HERE/'preflight.json');assert pre['numeric_repair_preflight']=='PASS'
    assert read(HERE/'NUMERIC_REPAIR_PREFLIGHT.json')['status']=='PASS'
    assert digest(HERE/'NUMERIC_REPAIR_PREFLIGHT.json')==pre['numeric_repair_preflight_SHA']
    d.run()
    write(HERE/'RECOVERY_PROVENANCE.json',dict(status='SUCCESS',parent=str(PARENT),
        parent_failure_preserved='failures/worker_20261004_212414',imported_done_blocks=171,
        newly_fitted_blocks=279,total_blocks=450,scientific_kernel_or_quality_changes=False,
        audit_change='Fixed dimensionless zero-sum cutoff replaced by derived roundoff/high-precision production audit.',
        full_actual_production_component_classification_audit=True,formal_response_study_completed=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    try:preflight() if args.preflight else run()
    except BaseException:
        failure=HERE/'failures'/datetime.now().strftime('preflight_%Y%m%d_%H%M%S');failure.mkdir(parents=True,exist_ok=False)
        (failure/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        for p in HERE.glob('*.py'):shutil.copy2(p,failure/p.name)
        write(failure/'source_hashes.json',{str(p):digest(p) for p in HERE.glob('*.py')});raise
