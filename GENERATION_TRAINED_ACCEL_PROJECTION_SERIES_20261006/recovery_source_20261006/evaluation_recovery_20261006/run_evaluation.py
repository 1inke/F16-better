"""Same-science saved-output reader repair; evaluation only, never retrain."""
from pathlib import Path
import sys, os, time, json, csv, argparse
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT));sys.path.insert(0,str(HERE))
import run_head as h
from resource_guard import guard
import numpy as np
import torch
h.m.RESULTS=HERE/'results'
def stage(name,**extra):
    h.write(HERE/'progress.json',dict(stage=name,at=h.datetime.now().astimezone().isoformat(),**extra))
    print(name,json.dumps(extra),flush=True)
h.stage=stage
def predictions():
    return [f'seed{s}_{a}_{p}.npz' for a in h.c.ARMS for s in (1,2,3) for p in h.c.POLICIES]
COPY_NAMES=predictions()+['summary.json','model_lock.json','per_seed_metrics.csv','aggregate_metrics.csv','bootstrap.csv','REPORT.md']
def lock_sources():
    paths=[p for p in HERE.iterdir() if p.suffix in ('.py','.ps1') or p.name in ('README.md','DESIGN.md','protocol.json')]
    return {str(p):h.sha(p) for p in paths if p.is_file()}
def parent_inputs():
    files={}
    for a in h.c.ARMS:
        for s in (1,2,3):
            d=h.checkpoint(s,a,'final').parent
            done=h.read(d/'complete.json')
            assert done['status']=='COMPLETE' and done['epochs']==80
            h.verify(done['model_hashes']);files.update(done['model_hashes'])
            with (d/'history.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            assert [int(r['epoch']) for r in rows]==list(range(1,81))
            objective=h.read(d/'objective_history.json')['rows'];assert len(objective)==80
            for name in ('complete.json','history.csv','objective_history.json','resume.pth'):
                files[str(d/name)]=h.sha(d/name)
    for name in COPY_NAMES:
        p=PARENT/'results'/name;assert p.is_file();files[str(p)]=h.sha(p)
    for name in ('stderr.log','worker_failure.txt','independent_worker_state.json'):
        p=PARENT/name;assert p.is_file();files[str(p)]=h.sha(p)
    return files
def saved_prediction(path):
    path=Path(path).resolve()
    allowed={ (h.m.RESULTS/n).resolve() for n in predictions() }
    if path not in allowed:raise ValueError('Saved-output reader whitelist rejected path')
    return h.RAW_LOAD(path)
def copy_and_validate():
    import shutil
    h.m.RESULTS.mkdir(exist_ok=True)
    for name in COPY_NAMES:
        source=PARENT/'results'/name;dest=h.m.RESULTS/name
        if dest.exists():assert h.sha(dest)==h.sha(source),str(dest)
        else:shutil.copy2(source,dest)
        assert h.sha(dest)==h.sha(source)
    cache=h.load_cache(h.p.DEV_CACHE);base,speed,gt=h.prepared(cache)[1:4]
    assert len(np.unique(cache['sample_id']))==len(cache['sample_id'])==1516
    maxima=0.;hashes={}
    for name in predictions():
        guard.check();dest=h.m.RESULTS/name;data=saved_prediction(dest)
        assert np.array_equal(data['sample_id'],cache['sample_id'])
        assert data['trajectory'].shape==(1516,20,12,2)
        assert data['trajectory'].dtype==np.float64 and np.isfinite(data['trajectory']).all()
        metrics=h.r.generation_metrics(data['trajectory'],base,speed,gt)
        for key,val in metrics.items():
            diff=float(np.max(abs(val-data[key])));maxima=max(maxima,diff)
            assert diff<=1e-6,(name,key,diff)
        hashes[str(dest)]=h.sha(dest)
    rejected=False
    try:saved_prediction(PARENT/'results'/predictions()[0])
    except ValueError:rejected=True
    assert rejected
    return dict(status='PASS',predictions=18,explicit_IDs=1516,finite_K20=True,
        original_saved_outputs_copied_exact=True,sample_metric_max=maxima,
        strict_output_whitelist_unknown_path_rejected=True,dataset_loader_unchanged=True,
        output_SHA=hashes,no_optimizer=True,no_training=True)
def parent_gate():
    p=h.read(PARENT/'preflight.json');assert p['status']=='PASS' and len(p['source_hashes'])==541
    h.verify(p['source_hashes']);old=h.old_manifest();assert old==31968
    return p['source_hashes'],old
def frozen_preflight(device):
    cache,cfg,fit,cal,values=h.prepared_train(device)
    timing={}
    for a in h.c.ARMS:
        cp=h.checkpoint(1,a,'final');saved=torch.load(cp,map_location='cpu',weights_only=False)
        model=h.ProjectionHead(940,saved['config']).to(device).eval()
        model.load_state_dict(saved['state_dict']);h.frozen_audit(model)
        before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        seconds=[]
        with torch.no_grad():
            for batch in range(8):
                idx=fit[batch*128:(batch+1)*128];guard.check()
                o,b,s=[torch.from_numpy(v[idx]).to(device) for v in values[:3]]
                torch.cuda.synchronize();t=time.perf_counter();out,res=model(o,b,s);torch.cuda.synchronize()
                seconds.append(time.perf_counter()-t)
                assert out.shape==(128,20,12,2) and out.dtype==torch.float64 and torch.isfinite(out).all()
                raw,parts=model.routed_components(o,b,s)
                if bool(model.native_fallback):raw=torch.zeros_like(raw)
                projected,up,info=h.projection.project(b,raw,parts['limit'])
                from preflight_impl import independent_physics
                independent_physics(h,b,raw,projected,up,parts['limit'],info)
                expected=b.double()+raw.double().cumsum(2) if a=='zero' else projected
                assert torch.allclose(out,expected,atol=2e-5,rtol=0)
        for k,v in model.state_dict().items():assert torch.equal(v.detach().cpu(),before[k]),k
        h.frozen_audit(model);timing[a]=float(np.mean(seconds))
        del saved,model,before
    return dict(status='PASS',actual_frozen_FIT128_batches_each_arm=8,mean_seconds=timing,
        full_weights_bool_scalers_exact=True,strict_projection_physics=True,no_backward=True,no_optimizer=True)
def preflight(device):
    start=time.perf_counter();guard.check();stage('PREFLIGHT_SOURCE_OLD_AND_COMPLETED_SHA')
    sources,old=parent_gate();inputs=parent_inputs();h.verify(inputs)
    stage('PREFLIGHT_SAVED_OUTPUT_READER');reader=copy_and_validate()
    stage('PREFLIGHT_FROZEN_FORWARD');forward=frozen_preflight(device)
    h.verify(sources);h.verify(inputs);h.old_manifest();new=lock_sources()
    result=dict(status='PASS',parent_sources=sources,source_hashes=new,parent_inputs=inputs,
        old_SHA_files=old,completed_trainings=6,complete_epochs=80,reader=reader,
        frozen_forward=forward,seconds=time.perf_counter()-start,resource=guard.values(),
        evaluation_only=True,no_training_resume_claim=True)
    h.write(HERE/'preflight.json',result);guard.save('PREFLIGHT_PASS')
    stage('PREFLIGHT_PASS',seconds=result['seconds'])
def run(device):
    p=h.read(HERE/'preflight.json');assert p['status']=='PASS'
    stage('INITIAL_SOURCE_OLD_COMPLETED_SHA');h.verify(p['source_hashes']);parent_gate();h.verify(p['parent_inputs'])
    assert not (h.m.RESULTS/'exit_status.json').exists()
    stage('SAVED_OUTPUT_READER_RECOVERY')
    # Preflight copies remain exact before remaining audits update cloned summary only.
    reader=copy_and_validate();h.write(h.m.RESULTS/'audit.json',reader)
    stage('TEMPORAL_AND_PROJECTION_REPLAY');h.temporal_projection_outputs(device)
    import completion_audit,quality_audit,physical_population_audit
    stage('SAVED_METRICS_GROUP_CI_PROJECTION_AUDIT');metrics=completion_audit.audit(h)
    stage('SAVED_QUALITY_CLUSTER_GLOBAL_MOTION_AUDIT');quality=quality_audit.audit(h)
    stage('PHYSICAL_POPULATION_AND_OBJECTIVE_HISTORY_AUDIT');physical=physical_population_audit.audit(h,device)
    stage('FINAL_SOURCE_OLD_COMPLETED_SHA')
    h.verify(p['source_hashes']);parent_gate();h.verify(p['parent_inputs']);guard.save('SUCCESS')
    result=dict(status='SUCCESS',evaluation_only=True,no_optimizer=True,no_training=True,
        six_completed_trainings_reused_SHA_exact=True,source541_and_old31968_unchanged=True,
        frozen_all_weights_bool_scalers_exact=True,metric_audit=metrics['status'],
        quality_audit=quality,physical_audit=physical['status'],finished_at=h.datetime.now().astimezone().isoformat())
    h.write(h.m.RESULTS/'exit_status.json',result);h.write(HERE/'RECOVERY_RESULT.json',result)
    stage('SUCCESS')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    try:
        (preflight if args.action=='preflight' else run)(torch.device(args.device))
    except BaseException:
        guard.save('FAILURE');raise
