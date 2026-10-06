"""Evaluation-only repair; original production source/models/failure remain untouched."""
from pathlib import Path
import importlib.util, sys, json, time, os, traceback, msvcrt
from datetime import datetime
HERE=Path(__file__).resolve().parent
OUT=HERE/'evaluation_repair'
def record(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    lock=(HERE/'independent_worker.lock').open('a+b');lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    sys.stdout=(OUT/'stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(OUT/'stderr.log').open('a',encoding='utf-8',buffering=1)
    state=dict(status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),started_at=datetime.now().astimezone().isoformat(),training_invoked=False)
    record('worker_state.json',state)
    try:
        spec=importlib.util.spec_from_file_location('upstream_eval_repair',HERE/'run_head.py')
        a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
        hashes=json.loads((HERE/'preflight.json').read_text())['source_hashes']
        for path,digest in hashes.items():assert a.p.sha256(Path(path))==digest,('source changed',path)
        complete=list((HERE/'results/models').rglob('complete.json'));assert len(complete)==6
        locked=json.loads((HERE/'results/model_lock.json').read_text())['models'];assert len(locked)==18
        for path,digest in locked.items():assert a.p.sha256(Path(path))==digest,('model changed',path)
        for path in complete:
            saved=json.loads(path.read_text());assert saved['status']=='COMPLETE' and saved['epochs']==80
            for model,digest in saved['model_hashes'].items():assert locked[model]==digest
        # ETH dev IDs index the original train file, exactly as native.extract does.
        original_fold_data=a.n.f.fold_data
        def dev_from_train(fold,split):
            assert fold=='eth' and split in ('train','dev'),('forbidden data split',fold,split)
            return original_fold_data(fold,'train')
        a.n.f.fold_data=dev_from_train
        (OUT/'results').mkdir(exist_ok=True)
        a.m.RESULTS=OUT/'results'  # checkpoints stay at the original c.HERE/results/models
        record('repair_provenance.json',dict(training_invoked=False,models_verified=18,complete_trainings=6,production_sources_unchanged=True,source_mapping='dev sample_id subset of eth train, same as native.extract',original_failure=json.loads((HERE/'results/exit_status.json').read_text()),original_model_hashes=locked,repair_script_sha=a.p.sha256(Path(__file__))))
        print('EVALUATION ONLY: six complete trainings / 18 hashes verified',flush=True)
        a.prepare_backend(a.torch.device('cuda:0'))
        a.evaluate(a.torch.device('cuda:0'))
        cache=a.p.load_npz(a.p.DEV_CACHE);dim=940
        values=a.r.prepared(cache,False,mean=a.np.zeros((1,dim),a.np.float32),std=a.np.ones((1,dim),a.np.float32))
        base,speed,gt=values[1:4]
        files=list(a.m.RESULTS.glob('seed*.npz'));assert len(files)==18
        for seed in (1,2,3):
            for arm in a.c.ARMS:
                for tag in a.c.POLICIES:
                    saved=a.n.RAW_LOAD(a.m.RESULTS/f'seed{seed}_{arm}_{tag}.npz')
                    assert a.np.array_equal(saved['sample_id'],cache['sample_id'])
                    assert saved['trajectory'].shape==(len(gt),20,12,2) and a.np.isfinite(saved['trajectory']).all()
                    actual=a.r.generation_metrics(saved['trajectory'],base,speed,gt)
                    for key,value in actual.items():a.np.testing.assert_allclose(saved[key],value,rtol=0,atol=1e-6)
        old=json.loads((HERE/'old_results_manifest.json').read_text())['files']
        for path,digest in old.items():assert a.p.sha256(Path(path))==digest,('old result changed',path)
        for path,digest in locked.items():assert a.p.sha256(Path(path))==digest
        record('independent_audit.json',dict(status='PASS',models_sha_verified=18,explicit_prediction_ids_verified=18,metrics_independently_recomputed=True,old_results_unchanged=len(old),production_source_hashes_unchanged=True,training_invoked=False))
        record('results/exit_status.json',dict(status='SUCCESS',time=time.time(),evaluation_only=True))
        state.update(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat());record('worker_state.json',state)
        print('INDEPENDENT AUDIT PASS / all old results and trained models unchanged',flush=True)
    except BaseException:
        traceback.print_exc();state.update(status='FAILURE',finished_at=datetime.now().astimezone().isoformat());record('worker_state.json',state);raise
    finally:
        sys.stdout.flush();sys.stderr.flush();lock.close()
