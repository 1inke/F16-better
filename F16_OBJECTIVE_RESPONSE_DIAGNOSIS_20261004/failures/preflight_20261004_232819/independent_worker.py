from pathlib import Path
from datetime import datetime
import os,sys,msvcrt,traceback,shutil
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    os.chdir(HERE);lock=(HERE/'independent_worker.lock').open('a+b')
    if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    sys.stdout=(HERE/'stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(HERE/'stderr.log').open('a',encoding='utf-8',buffering=1)
    import diagnosis as d
    state=dict(status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),started_at=datetime.now().astimezone().isoformat(),task=d.TASK,
      model_training=False,model_forward=False,path_tensor_autograd_only=True)
    d.write(HERE/'independent_worker_state.json',state);print('[INDEPENDENT] '+str(state),flush=True)
    try:d.run();state.update(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat())
    except BaseException:
        traceback.print_exc();failure=HERE/'failures'/datetime.now().strftime('worker_%Y%m%d_%H%M%S');failure.mkdir(parents=True,exist_ok=False)
        (failure/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        for p in HERE.glob('*.py'):shutil.copy2(p,failure/p.name)
        d.write(failure/'source_hashes.json',d.sources());state.update(status='FAILURE',failure=str(failure),finished_at=datetime.now().astimezone().isoformat())
        d.write(HERE/'exit_status.json',dict(status='FAILURE',failure=str(failure),worker_pid=state['pid']));raise
    finally:d.write(HERE/'independent_worker_state.json',state);sys.stdout.flush();sys.stderr.flush();lock.close()
