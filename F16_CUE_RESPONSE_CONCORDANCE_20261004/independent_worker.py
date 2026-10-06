"""One independent on-demand diagnostic worker with exclusive byte lock."""
from pathlib import Path
from datetime import datetime
import os, sys, traceback, msvcrt
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    os.chdir(HERE)
    lock=(HERE/'independent_worker.lock').open('a+b')
    if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    sys.stdout=(HERE/'stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(HERE/'stderr.log').open('a',encoding='utf-8',buffering=1)
    import diagnosis
    state=dict(status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),started_at=datetime.now().astimezone().isoformat(),
      task='Trajectory-CueConcordance-20261004',launcher='Windows Task Scheduler on demand, no triggers',optimizer=False,model_forward=False)
    diagnosis.write(HERE/'independent_worker_state.json',state)
    print('[INDEPENDENT] '+str(state),flush=True)
    try:
        diagnosis.run();state.update(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat())
    except BaseException:
        traceback.print_exc()
        failure=HERE/'failures'/datetime.now().strftime('worker_%Y%m%d_%H%M%S');failure.mkdir(parents=True,exist_ok=False)
        (failure/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        diagnosis.write(failure/'worker_state.json',state)
        diagnosis.write(failure/'source_hashes.json',{str(p):diagnosis.digest(p) for p in HERE.glob('*.py')})
        state.update(status='FAILURE',finished_at=datetime.now().astimezone().isoformat(),failure=str(failure))
        diagnosis.write(HERE/'exit_status.json',dict(status='FAILURE',worker_pid=state['pid'],failure=str(failure)))
        raise
    finally:
        diagnosis.write(HERE/'independent_worker_state.json',state)
        sys.stdout.flush();sys.stderr.flush();lock.close()
