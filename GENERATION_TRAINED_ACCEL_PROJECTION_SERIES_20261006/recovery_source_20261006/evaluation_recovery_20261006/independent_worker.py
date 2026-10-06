"""Evaluation-only sole OS byte-locked worker, never calls train."""
from pathlib import Path
from datetime import datetime
import os,sys,json,msvcrt,traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
def record(v):(HERE/'independent_worker_state.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
if __name__=='__main__':
    os.chdir(HERE);lock=(HERE/'independent_worker.lock').open('a+b')
    if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    sys.stdout=(HERE/'stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(HERE/'stderr.log').open('a',encoding='utf-8',buffering=1)
    state=dict(pid=os.getpid(),parent_pid=os.getppid(),status='RUNNING',evaluation_only=True,
        started_at=datetime.now().astimezone().isoformat(),stage='INITIAL_SOURCE_OLD_COMPLETED_SHA');record(state)
    try:
        import run_evaluation as e
        e.run(e.torch.device('cuda:0'))
        state.update(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat());record(state)
    except BaseException:
        traceback.print_exc();state.update(status='FAILURE',finished_at=datetime.now().astimezone().isoformat());record(state)
        (HERE/'worker_failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:sys.stdout.flush();sys.stderr.flush();lock.close()
