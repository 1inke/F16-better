from pathlib import Path
import os,sys,json,runpy,traceback,msvcrt
from datetime import datetime
HERE=Path(__file__).resolve().parent
def record(state):(HERE/'independent_worker_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
if __name__=='__main__':
    os.chdir(HERE);lock=(HERE/'independent_worker.lock').open('a+b')
    if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    sys.stdout=(HERE/'stdout.log').open('a',encoding='utf-8',buffering=1);sys.stderr=(HERE/'stderr.log').open('a',encoding='utf-8',buffering=1)
    state=dict(pid=os.getpid(),parent_pid=os.getppid(),started_at=datetime.now().astimezone().isoformat(),status='RUNNING',launcher='Windows Task Scheduler on-demand/no triggers');record(state)
    print('[INDEPENDENT] '+json.dumps(state),flush=True)
    try:
        sys.argv=[str(HERE/'run_native.py'),'--action','run','--device','cuda:0'];runpy.run_path(str(HERE/'run_native.py'),run_name='__main__')
        state.update(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat());record(state)
    except BaseException:
        traceback.print_exc();state.update(status='FAILURE',finished_at=datetime.now().astimezone().isoformat());record(state);raise
    finally:sys.stdout.flush();sys.stderr.flush();lock.close()
