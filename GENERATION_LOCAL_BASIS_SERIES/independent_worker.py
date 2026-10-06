"""On-demand Windows Task Scheduler worker; independent of the Codex process tree."""
from pathlib import Path
import sys
import os
import json
import time
import runpy
import traceback
import msvcrt
from datetime import datetime

HERE=Path(__file__).resolve().parent
def record(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2),encoding='utf-8')

if __name__=='__main__':
    if '--probe' in sys.argv:
        record('detach_probe.json',dict(pid=os.getpid(),parent_pid=os.getppid(),started_at=datetime.now().astimezone().isoformat(),status='STARTED'))
        time.sleep(20)
        record('detach_probe.json',dict(pid=os.getpid(),parent_pid=os.getppid(),finished_at=datetime.now().astimezone().isoformat(),status='SURVIVED_LAUNCHER_EXIT'))
        sys.exit(0)
    os.chdir(HERE)
    # Byte lock stays owned by this independent worker. A second worker must exit.
    lock=(HERE/'independent_worker.lock').open('a+b')
    if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    sys.stdout=(HERE/'stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(HERE/'stderr.log').open('a',encoding='utf-8',buffering=1)
    state=dict(pid=os.getpid(),parent_pid=os.getppid(),started_at=datetime.now().astimezone().isoformat(),status='RUNNING',launcher='Windows Task Scheduler on-demand/no triggers')
    record('independent_worker_state.json',state)
    print('[INDEPENDENT RESUME] '+json.dumps(state),flush=True)
    try:
        sys.argv=[str(HERE/'run_native.py'),'--action','run','--device','cuda:0']
        runpy.run_path(str(HERE/'run_native.py'),run_name='__main__')
        state.update(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat());record('independent_worker_state.json',state)
    except BaseException:
        traceback.print_exc();state.update(status='FAILURE',finished_at=datetime.now().astimezone().isoformat());record('independent_worker_state.json',state)
        raise
    finally:
        sys.stdout.flush();sys.stderr.flush();lock.close()

