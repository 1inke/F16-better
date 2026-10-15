import sys,os,json,traceback,msvcrt
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
H=Path(__file__).resolve().parent
def write(n,v):(H/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
assert not (H/'worker_state.json').exists(),'No duplicate worker'
lock=(H/'exclusive.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
sys.stdout=(H/'worker_stdout.log').open('x',encoding='utf-8',buffering=1);sys.stderr=(H/'worker_stderr.log').open('x',encoding='utf-8',buffering=1)
state=dict(status='RUNNING',pid=os.getpid(),parent=os.getppid(),started_at=datetime.now().astimezone().isoformat());write('worker_state.json',state)
try:
 import run
 run.main()
 state.update(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat());write('exit_status.json',dict(state));write('worker_state.json',state)
except Exception:
 traceback.print_exc();state.update(status='FAILURE',finished_at=datetime.now().astimezone().isoformat(),traceback=traceback.format_exc());write('exit_status.json',dict(state));write('worker_state.json',state);sys.exit(1)
