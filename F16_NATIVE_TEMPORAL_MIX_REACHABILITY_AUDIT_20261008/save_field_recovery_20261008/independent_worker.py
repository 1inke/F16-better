import sys,os,json,time,traceback,msvcrt
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
f=(H/'exclusive.lock').open('a+b');f.seek(0);f.write(b'1');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
def write(name,value):
 p=H/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(t,p)
state=dict(status='RUNNING',pid=os.getpid(),parent=os.getppid(),started_at=datetime.now().astimezone().isoformat(),type='NONFORMAL_FROZEN_SAVED_REACHABILITY_NO_TRAINING')
write('worker_state.json',state)
try:
 with (H/'worker_stdout.log').open('a',encoding='utf-8',buffering=1) as out,(H/'worker_stderr.log').open('a',encoding='utf-8',buffering=1) as err:
  sys.stdout=out;sys.stderr=err
  import run
  run.main()
 state.update(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat());write('exit_status.json',state);write('worker_state.json',state)
except BaseException:
 state.update(status='FAILURE',finished_at=datetime.now().astimezone().isoformat(),traceback=traceback.format_exc());write('exit_status.json',state);write('worker_state.json',state);raise
