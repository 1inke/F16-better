import os,sys,msvcrt,traceback,contextlib
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
lock=open(H/'worker.lock','a+b');lock.seek(0)
if not lock.read(1):lock.write(b'1');lock.flush()
lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
with open(H/'worker_stdout.log','a',encoding='utf8',buffering=1) as out,open(H/'worker_stderr.log','a',encoding='utf8',buffering=1) as err,contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
 import common as c
 started=c.now();c.write('worker_state.json',dict(status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),started_at=started))
 try:
  import run
  run.main();c.write('exit_status.json',dict(status='SUCCESS',started_at=started,finished_at=c.now(),code=0));c.write('worker_state.json',dict(status='SUCCESS',pid=os.getpid(),parent_pid=os.getppid(),started_at=started,finished_at=c.now()))
 except BaseException as e:
  traceback.print_exc();c.write('exit_status.json',dict(status='FAILURE',started_at=started,finished_at=c.now(),error=repr(e),traceback=traceback.format_exc(),code=1));c.write('worker_state.json',dict(status='FAILURE',pid=os.getpid(),parent_pid=os.getppid(),started_at=started,finished_at=c.now(),error=repr(e)));raise
