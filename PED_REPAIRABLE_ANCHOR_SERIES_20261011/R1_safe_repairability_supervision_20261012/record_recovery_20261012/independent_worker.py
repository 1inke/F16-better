import os,sys,time,msvcrt,traceback
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
lock=(H/"exclusive.lock").open("a+b");lock.seek(0)
if lock.read(1)==b"":lock.write(b"0");lock.flush()
lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
sys.stdout=(H/"worker_stdout.log").open("a",encoding="utf-8",buffering=1);sys.stderr=(H/"worker_stderr.log").open("a",encoding="utf-8",buffering=1)
import common as c
state=dict(status="RUNNING",pid=os.getpid(),parent=os.getppid(),started_at=c.now(),kind="GT_ORACLE_LABEL_AND_CPU_LINEAR_SELECTOR_NOT_NEURAL")
c.write("worker_state.json",state)
try:
    import run
    run.main();state.update(status="SUCCESS",finished_at=c.now())
except BaseException:
    traceback.print_exc();state.update(status="FAILURE",finished_at=c.now())
    c.write("WORKER_FAILURE_"+str(time.time_ns())+".json",dict(traceback=traceback.format_exc(),resource=c.RESOURCE));raise
finally:
    c.write("worker_state.json",state);c.write("exit_status.json",state)
