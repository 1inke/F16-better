"""One OS byte-lock independent S1 worker, no retries or old-worker access."""
import os,sys,time,msvcrt,traceback
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
lock=(H/"exclusive.lock").open("a+b");lock.seek(0)
if lock.read(1)==b"":lock.write(b"0");lock.flush()
lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
sys.stdout=(H/"worker_stdout.log").open("a",encoding="utf-8",buffering=1);sys.stderr=(H/"worker_stderr.log").open("a",encoding="utf-8",buffering=1)
import common as c
started=c.now();envelope=dict(status="RUNNING",pid=os.getpid(),parent=os.getppid(),started_at=started,type="CPU_TRAIN_ONLY_SELECTOR_NOT_NEURAL")
c.write("worker_state.json",envelope)
try:
    import run
    run.main();envelope.update(status="SUCCESS",finished_at=c.now())
except BaseException:
    traceback.print_exc();envelope.update(status="FAILURE",finished_at=c.now());c.write("WORKER_FAILURE_"+str(time.time_ns())+".json",dict(traceback=traceback.format_exc(),resource=c.RESOURCE));raise
finally:
    c.write("worker_state.json",envelope);c.write("exit_status.json",envelope)
