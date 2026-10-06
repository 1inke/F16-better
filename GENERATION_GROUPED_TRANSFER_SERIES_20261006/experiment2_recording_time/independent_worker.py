"""Unique OS worker with exclusive byte lock; no periodic trigger required."""
import os,sys,time,json,msvcrt,traceback
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
def write(name,value):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
if __name__=='__main__':
    started=datetime.now().astimezone().isoformat();lock=(HERE/'worker.lock').open('a+b');lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('Another worker owns this experiment byte lock')
    if (HERE/'results/exit_status.json').exists() and json.loads((HERE/'results/exit_status.json').read_text())['status']=='SUCCESS':raise RuntimeError('Already complete; no new worker')
    logs=HERE/'logs';logs.mkdir(exist_ok=True)
    with (logs/'worker_stdout.log').open('a',encoding='utf-8',buffering=1) as out,(logs/'worker_stderr.log').open('a',encoding='utf-8',buffering=1) as err:
        sys.stdout=out;sys.stderr=err
        write('worker_state.json',dict(status='RUNNING',pid=os.getpid(),parent=os.getppid(),started_at=started))
        try:
            import runner
            runner.main();finished=datetime.now().astimezone().isoformat()
            write('results/exit_status.json',dict(status='SUCCESS',started_at=started,finished_at=finished));write('worker_state.json',dict(status='SUCCESS',pid=os.getpid(),parent=os.getppid(),started_at=started,finished_at=finished))
        except Exception:
            trace=traceback.format_exc();print(trace,file=err,flush=True)
            write('results/exit_status.json',dict(status='FAILURE',started_at=started,finished_at=datetime.now().astimezone().isoformat(),error=trace));write('worker_state.json',dict(status='FAILURE',pid=os.getpid(),parent=os.getppid(),started_at=started));raise
