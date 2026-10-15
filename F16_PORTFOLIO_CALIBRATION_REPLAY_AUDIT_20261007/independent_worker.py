import os,sys,time,json,msvcrt,traceback
from pathlib import Path
from datetime import datetime
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
def write(name,value):
    p=HERE/name;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(q,p)
if __name__=='__main__':
    assert not (HERE/'worker_state.json').exists(),'Unique launch only'
    lock=(HERE/'worker.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    started=datetime.now().astimezone().isoformat();logs=HERE/'logs';logs.mkdir(exist_ok=True)
    with (logs/'worker_stdout.log').open('x',encoding='utf-8',buffering=1) as out,(logs/'worker_stderr.log').open('x',encoding='utf-8',buffering=1) as err:
        sys.stdout=out;sys.stderr=err;write('worker_state.json',dict(status='RUNNING',pid=os.getpid(),parent=os.getppid(),started_at=started))
        try:
            import audit
            audit.main();finished=datetime.now().astimezone().isoformat();write('exit_status.json',dict(status='SUCCESS',started_at=started,finished_at=finished));write('worker_state.json',dict(status='SUCCESS',pid=os.getpid(),parent=os.getppid(),started_at=started,finished_at=finished))
        except Exception:
            trace=traceback.format_exc();print(trace,file=err,flush=True);write('exit_status.json',dict(status='FAILURE',finished_at=datetime.now().astimezone().isoformat(),traceback=trace));write('worker_state.json',dict(status='FAILURE',pid=os.getpid(),parent=os.getppid(),started_at=started));raise
