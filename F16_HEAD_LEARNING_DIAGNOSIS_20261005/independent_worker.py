from pathlib import Path
from datetime import datetime
import json,os,sys,msvcrt,traceback,shutil
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
def state_write(value):(HERE/'independent_worker_state.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
if __name__=='__main__':
    os.chdir(HERE);lock=(HERE/'independent_worker.lock').open('a+b')
    if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    sys.stdout=(HERE/'stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(HERE/'stderr.log').open('a',encoding='utf-8',buffering=1)
    state=dict(status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),started_at=datetime.now().astimezone().isoformat(),task='Trajectory-HeadLearning-Diagnosis-20261005',formal_trial=False)
    state_write(state);print('[INDEPENDENT] '+json.dumps(state),flush=True)
    try:
        import diagnosis as d
        d.run(d.torch.device('cuda:0'));state.update(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat())
    except BaseException:
        traceback.print_exc();failure=HERE/'failures'/datetime.now().strftime('worker_%Y%m%d_%H%M%S');failure.mkdir(parents=True,exist_ok=False)
        (failure/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        for p in HERE.glob('*.py'):shutil.copy2(p,failure/p.name)
        state.update(status='FAILURE',failure=str(failure),finished_at=datetime.now().astimezone().isoformat())
        (HERE/'exit_status.json').write_text(json.dumps(dict(status='FAILURE',failure=str(failure)),indent=2),encoding='utf-8');raise
    finally:state_write(state);sys.stdout.flush();sys.stderr.flush();lock.close()
