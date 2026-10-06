from pathlib import Path
from datetime import datetime
import os,sys,msvcrt,traceback,json,psutil
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    os.chdir(HERE)
    preflight='--preflight' in sys.argv
    lock=(HERE/('preflight.lock' if preflight else 'independent_worker.lock')).open('a+b')
    if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    prefix='preflight' if preflight else 'worker'
    sys.stdout=(HERE/(prefix+'_stdout.log')).open('a',encoding='utf-8',buffering=1)
    sys.stderr=(HERE/(prefix+'_stderr.log')).open('a',encoding='utf-8',buffering=1)
    state=dict(pid=os.getpid(),parent_pid=os.getppid(),create_time=psutil.Process().create_time(),started_at=datetime.now().astimezone().isoformat(),status='RUNNING',scope='PRIMARY_PARAMETER_DIAGNOSIS_NO_OPTIMIZER')
    path=HERE/(prefix+'_state.json')
    def record():path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    record()
    try:
        import diagnosis
        (diagnosis.preflight if preflight else diagnosis.run)()
        state.update(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat());record()
    except BaseException:
        traceback.print_exc()
        state.update(status='FAILURE',finished_at=datetime.now().astimezone().isoformat(),failure='STOP_PRESERVE_PROVENANCE_WAIT_HUMAN');record()
        if 'diagnosis' in sys.modules:
            diagnosis.guard.save('FAILURE')
        raise
    finally:sys.stdout.flush();sys.stderr.flush();lock.close()
