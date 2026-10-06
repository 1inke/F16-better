from pathlib import Path
import os,sys,json,runpy,traceback,msvcrt
from datetime import datetime
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    os.chdir(HERE);lock=(HERE/'worker.lock').open('a+b')
    if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    sys.stdout=(HERE/'stdout.log').open('a',encoding='utf-8',buffering=1);sys.stderr=(HERE/'stderr.log').open('a',encoding='utf-8',buffering=1)
    state=dict(pid=os.getpid(),parent_pid=os.getppid(),start=datetime.now().astimezone().isoformat(),status='RUNNING')
    def record():(HERE/'worker_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    record()
    try:
        sys.argv=[str(HERE/'run_diagnosis.py')];runpy.run_path(sys.argv[0],run_name='__main__');state['status']='SUCCESS'
    except BaseException:state['status']='FAILURE';traceback.print_exc();raise
    finally:state['finish']=datetime.now().astimezone().isoformat();record();lock.close()
