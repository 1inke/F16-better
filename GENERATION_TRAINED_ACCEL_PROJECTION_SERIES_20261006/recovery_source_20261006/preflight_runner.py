"""One exclusive gate workflow; preserve every failure and its source snapshot."""
from pathlib import Path
import os,sys,runpy,msvcrt,traceback,time,json,shutil
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    os.chdir(HERE);lock=(HERE/'preflight.lock').open('a+b')
    if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    sys.stdout=(HERE/'preflight_stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(HERE/'preflight_stderr.log').open('a',encoding='utf-8',buffering=1)
    try:
        import run_head as h
        sys.argv=[str(HERE/'run_head.py'),'--action','preflight','--device','cuda:0'];h.main()
        runpy.run_path(str(HERE/'test_resume.py'),run_name='__main__')
        runpy.run_path(str(HERE/'benchmark.py'),run_name='__main__')
        h.verify(h.read(HERE/'preflight.json')['source_hashes']);h.old_manifest();h.guard.save('ALL_GATES_PASS')
        h.write(HERE/'gate_workflow.json',dict(status='PASS',production_started=False,completed_at=time.time()))
    except BaseException:
        traceback.print_exc();failure=HERE/'failures'/('preflight_'+str(time.time_ns()));failure.mkdir(parents=True)
        for path in list(HERE.glob('*.py'))+list(HERE.glob('*.json')):shutil.copy2(path,failure/path.name)
        (failure/'TRACE.txt').write_text(traceback.format_exc(),encoding='utf-8')
        (HERE/'gate_workflow.json').write_text(json.dumps(dict(status='STOP_PREFLIGHT_WAIT_HUMAN',failure=str(failure),at=time.time()),indent=2),encoding='utf-8');raise
    finally:sys.stdout.flush();sys.stderr.flush();lock.close()
