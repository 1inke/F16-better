"""Entrypoint with an exclusive preflight lock, before scientific imports."""
import sys, msvcrt
from pathlib import Path

if __name__=='__main__':
    lock=None
    try:
        if '--action' in sys.argv and sys.argv[sys.argv.index('--action')+1]=='preflight':
            lock=(Path(__file__).resolve().parent/'preflight.lock').open('a+b')
            if lock.seek(0,2)==0:
                lock.write(b'0');lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        import run_head
        run_head.main()
    finally:
        if lock is not None:lock.close()
