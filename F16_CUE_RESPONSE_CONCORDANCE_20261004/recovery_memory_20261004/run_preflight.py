"""Preserve full-size resource preflight logs and any failure provenance."""
from pathlib import Path
from datetime import datetime
import sys,traceback
sys.dont_write_bytecode=True
import diagnosis
HERE=Path(__file__).resolve().parent
assert not (HERE/'preflight_stdout.log').exists(),'Preserve earlier preflight logs'
sys.stdout=(HERE/'preflight_stdout.log').open('x',encoding='utf-8',buffering=1)
sys.stderr=(HERE/'preflight_stderr.log').open('x',encoding='utf-8',buffering=1)
try:diagnosis.preflight()
except BaseException:
    traceback.print_exc()
    folder=HERE/'failures'/datetime.now().strftime('preflight_%Y%m%d_%H%M%S')
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    diagnosis.write(folder/'source_hashes.json',{str(p):diagnosis.digest(p) for p in HERE.glob('*.py')})
    raise
finally:sys.stdout.flush();sys.stderr.flush()
