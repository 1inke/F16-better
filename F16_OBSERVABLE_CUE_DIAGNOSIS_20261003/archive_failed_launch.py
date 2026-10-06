"""Preserve the actual failed launch, infrastructure source and SHA provenance."""
from common import *
import shutil
state=read('independent_worker_state.json')
assert state['status']=='FAILURE'
destination=HERE/'failures/launch1_progress_io_failure'
assert not destination.exists(), 'Failure already archived; do not overwrite evidence'
destination.mkdir(parents=True)
hashes={}
for name in ('stderr.log','stdout.log','independent_worker_state.json','exit_status.json','progress.json','progress.json.tmp','preflight.json','common.py','native.py','independent_worker.py','run_diagnosis.py'):
    path=HERE/name
    if path.exists():
        shutil.copy2(path,destination/name);hashes[name]=sha(destination/name)
write('failures/launch1_progress_io_failure/provenance.json',dict(status='PRESERVED_REAL_FAILURE',original_worker=state,
    hashes=hashes,optimizer_reader_jobs_completed=0,stage='Native representation extraction, progress atomic replace WinError5',
    no_legacy_model_or_result_modified=True,root_cause='File access/replace failed; sharing versus persistent permission not conclusively established'))
