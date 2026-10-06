"""Read-only old-file SHA closure after resource stop; no torch/model/optimizer."""
from pathlib import Path
import hashlib,json,time
HERE=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for z in iter(lambda:f.read(2**20),b''):h.update(z)
    return h.hexdigest()
start=time.perf_counter();files=json.loads((HERE/'old_results_manifest.json').read_text(encoding='utf-8'))['files']
for path,digest in files.items():assert sha(path)==digest,path
report=dict(status='PASS',old_files_unchanged=len(files),seconds=time.perf_counter()-start,no_torch_or_training=True,resource_failure_not_retried=True)
(HERE/'prelaunch_review/OLD_PRESERVATION_AFTER_STOP.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
