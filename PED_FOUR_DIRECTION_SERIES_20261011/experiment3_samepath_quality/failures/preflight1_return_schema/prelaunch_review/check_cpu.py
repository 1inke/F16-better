"""Current source and actual two saved128 CPU prelaunch review; no Torch."""
import sys,importlib.util,time
from pathlib import Path
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent.parent
s=importlib.util.spec_from_file_location('cpu_review',H/'completion_review/audit_saved.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def main():
 start=time.perf_counter();a.guard();files=a.read(H/'source_manifest.json')['files'];a.verify(files);checks=[]
 for arm in a.ARMS:
  path=H/'prelaunch_review'/f'{arm}_smoke128.npz';m=H/'prelaunch_review'/f'{arm}_smoke128_metrics.json';before={str(p):a.sha(p) for p in (path,m)}
  z=a.npz(path);met,ass,vectors,clusters,bounds=a.audit_job(z,arm);saved=a.read(m);error=max(a.compare(met,saved['metrics']),a.compare(ass,saved['assignment']));assert error<=1e-12
  a.verify(before);checks.append(dict(arm=arm,rows=len(z['sample_id']),error=error,bounds=bounds,full_saved_SciPy_quality_motion_controls=True,saved_SHA=before))
 a.verify(files);a.guard();a.write('PRELAUNCH_CPU_REVIEW.json',dict(status='PASS',source_count=len(files),source_initial_final_PASS=True,checks=checks,no_torch_optimizer_forward=True,elapsed_seconds=time.perf_counter()-start,resource=a.resource,not_external_review=True));print('PRELAUNCH_CPU_REVIEW_PASS',checks)
if __name__=='__main__':main()
