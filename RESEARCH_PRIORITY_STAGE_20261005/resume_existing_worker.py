"""Human-authorized one-shot resume, no model forward or scientific edits."""
from pathlib import Path
import json, hashlib, psutil, ctypes, csv, datetime

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / 'GENERATION_DISCREPANCY_NORMALIZATION_SERIES_20261005'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

class Mem(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (key, ctypes.c_ulonglong) for key in ('total_phys', 'avail_phys',
        'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended')]

state = json.loads((HERE / 'runtime_state.json').read_text(encoding='utf-8'))
assert state['status'] == 'PAUSED_BY_HUMAN_PROCESS_SUSPENDED'
pause = state['human_pause']
p = psutil.Process(state['worker_pid'])
identity = p.as_dict(attrs=['pid', 'ppid', 'name', 'cmdline', 'create_time', 'status'])
assert identity['pid'] == 23612 and identity['ppid'] == 2728
assert identity['name'].lower() == 'python.exe'
assert abs(identity['create_time'] - pause['process_create_time']) < 1e-5
assert any(str(HERE).lower() in arg.lower() and 'independent_worker.py' in arg for arg in identity['cmdline'])
assert identity['status'] == psutil.STATUS_STOPPED, identity
sources = json.loads((HERE / 'preflight.json').read_text(encoding='utf-8'))['source_hashes']
assert len(sources) == 402
for path, digest in sources.items():
    assert sha(path) == digest, ('SOURCE_SHA_MISMATCH_STOP', path)
assert not (HERE / 'results' / 'exit_status.json').exists(), 'Unexpected saved exit: STOP'
histories = []
for arm in ('zero', 'mean_std'):
    for seed in (1,2,3):
        folder = HERE / 'results' / 'models' / arm / ('seed_' + str(seed)) / 'balanced_protected'
        file = folder / 'history.csv'
        with file.open(encoding='utf-8-sig', newline='') if file.exists() else __import__('contextlib').nullcontext(None) as stream:
            rows = list(csv.DictReader(stream)) if stream else []
        histories.append(dict(arm=arm, seed=seed, epoch=int(rows[-1]['epoch']) if rows else 0,
                              complete=(folder/'complete.json').exists()))
memory = p.memory_info()
m = Mem(); m.length = ctypes.sizeof(m)
assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
assert max(memory.rss, memory.private) <= 6 * 2**30, 'PROCESS_MEMORY_STOP'
assert min(m.avail_phys, m.avail_page) >= 6 * 2**30, 'SYSTEM_RESERVE_STOP'
# All checks above are read-only. Do not terminate/relaunch/modify old sources.
p.resume()
now = datetime.datetime.now().astimezone()
after = p.status()
assert after != psutil.STATUS_STOPPED
query = now + datetime.timedelta(minutes=15)
deadline = now + datetime.timedelta(minutes=30)
ceilquery = (query + datetime.timedelta(seconds=1)).replace(microsecond=0)
print(json.dumps(dict(action='HUMAN_AUTHORIZED_SAME_PROCESS_RESUME', identity=identity,
                     status_after=after, resumed_at=now.isoformat(), new_worker=False,
                     original_process_create_time=identity['create_time'], source_SHA_count=len(sources),
                     source_SHA_status='PASS', histories_before_resume=histories,
                     estimated_remaining_minutes=[20,35], planning_remaining_minutes=30,
                     query_at=query.isoformat(), automation_query_at=ceilquery.isoformat(),
                     analysis_deadline=deadline.isoformat(),
                     memory=dict(RSS_GiB=memory.rss/2**30, private_GiB=memory.private/2**30,
                                 available_physical_GiB=m.avail_phys/2**30,
                                 available_commit_GiB=m.avail_page/2**30),
                     original_optimizer_RNG_in_memory_preserved=True,
                     no_model_forward_or_retraining=True), ensure_ascii=False))
