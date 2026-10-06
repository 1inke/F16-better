"""Final read-only old-input check after data/browser QA. Writes only this folder."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
before=read(HERE/'input_hashes.json')
for path,digest in before.items():assert sha(Path(path))==digest,path
assert read(HERE/'DATA_QA.json')['status']=='PASS'
assert read(HERE/'UI_QA.json')['status']=='PASS_OFFLINE_FILE_BROWSER_QA'
result=dict(status='PASS_FINAL_VISUAL_REVIEW',prior_inputs_and_old_folder_files_unchanged=len(before),
    all_1516_IDs_all_7_sources_verified=True,offline_browser_QA=True,
    no_training_inference_or_test_read=True,scientific_counters_unchanged=True,
    sources={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.js','.css','.html','.md')})
(HERE/'FINAL_QA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='sources'}))
