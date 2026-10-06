"""Read-only checkpoint FIT/CAL diagnosis; no optimizer or DEV/test forward."""
from pathlib import Path
import importlib.util,sys,json,traceback,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
h=module('rank_learning_head',ROOT/'GENERATION_LOW_RANK_CONDITIONING_SERIES/run_head.py')
d=module('rank_learning_diagnostic',ROOT/'GENERATION_CONDITIONING_LEARNING_DIAGNOSIS/run_diagnosis.py')
d.HERE=HERE;d.ROOT=ROOT;d.PARENT=ROOT/'GENERATION_LOW_RANK_CONDITIONING_SERIES'
d.h=h;d.p=h.p;d.m=h.m;d.r=h.r;h.ConditioningHead=h.LowRankHead
def main():
    lock=json.loads((d.PARENT/'results/model_lock.json').read_text());assert len(lock['models'])==18 and not lock['dev_used_for_selection'] and not lock['test_access']
    for path,digest in lock['models'].items():assert h.p.sha256(Path(path))==digest,path
    audit=json.loads((d.PARENT/'results/independent_audit.json').read_text());assert audit['status']=='PASS' and audit['metrics_audit']['metrics_recomputed'] and audit['old_files_unchanged']==14513
    assert json.loads((d.PARENT/'results/exit_status.json').read_text())['status']=='SUCCESS'
    d.main()
    result=json.loads((HERE/'RESULT.json').read_text())
    result['arm_labels']={'zero':'dense_conditioning','mean_std':'rank8_conditioning'}
    for path in (Path(__file__),ROOT/'GENERATION_CONDITIONING_LEARNING_DIAGNOSIS/run_diagnosis.py',ROOT/'GENERATION_CONDITIONING_LEARNING_DIAGNOSIS/analyze.py'):
        result['source_hashes'][str(path)]=h.p.sha256(path)
    h.m.write_json(HERE/'RESULT.json',result)
    analysis=module('rank_learning_analysis',ROOT/'GENERATION_CONDITIONING_LEARNING_DIAGNOSIS/analyze.py');analysis.HERE=HERE;analysis.main()
    summary=json.loads((d.PARENT/'results/summary.json').read_text())
    h.m.write_json(HERE/'FORMAL_RESULT_REFERENCE.json',dict(status='PASS_REFERENCE',summary=summary,audit=audit,model_lock=lock,source_sha=h.p.sha256(d.PARENT/'results/summary.json')))
    d.manifest()
    print('FINAL SOURCE/MODEL/ALL PRIOR HASH PASS',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        path=HERE/('failure_'+str(time.time_ns())+'.txt');path.write_text(traceback.format_exc(),encoding='utf-8');raise
