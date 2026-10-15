import sys,json,hashlib,re,time
sys.dont_write_bytecode=True
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;ROOT=P.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    T=time.perf_counter();files={};output_count=0
    for name in ('input_manifest.json','forward_input_manifest.json','audit_input_manifest.json','supplement_input_manifest.json','tables_input_manifest.json','random_heading_supplement/input_manifest.json','random_heading_supplement/review_input_manifest.json'):
        for p,s in read(H/name)['files'].items():
            if p in files:assert files[p]==s,('MANIFEST_CONFLICT',p)
            files[p]=s
    for p,s in files.items():assert sha(p)==s,('FINAL_INPUT_SOURCE_SHA',p)
    for result in ('RESULT_SAVED.json','RESULT_FORWARD.json','random_heading_supplement/RESULT.json'):
        r=read(H/result)
        for p,s in r['output_sha'].items():assert sha(p)==s;output_count+=1
        for p,s in r.get('fit_sha',{}).items():assert sha(p)==s
        for name,s in r.get('summary_sha',{}).items():assert sha(H/(name+'.json'))==s
    for result in ('REVIEW.json','SUPPLEMENT_REVIEW.json','TABLES_REVIEW.json','random_heading_supplement/REVIEW.json'):
        r=read(H/result);assert r['status'].startswith('PASS')
        for name,s in r.get('summary_sha',{}).items():assert sha((H/result).parent/(name+'.json'))==s
        for p,s in r.get('new_output_sha',{}).items():assert sha(p)==s
    frozen=read(H/'FROZEN_AUDIT.json');assert len(frozen['records'])==15 and len({(r['fold'],r['seed']) for r in frozen['records']})==15
    numeric=read(H/'NUMERIC_REVISION.json');assert len(numeric['records'])==5 and all(r['fixed_k']==16 and r['double_closest_error_m']<=1e-12 and r['cast_closest_error_m']<=r['budget_m'] for r in numeric['records'])
    assert read(P/'model_diagnosis_20261009/forward_diagnostics/protocol.json')['cache_alignment_gate']==.00002
    assert 'STOP_MIRROR_CLOSEST' in read(P/'model_diagnosis_20261009/forward_diagnostics/FAILURE.json')['traceback']
    control=read(H/'runtime_state.json');assert not control['active_worker'] and control['current_budget_plan_complete'] and not control['whole_E0_E9_series_complete'] and control['automation_status']=='PAUSED_UNCHANGED';ledger=read(ROOT/'AUTONOMOUS_GENERATION_CYCLE.json')['ped_motion_understanding_E0_E6_completion_20261009'];assert ledger['long_training_processes_remain_suspended'] and ledger['automation_synchronization']=='PAUSED_BY_EXPLICIT_HUMAN' and not ledger['whole_series_complete']
    expect=dict(G1=540,G2=420,G3=90,G4=225,G5_HEADING=825,G5_GEOMETRY=750,G5_SOCIAL=90,G6_POPULATIONS=120,G7=135)
    counts={k:len(read(H/(k+'.json'))['records']) for k in expect};assert counts==expect,(counts,expect)
    report=(H/'REPORT.md').read_text(encoding='utf-8');links=re.findall(r'\]\(<([^>]+)>\)',report)
    for p in links:assert Path(p).is_file(),('REPORT_LINK',p)
    assert len(read(H/'random_heading_supplement/SUMMARY.json')['records'])==150
    out=dict(status='PASS_V3_G1_G7_REPORT_AND_ALL15_FROZEN_COMPLETION_DELIVERY',record_counts=counts,unique_actual_input_source_SHA_verified=len(files),saved_output_SHA_entries_checked=output_count,report_links_checked=len(links),report_sha=sha(H/'REPORT.md'),memory_sha=sha(ROOT/'.memory'),ledger_sha=sha(ROOT/'AUTONOMOUS_GENERATION_CYCLE.json'),current_sources_sha={str(p):sha(p) for p in H.glob('*.py')},E3_automation_still_paused=True,T3_lite_not_triggered=True,E8_neural_gate_not_relaxed=True,elapsed_seconds=time.perf_counter()-T)
    (H/'FINAL_REVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(out['status'],counts,len(files))
if __name__=='__main__':main()
