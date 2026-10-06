"""CPU prelaunch integrity/field review, not model forward or second gradient test."""
from pathlib import Path
import ast, hashlib, json, math, numpy as np
HERE=Path(__file__).resolve().parent;SERIES=HERE.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
assert not (HERE/'REVIEW.json').exists(),'PRESERVE_EXISTING'
pf=read(SERIES/'preflight.json');state=read(SERIES/'preflight_state.json');proto=read(SERIES/'protocol.json')
assert pf['status']=='PASS' and state['status']=='SUCCESS'
assert proto['optimizer']==False and proto['weight_buffer_update']==False and proto['DEV']==False
assert proto['newhead_parameter_count']==45798 and proto['FIT']==25912 and proto['CAL']==2879
for path,digest in pf['source_hashes'].items():assert sha(path)==digest,path
assert pf['models']==18 and pf['analysis_SHA_resume']=='PASS_SKIP_BADSOURCE_BADOUTPUT_REJECT_NO_OPTIMIZER'
assert len(pf['actual_GPU_reports'])==3
max_chunk=0.;max_scalar=0.;max_align=0.
for r in pf['actual_GPU_reports']:
    assert r['path_alignment']<=2e-5 and r['scalar_error']<=2e-5
    max_scalar=max(max_scalar,r['scalar_error']);max_align=max(max_align,r['path_alignment'])
    assert len(r['gradient_chunk_checks'])==6
    for test in r['gradient_chunk_checks']:
        assert test['error']<=test['tolerance'];max_chunk=max(max_chunk,test['error'])
    assert any(v['passed'] and v['winner_switch']==0 for v in r['numerical_float64_fd'])
    for v in r['numerical_float64_fd']:assert math.isfinite(v['analytic']) and math.isfinite(v['numeric'])
# Inspect actual worker branch for source/output SHA resume validation.
tree=ast.parse((SERIES/'diagnosis.py').read_text(encoding='utf-8'))
complete=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='complete_job')
env=dict(sha=sha);exec(compile(ast.Module(body=[complete],type_ignores=[]),'<production-complete-job>','exec'),env)
fixture=HERE/'fixture.json';fixture.write_text('{"fixture":"prelaunch"}',encoding='utf-8')
lock={'fixture':'fixed'};done={'status':'SUCCESS','source_hashes':lock,'outputs':{str(fixture):sha(fixture)}}
assert env['complete_job'](done,lock)
for name in ('badsource','badoutput'):
    bad=json.loads(json.dumps(done))
    if name=='badsource':bad['source_hashes']={}
    else:bad['outputs'][str(fixture)]='0'*64
    try:env['complete_job'](bad,lock)
    except AssertionError:pass
    else:raise AssertionError(name+' not rejected')
inputs={}
for path in [SERIES/'preflight.json',SERIES/'preflight_state.json',SERIES/'protocol.json',SERIES/'SETUP_PROVENANCE.md']+list((SERIES/'failures').rglob('*')):
    if path.is_file():inputs[str(path)]=sha(path)
result=dict(status='PASS_CPU_PRELAUNCH_REVIEW_NO_FORWARD',sources=len(pf['source_hashes']),old_files=pf['old_files'],models=18,
    actual_preflight_reports_inspected=3,path_alignment_max=max_align,scalar_error_max=max_scalar,chunk_gradient_error_max=max_chunk,
    actual_production_AST_SHA_skip_badsource_badoutput='PASS',original_primary_only=True,no_optimizer_weights_update=True,
    no_GPU_forward=True,no_second_gradient_claim=True,provenance_SHA=inputs,
    field_interpretation='Local Euclidean primary derivatives not total/AdamW actual update; nearzero UNDETERMINED; CAL selection/scene/window limits unchanged')
(HERE/'REVIEW.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)

