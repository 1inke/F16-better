import os,sys,time,json,re,hashlib,traceback
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent;ROOT=P.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    s=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):s.update(b)
    return s.hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main():
    files={};counts={};partial_files=0
    for name in ['input_manifest.json','review_input_manifest.json','supplement_input_manifest.json','probes/input_manifest.json','partial_forward_review/input_manifest.json']:
        data=read(H/name)
        for p,s in data['files'].items():
            if p in files:assert files[p]==s
            files[p]=s
    # Exact previous source/calibration/checkpoint manifest, not historical all files.
    for p,s in read(P/'source_manifest.json')['files'].items():
        if p in files:assert files[p]==s
        files[p]=s
    for p,s in files.items():assert sha(p)==s,('FINAL_INPUT_SHA',p)
    for folder in [H,H/'probes']:
        r=read(folder/'RESULT.json')
        for p,s in r['output_sha'].items():assert sha(p)==s
        for name,s in r['summary_sha'].items():assert sha(folder/(name+'.json'))==s
    assert read(H/'REVIEW.json')['status'].startswith('PASS') and read(H/'SUPPLEMENT_REVIEW.json')['status'].startswith('PASS') and read(H/'partial_forward_review/REVIEW.json')['status'].startswith('PASS_PARTIAL')
    for name in ('D1a','D2','D3','D6','D7','D5c'):counts[name]=len(read(H/(name+'.json'))['records'])
    for fold in ('eth','hotel'):
        raw=npz(H/f'forward_diagnostics/RAW_{fold}.npz');n=len(raw['sample_id']);normal=raw['normal']
        for seed in (1,2,3):
            base=npz(H/f'forward_diagnostics/{fold}_seed{seed}_original.npz');social=npz(H/f'forward_diagnostics/{fold}_seed{seed}_SOCIAL_paths.npz');old=npz(P/f'stage2_raw_interventions/jobs/{fold}_seed{seed}/noNbr/outputs.npz');assert max(abs(base['paths']-old['base_paths']).max(),abs(base['logits']-old['base_logits']).max())<=2e-5
            for method in ('top1','meanK'):
                b=base['paths'][np.arange(n),base['logits'].argmax(1)] if method=='top1' else base['paths'].mean(1);m=social['mirror_paths'][np.arange(n),social['mirror_logits'].argmax(1)] if method=='top1' else social['mirror_paths'].mean(1);nn=social['noNbr_paths'][np.arange(n),social['noNbr_logits'].argmax(1)] if method=='top1' else social['noNbr_paths'].mean(1);s=npz(H/f'forward_diagnostics/{fold}_seed{seed}_SOCIAL_{method}.npz');assert np.array_equal(((b[:,-1]-nn[:,-1])*normal).sum(1),s['l']);assert np.array_equal(((m[:,-1]-nn[:,-1])*normal).sum(1),s['lprime']);partial_files+=1
    report=(H/'REPORT.md').read_text(encoding='utf-8');targets=re.findall(r'\]\(<([^>]+)>\)|\]\(([^)]+)\)',report);links=[]
    for left,right in targets:
        path=left or right
        if not path.startswith('https:'):
            path=re.sub(r':\d+$','',path);assert Path(path).is_file(),('BROKEN_REPORT_LINK',path);links.append(path)
    control=read(H/'runtime_state.json');root=read(P/'runtime_state.json');ledger=read(ROOT/'AUTONOMOUS_GENERATION_CYCLE.json')['ped_motion_understanding_E0_E6_completion_20261009'];assert control['whole_plan_complete'] is False and control['active_worker'] is False and control['E3_processes_remain_suspended']==[35588,14516] and control['automation_status']=='PAUSED_UNCHANGED';assert root['whole_series_complete'] is False and root['active_worker'] is False;assert ledger['long_training_processes_remain_suspended'] is True and ledger['automation_synchronization']=='PAUSED_BY_EXPLICIT_HUMAN'
    failure=read(H/'forward_diagnostics/FAILURE.json');assert 'STOP_MIRROR_CLOSEST' in failure['traceback'];assert read(H/'forward_diagnostics/protocol.json')['cache_alignment_gate']==.00002
    output=dict(status='PASS_REPORT_DELIVERY_T0_T1_AND_PARTIAL_T2_FAILURE_DISCLOSED',records=counts,input_source_files_verified=len(files),report_links_checked=len(links),partial_lateral_path_reconstructions=partial_files,old_SOURCE_model_SHA_unchanged=True,E3_and_automation_remain_paused=True,T2_failed_not_faked_complete=True,report_sha=sha(H/'REPORT.md'),memory_sha=sha(ROOT/'.memory'),runtime_sha=sha(H/'runtime_state.json'),ledger_sha=sha(ROOT/'AUTONOMOUS_GENERATION_CYCLE.json'),verification_source_sha_after_execution=sha(Path(__file__)))
    (H/'FINAL_REVIEW.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8');print('FINAL QA PASS',output['records'],len(files),len(links))
if __name__=='__main__':main()
