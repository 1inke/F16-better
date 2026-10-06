"""Read-only cycle7 audit/review; no training or edits to prior outputs."""
from pathlib import Path
import json,hashlib
import torch
ROOT=Path(__file__).resolve().parent
FOLDERS=['GENERATION_LOCAL_BASIS_SERIES','GENERATION_BASIS_CONDITION_SERIES','GENERATION_BASIS_COMPOSITION_SERIES']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
items=[]
for name in FOLDERS:
    folder=ROOT.parent/name;results=folder/'results'
    audits={k:json.loads((results/(k+'.json')).read_text()) for k in ['audit','frozen_audit','independent_audit']}
    assert all(v['status']=='PASS' for v in audits.values())
    assert json.loads((results/'exit_status.json').read_text())['status']=='SUCCESS'
    lock=json.loads((results/'model_lock.json').read_text());assert len(lock['models'])==18
    for path,digest in lock['models'].items():assert sha(path)==digest,path
    preflight=json.loads((folder/'preflight.json').read_text())
    for path,digest in preflight['source_hashes'].items():assert sha(path)==digest,path
    manifest=json.loads((folder/'old_results_manifest.json').read_text())
    for path,digest in manifest['files'].items():assert sha(path)==digest,path
    histories=[]
    for arm in ['zero','mean_std']:
        for seed in [1,2,3]:
            base=results/'models'/arm/('seed_'+str(seed))/'balanced_protected'
            resume=torch.load(base/'resume.pth',map_location='cpu',weights_only=False)
            final=torch.load(base/'constrained.pth',map_location='cpu',weights_only=False)
            assert resume['epoch']==80
            history=resume['history']
            histories.append(dict(arm=arm,seed=seed,selected_epoch=final.get('epoch'),calibration=final.get('calibration'),history=history))
    summary=json.loads((results/'summary.json').read_text())
    motion=[]
    for row in summary['aggregate']:
        if row['policy']=='constrained':
            for metric in ['speed_audit_p95','acceleration_audit_mean','acceleration_audit_p95','jerk_audit_mean','jerk_audit_p95']:
                value=row[metric];native=summary['native'][row['group']][metric]
                motion.append(dict(arm=row['arm'],group=row['group'],metric=metric,relative_harm_percent=100*(value/native-1)))
    items.append(dict(name=name,summary=summary,audits=audits,source_hashes=preflight['source_hashes'],old_file_count=len(manifest['files']),histories=histories,motion=motion,worker=json.loads((folder/'independent_worker_state.json').read_text())))
report=dict(status='PASS_COMPLETE_CYCLE_REVIEW',cycle=7,experiments=3,fresh_trainings=18,epochs_per_training=80,policy_entries=54,not_independent_trials=54,major=False,items=items,limitations=['Repeated controls/seeds/split are not independent generalization','Only three seeds and one repeatedly inspected ETHdev','Function-class, optimization and effective-capacity changes are coupled','No proof loss/encoder/intent unpredictability is the root cause'])
(ROOT/'CYCLE_7_REVIEW.json').write_text(json.dumps(report,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')
print('PASS 3experiments18fresh80epoch54policyentries; allmodel/source/priorhash verified; fullsummaries/history/CIs saved')
for item in items:
    print(item['name'])
    for row in item['histories']:
        hist=row['history']
        print(row['arm'],row['seed'],'selected',row['selected_epoch'],'firstlastloss',hist[0].get('train_loss'),hist[-1].get('train_loss'))
    print('max_motion_harm',max((r['relative_harm_percent'],r['arm'],r['group'],r['metric']) for r in item['motion']))
