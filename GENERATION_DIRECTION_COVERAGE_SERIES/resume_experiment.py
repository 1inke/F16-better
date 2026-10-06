"""Verified arm-level resume; does not modify the locked scientific protocol."""
from pathlib import Path
import importlib.util
import sys
import argparse
import json
import csv
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ENTRY=HERE/'run_direction_coverage.py'
spec=importlib.util.spec_from_file_location('coverage_resume',ENTRY)
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
AUDIT=HERE/'resume_preflight.json'


def inspect():
    cfg=json.loads((HERE/'config.json').read_text())
    original=json.loads((HERE/'preflight.json').read_text())
    for path,digest in original['hashes'].items():
        if r.p.sha256(Path(path))!=digest:raise RuntimeError('Original source/data lock changed: '+path)
    if (r.RESULTS/'summary.json').exists():raise RuntimeError('Experiment already complete; do not overwrite evaluation')
    cache=r.p.load_npz(r.p.TRAIN_CACHE)
    fit,cal=r.p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    values=r.prepared(cache,False,fit=fit)
    complete=[];pending=[];hashes={}
    for arm in r.ARMS:
        files=[r.MODELS/f'{arm}_{tag}.pth' for tag in ('best','final')]+[r.MODELS/f'{arm}_history.csv']
        present=[f.exists() for f in files]
        if any(present) and not all(present):raise RuntimeError('Partial saved arm; inspect manually: '+arm)
        if not any(present):pending.append(arm);continue
        history=list(csv.DictReader(files[2].open(encoding='utf-8-sig')))
        assert len(history)==cfg['epochs'] and [int(x['epoch']) for x in history]==list(range(1,cfg['epochs']+1))
        for path in files[:2]:
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['config']==cfg and saved['arm']==arm and saved['dev_used_for_selection'] is False
            assert saved['input_dim']==values[0].shape[1]
            assert np.array_equal(saved['mean'],values[4]) and np.array_equal(saved['std'],values[5])
            epoch=int(saved['best_epoch']);assert 1<=epoch<=cfg['epochs']
            assert abs(float(history[epoch-1]['cal_key'])-saved['best_cal_key'])<1e-8
            assert abs(min(float(x['cal_key']) for x in history)-saved['best_cal_key'])<1e-8
            model=r.m.Correction(saved['input_dim'],cfg);model.load_state_dict(saved['state_dict'])
            assert all(torch.isfinite(v).all() for v in saved['state_dict'].values())
        complete.append(arm)
        hashes.update({str(f):r.p.sha256(f) for f in files})
    if not pending:raise RuntimeError('All arms exist; this resume is scoped to unfinished training')
    report={'status':'PASS','complete_arms':complete,'pending_arms':pending,'preserved_hashes':hashes,
            'original_source_hashes':original['hashes'],'resume_source_hash':r.p.sha256(Path(__file__)),
            'fit_n':len(fit),'calibration_n':len(cal),'test_access':False,
            'unfinished_arm_policy':'Restart from same seed/initialization; no epoch checkpoint exists'}
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    current=inspect()
    if args.action=='preflight':
        AUDIT.write_text(json.dumps(current,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in current.items() if 'hash' not in k},indent=2),flush=True);return
    previous=json.loads(AUDIT.read_text())
    if current!=previous:raise RuntimeError('Resume lock/state changed after preflight')
    original_train=r.m.train
    def verified_train(arm,*params,**kwargs):
        if arm in previous['complete_arms']:
            for path,digest in previous['preserved_hashes'].items():
                if r.p.sha256(Path(path))!=digest:raise RuntimeError('Completed artifact changed: '+path)
            print('[SKIP VERIFIED COMPLETE] '+arm,flush=True);return
        return original_train(arm,*params,**kwargs)
    r.m.train=verified_train
    sys.argv=[str(ENTRY),'--action','run','--device',args.device]
    r.main()
    for path,digest in previous['preserved_hashes'].items():
        if r.p.sha256(Path(path))!=digest:raise RuntimeError('Completed artifact unexpectedly changed')
    print('RESUME COMPLETE; original completed arms preserved exactly',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        r.RESULTS.mkdir(exist_ok=True)
        (r.RESULTS/'resume_failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
