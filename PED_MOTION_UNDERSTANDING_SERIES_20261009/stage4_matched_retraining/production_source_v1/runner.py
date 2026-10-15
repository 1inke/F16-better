import argparse,json,os,sys
from pathlib import Path
import torch
from controls import H,P,W,ProductionRecovery,load,digest
from epoch_recovery import InterruptedForFixture

def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--input-branch',choices=('full','last2','noNbr'),required=True);p.add_argument('--source-binding',type=Path,required=True)
    p.add_argument('--resume-epoch',action='store_true');p.add_argument('--fixture-stop-after',type=int);p.add_argument('--fixture-subset-rows',type=int);p.add_argument('--production-gate',type=Path)
    e,original=p.parse_known_args();root=Path(original[original.index('--output-root')+1]).resolve()
    assert root.is_relative_to(H) and '--force' not in original
    fixture=e.fixture_subset_rows is not None
    if fixture:assert root.is_relative_to(H/'preflight_runs') and 21<=e.fixture_subset_rows<=513
    else:
        assert e.production_gate is not None
        gate=json.loads(e.production_gate.read_text());assert gate['status']=='PASS_ALL_ACTUAL_PREFLIGHT_AND_FINAL_SOURCE' and gate['binding_sha']==digest(e.source_binding)
        assert root.is_relative_to(H/'runs') and '--test-data-path' in original
        fold=os.environ['E3_FOLD'];stage=os.environ['E3_STAGE'];seed=int(original[original.index('--seed')+1])
        assert fold in gate['folds'] and seed in (1,2,3) and e.input_branch in ('full','last2','noNbr')
        assert int(original[original.index('--epochs')+1])==dict(h3=30,v2=20)[stage]
        if stage=='v2':
            source=Path(os.environ['EXP143_SOURCE_CHECKPOINT']).resolve();assert source==H/'runs'/fold/f'seed_{seed}'/e.input_branch/'h3/f16_temporal_social_cross'/f'seed_{seed}'/'val_scene_-1/best_selected.pth'
            done=json.loads((source.parent/'done.json').read_text());assert done['status']=='SUCCESS' and done['outputs'][str(source)]==digest(source)
    recovery=ProductionRecovery(H,e.input_branch,e.source_binding,e.resume_epoch,e.fixture_stop_after,fixture)
    module,_=load(e.input_branch,recovery,e.fixture_subset_rows);sys.argv=[str(P/'isolated_trainer.py'),*original]
    try:module.run()
    except InterruptedForFixture as x:print(json.dumps(dict(intentional_fixture_interruption=str(x))))
if __name__=='__main__':main()
