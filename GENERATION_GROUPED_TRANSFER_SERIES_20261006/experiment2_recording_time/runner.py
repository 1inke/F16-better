"""Sole scientific production runner; completed seed reuse only after SHA verification."""
import adapter as h
import torch
def main():
    pre=h.read(h.HERE/'preflight.json');assert pre['status']=='PASS' and pre['resume_status']=='PASS';h.verify(pre['source_hashes'])
    assert h.read(h.HERE/'resume_check.json')['status']=='PASS'
    h.stage('INITIAL_SOURCE_OLD_SHA');h.verify(h.read(h.HERE/'old_results_manifest.json')['files'])
    cache,values,splits=h.load_data();h.initialize_classes();device=torch.device('cuda:0')
    for arm in ('random','grouped'):
        h.m.MODELS=h.HERE/'results/models'/arm
        for seed in (1,2,3):
            directory=h.m.MODELS/f'seed_{seed}/balanced_protected';complete=directory/'complete.json'
            if complete.exists():
                manifest=h.read(complete);assert manifest['status']=='COMPLETE' and manifest['epochs']==80;h.verify(manifest['model_hashes']);h.stage('SKIP_VERIFIED_COMPLETED_SEED',arm=arm,seed=seed);continue
            h.stage('TRAIN',arm=arm,seed=seed);h.train(seed,arm,cache,values,splits[arm+'_train_index'],splits['inner_cal_index'],device)
    from evaluate import run
    h.stage('EVALUATE_ALL_SAVED_AUDITS');run(cache,values,splits,device)
    h.stage('FINAL_SOURCE_OLD_SHA');h.verify(pre['source_hashes']);old=h.read(h.HERE/'old_results_manifest.json')['files'];h.verify(old)
    h.write(h.HERE/'results/PRESERVATION_AUDIT.json',dict(status='PASS',sources=len(pre['source_hashes']),old_files=len(old),unchanged=True));h.guard.save('PRODUCTION_PASS')
if __name__=='__main__':main()
