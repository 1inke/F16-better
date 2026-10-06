"""Read-only audit and mandatory three-experiment review; no optimizer."""
from pathlib import Path
import importlib.util,sys,json,csv,time,traceback
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
FOLDERS=['GENERATION_PAIRWISE_HISTORY_SERIES','GENERATION_NEIGHBOR_READOUT_SERIES','GENERATION_NEIGHBOR_MESSAGE_SERIES']
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
    assert not(HERE/'REVIEW.json').exists(),'Already complete; preserve review'
    started=time.time();records=[];hashes={};sources={};predictions={}
    for index,folder in enumerate(FOLDERS):
        base=ROOT/folder;spec=importlib.util.spec_from_file_location('cycle10_source_'+str(index),base/'run_head.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
        def digest(path):
            path=Path(path);key=str(path)
            if key not in hashes:hashes[key]=h.p.sha256(path)
            return hashes[key]
        pre=read(base/'preflight.json')
        for path,sha in pre['source_hashes'].items():assert digest(path)==sha,path
        for path,sha in read(base/'old_results_manifest.json')['files'].items():assert digest(path)==sha,path
        for path,sha in read(base/'results/model_lock.json')['models'].items():assert digest(path)==sha,path
        assert read(base/'results/exit_status.json')['status']=='SUCCESS'
        audit=read(base/'results/independent_audit.json');assert audit['status']=='PASS' and audit['metrics_audit']['metrics_recomputed']
        summary=read(base/'results/summary.json');assert len(summary['aggregate'])==24
        cache=h.p.load_npz(h.p.DEV_CACHE);dim=cache['motion'].shape[1]+cache['social'].shape[1]
        values=h.r.prepared(cache,False,mean=np.zeros((1,dim),np.float32),std=np.ones((1,dim),np.float32));native,speed,future=values[1:4]
        _,_,original=h.n.prepare_backend(torch.device('cpu'));del original
        modelclass=h.PairwiseHead
        metricsets={};histories=[]
        for arm in h.c.ARMS:
            for seed in (1,2,3):
                modeldir=base/f'results/models/{arm}/seed_{seed}/balanced_protected'
                with (modeldir/'history.csv').open(encoding='utf-8-sig',newline='') as stream:curve=list(csv.DictReader(stream))
                assert len(curve)==80 and [int(row['epoch']) for row in curve]==list(range(1,81))
                histories.append(dict(arm=arm,seed=seed,rows=curve))
                for policy in ('constrained','legacy','final'):
                    path=modeldir/(policy+'.pth');saved=torch.load(path,map_location='cpu',weights_only=False)
                    model=modelclass(saved['input_dim'],saved['config']);model.load_state_dict(saved['state_dict']);h.frozen_audit(model)
                    path=base/f'results/seed{seed}_{arm}_{policy}.npz';data=h.p.load_npz(path)
                    assert np.array_equal(data['sample_id'],cache['sample_id']) and data['trajectory'].shape==(1516,20,12,2) and np.isfinite(data['trajectory']).all()
                    actual=h.r.generation_metrics(data['trajectory'],native,speed,future)
                    for key,value in actual.items():assert np.allclose(value,data[key],atol=1e-6,rtol=1e-6),(folder,seed,arm,policy,key)
                    metricsets[seed,arm,policy]=actual;predictions[folder,seed,arm,policy]=data['trajectory'];digest(path)
        for row in summary['aggregate']:
            mask=next(mask for name,mask in h.m.masks(cache['change_group']) if name==row['group'])
            for key in metricsets[1,row['arm'],row['policy']]:
                actual=float(np.mean([metricsets[seed,row['arm'],row['policy']][key][mask].mean() for seed in (1,2,3)]))
                assert np.isclose(actual,row[key],atol=1e-6,rtol=1e-6),(folder,row['group'],key)
        for path in base.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.name not in ('runtime_state.json','independent_worker.lock'):digest(path)
        sources[str(base/'results/summary.json')]=digest(base/'results/summary.json')
        records.append(dict(folder=folder,summary=summary,audit=audit,history=histories,preflight_parameters=pre['parameters_trainable']))
        print('PASS18models/18predictions/allmetrics/frozen/history/source '+folder,flush=True)
    repeats=[]
    for policy in ('constrained','legacy','final'):
        repeats.append(dict(policy=policy,uniform_readout_vs_first_actualhistory_exact=all(np.array_equal(predictions[FOLDERS[1],seed,'zero',policy],predictions[FOLDERS[0],seed,'mean_std',policy]) for seed in (1,2,3)),message_control_vs_first_actualhistory_exact=all(np.array_equal(predictions[FOLDERS[2],seed,'zero',policy],predictions[FOLDERS[0],seed,'mean_std',policy]) for seed in (1,2,3))))
    for path,sha in hashes.items():assert h.p.sha256(Path(path))==sha,path
    result=dict(status='PASS_MANDATORY_CYCLE_10_REVIEW',experiments=records,formal_experiments=3,fresh_trainings=18,epochs_per_training=80,policy_entries=54,independent_trainings_not_54=True,repeated_controls=repeats,source_hashes=sources,all_prior_and_current_files_unchanged=len(hashes),limitations='Repeated ETH DEV exploratory/only3seeds; controls repeat effective architectures/seeds not independent generalization. Eighteen fresh trainings and54policyentries not54independenttrainings. Same nominalcount not equal function/effectivecapacity. All major gates unchanged; no newbaseline/intent/bottleneckcauseclaim.',next_direction='Stop raw-neighbor pooling/attention/message-placement/coeff/epoch local variants. Thirty formal experiments no major. First audit historical output-parameterization experiments and FIT/CAL learning limits before preregistering a genuinely different generator-head direction; no fourth local variation. Same permitted newhead/lastF16block/contextmode scope not expanded, oldlossguards/data/nativeK20 preserved unless new specifichuman permission. New artifacts only newfolder.')
    (HERE/'REVIEW.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (HERE/'all_files_manifest.json').write_text(json.dumps(dict(files=hashes)),encoding='utf-8')
    print('REVIEW PASS '+str(len(hashes))+' files '+str(time.time()-started)+'s',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (HERE/('failure_'+str(time.time_ns())+'.txt')).write_text(traceback.format_exc(),encoding='utf-8');raise
