"""Official raw indices and target identities; no quality-dependent split changes."""
import importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
from controls import H,P,S,W,atomic_json,digest,guard,verify_binding

def main():
    guard();parent=json.loads((P/'source_input_binding.json').read_text());verify_binding(P/'source_input_binding.json');locks=parent['files'].copy()
    spec=importlib.util.spec_from_file_location('e3_identity_only',S/'stage3_knn_memory/knn.py');kn=importlib.util.module_from_spec(spec);spec.loader.exec_module(kn)
    readiness=json.loads((P/'SOURCE_READINESS.json').read_text());rows=[];(H/'identity').mkdir(exist_ok=True)
    for fold,data in readiness['data'].items():
        guard();raw=np.load(data['train_data_path']['path'],mmap_mode='r');full=np.asarray(raw[:,0,:,-2:],dtype=np.float32)
        scenes=np.asarray(raw[:,0,0,0],dtype=int);vals=[]
        for scene in np.unique(scenes):
            candidates=np.flatnonzero(scenes==scene);n=max(1,min(len(candidates)-1,int(round(len(candidates)*.05))));vals.extend(candidates[-n:])
        val=np.sort(vals);mask=np.zeros(len(raw),bool);mask[val]=True;train=np.flatnonzero(~mask)
        ident=kn.train_identity(fold,full);bank=kn.a.load(S/f'stage3_knn_memory/data/{fold}_bank.npz')
        assert np.array_equal(bank['ids'],train) and np.array_equal(bank['track'],ident['track'][train])
        test=kn.a.load(S/f'data/{fold}_evaluation.npz');test_raw=np.load(data['test_data_path']['path'],mmap_mode='r');test_num=np.load(data['test_num_path']['path'],mmap_mode='r')
        assert np.array_equal(np.asarray(test_raw[:,:,:8,-2:],dtype=np.float32),test['obs'])
        assert np.array_equal(np.asarray(test_raw[:,0,8:,-2:],dtype=np.float32)-test['obs'][:,0,-1,None],test['gt'])
        assert np.array_equal(np.minimum(test_num,9),test['num_valid'] if 'num_valid' in test else test['valid'])
        assert not set(ident['track']).intersection(test['track']),('STOP_TRAIN_TEST_TARGET_ID',fold)
        shared=set(ident['track'][train]).intersection(ident['track'][val]);overlap=0
        for track in shared:
            ts=ident['start'][train[ident['track'][train]==track]];vs=ident['start'][val[ident['track'][val]==track]];ts=ts[ts>=0];vs=vs[vs>=0]
            overlap+=sum(bool(np.any(abs(ts-v)<20)) for v in vs)
        path=H/f'identity/{fold}.npz';np.savez_compressed(path,all_ids=np.arange(len(raw)),train=train,validation=val,test_ids=test['ids'],test_track=test['track'],**ident)
        locks[str(path)]=digest(path)
        rows.append(dict(fold=fold,train=len(train),validation=len(val),test=len(test['ids']),raw_meter_dt04_cache_EXACT=True,train_test_target_intersection=0,train_validation_shared_targets=len(shared),validation_known20index_target_overlap=overlap,ambiguous_train_time_n=ident['ambiguous_time_n'],not_context_disjoint=True))
    for path in H.glob('*'):
        if path.is_file() and path.suffix in ('.py','.ps1','.md','.json') and path.name in ('DESIGN.md','protocol.json') or path.is_file() and path.suffix in ('.py','.ps1'):
            locks[str(path)]=digest(path)
    # Export/audit consumes fixed E1 strata and world-relative identities.
    for fold in readiness['data']:
        for path in (S/f'data/{fold}_evaluation.npz',S/f'jobs/{fold}_seed1/outputs.npz',S/f'calibration/{fold}.json'):locks.setdefault(str(path),digest(path))
    atomic_json(dict(status='PASS_OFFICIAL_RAW_ID_SPLIT_AUDIT',records=rows,unchanged_original_scene_tail_split=True),H/'IDENTITY_AUDIT.json')
    atomic_json(dict(files=locks,inherited_original_digests=len(parent['files']),historical_all_files_audit=False),H/'source_input_binding.json')
    verify_binding(H/'source_input_binding.json');print(json.dumps(dict(status='PASS_ID_SOURCE_LOCK',files=len(locks),records=rows)))
if __name__=='__main__':main()
