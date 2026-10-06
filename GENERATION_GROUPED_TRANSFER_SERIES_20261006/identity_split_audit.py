"""Actual FIT-only identity/matched split audit. No model forward or optimizer."""
from pathlib import Path
from datetime import datetime
import ctypes, hashlib, json, os, sys, time, traceback
sys.dont_write_bytecode = True
import numpy as np
import psutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
TAG='20261006-grouped-transfer-v1'
GIB=2**30
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
peak_rss=0;peak_private=0;min_phys=2**64;min_commit=2**64
def guard():
    global peak_rss,peak_private,min_phys,min_commit
    m=Memory();m.length=ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    p=psutil.Process().memory_info()
    peak_rss=max(peak_rss,p.peak_wset);peak_private=max(peak_private,p.peak_pagefile)
    min_phys=min(min_phys,m.avail_phys);min_commit=min(min_commit,m.avail_page)
    if max(p.rss,p.private)>6*GIB or min(m.avail_phys,m.avail_page)<6*GIB:
        raise MemoryError('STOP_RESOURCE process6GiB and systemphysical+commit reserve6GiB')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)
def rank(value,purpose):return hashlib.sha256((TAG+'|'+purpose+'|'+str(value)).encode()).hexdigest()
def ordered(indices,values,purpose):return np.array(sorted(indices,key=lambda i:(rank(values[i],purpose),int(i))),np.int64)
def verify(files,stage):
    for i,(p,digest) in enumerate(files.items()):
        if i%1000==0:guard()
        assert sha(p)==digest,('STOP_SOURCE',p)
    print(stage,len(files),flush=True)
def summary(rows,scene,sequence,groups):
    return dict(n=len(rows),scene={str(s):int(np.sum(scene[rows]==s)) for s in np.unique(scene)},
                sequence={str(s):int(np.sum(sequence[rows]==s)) for s in np.unique(sequence)},
                loss_group={str(g):int(np.sum(groups[rows]==g)) for g in (0,1,2)})
def exposure(test,train,keys,sequence,starts):
    target=set(keys[train]);by_target={};by_seq={}
    for i in train:
        if starts[i]>=0:
            by_target.setdefault(keys[i],[]).append(int(starts[i]))
            by_seq.setdefault(sequence[i],[]).append(int(starts[i]))
    by_target={k:np.sort(v) for k,v in by_target.items()};by_seq={k:np.sort(v) for k,v in by_seq.items()}
    def intersects(table,i):
        if starts[i]<0:return None
        v=table.get(i[0] if isinstance(i,tuple) else '',[])
    same_track=sum(keys[i] in target for i in test)
    known=[i for i in test if starts[i]>=0];track_overlap=0;seq_overlap=0
    for i in known:
        for table,key,which in ((by_target,keys[i],0),(by_seq,sequence[i],1)):
            values=table.get(key,np.array([],np.int64))
            overlap=np.searchsorted(values,starts[i]+19,side='right')>np.searchsorted(values,starts[i]-19,side='left')
            if which==0:track_overlap+=int(overlap)
            else:seq_overlap+=int(overlap)
    return dict(evaluation_windows=len(test),same_target_track_exposed=int(same_track),
        known_start_windows=len(known),unknown_start_windows=len(test)-len(known),
        same_target_full20frame_overlap=int(track_overlap),same_sequence_full20frame_overlap=int(seq_overlap),
        sequence_overlap_is_context_time_proxy_not_neighbor_identity=True)
def main():
    started=time.perf_counter();guard()
    output=HERE/'identity_audit';assert not (output/'RESULT.json').exists(),'Do not overwrite completed evidence'
    output.mkdir(exist_ok=True)
    inherited=ROOT/'GENERATION_TRAINED_ACCEL_PROJECTION_SERIES_20261006/recovery_source_20261006/old_results_manifest.json'
    files=dict(read(inherited)['files']);verify(files,'INHERITED_SHA')
    prefixes=('GENERATION_','F16_','HEAD_','PULSE_','RESEARCH_','TWO_STAGE_')
    for folder in ROOT.iterdir():
        if folder==HERE or not folder.is_dir() or not folder.name.startswith(prefixes):continue
        for p in folder.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.lock','.tmp','.pending') or p.name=='runtime_state.json':continue
            if str(p) not in files:files[str(p)]=sha(p)
    inputs=[ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz',ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata_audit.json',
            ROOT/'GENERATION_CANDIDATE_ROUTED_MAPPING_SERIES_20261005/state_scalers.npz',
            ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz',
            ROOT/'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES/directional_regime_pilot.py']
    for p in inputs:files.setdefault(str(p),sha(p))
    verify(files,'INITIAL_ALL_OLD_SHA')
    write(HERE/'old_results_manifest.json',dict(files=files,all_prior_digests_preserved=True,controls_excluded=True))
    sources={str(p):sha(p) for p in [Path(__file__),HERE/'README.md',HERE/'DESIGN.md',HERE/'protocol.json']}
    write(output/'source_lock.json',dict(files=sources))
    with np.load(inputs[2],allow_pickle=False) as z:ids=z['fit_id'].copy()
    with np.load(inputs[3],allow_pickle=False) as z:
        cache_ids=z['sample_id'];assert len(cache_ids)==28791 and np.array_equal(cache_ids[:25912],ids)
        original_cal_ids=cache_ids[25912:].copy();groups=z['change_group'][:25912].copy()
    assert len(ids)==25912 and len(np.unique(ids))==25912 and set(np.unique(groups))=={0,1,2}
    assert sha(inputs[0])==read(inputs[1])['sha256']
    with np.load(inputs[0],allow_pickle=False) as z:
        order=np.argsort(z['sample_id']);lookup=z['sample_id'][order]
        ix=order[np.searchsorted(lookup,ids)];assert np.array_equal(z['sample_id'][ix],ids)
        sequence=z['sequence'][ix].copy();ped=z['pedestrian_id'][ix].copy();scene=z['scene_family'][ix].copy()
        starts=z['start_frame'][ix].copy();ambiguous=z['ambiguous_track_time'][ix].copy()
        ci=order[np.searchsorted(lookup,original_cal_ids)];assert np.array_equal(z['sample_id'][ci],original_cal_ids)
        calkeys=set(f'{s}|{int(p)}' for s,p in zip(z['sequence'][ci],z['pedestrian_id'][ci]))
        original_cal_scenes={str(s):int(np.sum(z['scene_family'][ci]==s)) for s in np.unique(z['scene_family'][ci])}
    assert not np.any(ped<0) and not np.any(sequence=='ambiguous') and not np.any(scene=='unverified'),'STOP_IDENTITY'
    keys=np.array([f'{s}|{int(p)}' for s,p in zip(sequence,ped)])
    tracks=np.unique(keys);outer_keys=[];inner_keys=[]
    for seq in sorted(np.unique(sequence)):
        pool=sorted(set(keys[sequence==seq]),key=lambda x:(rank(x,'outer-track'),x))
        n=max(1,int(round(.2*len(pool))));assert len(pool)-n>=3
        outer_keys.extend(pool[:n]);rest=pool[n:]
        rest=sorted(rest,key=lambda x:(rank(x,'inner-track'),x));inner_keys.extend(rest[:max(1,int(round(.1*len(rest))))])
    outer_set=set(outer_keys);inner_set=set(inner_keys);assert not outer_set&inner_set
    anchors=[]
    for key in sorted(outer_set):
        rows=np.flatnonzero(keys==key);order=ordered(rows,ids,'outer-anchor')
        anchors.extend(order[:max(1,int(round(.2*len(rows))))].tolist())
    test=np.sort(anchors).astype(np.int64)
    inner=np.flatnonzero(np.isin(keys,list(inner_set)))
    grouped=np.flatnonzero(~np.isin(keys,list(outer_set|inner_set)))
    eligible=np.flatnonzero((~np.isin(keys,list(inner_set))) & (~np.isin(np.arange(len(ids)),test)))
    control=[]
    for seq in sorted(np.unique(sequence)):
        for g in (0,1,2):
            need=int(np.sum((sequence[grouped]==seq)&(groups[grouped]==g)))
            pool=eligible[(sequence[eligible]==seq)&(groups[eligible]==g)]
            assert len(pool)>=need,('STOP_MATCH',seq,g,need,len(pool))
            control.extend(ordered(pool,ids,'random-training')[:need].tolist())
    control=np.sort(control).astype(np.int64);common=np.intersect1d(grouped,control)
    assert len(control)==len(grouped) and len(common)>128
    assert not set(grouped)&set(test) and not set(control)&set(test)
    assert not set(grouped)&set(inner) and not set(control)&set(inner)
    assert not set(keys[grouped])&outer_set and not set(keys[grouped])&inner_set and not set(keys[control])&inner_set
    assert summary(grouped,scene,sequence,groups)==summary(control,scene,sequence,groups)
    for s in np.unique(scene):assert np.any(scene[test]==s) and np.any(scene[inner]==s)
    assert all(np.any(groups[x]==2) for x in (test,inner,grouped,control,common)),'STOP_EMPTY_CHANGE_GROUP'
    save=dict(sample_id=ids,sequence=sequence,pedestrian_id=ped,scene_family=scene,start_frame=starts,
              ambiguous_track_time=ambiguous,loss_change_group=groups,outer_eval_index=test,inner_cal_index=inner,
              grouped_train_index=grouped,random_train_index=control,common_statistics_index=common,
              outer_eval_ids=ids[test],inner_cal_ids=ids[inner],grouped_train_ids=ids[grouped],random_train_ids=ids[control],common_statistics_ids=ids[common])
    np.savez_compressed(output/'splits.npz',**save)
    # Independent saved ID/set reconstruction; no quality outputs are loaded.
    with np.load(output/'splits.npz',allow_pickle=False) as z:
        for k,v in save.items():assert np.array_equal(z[k],v),('STOP_SAVED_MAPPING',k)
    e_control=exposure(test,control,keys,sequence,starts);e_group=exposure(test,grouped,keys,sequence,starts)
    assert e_control['same_target_track_exposed']>0 and e_group['same_target_track_exposed']==0,'STOP_UNIDENTIFIABLE_TARGET_EXPOSURE'
    verify(sources,'FINAL_CURRENT_SOURCE_SHA');verify(files,'FINAL_ALL_OLD_SHA')
    report=dict(status='PASS_IDENTITY_MATCHED_TARGET_TRACK_SPLIT',formal=False,no_model_forward=True,no_optimizer=True,
        elapsed_seconds=time.perf_counter()-started,unique_ids=len(ids),tracks=len(tracks),ambiguous_time_windows=int(ambiguous.sum()),
        ambiguous_identity_windows=0,all_windows_accounted_for=True,unused_by_each_arm_reported_not_deleted=True,
        outer_target_tracks=len(outer_set),inner_calibration_tracks=len(inner_set),
        partitions={name:summary(rows,scene,sequence,groups) for name,rows in [('outer_evaluation',test),('inner_calibration',inner),('grouped_training',grouped),('random_training',control),('common_statistics',common)]},
        random_exposure=e_control,grouped_exposure=e_group,
        old_protocol=dict(kind='ordered_prefix_suffix',FIT_n=25912,CAL_n=2879,original_CAL_scenes=original_cal_scenes,
                          target_tracks_shared_between_FIT_and_CAL=len(set(keys)&calkeys),CAL_tracks=len(calkeys)),
        input_hashes={str(p):sha(p) for p in inputs},old_file_count=len(files),old_all_SHA_initial_final_PASS=True,
        current_sources_initial_final_PASS=True,saved_split_SHA=sha(output/'splits.npz'),source_files=sources,
        output_allocation_bytes=sum(v.nbytes for v in save.values()),
        limitations=['Frozen F16 already saw this original FIT pool; new-head adaptation transfer only.',
          'Target-track disjoint is not full neighboring-pedestrian/context independence; simultaneous sequence intervals reported.',
          '123-style ambiguous start-time records remain tracked; no fictitious exact interval identity.',
          'Only3scene families, highly imbalanced, same split reused3modelseeds, no independent generalization/SOTA or causal shared-interference claim.'],
        resource=dict(peak_RSS_GiB=peak_rss/GIB,peak_private_GiB=peak_private/GIB,min_physical_GiB=min_phys/GIB,min_commit_GiB=min_commit/GIB),
        process_budget_GiB=6,system_reserve_each_GiB=6,
        next='Implement precise first paired training/preflight; no training worker or ETA yet')
    write(output/'RESULT.json',report)
    write(output/'exit_status.json',dict(status='SUCCESS',at=datetime.now().astimezone().isoformat()))
    print(json.dumps({k:report[k] for k in ('status','tracks','ambiguous_time_windows','partitions','random_exposure','grouped_exposure','old_protocol','resource','elapsed_seconds')},ensure_ascii=False),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        directory=HERE/'failures'/'identity_split_audit_1';directory.mkdir(parents=True,exist_ok=True)
        (directory/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write(directory/'SOURCE.json',dict(source_SHA=sha(__file__),protocol_SHA=sha(HERE/'protocol.json'),failure_preserved=True))
        write(HERE/'identity_audit'/'exit_status.json',dict(status='FAILURE_STOP_WAIT_HUMAN',at=datetime.now().astimezone().isoformat()))
        raise
