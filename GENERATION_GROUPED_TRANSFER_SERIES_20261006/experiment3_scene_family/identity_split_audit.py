"""Registered50-source-index time split. Metadata only, no model or optimizer."""
from pathlib import Path
from datetime import datetime
import ctypes,hashlib,json,os,sys,time,traceback
sys.dont_write_bytecode=True
import numpy as np
import psutil
HERE=Path(__file__).resolve().parent;SERIES=HERE.parent;ROOT=SERIES.parent
TAG='20261006-grouped-scene-v1'
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
resource=dict(peak_RSS_GiB=0.,peak_private_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
def guard():
    m=Memory();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    p=psutil.Process().memory_info()
    resource['peak_RSS_GiB']=max(resource['peak_RSS_GiB'],p.peak_wset/2**30);resource['peak_private_GiB']=max(resource['peak_private_GiB'],p.peak_pagefile/2**30)
    resource['min_physical_GiB']=min(resource['min_physical_GiB'],m.avail_phys/2**30);resource['min_commit_GiB']=min(resource['min_commit_GiB'],m.avail_page/2**30);resource['checks']+=1
    assert max(p.rss,p.private)<=6*2**30 and min(m.avail_phys,m.avail_page)>=6*2**30,'STOP_RESOURCE'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.pending')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8');os.replace(tmp,p)
def verify(d,stage):
    for i,(p,h) in enumerate(d.items()):
        if i%512==0:guard()
        assert sha(p)==h,('STOP_SOURCE',p)
    print(stage,len(d),flush=True)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rank(purpose,value):return hashlib.sha256((TAG+'|'+purpose+'|'+str(value)).encode()).hexdigest()
def counts(ix,scene,seq,g):
    return dict(n=len(ix),scene={str(x):int((scene[ix]==x).sum()) for x in np.unique(scene)},sequence={str(x):int((seq[ix]==x).sum()) for x in np.unique(seq)},loss_group={str(x):int((g[ix]==x).sum()) for x in (0,1,2)})
def exposure(test,train,seq,t,ped):
    out=dict(n=len(test),same_sequence_20index_overlap=0,same_target_20index_overlap=0,same_target_exposure=0)
    byseq={str(x):np.sort(t[train][seq[train]==x]) for x in np.unique(seq)}
    keys=np.asarray([str(s)+'|'+str(p) for s,p in zip(seq,ped)]);pool=set(keys[train])
    bytarget={str(k):np.sort(t[train][keys[train]==k]) for k in np.unique(keys[train])}
    for i in test:
        out['same_target_exposure']+=int(keys[i] in pool)
        for table,k,name in ((byseq,str(seq[i]),'same_sequence_20index_overlap'),(bytarget,str(keys[i]),'same_target_20index_overlap')):
            a=table.get(k,np.array([],np.int64));out[name]+=int(np.searchsorted(a,t[i]+19,side='right')>np.searchsorted(a,t[i]-19,side='left'))
    return out
def main():
    guard();begin=time.perf_counter();out=HERE/'identity_audit';assert not (out/'RESULT.json').exists(),'Preserve completed split audit'
    files=dict(read(SERIES/'experiment2_recording_time/old_results_manifest.json')['files']);verify(files,'INHERITED_OLD_SHA')
    initial=len(files)
    prefixes=('GENERATION_','F16_','HEAD_','PULSE_','RESEARCH_','TWO_STAGE_')
    for folder in ROOT.iterdir():
        if not folder.is_dir() or not folder.name.startswith(prefixes):continue
        for p in folder.rglob('*'):
            if HERE==p or HERE in p.parents or not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.lock','.tmp','.pending') or p.name=='runtime_state.json':continue
            if str(p) not in files:files[str(p)]=sha(p)
    verify(files,'INITIAL_EXPANDED_OLD_SHA')
    write(HERE/'old_results_manifest.json',dict(files=files,inherited=len(files),all_prior_digests_preserved=True,added=len(files)-initial,controls_excluded=True))
    sources={str(p):sha(p) for p in (Path(__file__),HERE/'README.md',HERE/'DESIGN.md',HERE/'protocol.json')}
    write(out/'source_lock.json',dict(files=sources));z=load(SERIES/'identity_audit/splits.npz')
    ids=z['sample_id'];seq=z['sequence'];ped=z['pedestrian_id'];scene=z['scene_family'];t=z['start_frame'];g=z['loss_change_group'];unknown=(t<0)|z['ambiguous_track_time'];known=~unknown
    assert len(ids)==len(np.unique(ids))==25912 and not (ped<0).any() and (seq!='ambiguous').all()
    meta=load(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz');order=np.argsort(meta['sample_id']);ix=order[np.searchsorted(meta['sample_id'][order],ids)];assert np.array_equal(meta['sample_id'][ix],ids)
    for k,v in [('sequence',seq),('pedestrian_id',ped),('scene_family',scene),('start_frame',t),('ambiguous_track_time',z['ambiguous_track_time'])]:assert np.array_equal(meta[k][ix],v)
    assert sha(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata.npz')==read(ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/metadata_audit.json')['sha256']
    keys=np.asarray([str(s)+'|'+str(p) for s,p in zip(seq,ped)])
    test=[];inner=[]
    for sequence in sorted(np.unique(seq)):
        rows=np.flatnonzero(seq==sequence)
        if np.all(scene[rows]=='zara'):
            ordered=sorted(rows,key=lambda i:(rank('outer-anchor',int(ids[i])),int(i)))
            test.extend(ordered[:max(1,round(.2*len(rows)))])
        else:
            assert not (scene[rows]=='zara').any()
            tracks=sorted(set(keys[rows]),key=lambda k:(rank('inner-track',k),k))
            selected=tracks[:max(1,round(.1*len(tracks)))]
            inner.extend(rows[np.isin(keys[rows],selected)].tolist())
    test=np.sort(test);inner=np.sort(inner)
    grouped=np.flatnonzero((scene!='zara')&~np.isin(np.arange(len(ids)),inner))
    eligible=np.flatnonzero(~np.isin(np.arange(len(ids)),np.r_[test,inner]))
    random=[]
    for group in (0,1,2):
        n=int((g[grouped]==group).sum());rows=eligible[g[eligible]==group];assert len(rows)>=n
        random.extend(sorted(rows,key=lambda i:(rank('random-training',int(ids[i])),int(i)))[:n])
    random=np.sort(random);common=np.intersect1d(random,grouped)
    assert len(grouped)==len(random)>=10000 and len(common)>=5000 and len(test)>=1000 and (g[test]==2).sum()>=200
    assert counts(random,scene,seq,g)['loss_group']==counts(grouped,scene,seq,g)['loss_group']
    assert np.all(scene[test]=='zara') and not (scene[grouped]=='zara').any() and (scene[random]=='zara').any()
    for sequence in np.unique(seq):
        if np.all(scene[seq==sequence]=='zara'):assert (seq[test]==sequence).any()
        else:assert (seq[inner]==sequence).any()
    assert all((g[inner]==j).any() for j in (0,1,2))
    for train in (random,grouped):
        assert not np.isin(test,train).any() and not np.isin(inner,train).any()
        assert not np.isin(keys[inner],keys[train]).any()
    # Unknown times stay registered and eligible; temporal counts are known-only.
    er=exposure(test[known[test]],random[known[random]],seq,t,ped)
    eg=exposure(test[known[test]],grouped[known[grouped]],seq,t,ped)
    er['same_family_training_windows']=int((scene[random]=='zara').sum())
    eg['same_family_training_windows']=int((scene[grouped]=='zara').sum())
    assert eg['same_family_training_windows']==eg['same_target_exposure']==eg['same_sequence_20index_overlap']==0
    assert er['same_family_training_windows']>0
    # Verify immutable cache row mapping without model forward or using quality.
    cache=load(ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz')
    assert np.array_equal(cache['sample_id'][:25912],ids)
    del cache
    flag=lambda ix:np.isin(np.arange(len(ids)),ix)
    save=dict(sample_id=ids,sequence=seq,pedestrian_id=ped,scene_family=scene,start_frame=t,ambiguous_track_time=z['ambiguous_track_time'],loss_change_group=g,time_block=t//50,known_time_eligible=known,outer_eval_index=test,inner_cal_index=inner,random_train_index=random,grouped_train_index=grouped,common_statistics_index=common,outer_eval_ids=ids[test],inner_cal_ids=ids[inner],random_train_ids=ids[random],grouped_train_ids=ids[grouped],common_statistics_ids=ids[common],registry_random_used=flag(random),registry_grouped_used=flag(grouped),registry_outer=flag(test),registry_inner=flag(inner),registry_unused_both=~(flag(random)|flag(grouped)|flag(test)|flag(inner)))
    out.mkdir(exist_ok=True);np.savez_compressed(out/'splits.npz',**save)
    saved=load(out/'splits.npz');assert all(np.array_equal(v,saved[k]) for k,v in save.items())
    scale=read(HERE/'METADATA_SCALE_CHECK.json')
    assert (len(random),len(common),len(test),int((g[test]==2).sum()),len(inner),int(unknown.sum()))==(scale['train_each'],scale['common_train'],scale['outer'],scale['outer_change'],scale['inner'],scale['unknown_time_registry'])
    verify(sources,'FINAL_CURRENT_SOURCE_SHA');verify(files,'FINAL_EXPANDED_OLD_SHA');guard()
    report=dict(status='PASS_IDENTITY_MATCHED_SCENE_FAMILY_SPLIT',formal=False,no_model_forward=True,no_optimizer=True,tag=TAG,target_family='zara',unique_ids=len(ids),all_windows_in_registry=True,unknown_time_retained=int(unknown.sum()),unknown_not_unpredictable=True,unused_both=int(save['registry_unused_both'].sum()),outer_target_tracks=len(set(zip(seq[test],ped[test]))),outer_recordings=len(np.unique(seq[test])),partitions={name:counts(v,scene,seq,g) for name,v in [('random_training',random),('grouped_training',grouped),('outer_evaluation',test),('inner_calibration',inner),('common_statistics',common)]},random_exposure=er,grouped_exposure=eg,sharedinner_whole_target_disjoint=True,composition='EXACT_GLOBAL_LOSS_GROUP_NOT_SEQUENCE_OR_SCENE',all_initial_final_old_SHA=True,inherited_digests_count=initial,old_file_count=len(files),sources=sources,saved_split_SHA=sha(out/'splits.npz'),metadata_scale_matches_exact=True,resource=resource,elapsed_seconds=time.perf_counter()-begin,limitations=['frozenF16seen;newheadonly','oneheldzaraFamilyNOToriginalCALuniversity','family_target_time_context_distributions_coupled','unknown_times_not_zero_overlap','3recordings_not_independent_sceneCI','finiteoptimization_confound'])
    write(out/'RESULT.json',report);write(out/'exit_status.json',dict(status='SUCCESS',at=datetime.now().astimezone().isoformat()));print(json.dumps(report,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        d=HERE/'failures/identity_audit1';d.mkdir(parents=True,exist_ok=True);write(d/'FAILURE.json',dict(status='STOP_IDENTITY_SOURCE_RESOURCE',traceback=traceback.format_exc(),source_SHA=sha(__file__),resource=resource));raise
