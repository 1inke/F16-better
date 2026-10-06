"""Frozen saved-data diagnostic; all writes confined to this new folder."""
from pathlib import Path
from datetime import datetime
import argparse, hashlib, importlib.util, json, os, sys, time
sys.dont_write_bytecode = True
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003'
PRIOR = ROOT/'GENERATION_PULSE_RESPONSE_SERIES_20261004'

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

rules = module('locked_response_gap_rules', ROOT/'F16_RESPONSE_GAP_DIAGNOSIS_20261004/rules.py')
oldcue = module('locked_low_cue_rules', ROOT/'F16_LOW_CUE_FAILURE_STRATIFICATION_20261004/analyze.py')

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20), b''): h.update(block)
    return h.hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for i in range(6):
        try: os.replace(tmp, path); return
        except PermissionError as e:
            readonly = path.exists() and bool(getattr(path.stat(), 'st_file_attributes', 0)&1)
            if readonly or i==5 or getattr(e, 'winerror', None) not in (5,32): raise
            time.sleep(.02*2**i)
def npz(path, keys=None):
    with np.load(path, allow_pickle=False) as z: return {k:z[k] for k in (keys or z.files)}
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem+'.pending.npz')
    np.savez_compressed(tmp, **value); os.replace(tmp, path)
def stage(name, **details):
    record = dict(stage=name, at=datetime.now().astimezone().isoformat(), **details)
    write(HERE/'progress.json', record); print(json.dumps(record, ensure_ascii=False), flush=True)
def verify(files, name):
    began = time.perf_counter()
    for i, (path, expected) in enumerate(files.items()):
        assert Path(path).is_file() and digest(path)==expected, ('SHA changed', path)
        if i%4000==0: stage(name, files_done=i, files_total=len(files))
    return time.perf_counter()-began
def snapshot():
    path = HERE/'old_results_manifest.json'
    if path.exists(): return read(path)['files']
    files = dict(read(PRIOR/'old_results_manifest.json')['files'])
    folders = [PRIOR, ROOT/'F16_LOW_CUE_FAILURE_STRATIFICATION_20261004',
       ROOT/'PULSE_RESPONSE_CHANGE_MAGNITUDE_20261004', ROOT/'PULSE_RESPONSE_VISUAL_REVIEW_20261004']
    for folder in folders:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.pending','.tmp') and p.name!='runtime_state.json':
                files[str(p)] = digest(p)
    write(path, dict(files=files, inherited_count=19558, added_folders=[str(x) for x in folders],
      excludes=['.memory','AUTONOMOUS_GENERATION_CYCLE.json','runtime_state.json','locks/temp/bytecode'],
      unrelated_external_data_not_accessed=True))
    return files
def sources():
    paths = list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/'DESIGN.md',HERE/'README.md',HERE/'protocol.json',HERE/'old_results_manifest.json']
    paths += [ROOT/'F16_RESPONSE_GAP_DIAGNOSIS_20261004/rules.py', BASE/'common.py',
      ROOT/'F16_LOW_CUE_FAILURE_STRATIFICATION_20261004/analyze.py',
      ROOT/'F16_LOW_CUE_FAILURE_STRATIFICATION_20261004/CLASSIFICATION.json',
      ROOT/'PULSE_RESPONSE_VISUAL_REVIEW_20261004/data/exact_arrays.npz',BASE/'metadata.npz']
    if (HERE/'resume_check.json').exists(): paths.append(HERE/'resume_check.json')
    for name in ('train','dev'):
        paths += [BASE/f'features/{name}.npz',BASE/f'features/{name}_audit.json', BASE/f'labels/{name}.npz',
           ROOT/f'GENERATION_NATIVE_DECODER_SERIES/{name}_encoder.npz']
    # Lock original model/cache/scientific provenance without loading models.
    locked = dict(read(PRIOR/'preflight.json')['source_hashes'])
    locked.update({str(p):digest(p) for p in paths})
    return locked
def align(ids, values):
    order = np.argsort(values['sample_id']); sorted_ids = values['sample_id'][order]
    where = np.searchsorted(sorted_ids, ids)
    assert len(np.unique(sorted_ids))==len(sorted_ids) and np.all(where<len(sorted_ids))
    assert np.array_equal(sorted_ids[where], ids)
    return {k:v[order[where]] for k,v in values.items()}

def cues(r2, truth, past, future):
    previous = oldcue.hand_cues(r2)
    focal = r2[:,:16].reshape(-1,8,2).astype(np.float64)
    v = (focal[:,2:]-focal[:,:-2])/.8
    ref, recent = v[:,:2].mean(1), v[:,-2:]
    cross = ref[:,None,0]*recent[:,:,1]-ref[:,None,1]*recent[:,:,0]
    angle = np.degrees(np.arctan2(cross,(ref[:,None]*recent).sum(-1)))
    dv = np.linalg.norm(recent,axis=-1)-np.linalg.norm(ref,axis=-1)[:,None]
    turnsign = np.where((np.sign(angle)==np.sign(angle[:,:1])).all(1),np.sign(angle[:,0]),0).astype(np.int8)
    speedsign = np.where((np.sign(dv)==np.sign(dv[:,:1])).all(1),np.sign(dv[:,0]),0).astype(np.int8)
    target = previous['target_turn']|previous['target_speed']|previous['target_start']|previous['target_stop']
    definite = (previous['target_turn']&(turnsign!=0))|(previous['target_speed']&(speedsign!=0))|previous['target_start']|previous['target_stop']
    first = ((previous['target_turn']&(turnsign==truth['response_direction'])&(truth['kind']==1)&(turnsign!=0))|
             (previous['target_speed']&(speedsign==truth['response_direction'])&(truth['kind']==2)&(speedsign!=0))|
             (previous['target_start']&(truth['kind']==3))|(previous['target_stop']&(truth['kind']==4)))&truth['y'].astype(bool)
    joined = np.concatenate((past[:,-2:],future),1)
    fv = (joined[:,2:]-joined[:,:-2])/.8; fs = np.linalg.norm(fv,axis=-1)
    fr = (past[:,-1]-past[:,-3])/.8; frs = np.linalg.norm(fr,axis=-1)
    fa = np.degrees(np.arctan2(fr[:,None,0]*fv[:,:,1]-fr[:,None,1]*fv[:,:,0],(fr[:,None]*fv).sum(-1)))
    flags = np.stack(((frs[:,None]>=.2)&(fs>=.2)&(np.abs(fa)>=30),
        np.abs(fs-frs[:,None])>=np.maximum(.3,.5*frs[:,None]),
        (frs[:,None]<=.1)&(fs>=.3),(frs[:,None]>=.3)&(fs<=.1)), -1)
    persistent = flags[:,:-1]&flags[:,1:]
    turnany = (persistent[:,:,0]&(np.sign(fa[:,:-1])==turnsign[:,None])&(turnsign[:,None]!=0)).any(1)
    speedany = (persistent[:,:,1]&(np.sign(fs[:,:-1]-frs[:,None])==speedsign[:,None])&(speedsign[:,None]!=0)).any(1)
    anymatch = (previous['target_turn']&turnany)|(previous['target_speed']&speedany)|(
        previous['target_start']&persistent[:,:,2].any(1))|(previous['target_stop']&persistent[:,:,3].any(1))
    category = np.zeros(len(r2),np.int8)
    category[previous['any']&~target] = 1  # generic interaction risk only
    category[target&~definite] = 5       # inconsistent past signs
    category[definite] = 4               # unmatched, not necessarily useless
    category[definite&anymatch] = 3       # any future event, not dominant first
    category[first] = 2
    return dict(cue_category=category,cue_any=previous['any'],cue_target=target,cue_specific=definite,
      cue_first_compatible=first,cue_any_compatible=anymatch,cue_turn=previous['target_turn'],
      cue_speed=previous['target_speed'],cue_start=previous['target_start'],cue_stop=previous['target_stop'],
      cue_neighbor=previous['neighbor_approach'],cue_turn_sign=turnsign,cue_speed_sign=speedsign,
      cue_signed_recent_angles=angle,cue_signed_recent_speed_delta=dv)

def distances(paths):
    i,j = np.triu_indices(20,1)
    d = np.linalg.norm(paths[:,i].astype(np.float64)-paths[:,j].astype(np.float64),axis=-1)
    return d.mean(-1),d[:,:,-1]
def cluster_all(mean, endpoint, thresholds):
    assignments = np.empty((len(mean),len(thresholds),20),np.int8)
    for k,(at,ft) in enumerate(thresholds):
        for row in range(len(mean)):
            d = np.maximum(mean[row]/at,endpoint[row]/ft)
            assignments[row,k] = fcluster(linkage(d,method='complete'),t=1.,criterion='distance')-1
    return assignments
def quality_counts(ade,fde,clusters,thresholds):
    raw = np.empty((len(ade),len(thresholds)),np.int8)
    effective = np.empty((len(ade),len(thresholds),clusters.shape[1]),np.int8)
    for k,(at,ft) in enumerate(thresholds):
        good = (ade<=at)&(fde<=ft); raw[:,k] = good.sum(1)
        for d in range(clusters.shape[1]):
            for row in range(len(ade)):
                effective[row,k,d] = len(np.unique(clusters[row,d,good[row]]))
    return raw,effective

def load_populations():
    meta = npz(BASE/'metadata.npz')
    populations = {}
    for name in ('train','dev'):
        x = npz(BASE/f'features/{name}.npz',['sample_id','R2','paths','logits'])
        lab = npz(BASE/f'labels/{name}.npz')
        enc = npz(ROOT/f'GENERATION_NATIVE_DECODER_SERIES/{name}_encoder.npz',['sample_id','past'])
        assert np.array_equal(x['sample_id'],lab['sample_id']) and np.array_equal(x['sample_id'],enc['sample_id'])
        assert np.max(np.abs(x['R2'][:,:16].reshape(-1,8,2)-enc['past']))==0
        assert np.max(np.abs(enc['past'][:,-1]))<1e-6
        truth = rules.describe(enc['past'],lab['future'])
        for k in ('y','uncertain','event_types','kind','onset_step','direction'):
            assert np.array_equal(truth[k],lab[k]),('GT event reconstruction',name,k)
        original = dict(ids=x['sample_id'],r2=x['R2'],paths=x['paths'],logits=x['logits'],past=enc['past'],
          future=lab['future'],family=align(x['sample_id'],meta)['scene_family'],
          sequence=align(x['sample_id'],meta)['sequence'],track=align(x['sample_id'],meta)['pedestrian_id'],
          ambiguous_track=align(x['sample_id'],meta)['ambiguous_track_time'])
        if name=='train':
            assert len(x['sample_id'])==28791
            for label, sl in (('FIT_native',slice(0,25912)),('CAL_native',slice(25912,None))):
                populations[label] = {k:v[sl] for k,v in original.items()}
        else:
            assert len(x['sample_id'])==1516
            populations['DEV_native'] = original
    assert len(set(populations['FIT_native']['ids'])&set(populations['CAL_native']['ids']))==0
    assert not set(populations['DEV_native']['ids'])&set(np.concatenate([populations['FIT_native']['ids'],populations['CAL_native']['ids']]))
    saved = align(populations['DEV_native']['ids'],npz(ROOT/'PULSE_RESPONSE_VISUAL_REVIEW_20261004/data/exact_arrays.npz'))
    assert np.array_equal(saved['native'],populations['DEV_native']['paths'])
    assert np.array_equal(saved['future'],populations['DEV_native']['future'])
    for method in ('fixed','learned'):
        for seed in (1,2,3):
            populations[f'DEV_{method}{seed}'] = {**populations['DEV_native'],'paths':saved[f'{method}{seed}']}
    return populations

def calculate(x, p):
    ids = x['ids']; n = len(ids)
    assert x['paths'].shape==(n,20,12,2)
    assert all(np.isfinite(x[k]).all() for k in ('paths','past','future','r2','logits'))
    data = rules.calculate(x['past'],x['future'],x['paths'],x['logits'],ids,x['family'],p)
    data.update(cues(x['r2'],rules.describe(x['past'],x['future']),x['past'],x['future']))
    d = np.linalg.norm(x['paths'].astype(np.float64)-x['future'][:,None].astype(np.float64),axis=-1)
    data['ade'],data['fde'] = d.mean(-1),d[:,:,-1]
    mean,end = distances(x['paths']); clusters = cluster_all(mean,end,p['dedup_thresholds'])
    data['clusters'] = clusters
    data['raw_good_count'],data['effective_good_count'] = quality_counts(data['ade'],data['fde'],clusters,p['path_thresholds'])
    data['all_distinct_count'] = clusters.max(-1)+1
    data['sequence'],data['track'],data['ambiguous_track'] = x['sequence'],x['track'],x['ambiguous_track']
    for j,threshold in enumerate(p['path_thresholds']):
        data[f'class_{j}'] = rules.partition(data,p,threshold)[0]
        data[f'alternate_class_{j}'] = rules.partition(data,p,threshold,p['alternate_hierarchy'])[0]
    return data

CUE_NAMES = ['no_hand_cue','generic_neighbor_only','target_first_type_direction_compatible',
             'target_any_later_type_direction_compatible','target_unmatched','target_sign_ambiguous']
def summaries(data,p):
    clear = data['GT_event'].astype(bool)&~data['GT_uncertain'].astype(bool)
    groups = dict(all=np.ones(len(clear),bool),clear=clear,no_event=~data['GT_event'].astype(bool)&~data['GT_uncertain'],
      boundary=data['GT_uncertain'].astype(bool),explicit_cue_clear=clear&data['cue_any'],
      specific_compatible_clear=clear&data['cue_first_compatible'],generic_only_clear=clear&(data['cue_category']==1))
    for i,name in enumerate(CUE_NAMES): groups['clear_'+name] = clear&(data['cue_category']==i)
    for scene in np.unique(data['scene_family']):
        groups['scene_'+str(scene)] = data['scene_family']==scene
        groups['compatible_scene_'+str(scene)] = (data['scene_family']==scene)&clear&data['cue_first_compatible']
    for kind in (1,2,3,4): groups[f'compatible_GTkind{kind}'] = clear&data['cue_first_compatible']&(data['GT_kind']==kind)
    result = []
    categories, nested, components, good = rules.partition(data,p,p['path_thresholds'][0])
    for name,mask in groups.items():
        n = int(mask.sum())
        if not n: result.append(dict(group=name,n=0)); continue
        selected = good[np.arange(len(clear)),data['selected_slot']]
        met = dict(group=name,n=n,clear_n=int((mask&clear).sum()),
          candidate_failure=int((mask&~good.any(1)).sum()),good_but_selection_miss=int((mask&good.any(1)&~selected).sum()),
          selected_good=int((mask&selected).sum()),minADE=float(data['ade'][mask].min(1).mean()),minFDE=float(data['fde'][mask].min(1).mean()),
          first_cue_compatible=int((mask&data['cue_first_compatible']).sum()),any_cue_compatible=int((mask&data['cue_any_compatible']).sum()),
          response_gap_counts={label:int((mask&(categories==i)).sum()) for i,label in enumerate(rules.CLASS_NAMES)},
          alternate_gap_counts={label:int((mask&(data['alternate_class_0']==i)).sum()) for i,label in enumerate(rules.CLASS_NAMES)},
          nested_response_coverage={key:int((mask&clear&value).sum()) for key,value in nested.items()},
          verified_track_count=len(set(zip(data['sequence'][mask&~data['ambiguous_track']].tolist(),data['track'][mask&~data['ambiguous_track']].tolist()))),
          ambiguous_track_windows=int((mask&data['ambiguous_track']).sum()))
        metrics=[]
        for q,t in enumerate(p['path_thresholds']):
            raw = data['raw_good_count'][mask,q]
            eff = data['effective_good_count'][mask,q]
            metrics.append(dict(path_threshold=t,coverage=int((raw>0).sum()),raw_good_distribution=np.bincount(raw,minlength=21).tolist(),
              raw_mean=float(raw.mean()),effective_good_distribution=[np.bincount(eff[:,d],minlength=21).tolist() for d in range(eff.shape[1])],
              effective_mean=eff.mean(0).tolist(),covered_effective_mean=eff[raw>0].mean(0).tolist() if (raw>0).any() else None,
              effective_ge2=[int((eff[:,d]>=2).sum()) for d in range(eff.shape[1])],effective_ge3=[int((eff[:,d]>=3).sum()) for d in range(eff.shape[1])],
              distinct_all_mean=data['all_distinct_count'][mask].mean(0).tolist()))
        met['quality_and_dedup'] = metrics
        result.append(met)
    return result

def synthetic(p):
    same = np.zeros((1,20,12,2),float)
    a,b = distances(same); c = cluster_all(a,b,p['dedup_thresholds'])
    assert (c.max(-1)==0).all()
    separate = same.copy(); separate[0,:,:,0] = np.arange(20)[:,None]
    a,b = distances(separate); c = cluster_all(a,b,p['dedup_thresholds']); assert (c.max(-1)==19).all()
    # Complete linkage cannot merge .00,.08,.16 when diameter .16 exceeds .10.
    chain = np.zeros((20,12,2)); chain[:,:,0] = np.arange(20)[:,None]*10; chain[:3,:,0] = np.array([0,.08,.16])[:,None]
    a,b = distances(chain[None]); c = cluster_all(a,b,[[.1,.2]])
    assert len(np.unique(c[0,0,:3]))==2
    ade=np.ones((1,20));fde=ade.copy();ade[0,0]=.1;fde[0,1]=.1
    raw,eff=quality_counts(ade,fde,c,[[.2,.4]]);assert raw[0,0]==0 and not eff.any()
    p0=np.zeros((1,8,2),np.float32);p0[:,:,0]=np.arange(-7,1)*.4
    gt=np.zeros((1,12,2),np.float32);gt[0,:,0]=np.arange(1,13)*.4
    pred=np.repeat(gt[:,None],20,axis=1)
    info=rules.calculate(p0,gt,pred,np.zeros((1,20)),np.array([1]),np.array(['synthetic']),p)
    assert info['class_0'][0]==0
    return dict(identical_one_cluster=True,separated_twenty_clusters=True,complete_link_not_chain_merge=True,
      independent_minima_not_samepath=True,exact_GT_candidate_good=True)

def preflight():
    assert not (HERE/'preflight.json').exists(),'Preserve previous preflight'
    p=read(HERE/'protocol.json'); t=time.perf_counter()
    old=snapshot(); locks=sources(); hash_s=verify({**old,**locks},'PREFLIGHT_SOURCE_AND_OLD_SHA')
    pops=load_populations(); all_ids=np.concatenate([pops[k]['ids'] for k in ('FIT_native','CAL_native','DEV_native')]);assert len(np.unique(all_ids))==30307
    x=pops['FIT_native'];part={k:v[:256] for k,v in x.items()}
    began=time.perf_counter();out=calculate(part,p);summaries(out,p);bench=time.perf_counter()-began
    dev=pops['DEV_native']; cue=cues(dev['r2'],rules.describe(dev['past'],dev['future']),dev['past'],dev['future'])
    clear=rules.describe(dev['past'],dev['future'])['y'].astype(bool)
    dd=np.linalg.norm(dev['paths'].astype(float)-dev['future'][:,None].astype(float),axis=-1)
    good=(dd.mean(-1)<=.2)&(dd[:,:,-1]<=.4)
    selected=good[np.arange(len(good)),dev['logits'].argmax(1)]
    assert len(dev['ids'])==1516 and clear.sum()==469 and good.any(1).sum()==733
    assert (cue['cue_any']&clear).sum()==274 and (cue['cue_any']&clear&~good.any(1)).sum()==228
    assert (cue['cue_any']&clear&good.any(1)&~selected).sum()==41
    estimate=bench*(28791+7*1516)/256*2
    planning=max(5,int(np.ceil((estimate+hash_s*2+60)/60)))
    verify(locks,'PREFLIGHT_FINAL_SOURCE_SHA')
    report=dict(status='PASS',source_hashes=locks,old_files=len(old),hash_seconds=hash_s,benchmark_256_seconds=bench,
      estimated_compute_audit_seconds=estimate,eta_minutes=[max(3,planning-2),planning+4],planning_minutes=planning,
      tests=synthetic(p),IDs_unique=30307,FIT=25912,CAL=2879,DEV=1516,DEV_handcue=274,
      explicit_coordinate_alignment_exact=True,event_label_reconstruction_exact=True,
      optimizer=False,model_forward=False,GPU_preflight='NOT_APPLICABLE_SAVED_DATA_CPU',
      frozen='Original checkpoint and scientific sources SHA locked, no model instantiated',
      training_resume='NOT_APPLICABLE; deterministic completed-job SHA verification/skip tested before launch',
      preflight_seconds=time.perf_counter()-t)
    write(HERE/'preflight.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},ensure_ascii=False),flush=True)

def done_valid(name, locks):
    record=HERE/'jobs'/f'{name}.done.json'
    if not record.exists(): return False
    value=read(record)
    assert value['status']=='SUCCESS' and value['source_hashes']==locks,('Completed job provenance',name)
    for path, expected in value['outputs'].items(): assert digest(path)==expected,('Completed job output SHA',name,path)
    return True

def audit_job(x,data,p):
    # Independently recompute saved arithmetic and row-wise accounting. Event
    # semantics are the locked rule, not independently established human truth.
    assert np.array_equal(x['ids'],data['sample_id'])
    delta=0.;n=len(x['ids']);tri=np.triu_indices(20,1)
    independently_selected=x['logits'].argmax(1)
    assert np.array_equal(independently_selected,data['selected_slot'])
    for left in range(0,n,256):
        right=min(n,left+256)
        raw=x['paths'][left:right].astype(np.float64); truth=x['future'][left:right].astype(np.float64)
        d=np.sqrt(((raw-truth[:,None])**2).sum(-1));ade=np.sum(d,axis=2)/12;fde=d[:,:,-1]
        delta=max(delta,float(abs(ade-data['ade'][left:right]).max()),float(abs(fde-data['fde'][left:right]).max()))
        pair=np.sqrt(((raw[:,tri[0]]-raw[:,tri[1]])**2).sum(-1))
        means=pair.sum(-1)/12;ends=pair[:,:,-1]
        for row in range(right-left):
            i=left+row
            for k,(at,ft) in enumerate(p['dedup_thresholds']):
                cc=data['clusters'][i,k];same=cc[tri[0]]==cc[tri[1]]
                assert (means[row,same]<=at+1e-12).all() and (ends[row,same]<=ft+1e-12).all(),('Cluster diameter',i,k)
                assert len(set(cc.tolist()))==int(data['all_distinct_count'][i,k])
            for q,(at,ft) in enumerate(p['path_thresholds']):
                accepted=[j for j in range(20) if ade[row,j]<=at and fde[row,j]<=ft]
                assert len(accepted)==int(data['raw_good_count'][i,q])
                for k in range(len(p['dedup_thresholds'])):
                    unique=set(int(data['clusters'][i,k,j]) for j in accepted)
                    assert len(unique)==int(data['effective_good_count'][i,q,k])
                for order,key in ((p['hierarchy'],f'class_{q}'),(p['alternate_hierarchy'],f'alternate_class_{q}')):
                    if int(independently_selected[i]) in accepted: code=0
                    elif accepted: code=1
                    elif data['GT_uncertain'][i]: code=9
                    elif not data['GT_event'][i]: code=8
                    else:
                        surviving=list(range(20));code=7
                        for name in order:
                            keep=[]
                            for j in surviving:
                                if name=='event':ok=bool(data['candidate_event'][i,j])
                                elif name=='type':ok=bool(data['candidate_event'][i,j]) and data['candidate_kind'][i,j]==data['GT_kind'][i]
                                elif name=='direction':ok=data['candidate_direction'][i,j]==data['GT_direction'][i]
                                elif name=='timing':ok=abs(int(data['candidate_onset'][i,j])-int(data['GT_onset'][i]))<=2
                                else:
                                    tolerance=15. if data['GT_kind'][i]==1 else max(.15,.25*abs(float(data['GT_strength'][i])))
                                    ok=abs(float(data['candidate_strength'][i,j])-float(data['GT_strength'][i]))<=tolerance
                                if ok:keep.append(j)
                            surviving=keep
                            if not surviving:code=dict(event=2,type=3,timing=4,direction=5,strength=6)[name];break
                    assert code==int(data[key][i]),('Partition',i,key,code,int(data[key][i]))
    assert delta<1e-10
    # Earlier hand-cue function is an alternate implementation for all flags.
    previous=oldcue.hand_cues(x['r2'])
    for old,new in [('any','cue_any'),('target_turn','cue_turn'),('target_speed','cue_speed'),
                    ('target_start','cue_start'),('target_stop','cue_stop'),('neighbor_approach','cue_neighbor')]:
        assert np.array_equal(previous[old],data[new])
    first=np.zeros(n,bool)
    for i in range(n):
        if not data['GT_event'][i]:continue
        kind=int(data['GT_kind'][i]);sign=int(data['GT_direction'][i])
        first[i]=((kind==1 and data['cue_turn'][i] and data['cue_turn_sign'][i]!=0 and int(data['cue_turn_sign'][i])==sign) or
           (kind==2 and data['cue_speed'][i] and data['cue_speed_sign'][i]!=0 and int(data['cue_speed_sign'][i])==sign) or
           (kind==3 and data['cue_start'][i]) or (kind==4 and data['cue_stop'][i]))
    assert np.array_equal(first,data['cue_first_compatible'])
    return dict(status='PASS_SAVED_ARITHMETIC',n=n,K=20,metric_max_error=delta,
      all_cluster_diameters_checked=True,all_raw_effective_counts_checked=True,
      both_hierarchy_same_candidate_partition_checked=True,all_cue_flags_and_first_sign_checked=True,
      all_models_replayed=False,event_rule_not_human_ground_truth_validation=True)

def resume_test():
    locks=sources();name='resume_synthetic_fixture'
    folder=HERE/'resume_smoke';folder.mkdir(exist_ok=True)
    output=folder/'saved.npz'
    if output.exists():raise AssertionError('Preserve existing resume evidence')
    save(output,dict(sample_id=np.arange(3),quality=np.array([0,1,2])))
    before=digest(output)
    done=HERE/'jobs'/f'{name}.done.json'
    write(done,dict(status='SUCCESS',source_hashes=locks,outputs={str(output):before}))
    assert done_valid(name,locks) and digest(output)==before
    altered=dict(locks);altered['synthetic_bad_source']='mismatch'
    rejected=False
    try:done_valid(name,altered)
    except AssertionError:rejected=True
    assert rejected
    # Do not mutate corrupt/old evidence to test rejection.
    write(HERE/'resume_check.json',dict(status='PASS',analysis_job_resume_only=True,
      source_mismatch_rejected=True,completed_output_SHA_verified_skip=True,saved_output_unchanged=True,
      output_SHA=before,optimizer_RNG_training_resume='NOT_APPLICABLE_NO_TRAINING'))

def run():
    pre=read(HERE/'preflight.json');p=read(HERE/'protocol.json');assert pre['status']=='PASS'
    assert read(HERE/'resume_check.json')['status']=='PASS'
    locks=pre['source_hashes'];assert sources()==locks
    old=read(HERE/'old_results_manifest.json')['files'];verify({**old,**locks},'INITIAL_OLD_AND_SOURCE_SHA')
    populations=load_populations();job_summaries={};audit_rows=[];began=time.perf_counter();durations=[]
    names=list(populations)
    for index,name in enumerate(names):
        stage('JOB_ANALYSIS',active_job=name,completed_jobs=index,total_jobs=len(names),job_seconds=durations)
        started=time.perf_counter();x=populations[name]
        path=HERE/'jobs'/f'{name}.npz';summary=HERE/'jobs'/f'{name}.summary.json';audit=HERE/'jobs'/f'{name}.audit.json'
        if done_valid(name,locks):
            print('SHA VERIFIED SKIP '+name,flush=True)
        else:
            assert not path.exists(),'Incomplete job output exists; preserve and inspect before recovery'
            out=calculate(x,p);save(path,out)
            write(summary,summaries(out,p))
            write(audit,audit_job(x,out,p))
            write(HERE/'jobs'/f'{name}.done.json',dict(status='SUCCESS',source_hashes=locks,
              outputs={str(z):digest(z) for z in (path,summary,audit)}))
        assert done_valid(name,locks)
        job_summaries[name]=read(summary);audit_rows.append(read(audit));durations.append(time.perf_counter()-started)
    native=npz(HERE/'jobs/DEV_native.npz')
    primary_cue=native['cue_any']&native['GT_event'].astype(bool)&~native['GT_uncertain']
    covered=native['raw_good_count'][:,0]>0
    good=(native['ade']<=.2)&(native['fde']<=.4)
    picked=good[np.arange(len(good)),native['selected_slot']]
    assert int(primary_cue.sum())==274 and int((primary_cue&~covered).sum())==228 and int((primary_cue&covered&~picked).sum())==41
    cases=[]
    for i in np.flatnonzero(primary_cue):
        cases.append(dict(sample_id=int(native['sample_id'][i]),scene=str(native['scene_family'][i]),
          cue_class=CUE_NAMES[int(native['cue_category'][i])],GT_kind=int(native['GT_kind'][i]),
          first_type_direction_compatible=bool(native['cue_first_compatible'][i]),any_event_compatible=bool(native['cue_any_compatible'][i]),
          generation_good=bool(covered[i]),selected_good=bool(picked[i]),raw_good=int(native['raw_good_count'][i,0]),
          effective_good=int(native['effective_good_count'][i,0,0]),response_gap=rules.CLASS_NAMES[int(native['class_0'][i])]))
    write(HERE/'DEV_274_CASES.json',dict(cases=cases,scoring_pool_IDs=native['sample_id'][primary_cue&covered&~picked].tolist()))
    stage('FINAL_ALL_OLD_AND_SOURCE_SHA',completed_jobs=len(names),total_jobs=len(names),job_seconds=durations)
    verify({**old,**locks},'FINAL_ALL_OLD_AND_SOURCE_SHA')
    write(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=audit_rows,max_metric_error=max(x['metric_max_error'] for x in audit_rows),
      source_SHA_files=len(locks),all_old_SHA_unchanged=len(old),all_new_outputs_SHA_verified=True,
      model_forward=False,optimizer=False,original_checkpoints_sources_unchanged=True))
    result=dict(status='PASS_FROZEN_SAVED_DATA_CONCORDANCE',name=p['name'],formal_trial=False,
      primary=['FIT_native','CAL_native'],secondary=['DEV_native','DEV_fixed1','DEV_fixed2','DEV_fixed3','DEV_learned1','DEV_learned2','DEV_learned3'],
      total_analysis_jobs=9,all_summaries=job_summaries,source_hashes=locks,all_old_files_unchanged=len(old),
      elapsed_compute_and_audit_seconds=time.perf_counter()-began,job_seconds=durations,
      priority=p['priority'],path_thresholds=p['path_thresholds'],dedup_thresholds=p['dedup_thresholds'],
      limitations=['Operational risk/cue compatibility is not proof of predictability or beneficial causal native use.',
       'Neighbor-only conflict is generic; direction/type not inferred.',
       'Single GT, only3 scene families, overlapping windows, originalF16 seen data, repeatedDEV and future-completeness neighbor qualification bias.',
       'Kinematic coarse response match does not establish full path accuracy or psychological intention.',
       'Distinct good-cluster number is secondary and is not maximized; compact near-static modes need not be harmful.',
       'Original scorer argmax for pulse paths is inherited slot score, not rescored.',
       'No new generator training, no official data exclusion/new baseline/SOTA/major claim.'])
    write(HERE/'RESULT.json',result)
    write(HERE/'exit_status.json',dict(status='SUCCESS',at=datetime.now().astimezone().isoformat(),
      worker_pid=os.getpid(),result_status=result['status'],arithmetic_audit='PASS'))
    stage('COMPLETE_AWAITING_INTERPRETATION',completed_jobs=len(names),total_jobs=len(names))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume-test',action='store_true');args=parser.parse_args()
    if args.preflight:preflight()
    elif args.resume_test:resume_test()
    else:run()
