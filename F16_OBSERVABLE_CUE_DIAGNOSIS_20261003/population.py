from common import *

def family(name):
    if name.startswith('crowds_zara'): return 'zara'
    if name.startswith('students') or name == 'uni_examples': return 'university'
    return 'hotel'

def fingerprint(points): return np.round(points.astype(np.float32),4).tobytes()

def metadata():
    """Recover metadata from existing ETH-training-source labels, no other fold."""
    if (HERE/'metadata.npz').exists():
        audit=read('metadata_audit.json')
        assert sha(HERE/'metadata.npz')==audit['sha256']; return npz(HERE/'metadata.npz')
    stage('SOURCE_METADATA_RECOVERY')
    lookup={}
    for seq in LABEL_NAMES:
        path=ASTRA/f'datasets/eth_ucy/labels/eth/{seq}_train.txt'
        table=np.loadtxt(path,dtype=str)
        frame=table[:,0].astype(float).astype(int)
        ped=table[:,1].astype(float).astype(int)
        xy=table[:,[13,15]].astype(np.float32)
        for person in np.unique(ped):
            take=np.flatnonzero(ped==person); order=np.argsort(frame[take]);take=take[order]
            f=frame[take];p=xy[take]
            for start in range(len(take)-19):
                if not np.all(np.diff(f[start:start+20])==1): continue
                key=fingerprint(p[start:start+20])
                lookup.setdefault(key,[]).append((seq,int(person),int(f[start])))
    raw=np.load(RAW,mmap_mode='r')
    train,_,_=load_population('train');dev,_,_=load_population('dev')
    ids=np.concatenate((train['sample_id'],dev['sample_id'])).astype(int)
    families=[]; seqs=[];peds=[];starts=[]; ambiguity=[];matches=[]
    for sample in ids:
        candidates=lookup.get(fingerprint(raw[sample,0,:,-2:]),[])
        fams={family(x[0]) for x in candidates}
        families.append(next(iter(fams)) if len(fams)==1 else 'unverified')
        seqs.append(candidates[0][0] if len({x[0] for x in candidates})==1 else 'ambiguous')
        peds.append(candidates[0][1] if len({(x[0],x[1]) for x in candidates})==1 else -1)
        starts.append(candidates[0][2] if len(candidates)==1 else -1)
        ambiguity.append(len(candidates)!=1); matches.append(len(candidates))
    data=dict(sample_id=ids,scene_family=np.array(families),sequence=np.array(seqs),pedestrian_id=np.array(peds),
              start_frame=np.array(starts),ambiguous_track_time=np.array(ambiguity),matching_source_windows=np.array(matches))
    save_npz('metadata.npz',**data)
    counts={g:int((data['scene_family']==g).sum())for g in np.unique(data['scene_family'])}
    write('metadata_audit.json',dict(status='PASS_METADATA_WITH_EXPLICIT_AMBIGUITIES',sha256=sha(HERE/'metadata.npz'),
        n=len(ids),scene_family_counts=counts,unique_track_time_n=int((~data['ambiguous_track_time']).sum()),
        no_other_fold_or_heldtest=True,source_files=[f'{x}_train.txt'for x in LABEL_NAMES],
        method='Exact rounded4 world-XY full20-window source fingerprint; metadata only, not probe input. All zara recordings grouped together; all university/students grouped together.',
        split_strategy='Leave-one-verified-scene-family-out within original FIT. Coarsest scene-family grouping prevents all same-scene time/track window overlap across folds.',
        limitations='At most three scene families: too few for reliable cluster CIs or formal Holm/permutation p values; those are withheld. Frozen F16 previously trained on these data. This is exploratory probe transfer, not independent native generalization. Unverified family rows excluded from grouped OOF only, retained descriptively. Ambiguous track/time never treated as independent identities.'))
    return data

def local_future(ids):
    raw=np.load(RAW,mmap_mode='r')
    coords=np.array(raw[ids,0,:,-2:],dtype=np.float32)
    direction=coords[:,7]-coords[:,4]
    norm=np.linalg.norm(direction,axis=-1)
    unit=direction/np.maximum(norm[:,None],1e-8)
    unit[norm<=.03]=[1.,0.]
    relative=coords[:,8:]-coords[:,7,None]
    return np.stack((relative[:,:,0]*unit[:,None,0]+relative[:,:,1]*unit[:,None,1],
                     -relative[:,:,0]*unit[:,None,1]+relative[:,:,1]*unit[:,None,0]),-1)

def labels(split):
    destination=HERE/f'labels/{split}.npz'
    if destination.exists(): return npz(destination)
    cache,enc,_=load_population(split)
    future=local_future(cache['sample_id'])
    paths=cache['candidate'][:,:,:24].reshape(-1,20,12,2)
    errors=np.linalg.norm(paths-future[:,None],axis=-1)
    np.testing.assert_allclose(errors.mean(-1),cache['ade'],atol=2e-5,rtol=0)
    np.testing.assert_allclose(errors[:,:,-1],cache['fde'],atol=2e-5,rtol=0)
    data=events(enc['past'],future)
    for f in (.75,1.25):
        other=events(enc['past'],future,f)
        data[f'y_factor_{f}']=other['y'];data[f'uncertain_factor_{f}']=other['uncertain']
    data.update(sample_id=cache['sample_id'],future=future)
    save_npz(f'labels/{split}.npz',**data)
    return data

def candidate_audit():
    if (HERE/'candidate_audit.npz').exists()and (HERE/'candidate_audit.json').exists():
        assert read('candidate_audit.json')['status']=='PASS_ALL_NATIVE_1516_K20'
        return npz(HERE/'candidate_audit.npz')
    cache,enc,_=load_population('dev');lab=labels('dev')
    paths=cache['candidate'][:,:,:24].reshape(-1,20,12,2)
    errors=np.linalg.norm(paths-lab['future'][:,None],axis=-1)
    ade=errors.mean(-1);fde=errors[:,:,-1]
    selected=cache['logits'].argmax(-1)
    candidate_events=events(np.repeat(enc['past'],20,axis=0),paths.reshape(-1,12,2))
    event=candidate_events['y'].reshape(-1,20)
    onset=candidate_events['onset_step'].reshape(-1,20)
    kind=candidate_events['kind'].reshape(-1,20)
    types=candidate_events['event_types'].reshape(-1,20,4)
    matched_type=(types & lab['event_types'][:,None]).any(-1)
    matched_time=np.abs(onset-lab['onset_step'][:,None])<=2
    row=np.arange(len(ade))
    step=np.linalg.norm(np.diff(np.concatenate((enc['past'][:,-1:],lab['future']),1),axis=1),axis=-1)
    tags={}
    for at,ft in ((.2,.4),(.3,.5)):
        good=(ade<=at)&(fde<=ft)
        tags[f'coverage_{at}_{ft}']=good.any(1)
        tags[f'selected_good_{at}_{ft}']=good[row,selected]
    output=dict(sample_id=cache['sample_id'],ade=ade,fde=fde,selected=selected,selected_ade=ade[row,selected],
        selected_fde=fde[row,selected],early_error=errors[:,:,:4].mean(-1),middle_error=errors[:,:,4:8].mean(-1),
        late_error=errors[:,:,8:].mean(-1),joint_regret=(ade+.5*fde)[row,selected]-(ade+.5*fde).min(1),
        candidate_event=event,candidate_event_type=kind,candidate_onset=onset,
        event_type_time_coverage=(event.astype(bool)&matched_type&matched_time).any(1),
        selected_event_type_time=(event.astype(bool)&matched_type&matched_time)[row,selected],
        GT_event=lab['y'],GT_uncertain=lab['uncertain'],near_static=step.sum(1)<=.5,**tags)
    save_npz('candidate_audit.npz',**output)
    summaries=[]
    groups=[('all',np.ones(len(ade),bool)),('clear_event',lab['y'].astype(bool)),
        ('no_clear_event',~lab['y'].astype(bool)&~lab['uncertain']),('boundary_uncertain',lab['uncertain']),
        ('near_static',output['near_static'])]
    for name,mask in groups:
        if not mask.any(): continue
        summaries.append(dict(group=name,n=int(mask.sum()),mean_oracle_ADE=float(ade[mask].min(1).mean()),
            mean_oracle_FDE=float(fde[mask].min(1).mean()),mean_selected_ADE=float(output['selected_ade'][mask].mean()),
            mean_selected_FDE=float(output['selected_fde'][mask].mean()),
            SAMEPATH_20_40_rate=float(output['coverage_0.2_0.4'][mask].mean()),
            selected_20_40_rate=float(output['selected_good_0.2_0.4'][mask].mean()),
            event_type_time_coverage_rate=float(output['event_type_time_coverage'][mask].mean())))
    write('candidate_audit.json',dict(status='PASS_ALL_NATIVE_1516_K20',n=len(ade),K=20,source='native original logit argmax',
        rows=summaries,independent_minima_not_samepath=True,event_type_time_tolerance_steps=2,
        limitation='Event type/timing matches are not complete path accuracy. One realized GT cannot establish impossible futures or invalidate all alternative candidates.'))
    return output

def blind_materials():
    """Materials only. Never write human answers or claim unseen status verified."""
    if (HERE/'human_blind/past_only_cases.json').exists()and (HERE/'human_blind/reveal_after_phase1_locked.npz').exists():
        assert read('human_blind/past_only_cases.json')['status']=='MATERIALS_ONLY_NOT_HUMAN_ANNOTATIONS'
        return
    cache,enc,obs=load_population('dev')
    exclude=set()
    annotation=Path('C:/Users/linke/Downloads/human_trajectory_observations (1).json')
    if annotation.exists():
        def walk(x):
            if isinstance(x,dict):
                if 'sample_id'in x: exclude.add(int(x['sample_id']))
                for y in x.values(): walk(y)
            elif isinstance(x,list):
                for y in x: walk(y)
        walk(json.loads(annotation.read_text(encoding='utf-8-sig')))
    eligible=np.flatnonzero(~np.isin(cache['sample_id'],list(exclude)))
    rng=np.random.default_rng(SEED);take=rng.choice(eligible,min(120,len(eligible)),replace=False)
    local=obs[take,160:304].reshape(-1,9,8,2)
    write('human_blind/past_only_cases.json',dict(status='MATERIALS_ONLY_NOT_HUMAN_ANNOTATIONS',
        sampling='Uniform without replacement among non-annotated native1516 IDs',
        inclusion_probability=len(take)/len(eligible),already_seen_status='Unknown: reviewer must disclose any prior exposure',
        excluded_previously_annotated_ids=sorted(exclude),dt=.4,
        cases=[dict(sample_id=int(cache['sample_id'][i]),past_local=local[j].tolist(),num_valid=int(obs[i,304]))for j,i in enumerate(take)],
        phase1_fields=['probability_of_change','event_type','onset_bin','cue_observed_time','cue_description','prior_GT_exposure'],
        annotations=[]))
    save_npz('human_blind/reveal_after_phase1_locked.npz',sample_id=cache['sample_id'][take],future=labels('dev')['future'][take],
             native=cache['candidate'][take,:,:24].reshape(-1,20,12,2),logits=cache['logits'][take])
