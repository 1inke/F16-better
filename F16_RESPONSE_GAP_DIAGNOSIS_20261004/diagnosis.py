"""New-folder-only deterministic saved-output analysis; no torch/model/optimizer."""
from pathlib import Path
from datetime import datetime
import argparse, hashlib, json, os, sys, time, traceback
sys.dont_write_bytecode=True
import numpy as np
import rules
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
PREVIOUS=ROOT/'GENERATION_TIME_RESPONSE_SERIES_20261004'
CUES=ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003'
CACHE=ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):value.update(block)
    return value.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(6):
        try:os.replace(temporary,path);return
        except PermissionError as error:
            readonly=path.exists() and bool(getattr(path.stat(),'st_file_attributes',0)&1)
            if readonly or attempt==5 or getattr(error,'winerror',None) not in (5,32):raise
            time.sleep(.02*2**attempt)
def stage(name,**details):
    record=dict(stage=name,at=datetime.now().astimezone().isoformat(),**details)
    write(HERE/'progress.json',record);print(json.dumps(record,ensure_ascii=False),flush=True)
def verify(files,label):
    start=time.perf_counter()
    for index,(name,sha) in enumerate(files.items()):
        assert Path(name).is_file() and digest(name)==sha,('Input SHA changed',name)
        if index%4000==0:stage(label,completed_files=index,total_files=len(files))
    return time.perf_counter()-start
def snapshot():
    path=HERE/'old_results_manifest.json'
    if path.exists():return read(path)['files']
    files=dict(read(PREVIOUS/'old_results_manifest.json')['files'])
    for folder in (PREVIOUS,):
        for source in folder.rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix not in ('.lock','.tmp','.pending') and source.name!='runtime_state.json':
                files[str(source)]=digest(source)
    write(path,dict(files=files,inherited_count=18872,added_folders=[str(PREVIOUS)],
      excludes=['.memory','AUTONOMOUS_GENERATION_CYCLE.json','runtime_state.json','lock/temp/bytecode'],
      source_models_and_scientific_results_immutable=True))
    return files
def sources():
    files=dict(read(PREVIOUS/'preflight.json')['source_hashes'])
    for path in list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/'README.md',HERE/'DESIGN.md',HERE/'protocol.json',HERE/'old_results_manifest.json']:
        files[str(path)]=digest(path)
    for path in (CUES/'metadata.npz',CUES/'metadata_audit.json',CUES/'labels/train.npz',CUES/'labels/dev.npz'):
        files[str(path)]=digest(path)
    return files
def npz(path,keys=None):
    with np.load(path,allow_pickle=False) as values:return {key:values[key] for key in (keys or values.files)}
def family(ids):
    meta=npz(CUES/'metadata.npz',['sample_id','scene_family'])
    order=np.argsort(meta['sample_id']);sorted_ids=meta['sample_id'][order]
    where=np.searchsorted(sorted_ids,ids)
    assert np.array_equal(sorted_ids[where],ids)
    return meta['scene_family'][order[where]]
def population(name):
    stage('LOAD_SAVED_INPUTS',population=name)
    cache=npz(CACHE/f'eth_{name}.npz',['sample_id','candidate','logits','ade','fde'])
    past=npz(ROOT/f'GENERATION_NATIVE_DECODER_SERIES/{name}_encoder.npz',['sample_id','past'])
    labels=npz(CUES/f'labels/{name}.npz',['sample_id','future'])
    assert np.array_equal(cache['sample_id'],past['sample_id']) and np.array_equal(cache['sample_id'],labels['sample_id'])
    paths=cache['candidate'][:,:,:24].reshape(-1,20,12,2)
    error=np.linalg.norm(paths-labels['future'][:,None],axis=-1)
    difference=max(float(np.abs(error.mean(-1)-cache['ade']).max()),float(np.abs(error[:,:,-1]-cache['fde']).max()))
    assert difference<2e-5,('Coordinate alignment',name,difference)
    return dict(past=past['past'],future=labels['future'],paths=paths,logits=cache['logits'],
                ids=cache['sample_id'],family=family(cache['sample_id']),cached_metric_max_delta=difference)
def subset(source,indices):
    return {name:value[indices] for name,value in source.items() if isinstance(value,np.ndarray)}
def inputs():
    train=population('train');dev=population('dev')
    assert len(train['ids'])==28791 and len(dev['ids'])==1516
    assert not set(train['ids'])&set(dev['ids'])
    fit=np.arange(25912);cal=np.arange(25912,28791)
    assert not set(train['ids'][fit])&set(train['ids'][cal])
    return (dict(FIT_native=subset(train,fit),CAL_native=subset(train,cal),DEV_native=subset(dev,np.arange(1516))),
      dict(train_cached_metric_max_delta=train['cached_metric_max_delta'],dev_cached_metric_max_delta=dev['cached_metric_max_delta']))

def synthetic(protocol):
    past=np.zeros((2,8,2),np.float32);past[:,:,0]=np.arange(-7,1)[None]*.4
    future=np.zeros((2,12,2),np.float32);future[:,:,0]=np.arange(1,13)[None]*.4
    future[1,:,1]=np.arange(1,13)*.4
    paths=np.repeat(future[:,None],20,axis=1);logits=np.zeros((2,20),np.float32)
    data=rules.calculate(past,future,paths,logits,np.array([1,2]),np.array(['synthetic']*2),protocol)
    assert (data['class_0']==0).all() and data['GT_event'][1]==1
    # Separate minima must not become a same-path adequate candidate.
    fake={key:value.copy() for key,value in data.items()}
    fake['ade'][:]=1.;fake['fde'][:]=1.;fake['ade'][:,0]=.1;fake['fde'][:,1]=.1
    assert not rules.partition(fake,protocol,[.2,.4])[3].any()
    # Individually matched response components from DIFFERENT candidates cannot
    # become a complete-response match. All type-matched candidates have late onset.
    fake['GT_event'][:]=1;fake['GT_uncertain'][:]=False;fake['GT_kind'][:]=1;fake['GT_onset'][:]=1
    fake['candidate_event'][:]=1;fake['candidate_kind'][:]=2;fake['candidate_kind'][:,0]=1
    fake['candidate_onset'][:]=1;fake['candidate_onset'][:,0]=10
    assert (rules.partition(fake,protocol,[.2,.4])[0]==4).all()
    permutation=np.arange(19,-1,-1)
    changed=rules.calculate(past,future,paths[:,permutation],logits[:,permutation],np.array([1,2]),np.array(['synthetic']*2),protocol)
    assert np.array_equal(data['class_0'],changed['class_0'])
    return dict(exact_GT_path_adequate=True,separate_minima_not_samepath=True,
      response_components_must_coincide_in_one_candidate=True,permutation_coverage_invariant=True,
      adequacy_has_priority_over_event_thresholds=True)

def preflight():
    assert not (HERE/'preflight.json').exists(),'Preserve an existing preflight; inspect failure provenance before retry'
    protocol=read(HERE/'protocol.json');start=time.perf_counter()
    old=snapshot();locks=sources()
    combined={**old,**locks};hash_seconds=verify(combined,'PREFLIGHT_ALL_OLD_AND_SOURCE_SHA')
    data,alignment=inputs()
    benchmark_start=time.perf_counter()
    part=subset(data['FIT_native'],np.arange(512))
    check=rules.calculate(part['past'],part['future'],part['paths'],part['logits'],part['ids'],part['family'],protocol)
    rows=rules.summary(check,protocol);benchmark_seconds=time.perf_counter()-benchmark_start
    assert len(rows)>0 and len(check['sample_id'])==512
    tests=synthetic(protocol)
    sources_again=sources();assert sources_again==locks
    # Estimated full workload: 28,791 native primary rows + 19*1,516 secondary rows,
    # two all-file hash passes, independent direct-metric audit, saved NPZ IO.
    compute_estimate=benchmark_seconds*(28791+19*1516)/512*2
    planning=max(3,int(np.ceil((2*hash_seconds+compute_estimate+45)/60)))
    result=dict(status='PASS',source_hashes=locks,old_files=len(old),combined_hash_seconds=hash_seconds,
      preflight_seconds=time.perf_counter()-start,benchmark_512_seconds=benchmark_seconds,
      eta_minutes=[max(2,planning-1),planning+3],planning_minutes=planning,
      estimated_compute_audit_seconds=compute_estimate,tests=tests,alignment=alignment,
      FIT_n=25912,CAL_n=2879,DEV_secondary_n=1516,K=20,optimizer=False,model_forward=False,
      GPU_preflight='NOT_APPLICABLE: pure saved-output CPU analysis',
      frozen_integrity='No model instantiated or mutated; all original checkpoint SHA locked',
      training_resume='NOT_APPLICABLE: no training; completed analysis-job SHA verified skip')
    write(HERE/'preflight.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},ensure_ascii=False),flush=True)

def independent_partition(data,protocol,threshold,order):
    # Independent sample-loop accounting; does not call rules.partition/masks.
    result=[]
    for row in range(len(data['sample_id'])):
        adequate=(data['ade'][row]<=threshold[0])&(data['fde'][row]<=threshold[1])
        if adequate[data['selected_slot'][row]]:result.append(0);continue
        if adequate.any():result.append(1);continue
        if data['GT_uncertain'][row]:result.append(9);continue
        if not data['GT_event'][row]:result.append(8);continue
        surviving=set(range(20));code=7
        for stage_name in order:
            accepted=set()
            for slot in surviving:
                if stage_name=='event':yes=bool(data['candidate_event'][row,slot])
                elif stage_name=='type':yes=bool(data['candidate_event'][row,slot]) and data['candidate_kind'][row,slot]==data['GT_kind'][row]
                elif stage_name=='timing':yes=abs(int(data['candidate_onset'][row,slot])-int(data['GT_onset'][row]))<=protocol['onset_tolerance_steps']
                elif stage_name=='direction':yes=data['candidate_direction'][row,slot]==data['GT_direction'][row]
                else:
                    limit=protocol['turn_strength_tolerance_degrees'] if data['GT_kind'][row]==1 else max(protocol['speed_strength_absolute_tolerance_mps'],protocol['speed_strength_relative_tolerance']*abs(float(data['GT_strength'][row])))
                    yes=abs(float(data['candidate_strength'][row,slot])-float(data['GT_strength'][row]))<=limit
                if yes:accepted.add(slot)
            surviving=accepted
            if not surviving:
                code=dict(event=2,type=3,timing=4,direction=5,strength=6)[stage_name];break
        result.append(code)
    return np.array(result,np.int8)

def audit_job(data,source,protocol):
    error=np.linalg.norm(source['paths']-source['future'][:,None],axis=-1)
    difference=max(float(np.abs(data['ade']-error.mean(-1)).max()),float(np.abs(data['fde']-error[:,:,-1]).max()))
    assert difference<1e-6
    assert np.array_equal(data['sample_id'],source['ids']) and len(np.unique(data['sample_id']))==len(source['ids'])
    assert source['paths'].shape==(len(source['ids']),20,12,2) and np.isfinite(source['paths']).all()
    for key,actual in [('early_error',error[:,:,:4].mean(-1)),('middle_error',error[:,:,4:8].mean(-1)),('late_error',error[:,:,8:].mean(-1))]:
        assert np.max(np.abs(data[key]-actual))<1e-6,key
    # Independently derive physical summary fields from the reused ORIGINAL event
    # rule. This checks our wrapper/reshape/arithmetic, NOT event-label validity.
    predicted=rules.old.events(np.repeat(source['past'],20,axis=0),source['paths'].reshape(-1,12,2))
    truth=rules.old.events(source['past'],source['future'])
    assert np.array_equal(data['candidate_kind'],predicted['kind'].reshape(-1,20))
    assert np.array_equal(data['GT_kind'],truth['kind'])
    speed=np.abs(predicted['future_speed'].reshape(-1,20,12)-truth['future_speed'][:,None]).mean(-1)
    angle=np.abs(((predicted['signed_angle'].reshape(-1,20,12)-truth['signed_angle'][:,None]+180)%360)-180).mean(-1)
    assert np.max(np.abs(data['speed_curve_mae']-speed))<1e-6
    assert np.max(np.abs(data['angle_curve_mae']-angle))<1e-6
    for index,threshold in enumerate(protocol['path_thresholds']):
        for label,order in [('class_',protocol['hierarchy']),('alternate_class_',protocol['alternate_hierarchy'])]:
            rebuilt=independent_partition(data,protocol,threshold,order)
            assert np.array_equal(rebuilt,data[label+str(index)]),(label,index)
    rows=rules.summary(data,protocol)
    for value in rows:
        name=value['group'];clear=data['GT_event'].astype(bool)&~data['GT_uncertain'].astype(bool)
        if name=='overall':mask=np.ones(len(clear),bool)
        elif name=='clear_event':mask=clear
        elif name=='no_clear_event':mask=~data['GT_event'].astype(bool)&~data['GT_uncertain'].astype(bool)
        elif name=='boundary_uncertain':mask=data['GT_uncertain'].astype(bool)
        elif name.startswith('clear_scene_'):mask=clear&(data['scene_family']==name[len('clear_scene_'):])
        elif name.startswith('scene_'):mask=data['scene_family']==name[len('scene_'):]
        else:mask=clear&(data['GT_kind']==int(name[len('GT_kind_'):]))
        assert int(mask.sum())==value['n']
        selected=data['selected_slot'];ar=np.arange(len(clear));joint=(error.mean(-1)+.5*error[:,:,-1]).argmin(1)
        expected=dict(independent_min_ADE=error.mean(-1).min(1)[mask].mean(),independent_min_FDE=error[:,:,-1].min(1)[mask].mean(),
          selected_ADE=error.mean(-1)[ar,selected][mask].mean(),selected_FDE=error[:,:,-1][ar,selected][mask].mean(),
          joint_oracle_ADE=error.mean(-1)[ar,joint][mask].mean(),joint_oracle_FDE=error[:,:,-1][ar,joint][mask].mean(),
          joint_oracle_early=error[:,:,:4].mean(-1)[ar,joint][mask].mean(),joint_oracle_middle=error[:,:,4:8].mean(-1)[ar,joint][mask].mean(),
          joint_oracle_late=error[:,:,8:].mean(-1)[ar,joint][mask].mean(),joint_oracle_speed_curve_mae=speed[ar,joint][mask].mean(),
          joint_oracle_angle_curve_mae=angle[ar,joint][mask].mean())
        for key,actual in expected.items():assert abs(float(actual)-value[key])<1e-6,(name,key)
        for index in (0,1):
            item=value['threshold_'+str(index)]
            assert sum(item['partition_counts'].values())==value['n']
            assert sum(item['alternate_order_partition_counts'].values())==value['n']
            assert item['partition_counts']['adequate_selected']==item['selected_adequate_n']
            assert item['partition_counts']['adequate_selected']+item['partition_counts']['adequate_present_selection_miss']==item['samepath_adequate_n']
    return dict(status='PASS',n=len(source['ids']),K=20,direct_metric_max_error=difference,
      independent_scalar_partitions=True,event_label_truth_independently_verified=False,
      scope='Direct coordinates and saved component scalar arithmetic; no model replay, causal or event-label truth certification')

def jobs(data):
    yield 'FIT_native',data['FIT_native'],'PRIMARY_FIT'
    yield 'CAL_native',data['CAL_native'],'PRIMARY_CAL'
    yield 'DEV_native',data['DEV_native'],'SECONDARY_DEV_ALREADY_EXPLORED'
    paths=sorted((PREVIOUS/'results').glob('seed*.npz'));assert len(paths)==18
    for path in paths:
        record=npz(path,['sample_id','trajectory'])
        assert np.array_equal(record['sample_id'],data['DEV_native']['ids'])
        source=dict(data['DEV_native']);source['paths']=record['trajectory']
        yield path.stem,source,'SECONDARY_DEV_ALREADY_EXPLORED'

def run():
    assert not (HERE/'RESULT.json').exists(),'Preserve completed diagnostic outputs'
    pf=read(HERE/'preflight.json');assert pf['status']=='PASS'
    verify(pf['source_hashes'],'RUN_SOURCE_SHA')
    old=read(HERE/'old_results_manifest.json')['files'];verify(old,'RUN_OLD_SHA_BEFORE')
    protocol=read(HERE/'protocol.json');data,alignment=inputs()
    summaries=[];audits=[];outputs={};all_saved={}
    for position,(name,source,scope) in enumerate(jobs(data)):
        destination=HERE/'jobs'/f'{name}.npz';record_path=HERE/'jobs'/f'{name}.json'
        stage('RESPONSE_COMPONENT_ACCOUNTING',completed_jobs=position,total_jobs=21,active_job=name)
        if destination.exists() or record_path.exists():
            assert destination.exists(),'Record without saved job: preserve provenance'
            saved=npz(destination)
            if record_path.exists():
                record=read(record_path);assert record['status']=='PASS' and digest(destination)==record['sha256']
            else:
                checked=audit_job(saved,source,protocol)
                record=dict(status='PASS',sha256=digest(destination),scope=scope,audit=checked,
                            recovery='Existing NPZ independently verified; only missing record added, no NPZ overwrite')
                write(record_path,record)
        else:
            chunks=[]
            for begin in range(0,len(source['ids']),protocol['batch']):
                index=np.arange(begin,min(begin+protocol['batch'],len(source['ids'])))
                part=subset(source,index)
                chunks.append(rules.calculate(part['past'],part['future'],part['paths'],part['logits'],part['ids'],part['family'],protocol))
                if begin%4096==0:stage('RESPONSE_COMPONENT_ACCOUNTING',completed_jobs=position,total_jobs=21,active_job=name,completed_rows=begin,total_rows=len(source['ids']))
            saved={key:np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
            checked=audit_job(saved,source,protocol)
            destination.parent.mkdir(exist_ok=True)
            temporary=destination.with_name(destination.name+'.pending.npz')
            np.savez_compressed(temporary,**saved);os.replace(temporary,destination)
            record=dict(status='PASS',sha256=digest(destination),scope=scope,audit=checked)
            write(record_path,record)
        checked=audit_job(npz(destination),source,protocol)
        rows=rules.summary(saved,protocol);outputs[name]=record['sha256'];all_saved[name]=saved
        summaries.append(dict(job=name,scope=scope,selected_slot_source='original native logit argmax (inherited for temporal-response paths)',rows=rows))
        audits.append(dict(job=name,**checked))
    comparisons=[]
    reference=all_saved['DEV_native']
    clear=reference['GT_event'].astype(bool)&~reference['GT_uncertain'].astype(bool)
    for name,treated in all_saved.items():
        if not name.startswith('seed'):continue
        control_name=name.replace('_mean_std_','_zero_') if '_mean_std_' in name else None
        references=[('native',reference)]+([(control_name,all_saved[control_name])] if control_name else [])
        for reference_name,base in references:
            for index,threshold in enumerate(protocol['path_thresholds']):
                g1=(base['ade']<=threshold[0])&(base['fde']<=threshold[1]);g2=(treated['ade']<=threshold[0])&(treated['fde']<=threshold[1])
                for label,mask in [('overall',np.ones(len(clear),bool)),('clear_event',clear)]:
                    before=g1.any(1);after=g2.any(1)
                    comparisons.append(dict(job=name,reference=reference_name,group=label,threshold_index=index,n=int(mask.sum()),
                        rescued=int((mask&~before&after).sum()),lost=int((mask&before&~after).sum()),
                        gap_partition_transition_matrix=np.array([[int((mask&(base['class_'+str(index)]==i)&(treated['class_'+str(index)]==j)).sum()) for j in range(10)] for i in range(10)]).tolist()))
    stage('FINAL_SOURCE_AND_ALL_OLD_SHA',completed_jobs=21,total_jobs=21)
    final_source_seconds=verify(pf['source_hashes'],'FINAL_SOURCE_SHA')
    final_old_seconds=verify(old,'FINAL_ALL_OLD_SHA')
    for name,sha in outputs.items():assert digest(HERE/'jobs'/f'{name}.npz')==sha
    audit=dict(status='PASS',jobs=audits,total_jobs=21,old_files_unchanged=len(old),sources_unchanged=len(pf['source_hashes']),
      final_source_seconds=final_source_seconds,final_old_SHA_seconds=final_old_seconds,all_saved_job_SHA_unchanged=True,
      all_original_models_not_instantiated_or_mutated=True,model_replay=False,optimizer=False,GT_forward=False,
      independent_scope='Every saved ADE/FDE and primary/alternate partition reconstructed from source coordinates and scalar components; group summary arithmetic; event-rule truth not independently certified')
    write(HERE/'independent_audit.json',audit)
    result=dict(status='PASS_DIAGNOSTIC_ONLY',formal_trial=False,major=False,optimizer=False,model_forward=False,GT_forward=False,
      protocol=protocol,alignment=alignment,class_names=rules.CLASS_NAMES,summaries=summaries,comparisons=comparisons,outputs=outputs,
      old_files_unchanged=len(old),source_files_unchanged=len(pf['source_hashes']),audit='independent_audit.json',
      limitations=['Local exploratory accounting, not causal root-cause or future impossibility proof.',
        'Hierarchy depends on order; fixed alternate order and marginal counts also reported.',
        'Dominant-first-event != ANY event-type overlap used in earlier diagnostic.',
        'Strength tolerances operational/fixed before this analysis, not universal/SOTA criteria.',
        'GT used only offline; original model not retrained/replayed.',
        'Primary FIT/CAL native model previously trained on these sources; secondary DEV already repeatedly explored.',
        'Only three scene families/correlated windows; no reliable cluster confidence intervals or p values.',
        'Each policy and seed reuses IDs; never sum rescues as new independent people.',
        'Event auxiliaries/scorer/architecture changes not combined; no next training embedded.'])
    write(HERE/'RESULT.json',result)
    lines=['# Response-gap diagnosis: saved-output completion','',
      'Operational diagnosis only. Read DESIGN.md and independent_audit.json before interpreting.',
      'AI-assisted design/code; no external independent peer review/public preregistration.',
      '','## Primary original FIT/CAL','']
    for item in summaries[:2]:
        row=next(value for value in item['rows'] if value['group']=='clear_event')
        lines.append('### '+item['job']+' — clear events N='+str(row['n']))
        lines.extend(['',json.dumps(row,ensure_ascii=False,indent=2),''])
    lines.extend(['## Interpretation pending','',
      'Completion follow-up must compare primary FIT/CAL strata and all fixed secondary outputs,',
      'consider marginal and alternate-order gaps, then specify one architecture hypothesis.',
      'Do not infer that hierarchical counts identify causes, or launch further training automatically.'])
    (HERE/'RESULT_ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
    write(HERE/'exit_status.json',dict(status='SUCCESS',finished_at=datetime.now().astimezone().isoformat(),completed_jobs=21))
    stage('COMPLETE',completed_jobs=21,total_jobs=21)
    print(json.dumps(dict(status=result['status'],old_files=len(old),sources=len(pf['source_hashes']))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    try:preflight() if args.preflight else run()
    except BaseException:
        target=HERE/'failures'/('preflight_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        target.mkdir(parents=True,exist_ok=False)
        (target/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write(target/'source_hashes.json',{str(path):digest(path) for path in HERE.glob('*.py')})
        raise
