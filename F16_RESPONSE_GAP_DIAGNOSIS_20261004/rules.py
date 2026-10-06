"""Pure CPU, posthoc candidate accounting. No model, training or inference GT route."""
from pathlib import Path
import importlib.util, sys
sys.dont_write_bytecode = True
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('response_gap_old_event_rules', ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/common.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

CLASS_NAMES = ('adequate_selected','adequate_present_selection_miss',
 'no_candidate_event','dominant_type_gap','onset_gap','direction_gap','strength_gap',
 'coarse_response_covered_path_inadequate','nonclear_path_inadequate','boundary_uncertain_path_inadequate')
STAGES = ('event','type','timing','direction','strength')

def describe(past, future):
    value = old.events(past, future)
    index = np.minimum(np.maximum(value['onset_step']-1, 0), 10)
    rows = np.arange(len(past))
    mean_angle = (value['signed_angle'][rows,index]+value['signed_angle'][rows,index+1])*.5
    mean_speed = (value['future_speed'][rows,index]+value['future_speed'][rows,index+1])*.5
    magnitude = np.where(value['kind']==1,np.abs(mean_angle),mean_speed-value['refspeed'])
    direction = np.where(value['kind']==1,np.sign(mean_angle),np.sign(mean_speed-value['refspeed'])).astype(np.int8)
    value['response_strength'] = np.where(value['y'],magnitude,0).astype(np.float32)
    value['response_direction'] = np.where(value['y'],direction,0).astype(np.int8)
    return value

def masks(data, protocol):
    event = data['candidate_event'].astype(bool)
    kind = event & (data['candidate_kind']==data['GT_kind'][:,None])
    time = np.abs(data['candidate_onset']-data['GT_onset'][:,None])<=protocol['onset_tolerance_steps']
    direct = data['candidate_direction']==data['GT_direction'][:,None]
    gt_strength=data['GT_strength'].astype(np.float64)
    tolerance = np.where(data['GT_kind']==1,protocol['turn_strength_tolerance_degrees'],
        np.maximum(protocol['speed_strength_absolute_tolerance_mps'],
          protocol['speed_strength_relative_tolerance']*np.abs(gt_strength)))
    strength = np.abs(data['candidate_strength'].astype(np.float64)-gt_strength[:,None])<=tolerance[:,None]
    return dict(event=event,type=kind,timing=time,direction=direct,strength=strength)

def partition(data, protocol, threshold, order=None):
    order = order or protocol['hierarchy']
    component = masks(data,protocol)
    clear=data['GT_event'].astype(bool)&~data['GT_uncertain'].astype(bool)
    adequate=(data['ade']<=threshold[0])&(data['fde']<=threshold[1])
    yes=adequate.any(1); picked=adequate[np.arange(len(yes)),data['selected_slot']]
    category=np.where(data['GT_uncertain'],9,8).astype(np.int8)
    category[yes&~picked]=1;category[picked]=0
    remaining=clear&~yes
    running=np.ones_like(adequate,bool)
    codes=dict(event=2,type=3,timing=4,direction=5,strength=6)
    stages={}
    for stage in order:
        running &= component[stage]
        coverage=running.any(1);stages[stage]=coverage.copy()
        fail=remaining&~coverage;category[fail]=codes[stage];remaining[fail]=False
    category[remaining]=7
    return category,stages,component,adequate

def calculate(past, future, paths, logits, ids, family, protocol):
    n=len(ids)
    assert paths.shape==(n,20,12,2) and past.shape==(n,8,2) and future.shape==(n,12,2)
    assert len(np.unique(ids))==n
    assert all(np.isfinite(v).all() for v in (paths,past,future,logits))
    # All source coordinates are relative to the forecast origin. No extra origin
    # shift is applied; original normalized encoder past ends at zero.
    assert np.max(np.abs(past[:,-1]))<1e-6
    truth=describe(past,future)
    predicted=describe(np.repeat(past,20,axis=0),paths.reshape(-1,12,2))
    err=np.linalg.norm(paths-future[:,None],axis=-1)
    ade=err.mean(-1);fde=err[:,:,-1]
    data=dict(sample_id=ids,scene_family=family,selected_slot=logits.argmax(1).astype(np.int8),
      ade=ade,fde=fde,early_error=err[:,:,:4].mean(-1),middle_error=err[:,:,4:8].mean(-1),late_error=err[:,:,8:].mean(-1),
      candidate_event=predicted['y'].reshape(n,20),candidate_kind=predicted['kind'].reshape(n,20),
      candidate_onset=predicted['onset_step'].reshape(n,20),candidate_direction=predicted['response_direction'].reshape(n,20),
      candidate_strength=predicted['response_strength'].reshape(n,20),candidate_types=predicted['event_types'].reshape(n,20,4),
      candidate_uncertain=predicted['uncertain'].reshape(n,20),GT_event=truth['y'],GT_uncertain=truth['uncertain'],
      GT_kind=truth['kind'],GT_onset=truth['onset_step'],GT_direction=truth['response_direction'],GT_strength=truth['response_strength'],
      GT_types=truth['event_types'],speed_curve_mae=np.abs(predicted['future_speed'].reshape(n,20,12)-truth['future_speed'][:,None]).mean(-1),
      angle_curve_mae=np.abs(((predicted['signed_angle'].reshape(n,20,12)-truth['signed_angle'][:,None]+180)%360)-180).mean(-1))
    for j,threshold in enumerate(protocol['path_thresholds']):
        data['class_'+str(j)]=partition(data,protocol,threshold)[0]
        data['alternate_class_'+str(j)]=partition(data,protocol,threshold,protocol['alternate_hierarchy'])[0]
    return data

def summary(data,protocol):
    clear=data['GT_event'].astype(bool)&~data['GT_uncertain'].astype(bool)
    groups={'overall':np.ones(len(clear),bool),'clear_event':clear,
      'no_clear_event':~data['GT_event'].astype(bool)&~data['GT_uncertain'].astype(bool),
      'boundary_uncertain':data['GT_uncertain'].astype(bool)}
    for name in np.unique(data['scene_family']):
        groups['scene_'+str(name)]=data['scene_family']==name
        groups['clear_scene_'+str(name)]=clear&(data['scene_family']==name)
    for kind in (1,2,3,4):groups['GT_kind_'+str(kind)]=clear&(data['GT_kind']==kind)
    rows=[]
    ar=np.arange(len(clear));slot=data['selected_slot']
    joint=(data['ade']+protocol['joint_FDE_weight']*data['fde']).argmin(1)
    category,stage,components,good=partition(data,protocol,protocol['path_thresholds'][0])
    overlap=(data['candidate_types'].astype(bool)&data['GT_types'][:,None].astype(bool)).any(-1)
    raw_type=components['type'];direction=raw_type&components['direction'];strength=raw_type&components['strength']
    for name,mask in groups.items():
        n=int(mask.sum())
        if not n:continue
        value=dict(group=name,n=n,independent_min_ADE=float(data['ade'][mask].min(1).mean()),
          independent_min_FDE=float(data['fde'][mask].min(1).mean()),
          selected_ADE=float(data['ade'][ar,slot][mask].mean()),selected_FDE=float(data['fde'][ar,slot][mask].mean()),
          joint_oracle_ADE=float(data['ade'][ar,joint][mask].mean()),joint_oracle_FDE=float(data['fde'][ar,joint][mask].mean()),
          joint_oracle_early=float(data['early_error'][ar,joint][mask].mean()),
          joint_oracle_middle=float(data['middle_error'][ar,joint][mask].mean()),joint_oracle_late=float(data['late_error'][ar,joint][mask].mean()),
          joint_oracle_speed_curve_mae=float(data['speed_curve_mae'][ar,joint][mask].mean()),
          joint_oracle_angle_curve_mae=float(data['angle_curve_mae'][ar,joint][mask].mean()),
          clear_event_n=int((clear&mask).sum()))
        for j,threshold in enumerate(protocol['path_thresholds']):
            cat,_,_,adequate=partition(data,protocol,threshold)
            alt=data['alternate_class_'+str(j)]
            value['threshold_'+str(j)]=dict(ADE=threshold[0],FDE=threshold[1],
              samepath_adequate_n=int(adequate[mask].any(1).sum()),selected_adequate_n=int(adequate[ar,slot][mask].sum()),
              partition_counts={label:int(((cat==i)&mask).sum()) for i,label in enumerate(CLASS_NAMES)},
              alternate_order_partition_counts={label:int(((alt==i)&mask).sum()) for i,label in enumerate(CLASS_NAMES)})
        cmask=clear&mask;cn=int(cmask.sum())
        value['event_accounting_denominator']=cn
        if cn:
            value['nested_coverage_counts']={s:int((cov&cmask).sum()) for s,cov in stage.items()}
            value['marginal_same_candidate_coverage_counts']=dict(any_event=int((components['event'].any(1)&cmask).sum()),
              dominant_type=int((raw_type.any(1)&cmask).sum()),any_multitype_overlap=int((overlap.any(1)&cmask).sum()),
              type_direction=int((direction.any(1)&cmask).sum()),type_strength=int((strength.any(1)&cmask).sum()),
              type_timing=int(((raw_type&components['timing']).any(1)&cmask).sum()),
              type_direction_strength=int(((direction&components['strength']).any(1)&cmask).sum()))
            first_same_type=np.where(raw_type,np.abs(data['candidate_onset']-data['GT_onset'][:,None]),999).min(1)
            eligible=cmask&raw_type.any(1)
            value['best_onset_error_given_any_same_type']=dict(n=int(eligible.sum()),
              mean_steps=float(first_same_type[eligible].mean()) if eligible.any() else None,
              missing_same_type_n=int((cmask&~raw_type.any(1)).sum()))
        rows.append(value)
    return rows
