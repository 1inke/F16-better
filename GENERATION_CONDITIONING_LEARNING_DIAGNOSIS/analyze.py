from pathlib import Path
import json,numpy as np
HERE=Path(__file__).resolve().parent
def main():
    result=json.loads((HERE/'RESULT.json').read_text());assert result['status']=='PASS_FIT_CAL_DESCRIPTIVE_ONLY'
    rows=result['rows'];aggregate=[]
    for split in ('FIT','CAL'):
        for policy in ('constrained','final'):
            for arm in ('zero','mean_std'):
                selected=[r for r in rows if r['split']==split and r['policy']==policy and r['arm']==arm];assert len(selected)==3
                for group in ('overall','stable','transition','change'):
                    gs=[next(g for g in row['groups'] if g['group']==group) for row in selected]
                    entry=dict(split=split,policy=policy,arm=arm,group=group,epochs=[r['epoch'] for r in selected],objective=float(np.mean([r['objective'] for r in selected])))
                    for key in ('minade20','minfde20','joint_oracle','acceleration_audit_mean','acceleration_audit_p95','jerk_audit_mean','jerk_audit_p95','endpoint_spread','duplicate_fraction_10cm'):
                        entry[key]=float(np.mean([g['metrics'][key] for g in gs]));entry['native_'+key]=float(np.mean([g['native'][key] for g in gs]))
                        if key not in ('duplicate_fraction_10cm',):entry['gain_pct_'+key]=100*(1-entry[key]/entry['native_'+key])
                    for key in ('raw_clip_fraction','raw_limit_max','residual_rms','context_drift_rms','mode_drift_rms'):entry[key]=float(np.mean([g[key] for g in gs]))
                    aggregate.append(entry)
    curves=[]
    for history in result['full_calibration_history']:
        a,b=history['rows'][0],history['rows'][-1]
        curves.append(dict(arm=history['arm'],seed=history['seed'],initial_loss=float(a['train_loss']),final_loss=float(b['train_loss']),train_loss_gain_pct=100*(1-float(b['train_loss'])/float(a['train_loss'])),first_cal_change_ade=float(a['change_minade20']),last_cal_change_ade=float(b['change_minade20']),first_cal_change_fde=float(a['change_minfde20']),last_cal_change_fde=float(b['change_minfde20'])))
    temporal=[]
    for row in rows:
        if row['split']=='CAL':
            values=np.array(row['correction_jerk_rms_by_step']);energy=values**2
            # Original anchor knots t4,t8 affect jerk differences around indices2,3,6,7.
            ids=[2,3,6,7];temporal.append(dict(arm=row['arm'],policy=row['policy'],seed=row['seed'],energy_share_at_anchor_knots=float(energy[ids].sum()/max(energy.sum(),1e-30)),profile=values.tolist()))
    output=dict(status='PASS_DESCRIPTIVE_ANALYSIS',aggregate=aggregate,learning_curves=curves,correction_time_profile=temporal,limitations='FIT subset/CAL paired descriptive; no causal attribution, scene-independent CI or production improvement guarantee. Native reference unchanged. No optimizer/dev/test.')
    (HERE/'ANALYSIS.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    for row in aggregate:
        if row['group'] in ('overall','change'):print(json.dumps({k:row[k] for k in ('split','policy','arm','group','epochs','gain_pct_minade20','gain_pct_minfde20','gain_pct_jerk_audit_p95','raw_clip_fraction','context_drift_rms','mode_drift_rms')}))
    print('CURVES '+json.dumps(curves));print('KNOTS '+json.dumps(temporal))
if __name__=='__main__':main()
