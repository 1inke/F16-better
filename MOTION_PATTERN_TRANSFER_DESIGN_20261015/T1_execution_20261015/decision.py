import numpy as np
SEEDS=(42,43,44)
def gates(records,P0,bootstrap):
 A=[next(r for r in records if r['seed']==s and r['arm']=='dynamic_winner') for s in SEEDS]
 B=[next(r for r in records if r['seed']==s and r['arm']=='fixed_motion_responsibility') for s in SEEDS]
 ac=np.array([r['splits']['outer']['change']['coverage'] for r in A]);bc=np.array([r['splits']['outer']['change']['coverage'] for r in B]);pc=P0['outer']['change']['coverage']
 cis={r['reference']:r for r in bootstrap['records'] if r['split']=='outer' and r['group']=='change'}
 parts=dict(B_A_coverage_each_positive=bool(np.all(bc>ac)),B_A_mean_coverage_atleast5pp=bool(np.mean(bc-ac)>=.05),B_A_48family_lower_positive=bool(cis['A_dynamic_winner']['CI_family48'][0][0]>0),B_ownP0_coverage_each_positive=bool(np.all(bc>pc)),B_ownP0_48family_lower_positive=bool(cis['own_P0']['CI_family48'][0][0]>0),B_change_ADEFDE_mean_not_worseA=all(np.mean([b['splits']['outer']['change'][k] for b in B])<=np.mean([a['splits']['outer']['change'][k] for a in A]) for k in ('minADE','minFDE')),B_protected_each_ADEFDE_within_A1percent=all(b['splits']['outer'][g][k]<=a['splits']['outer'][g][k]*1.01 for a,b in zip(A,B) for g in ('overall','stable','transition') for k in ('minADE','minFDE')))
 quality=dict(B_change_coverage_each_atleast_native=all(b['splits']['outer']['change']['rescued']>=b['splits']['outer']['change']['lost'] for b in B),B_change_each_netpositive_ownP0=bool(np.all(bc>pc)),B_protected_each_native_ADEFDE=False,B_all48_seed_split_group_fiveMotion_PASS=all(not b['splits'][s][g]['motion_failed'] for b in B for s in ('FIT','outer','inner','original_CAL') for g in ('overall','stable','transition','change')))
 # Error reference is reconstructed from the saved paired difference, never
 # inferred from population counts or independently selected minima.
 quality['B_protected_each_native_ADEFDE']=all(
  b['splits']['outer'][g][key]<=1.01*(b['splits']['outer'][g][key]-next(r for r in bootstrap['records'] if r['split']=='outer' and r['group']==g and r['reference']=='native20')['perseed_estimate'][i][pos])
  for i,b in enumerate(B) for g in ('overall','stable','transition') for key,pos in (('minADE',1),('minFDE',2)))
 S=all(parts.values());Q=all(quality.values());major={}
 for arm,rs in (('dynamic_winner',A),('fixed_motion_responsibility',B)):
  # This precomputed decision deliberately does not promote baseline: complete
  # native-vs-A CIs/correlation scrutiny is not in this T1 primary family.
  major[arm]=dict(all_fiveMotion=all(not r['splits'][s][g]['motion_failed'] for r in rs for s in ('FIT','outer','inner','original_CAL') for g in ('overall','stable','transition','change')),baseline_upgrade=False)
 return dict(S_T1=bool(S),S_T1_components=parts,Q=bool(Q),Q_components=quality,B_A_coverage_gain_pp=(100*(bc-ac)).tolist(),B_ownP0_coverage_gain_pp=(100*(bc-pc)).tolist(),T2_permitted=bool(S and Q),next_action='PRECISE_T2_PREFLIGHT' if S and Q else 'STOP_THIS_FIXED_RESPONSIBILITY_SERIES',major_constraints_not_relaxed=True,major_precheck=major,baseline_upgrade=False,whole_series_complete=False)
