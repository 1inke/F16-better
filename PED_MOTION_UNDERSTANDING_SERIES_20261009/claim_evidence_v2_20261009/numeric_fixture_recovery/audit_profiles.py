"""A3/A4 saved-profile review, plus source-based heading-fallback attribution."""
import sys,importlib.util,time
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('v2_profiles',H/'analyze_v2.py');v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
a=v.a;P=v.P;S=v.S;checks=0
def same(x,y):
    global checks
    checks+=1
    if x is None or y is None:assert x is None and y is None;return
    assert np.allclose(x,y,rtol=1e-12,atol=1e-10),(x,y)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
    manifest=a.read(H/'input_manifest.json')['files'];result=a.read(H/'RESULT.json')
    for p,d in manifest.items():a.guard();assert a.sha(p)==d
    A3=a.read(H/'A3.json')['records'];A4=a.read(H/'A4.json')['records'];cal=a.read(H/'CALIBRATION.json');heading=[]
    d=load(P/'data/univ_evaluation.npz');c1,c2,c3=v.cues(d['obs'],d['valid']);pattern=c1.astype(int)+2*c2.astype(int)+4*c3.astype(int)
    for seed in (1,2,3):
        z=load(H/f'A3_univ_seed{seed}.npz');r=load(H/f'A1A2_univ_seed{seed}.npz');assert np.array_equal(z['pattern'],pattern) and np.array_equal(z['sample_id'],d['ids'])
        for rec in [x for x in A3 if x['seed']==seed]:
            tag=rec['group'];non=z['mask_nonCV'];masks=dict(nonCV=non,P1=z['mask_P1'],nonCV_no_cue=non&(pattern==0),C1_overlap=non&c1,C2_overlap=non&c2,C3_overlap=non&c3)
            if tag.startswith('pattern_'):m=non&(pattern==int(tag[8:],2))
            elif tag.startswith('no_near_speed'):
                s,e=tag[len('no_near_speed'):].split('_CVerror');m=z['no_near']&(z['speed_bin']==int(s))&(z['CV_error_bin']==int(e))
            else:m=masks[tag]
            assert rec['n']==int(m.sum()) and rec['tracks']==len(np.unique(z['track'][m]))
            for key in ('top1','meanK','random','oracle'):
                expected=a.ci(r[key][m],r['den'][m],z['track'][m],True);same(expected['value'],rec['skills'][key]['value']);same(expected['CI'],rec['skills'][key]['CI'])
            same(z['delta_SSE'][m].sum(),rec['delta_SSE_sum'])
    for fold in v.Q['folds']:
        d=load(S/f'data/{fold}_selected.npz');nb=a.neighborhood(d['obs'],d['valid'],cal[fold]['R']);edge=v.edge_proxy(d);lag=np.linalg.norm(d['obs'][:,0,-1]-d['obs'][:,0,-4],axis=1);fallback=lag<=.03
        for seed in v.Q['seeds']:
            rot=load(v.V/f'N1_{fold}_seed{seed}_rotate.npz');mir=load(v.V/f'N1_{fold}_seed{seed}_mirror.npz');res=rot['residual'];features=dict(neighbor_count=d['valid']-1,near_count=nb['density'],speed_mps=nb['speed']/.4,empirical_box_edge_proxy=edge,residual_m=res)
            threshold=np.quantile(rot['dT'],.95);tail=rot['dT']>threshold
            if fold in ('univ','zara02'):
                for label,m in [('rotation_tail',tail),('rotation_remainder',~tail)]:
                    rec=next(x for x in A4 if x['fold']==fold and x['seed']==seed and x['group']==label)
                    for key,x in features.items():
                        rr=v.feature_stats(x,m)
                        for field in ('n','mean','median','p95'):same(rr[field],rec['features'][key][field])
                    heading.append(dict(fold=fold,seed=seed,group=label,n=int(m.sum()),tracks=len(np.unique(d['track'][m])),heading_fallback_n=int((m&fallback).sum()),heading_fallback_fraction=float(fallback[m].mean()),three_step_norm_mean=float(lag[m].mean()),zero_three_step_n=int((m&(lag==0)).sum()),fixed_world_axis_fallback_source='train_front_variant.py:343-363 plus stage1.py forward inverse',posthoc_source_attribution_not_new_forward_or_rootcause_proof=True))
            edges=np.quantile(res,[1/3,2/3]);b=np.searchsorted(edges,res,side='right');ratio=np.divide(mir['dT'],res,out=np.full(len(res),np.nan),where=res>0)
            for j in range(3):
                rec=next(x for x in A4 if x['fold']==fold and x['seed']==seed and x['group']==f'mirror_residual_bin{j}');m=b==j;ok=m&(res>0)
                for key,x,mask in [('path_difference',mir['dT'],m),('ratio',ratio,ok),('residual',res,m)]:
                    rr=v.feature_stats(x,mask)
                    for field in ('n','mean','median','p95'):same(rr[field],rec[key][field])
    source=P.parent/'FRONT_ENCODER_SERIES/train_front_variant.py';a.lock(source)
    a.write('A4_HEADING_FALLBACK.json',dict(records=heading,source_sha=a.sha(source),original_fallback_threshold_m=.03,not_map_boundary_proof=True,new_analysis_exploratory=True))
    for p,d in manifest.items():a.guard();assert a.sha(p)==d
    for p,d in result['output_sha'].items():assert a.sha(p)==d
    a.write('PROFILE_REVIEW.json',dict(status='PASS_ALL90_A3_AND57_A4_PROFILES',checks=checks,input_final_SHA_PASS=True,not_recompute_all_Hungarian_costs=True,no_torch_model=True,elapsed_seconds=time.perf_counter()-a.START,resource=a.RES))
    print('PROFILE AUDIT PASS',checks,flush=True)
if __name__=='__main__':main()
