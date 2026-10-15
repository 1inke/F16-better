"""CPU saved-array numerical review; no model/torch/optimizer."""
import sys,os,time,json
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parents[1];sys.path.insert(0,str(H));import run as r
def main():
    start=time.perf_counter();r.guard();result=r.read(H/'RESULT.json');assert result['status']=='PASS_DIAGNOSTIC_ONLY_REMOVED_TAIL_UTILITY';locks=r.read(H/'source_manifest.json')['files'];r.q.verify(locks);r.q.verify(result['output_sha']);maxerr=0.;rows=[];case_sets={}
    for rec in result['records']:
        tag,seed,split=rec['grouping'],rec['seed'],rec['split'];z=r.q.npz(r.q.P/'runs'/tag/f'seed_{seed}'/(split+'.npz'));v=r.q.npz(H/'jobs'/f'{tag}_{seed}_{split}.npz');assert np.array_equal(v['sample_id'],z['sample_id']);N=len(z['sample_id']);rowsidx=np.arange(N)[:,None]
        p=z['paths'].astype(float);e=np.sqrt(((p-z['gt'][:,None])**2).sum(-1));a=e.mean(-1);f=e[:,:,-1];j=np.maximum(a/.2,f/.4);good=(a<=.2)&(f<=.4)
        for name,saved_key,indices in (('native','native_utility',np.tile(np.arange(20),(N,1))),('replacement','replacement_utility',None)):
            if indices is None:
                aa=a[:,:20].copy();ff=f[:,:20].copy();jj=j[:,:20].copy();gg=good[:,:20].copy()
                for x,y in ((aa,a),(ff,f),(jj,j),(gg,good)):x[rowsidx,v['removed_slots']]=y[:,20:]
            else:aa,ff,jj,gg=a[:,:20],f[:,:20],j[:,:20],good[:,:20]
            u=np.empty((N,20,4))
            for k in range(20):
                keep=np.delete(np.arange(20),k);u[:,k,0]=gg.any(1).astype(int)-gg[:,keep].any(1).astype(int)
                for c,x in enumerate((aa,ff,jj),1):u[:,k,c]=np.min(x[:,keep],axis=1)-np.min(x,axis=1)
            maxerr=max(maxerr,float(np.abs(u-v[saved_key]).max()))
        maxerr=max(maxerr,float(np.abs(np.stack((a[:,:20],f[:,:20],j[:,:20]),-1)-v['native_errors']).max()),float(np.abs(np.stack((a[:,20:],f[:,20:],j[:,20:]),-1)-v['learned_errors']).max()))
        vel=np.concatenate((p[:,:,:1],p[:,:,1:]-p[:,:,:-1]),2)/.4;speed=np.sqrt((vel*vel).sum(-1));maxerr=max(maxerr,float(np.abs(speed-v['path_speed']).max()));assert maxerr<=1e-12
        assert r.compare(rec['summary'],r.summarize(z,v))==0
        exported=r.read(H/'jobs'/f'{tag}_{seed}_{split}.cases.json')['cases'];assert r.compare(exported,r.cases(z,v))==0
        cases={int(x['sample_id']):x for x in exported};lostids=set(int(x) for x in v['sample_id'][v['events'][:,0]]);rescueids=set(int(x) for x in v['sample_id'][v['events'][:,1]])
        case_sets[(tag,seed,split)]={'lost':lostids,'rescue':rescueids}
        for group,mask in r.q.masks(z['groups']):
            summaries={}
            for event in ('lost','rescue'):
                m=mask&v['events'][:,r.EVENTS.index(event)];ix=np.flatnonzero(m);ns=v['native_Jslot'][ix];ls=v['learned_Jslot'][ix]
                summaries[event]=dict(n=len(ix),IDs=v['sample_id'][m],tracks=int(np.unique(v['track'][m]).size),J_removed=int(v['native_J_in_removed'][m].sum()),Q_removed=int(v['native_Q_in_removed'][m].sum()),individual_coverage_positive=int((v['native_utility'][m,:,0]>0).any(1).sum()),near_common=int((v['redundancy'][m,0]<=1).sum()),distance_D=r.descript(v['redundancy'][m,0]),meanpath_separation=r.descript(v['redundancy'][m,1]),end_separation=r.descript(v['redundancy'][m,2]),GT_features=r.descript(v['GT_features'][m]),native_J_features=r.descript(v['path_features'][ix,ns]),learned_J_features=r.descript(v['path_features'][ix,20+ls]),all_learned_features=r.descript(v['path_features'][m,20:].reshape(-1,8)),native_J_error=r.descript(v['native_errors'][ix,ns]),learned_J_error=r.descript(v['learned_errors'][ix,ls]),flags_count=v['behavior_flags'][m].sum(0),direction_valid=int(v['learned_J_geometry'][m,3].sum()),turn_valid=int(v['learned_J_geometry'][m,6].sum()),native_slot_hist=np.bincount(ns,minlength=20),native_rank_hist=np.bincount(v['old_rank_by_slot'][ix,ns],minlength=21)[1:])
            rank=rec['summary'][group]['rank_utility'];total=np.sum([rank[str(i)]['sum'] for i in range(1,21)],0);bottom=np.sum([rank[str(i)]['sum'] for i in range(17,21)],0)
            unique_bot=int(((v['native_utility'][mask,:,0]>0)&(v['old_rank_by_slot'][mask]>=17)).any(1).sum());groupneeded=int(v['removed_joint_coverage_utility'][mask].sum())
            rows.append(dict(grouping=tag,seed=seed,split=split,group=group,n=int(mask.sum()),events=summaries,GT_prevalence=rec['summary'][group]['GT_behavior_prevalence'],native_unique_coverage_total=int(total[0]),bottom4_individual_coverage_windows=unique_bot,bottom4_group_necessary_windows=groupneeded,bottom4_utility_sum=bottom,native_total_utility_sum=total,speed_tail=rec['summary'][group]['speed_tail']))
        r.guard()
    unions=[]
    for tag in ('scene_family','target_track','recording_time'):
        for split in ('outer','inner','original_CAL'):
            for event in ('lost','rescue'):
                sets=[case_sets[(tag,seed,split)][event] for seed in (42,43,44)];unions.append(dict(grouping=tag,split=split,event=event,union_unique_ID=len(set.union(*sets)),intersection_all_three=len(set.intersection(*sets)),per_seed_counts=[len(x) for x in sets]))
    r.q.verify(locks);r.q.verify(result['output_sha']);r.guard();r.write('completion_review/SUMMARY.json',dict(rows=rows,unique_ID_seed_overlap=unions,feature_order=r.FEATURES,flag_order=['GTlow_average_speed','GTterminal_stop','learned_direction_error','learned_endpoint_overshoot','learned_arc_excess','GTstop_not_retained','turn_absolute_sum_error'],native_repeated_seeds_not_independent=True));r.write('completion_review/REVIEW.json',dict(status='PASS_CPU_SAVED_ONLY',jobs=27,output_files=108,independent_direct_original_path_error_speed_LOO_max_error=maxerr,all_cases_summaries_exact=True,source571_initial_final_PASS=True,all108outputSHA_PASS=True,no_torch_model_forward_backward_optimizer=True,not_external_review=True,elapsed_seconds=time.perf_counter()-start,resource=r.q.RESOURCE));print('REVIEW_PASS',maxerr,time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
