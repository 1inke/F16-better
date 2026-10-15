"""CPU descriptive saved-output contrasts only; not new forward/training or outcome selection."""
import sys,math
sys.dont_write_bytecode=True
import audit_saved as a
import numpy as np
def main():
    a.guard();result=a.read(a.H/'RESULT.json');sources=a.read(a.H/'source_manifest.json')['files'];a.verify(sources);a.verify(result['output_sha']);rows=[]
    for seed in a.SEEDS:
        for split in ('outer','inner','original_CAL'):
            a.guard();zs={k:a.npz(a.H/'runs'/k/f'seed_{seed}/{split}.npz') for k in a.ARMS}
            assert all(np.array_equal(zs['free12'][k],zs['bernstein12'][k]) for k in ('sample_id','gt','groups','native20','slots'))
            measures={}
            for arm,z in zs.items():
                e=a.error(z['refined20'],z['gt']);z['same_complete_good']=e[3].any(1);z['ADE']=e[0].min(1);z['FDE']=e[1].min(1);z['J']=e[2].min(1)
                measures[arm]=a.temporal_check(z,arm)
            for group,mask in a.masks(zs['free12']['groups']):
                n=int(mask.sum());f=zs['free12'];b=zs['bernstein12'];native=a.error(f['native20'][mask],f['gt'][mask]);ng=native[3].any(1);sg={arm:z['same_complete_good'][mask] for arm,z in zs.items()}
                temporal={}
                for arm,z in zs.items():
                    r=z['residual'][mask];u=z['bounded_controls'][mask];x=np.arange(12)/11.;B=np.stack([math.comb(11,i)*x**i*(1-x)**(11-i) for i in range(12)],1);w=u if arm=='free12' else np.einsum('ti,bkid->bktd',B,u)
                    v=np.diff(np.concatenate((np.zeros_like(r[:,:,:1]),r),2),axis=2)/.4;ac=np.diff(v,axis=2)/.4;j=np.diff(ac,axis=2)/.4;bound=.20*(np.arange(1,13)/12.)**2
                    temporal[arm]=dict(control_adjacent_difference_mean=float(np.linalg.norm(np.diff(u,axis=2),axis=-1).mean()),mixed_adjacent_difference_mean=float(np.linalg.norm(np.diff(w,axis=2),axis=-1).mean()),residual_accel_mean=float(np.linalg.norm(ac,axis=-1).mean()),residual_jerk_mean=float(np.linalg.norm(j,axis=-1).mean()),bound_ratio_quantiles=np.quantile(np.linalg.norm(r,axis=-1)/bound[None,None],[0,.5,.95,1]),near_bound_fraction_descriptive_1e_6=float((np.linalg.norm(r,axis=-1)/bound[None,None]>=1-1e-6).mean()),not_fullpath_motion=True)
                rows.append(dict(seed=seed,split=split,group=group,n=n,paired_coverage_binary_different=int((sg['free12']!=sg['bernstein12']).sum()),same_rescue_ids=np.array_equal(f['sample_id'][mask][sg['free12']&~ng],b['sample_id'][mask][sg['bernstein12']&~ng]),same_lost_ids=np.array_equal(f['sample_id'][mask][~sg['free12']&ng],b['sample_id'][mask][~sg['bernstein12']&ng]),paired_path_max_difference=float(abs(f['refined20'][mask]-b['refined20'][mask]).max()),temporal=temporal))
    a.verify(sources);a.verify(result['output_sha']);a.guard();a.write('PAIRED_DETAILS.json',dict(status='PASS',records=rows,all72_seed_split_group_records=True,no_model_forward_gradient_optimizer=True,not_additional_confirmatory_endpoint=True,source_output_SHA_initial_final_PASS=True,resource=a.resource));print('PAIRED_DETAILS_PASS',len(rows),max(x['paired_coverage_binary_different'] for x in rows),flush=True)
if __name__=='__main__':main()

