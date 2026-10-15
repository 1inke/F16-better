"""CPU saved-data review; reuse registered independent math, no model or torch."""
import sys,json,importlib.util,time
sys.dont_write_bytecode=True
from pathlib import Path
H=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('registered_saved_math',H/'run.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
np=q.np
def write(name,value):
    p=Path(__file__).parent/name
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8')
def main():
    started=q.now();t=time.time();q.guard()
    result=q.read(H/'RESULT.json');lock=q.read(H/'source_manifest.json')['files'];source=q.digest(lock)
    assert result['status']=='PASS_DIAGNOSTIC_ONLY_NATIVE_TEMPORAL_MIX_REACHABILITY'
    assert source==result['source_sha'] and len(lock)==1480
    assert len(result['records'])==60 and len(result['output_sha'])==120
    q.verify(lock);q.verify(result['output_sha'])
    records={(r['split'],r['partner']):r for r in result['records']}
    assert len(records)==60
    table=[];banks=[];details=[];native={};maxerr=0.;geometry=0.
    for split in q.SPLITS:
        z=q.inputs(split);nm,nx=q.metric(z,z['native20'],True);native[split]=nm
        ng=nx['vectors'][:,0].astype(bool);union=ng.copy();shapes=np.zeros(len(ng),bool)
        for j in range(20):
            q.guard();d=H/f'jobs/{split}/partner_{j:02d}';done=q.done_valid(d,source)
            assert done['status']=='PASS' and done['files']=={str(d/'paths.npz'):result['output_sha'][str(d/'paths.npz')],str(d/'metrics.json'):result['output_sha'][str(d/'metrics.json')]}
            saved=q.npz(d/'paths.npz');assert np.array_equal(saved['sample_id'],z['sample_id']) and np.array_equal(saved['slots'],z['slots']) and int(saved['partner'])==j
            assert saved['refined4'].shape==(len(ng),4,12,2) and np.isfinite(saved['refined4']).all()
            p=q.whole(z['native20'],z['slots'],saved['refined4']);im,ix=q.metric(z,p,True)
            original=q.read(d/'metrics.json');assert q.compare(original,records[(split,j)])==0
            for k in ix:assert np.array_equal(ix[k],saved[k]),('SAVED_METRIC_ARRAY',split,j,k)
            rms=q.shape_diagnostic(z['reference4'],saved['residual']);assert np.array_equal(rms,saved['shape_RMS_beyond_global_similarity'])
            assert np.array_equal(saved['refined4']-z['reference4'],saved['residual']) or np.max(abs(saved['refined4']-z['reference4']-saved['residual']))<=2e-15
            aa,ff,jj,cg,_=q.INDEP['error'](saved['refined4'],z['gt']);assert np.array_equal(cg,saved['generated_candidate_good'])
            ge,audit=q.physical_review(z,j,saved);geometry=max(geometry,ge);assert q.compare(audit,original['pointwise_roundoff_not_scientific_epsilon'])==0
            for group,mask in q.masks(z['groups']):
                stored=original['metrics'][group];core={k:stored[k] for k in im[group]}
                err=q.compare(im[group],core);assert err<=1e-12;maxerr=max(maxerr,err)
                good=ix['vectors'][:,0].astype(bool)
                extra=dict(candidate_nonzero_ge1mm_fraction=float((np.linalg.norm(saved['residual'][mask],axis=-1).max(-1)>=.001).mean()),candidate_temporal_shape_ge1mm_fraction=float((rms[mask]>=.001).mean()),alpha_zero_fraction=float((saved['alpha'][mask]==0).mean()),alpha_median=float(np.median(saved['alpha'][mask])),residual_norm_quantiles=np.quantile(np.linalg.norm(saved['residual'][mask],axis=-1),[0,.5,.95,1]),rescued_ids=z['sample_id'][mask&good&~ng].tolist(),lost_ids=z['sample_id'][mask&~good&ng].tolist())
                ee=q.compare(extra,{k:stored[k] for k in extra});assert ee<=1e-12;maxerr=max(maxerr,ee)
                row=dict(split=split,partner=j,group=group,**stored)
                row['delta_minADE']=stored['minADE']-nm[group]['minADE'];row['delta_minFDE']=stored['minFDE']-nm[group]['minFDE'];row['delta_J']=stored['J']-nm[group]['J']
                table.append(row)
            union|=cg.any(1);shapes|=(cg&(rms>=.001)).any(1)
            for i,k in np.argwhere(cg&~ng[:,None]):
                na,nf,nj,_,_=q.INDEP['error'](z['native20'][i:i+1],z['gt'][i:i+1]);ns=int(nj[0].argmin())
                details.append(dict(split=split,sample_id=int(z['sample_id'][i]),group=q.GROUPS[int(z['groups'][i])+1],partner=j,anchor_slot=int(z['slots'][i,k]),native_Jclosest_slot=ns,native_ADE=float(na[0,ns]),native_FDE=float(nf[0,ns]),native_J=float(nj[0,ns]),generated_ADE=float(aa[i,k]),generated_FDE=float(ff[i,k]),generated_J=float(jj[i,k]),shape_RMS=float(rms[i,k]),alpha=float(saved['alpha'][i,k]),position_lambda=float(saved['position_lambda'][i,k]),max_residual_norm=float(np.linalg.norm(saved['residual'][i,k],axis=-1).max()),posthoc_witness_not_GT_deployment_selection=True))
            del p,saved,ix,im
        for group,mask in q.masks(z['groups']):
            b=dict(split=split,group=group,n=int(mask.sum()),native_good=int((ng&mask).sum()),finite_bank_union_good=int((union&mask).sum()),finite_bank_rescued_ids=z['sample_id'][mask&union&~ng].tolist(),beyond_global_similarity_bank_rescued_ids=z['sample_id'][mask&shapes&~ng].tolist(),not_K20_deployment_or_major_gain=True,not_universal_reachability_bound=True)
            old=next(x for x in result['bank_witness'] if x['split']==split and x['group']==group);assert q.compare(b,old)==0;banks.append(b)
    q.verify(result['output_sha']);q.verify(lock);assert 'torch' not in sys.modules
    summary=[]
    for split in q.SPLITS:
        for group in q.GROUPS:
            rows=[x for x in table if x['split']==split and x['group']==group]
            keys=('good','rescued','lost','delta_minADE','delta_minFDE','delta_J','effective_good','effective_distinct','candidate_nonzero_ge1mm_fraction','candidate_temporal_shape_ge1mm_fraction','alpha_zero_fraction','alpha_median')
            summary.append(dict(split=split,group=group,native=native[split][group],ranges={k:[min(x[k] for x in rows),max(x[k] for x in rows)] for k in keys},all5_literal_pass_policies=sum(x['all5_literal_PASS'] for x in rows),policies=20))
    write('ALL_POLICY_GROUPS.json',table);write('WITNESS_DETAILS.json',details);write('SUMMARY.json',dict(summary=summary,bank_witness=banks,native=native))
    write('REVIEW.json',dict(status='PASS',started_at=started,finished_at=q.now(),elapsed_seconds=time.time()-t,scientific_jobs=60,output_SHA_count=120,policy_group_rows=240,related_source_count=1480,max_saved_quality_clusters_motion_diagnostic_error=maxerr,max_geometry_error=geometry,bank_witness_EXACT=True,source_output_initial_final_SHA_unchanged=True,all5_literal_PASS_group_rows=sum(x['all5_literal_PASS'] for x in table),no_torch_model_forward_backward_optimizer=True,reuses_registered_independent_numpy_SciPy_definitions=True,not_external_review_or_all_model_replay=True,formal_increment=0,resource=q.resource))
    print(json.dumps(dict(status='PASS',maxerror=maxerr,geometry=geometry,bank=banks,summary=summary,resource=q.resource)))
if __name__=='__main__':main()
