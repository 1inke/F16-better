import os,sys,time,importlib.util,gc,traceback
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent;F=P/'model_diagnosis_20261009/forward_diagnostics'
spec=importlib.util.spec_from_file_location('v3_parent_forward',F/'run.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
a=f.a;Q=a.read(H/'protocol.json');a.H=H;a.Q=Q;a.START=time.perf_counter();f.Q=Q
torch=f.torch

def closest(x,valid):
    x=x.astype(float);v=x[:,:,-1]-x[:,:,-2];rel=x[:,:1,-1]-x[:,1:,-1];dv=v[:,:1]-v[:,1:];r=rel[:,:,None]+dv[:,:,None]*np.arange(1,13)[None,None,:,None];legal=np.arange(1,9)[None]<valid[:,None];return np.where(legal,np.linalg.norm(r,axis=-1).min(2),np.inf).min(1)
def reflection(x,dtype=np.float64,cast=False):
    y=x.astype(dtype);v=y[:,0,-1]-y[:,0,-2];nn=np.linalg.norm(v,axis=1);u=np.divide(v,nn[:,None],out=np.zeros_like(v),where=nn[:,None]>0);n=np.stack((-u[:,1],u[:,0]),1);rel=y[:,1:]-y[:,:1,-1:];y[:,1:]-=2*(rel*n[:,None,None]).sum(-1)[...,None]*n[:,None,None];return y.astype(np.float32) if cast else y,u,n
def base_coarse(model,obs,valid):
    paths=[];coarses=[];logits=[];hasref=hasattr(model.decoder,'refiner') and bool(getattr(model.decoder,'use_refiner',False))
    for b in range(0,len(obs),128):
        a.guard();past=torch.from_numpy(np.array(obs[b:b+128],np.float32,copy=True)).cuda();nv=torch.from_numpy(np.array(valid[b:b+128],np.int64,copy=True)).cuda();u=f.unit(past,'original');c,ss=u[:,0],u[:,1];M=torch.stack((c,-ss,ss,c),-1).reshape(len(past),1,1,2,2);local=torch.matmul((past-past[:,:1,-1:]).unsqueeze(-2),M).squeeze(-2);center=local.mean(-2,keepdim=True)
        with torch.no_grad():
            pred=model(local-center,None,None,mode='testing',num_valid=nv,graph_loc=local)[2];lg=model.last_mode_logits[:,0].clone();coarse=getattr(model.decoder,'last_coarse_trajectories',None)
            assert coarse is not None and coarse.shape==pred.shape,('STOP_G4_COARSE_SCHEMA',type(model.decoder).__name__)
            def world(z):
                r=z[:,0]+center[:,0,None]-local[:,0,-1,None,None];return torch.stack((r[...,0]*u[:,None,None,0]-r[...,1]*u[:,None,None,1],r[...,0]*u[:,None,None,1]+r[...,1]*u[:,None,None,0]),-1).cpu().numpy().astype(float)
            paths.append(world(pred));coarses.append(world(coarse));logits.append(lg.cpu().numpy())
    p=np.concatenate(paths);c=np.concatenate(coarses);l=np.concatenate(logits);assert np.isfinite(p).all() and np.isfinite(c).all() and np.isfinite(l).all();return p,c,l,hasref
def geometry(obs,valid,fold):
    a.guard();x=obs.astype(float);v=x[:,0,-1]-x[:,0,-2];norm=np.linalg.norm(v,axis=1);defined=norm>=.02;ref,u,n=reflection(obs,np.float64);ref32,u32,n32=reflection(obs,np.float32);cast=ref.astype(np.float32);legal=defined&np.isfinite(closest(obs,valid));old=obs.astype(float).copy();vold=(obs[:,0,-1]-obs[:,0,-2]).astype(float);vo=np.linalg.norm(vold,axis=1);ou=np.divide(vold,vo[:,None],out=np.zeros_like(vold),where=vo[:,None]>0);on=np.stack((-ou[:,1],ou[:,0]),1);r=old[:,1:]-old[:,:1,-1:];old[:,1:]-=2*(r*on[:,None,None]).sum(-1)[...,None]*on[:,None,None];old=old.astype(np.float32)
    before=closest(obs,valid);after=closest(cast,valid);double=closest(ref,valid);oldc=closest(old,valid);M=float(abs(x).max());eps=np.finfo(np.float32).eps;budget=max(Q['geometry_absolute_floor_m'],Q['geometry_k']*eps*M);de=float(abs(before[legal]-double[legal]).max()) if legal.any() else 0;ce=float(abs(before[legal]-after[legal]).max()) if legal.any() else 0;old_err=float(abs(before[legal]-oldc[legal]).max()) if legal.any() else 0
    assert de<=Q['double_geometry_gate_m'],('STOP_DOUBLE_GEOMETRY',fold,de);assert ce<=budget,('STOP_REGISTERED_GEOMETRY_BUDGET',fold,ce,budget);assert np.array_equal(cast[:,0],obs[:,0]);assert np.isfinite(cast).all()
    rec=dict(fold=fold,M_world_m=M,eps32=float(eps),fixed_k=16,budget_m=budget,double_closest_error_m=de,cast_closest_error_m=ce,parent_axis_cast_reproduction_error_m=old_err,cast_error_over_epsM=ce/(eps*M) if M else None,construct_float32_vs_double_max_m=float(abs(ref32.astype(float)-ref).max()),double_to_float32_coordinate_max_m=float(abs(cast.astype(float)-ref).max()),float32_axis_vs_double_axis_max=float(abs(u-u32).max()),defined=int(defined.sum()),undefined=int((~defined).sum()),axis_numeric_change='all_float64_last_velocity_and_unit; original_intent_unchanged',cache_and_logit_gates_not_relaxed=True)
    return cast,u,n,defined,before,after,rec
def main():
    a.guard();assert torch.cuda.is_available()
    for manifest in (P/'source_manifest.json',P/'stage2_raw_interventions/source_manifest.json',P/'model_diagnosis_20261009/partial_forward_review/input_manifest.json'):
        a.lock(manifest)
        for p,d in a.read(manifest)['files'].items():
            if p in a.EXPECTED:assert a.EXPECTED[p]==d
            a.EXPECTED[p]=d
    for p,d in a.read(P/'source_manifest.json')['files'].items():a.lock(p,d)
    for p in (Path(__file__),H/'protocol.json',H/'DESIGN.md',F/'run.py',F/'FAILURE.json'):a.lock(p)
    numbers=[];audit=[]
    # Numerical qualification before loading any model. One fixed post-failure revision.
    for fold in Q['folds']:
        d=a.load(P/f'stage2_raw_interventions/data/{fold}_selected.npz');mi,u,n,defined,bc,ac,rec=geometry(d['obs'],d['valid'],fold);numbers.append(rec)
    a.write('NUMERIC_REVISION.json',dict(status='PASS_FIXED_K16_COORDINATE_CONSTRUCTION',records=numbers,post_failure_explicit_human_revision=True,parent_failure_preserved=True,E8_neural_backend_alignment_still_unresolved=True))
    for fold in Q['folds']:
        d=a.load(P/f'stage2_raw_interventions/data/{fold}_selected.npz');obs=d['obs'];nv=d['valid'];mi,u,n,defined,bc,ac,rec=geometry(obs,nv,fold);none=obs.copy();none[:,1:]=obs[:,:1];nonevalid=np.ones_like(nv);orig=a.load(P/f'jobs/{fold}_seed1/outputs.npz',keys=['sample_id','mask_nonCV_cue_positive']);mapping={int(v):i for i,v in enumerate(orig['sample_id'])};sel=np.array([mapping[int(v)] for v in d['ids']]);P1=orig['mask_nonCV_cue_positive'][sel];fallback=np.linalg.norm(obs[:,0,-1]-obs[:,0,-4],axis=-1)<=.03;C3=a.neighborhood(obs,nv)['C3'];a.save(f'RAW_{fold}.npz',sample_id=d['ids'],track=d['track'],GT=d['gt'],CV=a.cv(obs),obs=obs,valid=nv,mirror_neighbors=mi,axis=u,normal=n,axis_defined=defined,closest_original=bc,closest_mirror=ac,mask_P1=P1,fallback=fallback,C3=C3)
        for seed in Q['seeds']:
            a.guard();model,trainer,cfg=f.s.build(fold,seed);snap=f.s.snapshot(model);before_sha={k:a.sha(p) for k,p in []};base,coarse,logs,hasref=base_coarse(model,obs,nv);old=a.old_job(P/f'stage2_raw_interventions/jobs/{fold}_seed{seed}/noNbr',2);assert np.array_equal(old['sample_id'],d['ids']);ep=float(abs(base-old['base_paths']).max());el=float(abs(logs-old['base_logits']).max());assert max(ep,el)<=Q['cache_alignment_gate'],('STOP_MODEL_CACHE_ALIGNMENT',fold,seed,ep,el)
            a.save(f'G4_{fold}_seed{seed}.npz',sample_id=d['ids'],pre_paths=coarse,post_paths=base,logits=logs,has_refiner=np.array(hasref))
            if fold in ('eth','hotel'):
                # Full old interventions reused, not retrained or overwritten. Fresh base for frozen audit/G4 only.
                source='prior_immutable_partial_interventions'
                oldbase=a.load(F/f'{fold}_seed{seed}_original.npz');assert np.array_equal(base,oldbase['paths']) and np.array_equal(logs,oldbase['logits'])
            else:
                source='new_completed_missing_interventions'
                for name in Q['heading_variants']:
                    pp,ll=(base,logs) if name=='original' else f.forward(model,obs,nv,name);a.save(f'{fold}_seed{seed}_{name}.npz',sample_id=d['ids'],paths=pp,logits=ll)
                    for kind in ('rotate','mirror'):
                        tr,R=f.transform(obs,kind);qp,ql=f.forward(model,tr,nv,name);inv=np.einsum('...j,ji->...i',qp,R);same,matched=f.matching(pp,inv);a.save(f'{fold}_seed{seed}_{name}_{kind}.npz',sample_id=d['ids'],inverse_paths=inv,logits=ql,same_slot=same,matched=matched)
                angular=[]
                for angle in Q['angles_degrees']:
                    pp,ll=f.forward(model,obs,nv,'angle'+str(angle));angular.append(pp[np.arange(len(pp)),ll.argmax(1)]);a.save(f'{fold}_seed{seed}_angle{angle}.npz',sample_id=d['ids'],paths=pp,logits=ll)
                ens=np.mean(angular,axis=0);amp=np.linalg.norm(np.stack(angular)-ens[None],axis=-1).mean((0,2));a.save(f'{fold}_seed{seed}_TTA.npz',sample_id=d['ids'],top1_average=ens,angular_path_deviation=amp);mp,ml=f.forward(model,mi,nv,'original');npred,nl=f.forward(model,none,nonevalid,'original');a.save(f'{fold}_seed{seed}_SOCIAL_paths.npz',sample_id=d['ids'],mirror_paths=mp,mirror_logits=ml,noNbr_paths=npred,noNbr_logits=nl)
            f.s.exact(model,snap);audit.append(dict(fold=fold,seed=seed,all_parameter_bool_statbuffer_values_EXACT=True,state_tensors=len(snap),requires_grad_all_false=True,parameter_grads_all_none=True,base_cache_path_max=ep,base_cache_logits_max=el,has_refiner=hasref,G5_source=source));a.write('FROZEN_AUDIT_PROGRESS.json',dict(records=audit));del model,snap;gc.collect();torch.cuda.empty_cache();print('FROZEN',fold,seed,round(time.perf_counter()-a.START,2),flush=True)
    for p,digest in a.LOCKS.items():a.guard();assert a.sha(p)==digest,('FINAL_SOURCE_SHA',p)
    a.write('FROZEN_AUDIT.json',dict(status='PASS_ALL_15_MODELS_PARAMETER_BOOL_STATBUFFER_EXACT',records=audit));a.write('forward_input_manifest.json',dict(files=a.LOCKS));a.write('RESULT_FORWARD.json',dict(status='PASS_G4_G5_FROZEN_COMPLETION',models=15,new_full_intervention_models=9,old_full_intervention_models_reused=6,output_sha={str(p):a.sha(p) for p in H.glob('*.npz') if p.name.startswith(('G4_','RAW_','eth_','hotel_','univ_','zara'))},resource=a.RES,elapsed_seconds=time.perf_counter()-a.START,no_optimizer_backward_or_training=True))
if __name__=='__main__':
    try:main()
    except Exception:a.write('FAILURE_FORWARD.json',dict(status='STOP',traceback=traceback.format_exc(),resource=a.RES,elapsed_seconds=time.perf_counter()-a.START));raise
