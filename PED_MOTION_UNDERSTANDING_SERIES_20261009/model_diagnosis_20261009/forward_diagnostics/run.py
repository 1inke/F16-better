import os,sys,time,importlib.util,gc,traceback
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
H=Path(__file__).resolve().parent;P=H.parent.parent;START=time.perf_counter()
sp=importlib.util.spec_from_file_location('forward_saved_utils',P/'claim_evidence_followup_20261009/saved_reader_recovery/analyze_saved.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
Q=a.read(H/'protocol.json');a.H=H;a.Q=Q;a.START=START
sp=importlib.util.spec_from_file_location('forward_stage1',P/'stage1.py');s=importlib.util.module_from_spec(sp);sys.modules[sp.name]=s;sp.loader.exec_module(s)
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def unit(past,variant):
    d=past[:,0,-1]-past[:,0,-4];norm=torch.linalg.vector_norm(d,dim=-1);default=torch.zeros_like(d);default[:,0]=1;threshold=.03
    if variant=='threshold01':threshold=.01
    if variant=='threshold06':threshold=.06
    if variant=='long7':d=past[:,0,-1]-past[:,0,0];norm=torch.linalg.vector_norm(d,dim=-1)
    if variant=='backtrack':
        found=torch.zeros(len(past),dtype=torch.bool,device=past.device);out=default.clone()
        for j in range(7,0,-1):
            dd=past[:,0,j]-past[:,0,j-1];nn=torch.linalg.vector_norm(dd,dim=-1);ok=(nn>.03)&~found;out=torch.where(ok[:,None],dd/nn[:,None].clamp_min(1e-6),out);found|=ok
        return out
    u=torch.where((norm>threshold)[:,None],d/norm[:,None].clamp_min(1e-6),default)
    if variant.startswith('angle'):
        degree=float(variant[5:]);theta=degree*np.pi/180;fixed=torch.tensor([np.cos(theta),np.sin(theta)],dtype=past.dtype,device=past.device);u=torch.where((norm<=.03)[:,None],fixed,u)
    return u
def forward(model,obs,valid,variant):
    outputs=[];logits=[]
    for begin in range(0,len(obs),128):
        a.guard();past=torch.from_numpy(np.array(obs[begin:begin+128],np.float32,copy=True)).cuda();nv=torch.from_numpy(np.array(valid[begin:begin+128],np.int64,copy=True)).cuda();u=unit(past,variant);c,ss=u[:,0],u[:,1];matrix=torch.stack((c,-ss,ss,c),-1).reshape(len(past),1,1,2,2);origin=past[:,:1,-1:];local=torch.matmul((past-origin).unsqueeze(-2),matrix).squeeze(-2);center=local.mean(-2,keepdim=True)
        with torch.no_grad():pred=model(local-center,None,None,mode='testing',num_valid=nv,graph_loc=local)[2];lg=model.last_mode_logits[:,0].clone()
        rel=pred[:,0]+center[:,0,None]-local[:,0,-1,None,None];world=torch.stack((rel[...,0]*u[:,None,None,0]-rel[...,1]*u[:,None,None,1],rel[...,0]*u[:,None,None,1]+rel[...,1]*u[:,None,None,0]),-1);outputs.append(world.cpu().numpy().astype(float));logits.append(lg.cpu().numpy())
    pp=np.concatenate(outputs);ll=np.concatenate(logits);assert pp.shape==(len(obs),20,12,2) and np.isfinite(pp).all() and np.isfinite(ll).all();return pp,ll
def transform(x,kind):
    if kind=='rotate':
        th=Q['rotate_degrees']*np.pi/180;R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]],float)
    else:R=np.array([[1,0],[0,-1]],float)
    return np.einsum('...j,ij->...i',x.astype(float),R).astype(np.float32),R
def matching(p,q):
    same=np.linalg.norm(p-q,axis=-1).mean((1,2));value=np.empty(len(p))
    for begin in range(0,len(p),128):
        a.guard();C=np.linalg.norm(p[begin:begin+128,:,None]-q[begin:begin+128,None,:],axis=-1).mean(-1)
        for k,c in enumerate(C):rr,cc=linear_sum_assignment(c);value[begin+k]=c[rr,cc].mean()
    return same,value
def predictions(p,l):return p[np.arange(len(p)),l.argmax(1)],p.mean(1)
def metrics(p,GT,CV,t,m):
    aa,ff=a.err(p,GT);den=((GT-CV)**2).sum((1,2));num=((GT-p)**2).sum((1,2));return dict(n=int(m.sum()),tracks=len(np.unique(t[m])),skill=a.ci(num[m],den[m],t[m],True),ADE=a.mean_ci(aa[m],t[m]),FDE=a.mean_ci(ff[m],t[m]),coverage=a.mean_ci(((aa<=.2)&(ff<=.4))[m].astype(float),t[m]))
def main():
    a.guard();assert torch.cuda.is_available(),'STOP_NO_GPU'
    for mf in [P/'source_manifest.json',P/'stage2_raw_interventions/source_manifest.json',P/'stage2_raw_interventions/completion_review/input_manifest.json']:
        a.lock(mf)
        for p,d in a.read(mf)['files'].items():
            if p in a.EXPECTED:assert a.EXPECTED[p]==d
            a.EXPECTED[p]=d
    # Verify all current related source/calibration/checkpoint inputs before model use.
    for p,d in a.read(P/'source_manifest.json')['files'].items():a.lock(p,d)
    for p in [Path(__file__),H/'DESIGN.md',H/'protocol.json',P/'stage1.py']:a.lock(p)
    rows=[];social=[];audit=[];d1rows=[]
    for fold in Q['folds']:
        data=a.load(P/f'stage2_raw_interventions/data/{fold}_selected.npz');obs=data['obs'];valid=data['valid'];GT=data['gt'];CV=a.cv(obs);t=data['track'];fallback=np.linalg.norm(obs[:,0,-1]-obs[:,0,-4],axis=-1)<=.03
        originalall=a.load(P/f'jobs/{fold}_seed1/outputs.npz');evaldata=a.load(P/f'data/{fold}_evaluation.npz');lookup={int(ID):i for i,ID in enumerate(evaldata['ids'])};sel=np.array([lookup[int(ID)] for ID in data['ids']]);P1=originalall['mask_nonCV_cue_positive'][sel];C3=a.neighborhood(obs,valid)['C3'];groups=dict(all=np.ones(len(t),bool),P1=P1,fallback=fallback,no_fallback=~fallback)
        vel=(obs[:,0,-1]-obs[:,0,-2]).astype(float);nn=np.linalg.norm(vel,axis=1);defined=nn>=.02;u=np.divide(vel,nn[:,None],out=np.zeros_like(vel),where=nn[:,None]>0);normal=np.stack((-u[:,1],u[:,0]),1);mirror=obs.astype(float).copy();rel=mirror[:,1:]-mirror[:,:1,-1:];projection=(rel*normal[:,None,None]).sum(-1);mirror[:,1:]-=2*projection[...,None]*normal[:,None,None];mirror=mirror.astype(np.float32);none=obs.copy();none[:,1:]=obs[:,:1];nonevalid=np.ones_like(valid)
        before=a.neighborhood(obs,valid);after=a.neighborhood(mirror,valid);# independently compare target-relative CV closest distance under orthogonal reflection.
        def closest(x):
            vv=(x[:,:,-1]-x[:,:,-2]).astype(float);xx=x[:,:,-1].astype(float);pr=xx[:,:,None]+vv[:,:,None]*np.arange(1,13)[None,None,:,None];legal=np.arange(1,9)[None]<valid[:,None];return np.where(legal[:,:,None],np.linalg.norm(pr[:,1:]-pr[:,:1],axis=-1),np.inf).min((1,2))
        bclose=closest(obs);aclose=closest(mirror);ok=defined&np.isfinite(bclose);err=float(abs(bclose[ok]-aclose[ok]).max()) if ok.any() else 0.;assert err<=2e-5,('STOP_MIRROR_CLOSEST',fold,err)
        assert np.array_equal(mirror[:,0],obs[:,0]);a.save(f'RAW_{fold}.npz',sample_id=data['ids'],track=t,GT=GT,CV=CV,obs=obs,mirror_neighbors=mirror,valid=valid,axis=u,normal=normal,axis_defined=defined,closest_original=bclose,closest_mirror=aclose,fallback=fallback,mask_P1=P1,C3=C3)
        for seed in Q['seeds']:
            a.guard();model,trainer,cfg=s.build(fold,seed);snapshot=s.snapshot(model);old=a.old_job(P/f'stage2_raw_interventions/jobs/{fold}_seed{seed}/noNbr',2);assert np.array_equal(data['ids'],old['sample_id']);base,logs=forward(model,obs,valid,'original');ep=float(abs(base-old['base_paths']).max());el=float(abs(logs-old['base_logits']).max());assert max(ep,el)<=Q['cache_alignment_gate'],('STOP_GPU_CACHE_ALIGNMENT',fold,seed,ep,el)
            if not audit:a.write('PREFLIGHT.json',dict(status='PASS_FIRST_GPU_MODEL_ACTUAL_BASE_CACHE_ALIGNMENT',fold=fold,seed=seed,path_max=ep,logits_max=el,not_all_15_model_preflight=True))
            bp,bm=predictions(base,logs);angular=[]
            for name in Q['heading_variants']:
                pp,ll=(base,logs) if name=='original' else forward(model,obs,valid,name);top,mean=predictions(pp,ll);d1rows.extend(dict(fold=fold,seed=seed,variant=name,method=method,subset=tag,**metrics(pred,GT,CV,t,m)) for method,pred in [('top1',top),('meanK',mean)] for tag,m in groups.items())
                a.save(f'{fold}_seed{seed}_{name}.npz',sample_id=data['ids'],paths=pp,logits=ll)
                for tr in ('rotate','mirror'):
                    transformed,R=transform(obs,tr);qp,ql=forward(model,transformed,valid,name);inverse=np.einsum('...j,ji->...i',qp,R);same,matched=matching(pp,inverse)
                    for tag,m in groups.items():rows.append(dict(fold=fold,seed=seed,variant=name,transform=tr,subset=tag,n=int(m.sum()),same_slot_mean=a.mean_ci(same[m],t[m]),matched_mean=a.mean_ci(matched[m],t[m]),same_slot_p95=float(np.quantile(same[m],.95)) if m.any() else None,matched_p95=float(np.quantile(matched[m],.95)) if m.any() else None))
                    a.save(f'{fold}_seed{seed}_{name}_{tr}.npz',sample_id=data['ids'],inverse_paths=inverse,logits=ql,same_slot=same,matched=matched)
            for angle in Q['angles_degrees']:
                pp,ll=forward(model,obs,valid,'angle'+str(angle));ap=predictions(pp,ll)[0];angular.append(ap);a.save(f'{fold}_seed{seed}_angle{angle}.npz',sample_id=data['ids'],paths=pp,logits=ll)
            ensemble=np.mean(angular,axis=0);amp=np.linalg.norm(np.stack(angular)-ensemble[None],axis=-1).mean((0,2));a.save(f'{fold}_seed{seed}_TTA.npz',sample_id=data['ids'],top1_average=ensemble,angular_path_deviation=amp)
            for tag,m in groups.items():d1rows.append(dict(fold=fold,seed=seed,variant='four_fixed_orientation_top1_TTA_not_original_candidate',method='top1_average',subset=tag,angular_variation_mean=a.mean_ci(amp[m],t[m]),angular_variation_p95=float(np.quantile(amp[m],.95)) if m.any() else None,**metrics(ensemble,GT,CV,t,m)))
            mp,ml=forward(model,mirror,valid,'original');no,nl=forward(model,none,nonevalid,'original');nt,nm=predictions(no,nl);mt,mm=predictions(mp,ml)
            for method,bpred,mpred,npred in [('top1',bp,mt,nt),('meanK',bm,mm,nm)]:
                l=((bpred[:,-1]-npred[:,-1])*normal).sum(1);lp=((mpred[:,-1]-npred[:,-1])*normal).sum(1);den=abs(l)+abs(lp)
                for tag,m in [('all',defined),('P1',defined&P1),('C3',defined&C3)]:social.append(dict(fold=fold,seed=seed,method=method,subset=tag,n=int(m.sum()),undefined_axis=int((~defined).sum()),A=a.ci((l+lp)[m],den[m],t[m]),magnitude=a.mean_ci(den[m],t[m]),absolute_antisymmetry_error=a.ci(abs(l+lp)[m],den[m],t[m]),zero_response=int(((den==0)&m).sum())))
                a.save(f'{fold}_seed{seed}_SOCIAL_{method}.npz',sample_id=data['ids'],l=l,lprime=lp,axis_defined=defined)
            a.save(f'{fold}_seed{seed}_SOCIAL_paths.npz',sample_id=data['ids'],mirror_paths=mp,mirror_logits=ml,noNbr_paths=no,noNbr_logits=nl)
            s.exact(model,snapshot);audit.append(dict(fold=fold,seed=seed,base_cache_path_max=ep,base_cache_logits_max=el,all_weights_bool_statbuffers_exact=True,grad_none=True,mirror_closest_error=err));del model,snapshot;gc.collect();torch.cuda.empty_cache();print('FORWARD',fold,seed,round(time.perf_counter()-START,2),flush=True)
    for name,rr in [('D1b',d1rows),('D1b_GEOMETRY',rows),('D5b',social)]:a.emit(name,dict(records=rr,exploratory=True))
    for p,d in a.LOCKS.items():a.guard();assert a.sha(p)==d,('FINAL_SHA',p)
    a.write('FROZEN_AUDIT.json',dict(status='PASS_ALL_15_MODELS',records=audit));a.write('input_manifest.json',dict(files=a.LOCKS));a.write('RESULT.json',dict(status='PASS_FROZEN_D1B_D5B_GPU',models=15,input_files=len(a.LOCKS),output_sha={str(p):a.sha(p) for p in H.glob('*.npz')},summary_sha=a.OUT,resource=a.RES,elapsed_seconds=time.perf_counter()-START,no_training=True))
if __name__=='__main__':
    try:main()
    except Exception:a.write('FAILURE.json',dict(status='STOP',traceback=traceback.format_exc(),resource=a.RES,elapsed_seconds=time.perf_counter()-START));raise
