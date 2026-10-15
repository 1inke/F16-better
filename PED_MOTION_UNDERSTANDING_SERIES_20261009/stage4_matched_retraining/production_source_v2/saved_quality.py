"""Validation-selected model exports. Future is posthoc only, never forward."""
import importlib.util,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from controls import H,S,atomic_json,fingerprint,guard
spec=importlib.util.spec_from_file_location('e3_cpu_saved_math',S/'completion_review/audit_saved.py');math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)

def save(path,**kw):
    tmp=path.with_suffix('.writing.npz');np.savez_compressed(tmp,**kw);os.replace(tmp,path)
def compute(paths,GT,logits,obs,masks,delta):
    a,f=math.errors(paths,GT);cs=math.summary(paths,GT,math.cv(obs),delta);distinct,effective=math.cluster(paths,GT)
    top=paths[np.arange(len(paths)),logits.argmax(1)];mean=paths.mean(1);cv=math.cv(obs);den=((GT-cv)**2).sum((1,2));records=[]
    for name,m in masks.items():
        if not m.any():continue
        r=dict(subset=name,n=int(m.sum()),**{k:float(v[m].mean()) for k,v in cs.items()},distinct=float(distinct[m].mean()),effectivegood=float(effective[m].mean()),motion=math.motion(paths,m),points={})
        for method,p in dict(top1=top,meanK=mean).items():
            aa,ff=math.errors(p,GT);num=((GT-p)**2).sum((1,2));r['points'][method]=dict(ADE=float(aa[m].mean()),FDE=float(ff[m].mean()),skill=float(1-num[m].sum()/den[m].sum()) if den[m].sum()>0 else None)
        records.append(r)
    return dict(ADE=a,FDE=f,distinct=distinct,effectivegood=effective,**{'candidate_'+k:v for k,v in cs.items()}),records

def export_selected(recovery,s):
    model=s['evaluation_model'];model.eval();before=fingerprint(model);fold=os.environ.get('E3_FOLD','eth')
    loader=s['test_loader']
    if loader is None:loader=DataLoader(Subset(s['base_dataset'],s['val_indices'].tolist()),batch_size=s['args'].batch_size,shuffle=False,num_workers=0)
    selected_test=s['test_loader'] is not None;parts=[];ls=[];obs=[];gs=[];valid=[]
    atomic_json(dict(stage='SELECTED_POSTHOC_FORWARD',fold=fold,branch=recovery.branch,seed=s['args'].seed,pid=os.getpid(),time=time.time(),test=selected_test),H/'progress.json')
    with torch.inference_mode():
        for past,future,n,scene,_,_ in loader:
            recovery.resources.append(guard());o=past.clone();g=(future[:,0]-past[:,0,-1,None]).cpu().numpy().astype(float)
            past=past.to(s['device']);n=n.to(s['device']).int();future=future.to(s['device'])
            d=past[:,0,-1]-past[:,0,-4];sp=torch.linalg.vector_norm(d,dim=-1);u=torch.zeros_like(d);u[:,0]=1.;ok=sp>.03;u[ok]=d[ok]/sp[ok,None]
            headed,_=recovery.module.normalize_to_focal_heading(past,future);center=headed.mean(-2,keepdim=True)
            pred=model(headed-center,None,None,mode='testing',num_valid=n,graph_loc=headed)[2]+center.unsqueeze(-3)
            rel=pred[:,0]-headed[:,0,-1,None,None];world=torch.stack((rel[...,0]*u[:,None,None,0]-rel[...,1]*u[:,None,None,1],rel[...,0]*u[:,None,None,1]+rel[...,1]*u[:,None,None,0]),-1)
            parts.append(world.cpu().numpy().astype(float));ls.append(model.last_mode_logits[:,0].cpu().numpy());obs.append(o.numpy());gs.append(g);valid.append(n.cpu().numpy())
    p=np.concatenate(parts);l=np.concatenate(ls);o=np.concatenate(obs);g=np.concatenate(gs);n=np.concatenate(valid)
    assert p.shape==(len(g),20,12,2) and np.isfinite(p).all() and np.isfinite(l).all() and fingerprint(model)==before
    if selected_test:
        ref=math.load(S/f'data/{fold}_evaluation.npz');z=math.load(S/f'jobs/{fold}_seed1/outputs.npz')
        assert len(ref['ids'])==len(g) and np.array_equal(g,ref['gt']);ids=ref['ids'];tracks=ref['track'];masks={k[5:]:v for k,v in z.items() if k.startswith('mask_')}
        assert np.array_equal(z['sample_id'],ids);delta=math.read(S/f'calibration/{fold}.json')['delta_nonCV_candidate']
        # Information branch does not redefine original GT/E1 strata or P1 CV.
        original_obs=ref['obs'];cv=math.cv(original_obs)
    else:
        ids=np.asarray(s['val_indices']);tracks=np.array(['fixture_'+str(i) for i in ids]);masks=dict(all=np.ones(len(g),bool));delta=.2;original_obs=o;cv=math.cv(o)
    arrays,records=compute(p,g,l,original_obs,masks,delta)
    # Same original raw world-relative distances, with float32 heading roundoff.
    canonical=s['test_metrics'] if selected_test else s['selected_metrics']
    oracle=canonical['oracle'];a,f=arrays['ADE'],arrays['FDE']
    check=dict(min_ade_20=float(a.min(1).mean()),min_fde_20=float(f.min(1).mean()),same_path_fde=float(f[np.arange(len(f)),a.argmin(1)].mean()))
    dif={k:abs(v-oracle[k]) for k,v in check.items()};assert max(dif.values())<=2e-5,('STOP_SAVED_ORIGINAL_METRICS',dif)
    dest=s['run_dir'];npz=dest/'selected_predictions.npz';summary=dest/'saved_quality.json'
    save(npz,sample_id=ids,track=tracks,paths=p,logits=l,GT=g,CV=cv,original_obs=original_obs,model_obs=o,valid=n,**arrays,**{'mask_'+k:v for k,v in masks.items()})
    reread=math.load(npz);rarrays,rrecords=compute(reread['paths'],reread['GT'],reread['logits'],reread['original_obs'],{k[5:]:v for k,v in reread.items() if k.startswith('mask_')},delta)
    for k,v in arrays.items():assert np.array_equal(v,rarrays[k]),k
    assert records==rrecords
    atomic_json(dict(status='PASS_SAVED_QUALITY_CLUSTERS_MOTION',rows=len(p),K=20,selected_by_validation_only=True,GT_prediction_forward=False,all_weights_bool_statbuffers_EXACT=True,original_metrics_difference=dif,full_saved_arithmetic_EXACT=True,records=records,not_external_review=True,fixture=recovery.fixture),summary)
    return [npz,summary]


