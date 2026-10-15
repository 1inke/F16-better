import os,sys,time,json,importlib.util,gc,traceback
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent.parent;V=P/'claim_evidence_v2_20261009/numeric_fixture_recovery'
sp=importlib.util.spec_from_file_location('probe_saved_utils',P/'claim_evidence_followup_20261009/saved_reader_recovery/analyze_saved.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
sp=importlib.util.spec_from_file_location('probe_old_v2',V/'analyze_v2.py');v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v);v.a=a
Q=a.read(H/'protocol.json');a.Q=Q;a.H=H;a.START=time.perf_counter()
def features(obs,valid):
    x=obs.astype(float);vv=np.diff(x,axis=2);vel=vv[:,0];acc=np.diff(vel,axis=1);speed=np.linalg.norm(vel,axis=-1);dot=(vel[:,:-1]*vel[:,1:]).sum(-1);cross=vel[:,:-1,0]*vel[:,1:,1]-vel[:,:-1,1]*vel[:,1:,0];ang=np.arctan2(cross,dot);selfX=np.concatenate((vel.reshape(len(x),-1),acc.reshape(len(x),-1),speed,np.sin(ang),np.cos(ang),(speed<.02).mean(1)[:,None],x[:,0,-1]-x[:,0,0]),1)
    legal=np.arange(1,9)[None]<valid[:,None];rel=x[:,1:,-1]-x[:,:1,-1];vr=vv[:,1:,-1]-vv[:,:1,-1];diff=rel[:,:,None]+vr[:,:,None]*np.arange(1,13)[None,None,:,None];ds=np.linalg.norm(diff,axis=-1);dd=ds.min(2);ti=ds.argmin(2)+1;raw=np.concatenate((rel,vr,dd[:,:,None],ti[:,:,None]),2);n=legal.sum(1);mu=(raw*legal[:,:,None]).sum(1)/np.maximum(n[:,None],1);std=np.sqrt((((raw-mu[:,None])**2)*legal[:,:,None]).sum(1)/np.maximum(n[:,None],1));mini=np.where(legal,dd,np.inf).min(1);mini=np.where(n>0,mini,0);social=np.concatenate((mu,std,mini[:,None],n[:,None]),1)
    assert np.isfinite(selfX).all() and np.isfinite(social).all();return selfX,np.concatenate((selfX,social),1)
def fit(X,Y):
    mu=X.mean(0);sd=X.std(0);sd=np.where(sd<1e-6,1,sd);ym=Y.mean(0);z=(X-mu)/sd;W=np.linalg.solve(z.T@z/len(z)+np.eye(z.shape[1])*Q['ridge_alpha_normalized_covariance'],z.T@(Y-ym)/len(z));return mu,sd,ym,W
def predict(X,model):mu,sd,ym,W=model;return (X-mu)/sd@W+ym
def main():
    a.guard()
    for mf in [V/'input_manifest.json',P/'stage3_knn_memory/completion_review/input_manifest.json']:
        a.lock(mf);a.EXPECTED.update(a.read(mf)['files'])
    for p in [Path(__file__),H/'DESIGN.md',H/'protocol.json',V/'analyze_v2.py',P/'claim_evidence_followup_20261009/saved_reader_recovery/analyze_saved.py']:a.lock(p)
    X=np.arange(40.).reshape(20,2);Y=X[:,:1];m=fit(X,Y);assert np.isfinite(predict(X,m)).all();a.write('PREFLIGHT.json',dict(status='PASS_FIXED_RIDGE_FIXTURE',no_torch=True))
    r4=[];r5=[];info=[]
    for fold in Q['folds']:
        a.guard();b=a.load(P/f'stage3_knn_memory/data/{fold}_bank.npz');d=a.load(P/f'data/{fold}_evaluation.npz');assert not np.intersect1d(b['track'],d['track']).size
        rp=a.D/f'{fold}_data_train.npy';cp=a.D/f'{fold}_num_train.npy';a.lock(rp);a.lock(cp);raw=np.load(rp,mmap_mode='r');count=np.load(cp,mmap_mode='r');obs=np.array(raw[b['ids'],:,:8,-2:],np.float32);valid=np.minimum(count[b['ids']].astype(int),9);assert np.array_equal(obs[:,0],b['obs'])
        trainX=features(obs,valid);testX=features(d['obs'],d['valid']);cv=a.cv(obs);CV=a.cv(d['obs']);res=(b['gt']-cv).reshape(len(cv),24);mean=res.mean(0);cov=(res-mean).T@(res-mean)/len(res);evals,evec=np.linalg.eigh(cov);basis=evec[:,-2:][:,::-1];components=(res-mean)@basis;fraction=float(evals[-2:].sum()/evals.sum());trainnb=a.neighborhood(obs,valid);testnb=a.neighborhood(d['obs'],d['valid']);trmask=trainnb['C3']&trainnb['away_valid'];mask=testnb['C3']&testnb['away_valid'];g=((b['gt'][:,-1]-cv[:,-1])*trainnb['away']).sum(1);gg=((d['gt'][:,-1]-CV[:,-1])*testnb['away']).sum(1)
        pred=[];gp=[];parameters={};meanbaseline=float(g[trmask].mean());c1,c2,c3=v.cues(d['obs'],d['valid']);t=d['track']
        for j,name in enumerate(('self','self_social')):
            m=fit(trainX[j],components);p=CV+(predict(testX[j],m)@basis.T+mean).reshape(-1,12,2);gm=fit(trainX[j][trmask],g[trmask,None]);pg=predict(testX[j],gm)[:,0];pred.append(p);gp.append(pg)
            for target,model in [('PCA',m),('g',gm)]:
                for key,z in zip(('mu','std','target_mean','weights'),model):parameters[name+'_'+target+'_'+key]=z
            den=((d['gt']-CV)**2).sum((1,2));num=((p-d['gt'])**2).sum((1,2));gn=(pg-gg)**2;gden=(gg-meanbaseline)**2
            for seed in Q['seeds']:
                old=a.old_job(P/f'jobs/{fold}_seed{seed}');assert np.array_equal(d['ids'],old['sample_id']);tp=old['model_paths'][np.arange(len(t)),old['model_logits'].argmax(1)];tn=((tp-d['gt'])**2).sum((1,2));P1=old['mask_nonCV_cue_positive']
                for tag,mm in [('all',np.ones(len(t),bool)),('P1',P1),('C1',c1),('C2',c2),('nonCV_C1',old['mask_nonCV']&c1),('nonCV_C2',old['mask_nonCV']&c2)]:r4.append(dict(fold=fold,seed=seed,probe=name,subset=tag,n=int(mm.sum()),tracks=len(np.unique(t[mm])),skill=a.ci(num[mm],den[mm],t[mm],True),skill_gain_top1=a.ci((tn-num)[mm],den[mm],t[mm]),PCA_train_variance_fraction=fraction))
            r5.append(dict(fold=fold,probe=name,n=int(mask.sum()),tracks=len(np.unique(t[mask])),TRAIN_C3_n=int(trmask.sum()),TRAIN_g_mean=meanbaseline,skill_vs_TRAIN_mean=a.ci(gn[mask],gden[mask],t[mask],True),skill_vs_zero=a.ci(gn[mask],gg[mask]**2,t[mask],True),mean_absolute_error=a.mean_ci(abs(pg-gg)[mask],t[mask])))
        g0=(gp[0]-gg)**2;g1=(gp[1]-gg)**2;r5.append(dict(fold=fold,probe='social_minus_self',n=int(mask.sum()),paired_MSE_gain=a.mean_ci((g0-g1)[mask],t[mask]),gain_vs_TRAIN_mean=a.ci((g0-g1)[mask],((gg-meanbaseline)**2)[mask],t[mask])))
        a.save(f'PROBES_{fold}.npz',sample_id=d['ids'],track=t,GT=d['gt'],CV=CV,pred_self=pred[0],pred_social=pred[1],g=gg,g_self=gp[0],g_social=gp[1],g_train_mean=np.array(meanbaseline),C1=c1,C2=c2,C3=testnb['C3'],g_mask=mask,mask_P1=P1,train_ids=b['ids'],train_tracks=b['track'],PCA_basis=basis,PCA_mean=mean,PCA_train_variance_fraction=np.array(fraction),**parameters)
        info.append(dict(fold=fold,train_n=len(obs),test_n=len(t),features_self=trainX[0].shape[1],features_social=trainX[1].shape[1],PCA_train_variance_fraction=fraction,train_test_tracks_disjoint=True))
        del obs,raw,count;gc.collect();print('PROBE',fold,round(time.perf_counter()-a.START,2),flush=True)
    a.emit('D4',dict(records=r4));a.emit('D5a',dict(records=r5));a.emit('FIT_INFO',dict(records=info))
    for p,s in a.LOCKS.items():a.guard();assert a.sha(p)==s
    a.write('input_manifest.json',dict(files=a.LOCKS));a.write('RESULT.json',dict(status='PASS_FIXED_LIGHTWEIGHT_D4_D5A',no_neural_training_or_forward=True,output_sha={str(p):a.sha(p) for p in H.glob('*.npz')},summary_sha=a.OUT,input_files=len(a.LOCKS),resource=a.RES,elapsed_seconds=time.perf_counter()-a.START))
if __name__=='__main__':
    try:main()
    except Exception:a.write('FAILURE.json',dict(status='STOP',traceback=traceback.format_exc(),resource=a.RES));raise
