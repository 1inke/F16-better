"""FIT-only diagnostic readers, not changes to F16 or its generator."""
from common import *
import random, copy
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
from threadpoolctl import threadpool_limits
import joblib

FEATURES=('B1','R1','R2','E1','E2','E2_motion','E3','E4','social')
CS=(.1,1.,10.)

def metrics(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    assert np.isfinite(p).all()
    return dict(n=len(y),prevalence=float(np.mean(y)),AUPRC=float(average_precision_score(y,p))if y.any()else None,
                AUROC=float(roc_auc_score(y,p))if len(np.unique(y))==2 else None,
                Brier=float(np.mean((p-y)**2)),log_loss=float(log_loss(y,p,labels=[0,1])))

def transform_fit(x, mode):
    scaler=StandardScaler().fit(x)
    z=scaler.transform(x).astype(np.float32)
    pca=None
    if mode=='common64' and z.shape[1]>64:
        pca=PCA(n_components=min(64,len(z)-1),svd_solver='randomized',random_state=SEED).fit(z)
        z=pca.transform(z).astype(np.float32)
    return (scaler,pca),z

def transform(pair,x):
    scaler,pca=pair;z=scaler.transform(x).astype(np.float32)
    if pca is not None:z=pca.transform(z).astype(np.float32)
    assert np.isfinite(z).all();return z

def logistic(x,y,C):
    assert len(np.unique(y))==2
    model=LogisticRegression(C=C,solver='lbfgs',max_iter=400,tol=1e-5,random_state=SEED)
    model.fit(x,y)
    return model

def finished(key):
    path=HERE/f'jobs/{key}.done.json'
    if not path.exists():return None
    record=read(f'jobs/{key}.done.json')
    for filename,digest in record['hashes'].items():assert sha(HERE/filename)==digest,filename
    return npz(HERE/record['predictions']), record

def finish(key, ids, y,p, model_file, info):
    predictions=f'jobs/{key}.predictions.npz'
    save_npz(predictions,sample_id=ids,y=y,p=np.asarray(p,dtype=np.float64))
    files=[predictions,model_file]
    record=dict(status='SUCCESS',predictions=predictions,metrics=metrics(y,p),**info,
                hashes={f:sha(HERE/f) for f in files})
    write(f'jobs/{key}.done.json',record)
    return dict(sample_id=ids,y=y,p=p),record

def select_C(x,y,groups,mode):
    group_values=np.unique(groups)
    inner=[]
    for group in group_values:
        tr=groups!=group;va=~tr
        if tr.sum()<64 or len(np.unique(y[tr]))<2 or not va.any():continue
        pair,z=transform_fit(x[tr],mode)
        inner.append((z,y[tr],transform(pair,x[va]),y[va]))
    if not inner:return 1.,dict(status='INSUFFICIENT_INNER_GROUPS_C1_FIXED')
    scores=[]
    for C in CS:
        losses=[];weights=[]
        for a,b,c,d in inner:
            m=logistic(a,b,C)
            losses.append(log_loss(d,m.predict_proba(c)[:,1],labels=[0,1]));weights.append(len(d))
        scores.append(float(np.average(losses,weights=weights)))
    winner=int(np.argmin(scores))
    return CS[winner],dict(inner_groups=len(inner),Cs=list(CS),inner_log_losses=scores,
                          selection='FIT inner scene-family held-out log loss; no CAL/DEV choice')

class TinyProbe(nn.Module):
    def __init__(self,dimension):
        super().__init__();self.network=nn.Sequential(nn.Linear(dimension,32),nn.SiLU(),nn.Linear(32,1))
    def forward(self,x):return self.network(x).squeeze(-1)

def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def restore_rng(value):
    torch.set_rng_state(value['torch'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in value['cuda']]);np.random.set_state(value['numpy']);random.setstate(value['python'])

def train_mlp(z,y,seed,checkpoint=None,epochs=32,stop=None):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
    model=TinyProbe(z.shape[1]).to('cuda:0')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    start=0;history=[]
    if checkpoint and checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cuda:0',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng']);start=saved['completed_epochs'];history=saved['history']
    X=torch.tensor(z,device='cuda:0');Y=torch.tensor(y,dtype=torch.float32,device='cuda:0')
    for epoch in range(start,epochs if stop is None else min(stop,epochs)):
        order=torch.randperm(len(z),device='cuda:0')
        total=0.
        for begin in range(0,len(z),512):
            rows=order[begin:begin+512];optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(X[rows]),Y[rows])
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step();total+=float(loss)*len(rows)
        history.append(total/len(z))
        if checkpoint:
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),
                       completed_epochs=epoch+1,history=history,seed=seed,scaler_transformed_dimension=z.shape[1])
            tmp=checkpoint.with_name(checkpoint.name+'.tmp');torch.save(saved,tmp);os.replace(tmp,checkpoint)
    model.eval()
    return model,optimizer,history

def mlp_predict(model,x):
    parts=[]
    with torch.no_grad():
        for begin in range(0,len(x),2048):
            parts.append(torch.sigmoid(model(torch.tensor(x[begin:begin+2048],device='cuda:0'))).cpu().numpy())
    return np.concatenate(parts)

def nested_binary(features, labels, meta):
    (HERE/'jobs').mkdir(exist_ok=True)
    ids=features['sample_id'];y=labels['y'];uncertain=labels['uncertain']
    groups=meta['scene_family'][:len(ids)]
    base=(np.arange(len(ids))<FIT_N)&~uncertain&(groups!='unverified')
    all_groups=np.unique(groups[base])
    assert len(all_groups)>=2, 'No valid scene-grouped evaluation possible'
    totals=len(FEATURES)*2*len(all_groups)+len(FEATURES)*3*len(all_groups)+len(FEATURES)*2
    completed=0;records=[];prediction_sets={}
    prior_ids=[];prior_y=[];prior_p=[]
    for group in all_groups:
        tr=base&(groups!=group);va=base&(groups==group)
        prior_ids.extend(ids[va]);prior_y.extend(y[va]);prior_p.extend(np.full(int(va.sum()),y[tr].mean()))
    save_npz('OOF_prevalence_baseline.npz',sample_id=np.array(prior_ids),y=np.array(prior_y),p=np.array(prior_p))
    prediction_sets['OOF_prevalence_baseline']=dict(metrics=metrics(np.array(prior_y),np.array(prior_p)),file='OOF_prevalence_baseline.npz')
    with threadpool_limits(limits=4):
        for mode in ('full','common64'):
            for name in FEATURES:
                outputs=[]
                for group in all_groups:
                    key=f'logistic_{mode}_{name}_holdout_{group}'
                    hit=finished(key)
                    if hit is None:
                        tr=base&(groups!=group);va=base&(groups==group)
                        stage('PROBE_TRAINING',job=key,completed_jobs=completed,estimated_total_jobs=totals)
                        C,selection=select_C(features[name][tr],y[tr],groups[tr],mode)
                        pair,z=transform_fit(features[name][tr],mode)
                        model=logistic(z,y[tr],C)
                        p=model.predict_proba(transform(pair,features[name][va]))[:,1]
                        path=f'jobs/{key}.joblib';joblib.dump(dict(transform=pair,model=model),HERE/path)
                        hit=finish(key,ids[va],y[va],p,path,dict(probe=name,mode=mode,reader='L2_logistic',heldout_scene_family=str(group),
                                   C=C,selection=selection,train_n=int(tr.sum()),input_dim=features[name].shape[1],transformed_dim=z.shape[1]))
                    outputs.append(hit[0]);records.append(hit[1]);completed+=1
                cat={k:np.concatenate([o[k]for o in outputs])for k in ('sample_id','y','p')}
                key=f'OOF_logistic_{mode}_{name}';save_npz(f'{key}.npz',**cat)
                prediction_sets[key]=dict(metrics=metrics(cat['y'],cat['p']),file=f'{key}.npz')
        # Secondary finite-capacity nonlinear readers. No CAL early stopping.
        for name in FEATURES:
            for seed in (1,2,3):
                outputs=[]
                for group in all_groups:
                    key=f'mlp_full_{name}_seed{seed}_holdout_{group}'
                    hit=finished(key)
                    if hit is None:
                        tr=base&(groups!=group);va=base&(groups==group)
                        stage('PROBE_TRAINING',job=key,completed_jobs=completed,estimated_total_jobs=totals)
                        pair,z=transform_fit(features[name][tr],'full')
                        path=f'jobs/{key}.pt';checkpoint=HERE/path
                        model,_,history=train_mlp(z,y[tr],seed,checkpoint)
                        p=mlp_predict(model,transform(pair,features[name][va]))
                        tf=f'jobs/{key}.transform.joblib';joblib.dump(pair,HERE/tf)
                        hit=finish(key,ids[va],y[va],p,path,dict(probe=name,mode='full',reader='MLP32',seed=seed,
                                epochs=32,heldout_scene_family=str(group),train_n=int(tr.sum()),history=history,
                                transform_file=tf,transform_sha=sha(HERE/tf)))
                    else:
                        assert sha(HERE/hit[1]['transform_file'])==hit[1]['transform_sha']
                    outputs.append(hit[0]);records.append(hit[1]);completed+=1
                cat={k:np.concatenate([o[k]for o in outputs])for k in ('sample_id','y','p')}
                key=f'OOF_mlp_full_{name}_seed{seed}';save_npz(f'{key}.npz',**cat)
                prediction_sets[key]=dict(metrics=metrics(cat['y'],cat['p']),file=f'{key}.npz')
        # Final readers selected using only FIT grouped CV. CAL remains descriptive.
        # Reading CAL metrics does not alter C, epochs, scalers, features or labels.
        cal=(np.arange(len(ids))>=FIT_N)&~uncertain
        dev=npz(HERE/'features/dev.npz');devlabel=npz(HERE/'labels/dev.npz')
        dev_predictions={}
        for mode in ('full','common64'):
            for name in FEATURES:
                key=f'final_logistic_{mode}_{name}'
                hit=finished(key)
                if hit is None:
                    stage('FINAL_FIT_READER',job=key,completed_jobs=completed,estimated_total_jobs=totals)
                    C,selection=select_C(features[name][base],y[base],groups[base],mode)
                    pair,z=transform_fit(features[name][base],mode);model=logistic(z,y[base],C)
                    p=model.predict_proba(transform(pair,features[name][cal]))[:,1]
                    path=f'jobs/{key}.joblib';joblib.dump(dict(transform=pair,model=model),HERE/path)
                    hit=finish(key,ids[cal],y[cal],p,path,dict(probe=name,mode=mode,reader='L2_logistic',C=C,
                        selection=selection,evaluation='CAL descriptive; original overlap is not purged',train_n=int(base.sum())))
                saved=joblib.load(HERE/f'jobs/{key}.joblib')
                dev_predictions[key]=saved['model'].predict_proba(transform(saved['transform'],dev[name]))[:,1]
                records.append(hit[1]);completed+=1
        save_npz('dev_risk_readouts.npz',sample_id=dev['sample_id'],y=devlabel['y'],**dev_predictions)
    write('binary_probes.json',dict(status='PASS_EXPLORATORY_FIT_SCENE_TRANSFER',feature_names=list(FEATURES),
        scene_families=all_groups.tolist(),grouped_OOF_n=int(base.sum()),excluded_uncertain_FIT_n=int(uncertain[:FIT_N].sum()),
        excluded_unverified_FIT_n=int((groups[:FIT_N]=='unverified').sum()),
        fit_prevalence=float(y[base].mean()),outputs=prediction_sets,jobs=records,
        CI=None,Holm_p_values=None,reason='Only three independent scene families. No well-powered group CI or exchangeability-backed permutation null is claimed.',
        limitation='Frozen F16 already saw its original training data. Reader transfer among existing scenes is not new native held-out generalization; feature dimensions and optimization differ.'))
    return base,groups

def secondary_and_controls(features,lab,base,groups):
    """Fixed-C secondary questions; not timing/full-path claims from risk AUPRC."""
    ids=features['sample_id'];rows=[]
    with threadpool_limits(limits=4):
        for name in ('R2','E2_motion','E3'):
            for target in ('kind','onset_bin','direction'):
                y=lab[target];positive=base & lab['y'].astype(bool)
                if target=='direction':positive &= lab['event_types'][:,0].astype(bool)
                predictions=[];truth=[];samples=[]
                for group in np.unique(groups[base]):
                    tr=positive&(groups!=group);va=positive&(groups==group)
                    if not va.any() or len(np.unique(y[tr]))<2:continue
                    pair,z=transform_fit(features[name][tr],'full')
                    model=LogisticRegression(C=1,solver='lbfgs',max_iter=400,random_state=SEED).fit(z,y[tr])
                    pred=model.predict(transform(pair,features[name][va]))
                    predictions.extend(pred);truth.extend(y[va]);samples.extend(ids[va])
                accuracy=float(np.mean(np.array(predictions)==truth))if truth else None
                rows.append(dict(probe=name,target=target,n=len(truth),accuracy=accuracy,
                    classes=np.unique(truth).tolist(),C_fixed=1.,conditioning='Actual future event positives only; not deployable event detection+timing accuracy'))
                save_npz(f'secondary/{name}_{target}.npz',sample_id=np.array(samples),truth=np.array(truth),prediction=np.array(predictions))
        controls=[]
        rng=np.random.default_rng(SEED)
        for name in ('R1','R2','E2_motion','E3'):
            ys=lab['y'].copy()
            for group in np.unique(groups[base]):
                ix=np.flatnonzero(base&(groups==group));ys[ix]=rng.permutation(ys[ix])
            prediction=[];truth=[]
            for group in np.unique(groups[base]):
                tr=base&(groups!=group);va=base&(groups==group)
                pair,z=transform_fit(features[name][tr],'full');model=logistic(z,ys[tr],1.)
                prediction.extend(model.predict_proba(transform(pair,features[name][va]))[:,1]);truth.extend(lab['y'][va])
            controls.append(dict(probe=name,metrics=metrics(np.array(truth),np.array(prediction)),
                role='Destructive within-training-scene label randomization sanity control, NOT an exchangeability-validated permutation p value'))
    write('secondary_and_controls.json',dict(status='PASS_DESCRIPTIVE_ONLY',secondary=rows,label_randomization=controls,
          group_permutation_inference='WITHHELD: variable-size correlated scene families and only three groups; formal exchangeability not established'))

def independent_prediction_audit():
    rows=[]
    for file in sorted((HERE/'jobs').glob('*.done.json')):
        record=json.loads(file.read_text())
        for name,digest in record['hashes'].items():assert sha(HERE/name)==digest
        if 'transform_file'in record:assert sha(HERE/record['transform_file'])==record['transform_sha']
        data=npz(HERE/record['predictions']);m=metrics(data['y'],data['p'])
        for key,value in m.items():
            assert value==record['metrics'][key]or(value is not None and abs(value-record['metrics'][key])<1e-12),(file,key)
        assert len(np.unique(data['sample_id']))==len(data['sample_id'])
        rows.append(dict(job=file.stem,metrics=m))
    write('prediction_audit.json',dict(status='PASS',jobs=len(rows),rows=rows,
        audit_scope='Saved prediction IDs/finiteness/metrics/SHA independently recomputed; not all models replayed a second time'))
    return len(rows)
