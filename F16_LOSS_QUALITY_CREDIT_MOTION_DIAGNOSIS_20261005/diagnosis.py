"""Loss-quality-credit-motion diagnostic worker; scientific parameters never updated."""
from pathlib import Path
import json,gc,time
import numpy as np
import torch
import analysis_core as c
from resource_guard import guard
HERE=c.HERE;ROOT=c.ROOT;START=time.perf_counter()
def old_snapshot():
    p=HERE/'old_results_manifest.json'
    if p.exists():return c.read(p)['files']
    prior=ROOT/'F16_PARAMETER_TRANSFER_CONCORDANCE_20261005/old_results_manifest.json'
    files=dict(c.read(prior)['files']);c.verify(files,'INHERITED_OLD_SHA')
    for folder in ROOT.iterdir():
        if folder==HERE or not folder.is_dir() or not folder.name.startswith(('GENERATION_','F16_','HEAD_','PULSE_')):continue
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.tmp','.pending') and p.name!='runtime_state.json' and str(p) not in files:files[str(p)]=c.sha(p)
    c.write(HERE/'old_results_manifest.json',dict(files=files,prior_digests_preserved=True,excluded_controls=['.memory','AUTONOMOUS_GENERATION_CYCLE.json','runtime_state.json','locks/tmp/bytecode']))
    return files
snapshot_old=old_snapshot
def sources(models):
    files=dict(c.read(ROOT/'F16_PARAMETER_TRANSFER_CONCORDANCE_20261005/preflight.json')['source_hashes'])
    c.verify(files,'INHERITED_SOURCE_SHA')
    paths=list(HERE.glob('*.py'))+list(HERE.glob('*.ps1'))+[HERE/p for p in ('README.md','DESIGN.md','DECISION_PLAN.md','protocol.json','old_results_manifest.json')]
    paths+=[ROOT/'F16_PARAMETER_TRANSFER_CONCORDANCE_20261005/diagnosis.py',ROOT/'F16_PARAMETER_TRANSFER_CONCORDANCE_20261005/RESULT.json']
    for p in paths:files[str(p)]=c.sha(p)
    for m in models:files[m['path']]=m['sha256']
    return files
def completed(done,src):
    assert done['status']=='SUCCESS' and done['sources']==src
    for p,h in done['outputs'].items():assert c.sha(p)==h
    return True
def done(key,src,files,record):
    vals={str(p):c.sha(p) for p in files}
    c.write(HERE/'jobs'/(key+'.done.json'),dict(status='SUCCESS',sources=src,outputs=vals,record=record))
    return vals
def npz(path,v):
    np.savez_compressed(path,**v)
    with np.load(path,allow_pickle=False) as z:
        for k,x in v.items():assert equal(x,z[k]),k

def equal(a,b):
    return np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b)

def audit_geometry(v,data,paths,split):
    a,b=c.d.SPLITS[split]
    rebuilt,motion=c.d.metrics(paths,data['future'][a:b],data['logits'][a:b])
    for k,x in rebuilt.items():assert equal(v[k],x),('FULL_GEOMETRY_RECOMPUTE',split,k)
    del rebuilt,motion
def audit_snap(v):
    a=v['ADE20'];f=v['FDE20'];j=np.maximum(a/.2,f/.4);good=(a<=.2)&(f<=.4)
    assert np.array_equal(v['J_joint'],j.min(-1))
    assert np.array_equal(v['coverage'],good.any(-1))
    assert np.array_equal(v['raw_good'],good.sum(-1))
    winner=(a+.5*f).argmin(-1)
    assert np.array_equal(v['quality_winner'],winner)
    assert np.array_equal(v['winner_good'],good[np.arange(len(good)),winner])
    no=np.arange(20)[None,:]!=winner[:,None];assert np.max(v['primary_credit20'][no])==0
    assert np.isfinite(a).all() and np.isfinite(f).all()
    assert np.allclose(v['L_total'],sum(v['L_'+k] for k in c.TERMS if k!='total'),atol=1e-12,rtol=0)
    assert set(np.unique(v['group']))<={0,1,2}
def load_job(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def snapshot_summary(v):
    rows=[]
    for name,mask in c.d.masks(v['group']):
        ws=v['witness_slot'][mask]>=0;dl=v['witness_delta_L'][mask];zero=v['witness_direct_primary_credit'][mask]==0
        rows.append(dict(group=name,n=int(mask.sum()),coverage=int(v['coverage'][mask].sum()),winner_good=int(v['winner_good'][mask].sum()),good_present_winner_bad=int(v['good_present_winner_bad'][mask].sum()),nearest_joint_not_qualitywinner=int((v['J_slot'][mask]!=v['quality_winner'][mask]).sum()),witness_available=int(ws.sum()),witness_finite_loss_down=int((ws&(dl<-1e-8)).sum()),witness_finite_loss_down_no_direct_primary=int((ws&(dl<-1e-8)&zero).sum()),primary_coordinate_credit_nonwinner_max=0.,primary_total=float(v['L_primary'][mask].mean()),total_loss=float(v['L_total'][mask].mean())))
    return rows
def preflight():
    import preflight_impl
    preflight_impl.run(__import__(__name__))
def run():
    pf=c.read(HERE/'preflight.json');assert pf['status']=='PASS'
    src=pf['source_hashes'];old=old_snapshot();c.verify(src,'INITIAL_SOURCE_SHA');c.verify(old,'INITIAL_OLD_SHA')
    data=c.initialize();witness=c.load_witness(data);models=c.d.checkpoints();(HERE/'jobs').mkdir(exist_ok=True)
    records=[];frozen=[]
    for split,(a,b) in c.d.SPLITS.items():
        key='native_'+split;p=HERE/'jobs'/(key+'.npz');dp=HERE/'jobs'/(key+'.done.json')
        if dp.exists():
            v=c.read(dp);completed(v,src);records.append(v['record']);continue
        c.stage('NATIVE_SAVED_SNAPSHOT',split=split,completed_jobs=len(records))
        s=c.snapshot(data,data['base'][a:b],split,witness);audit_snap(s);npz(p,s)
        rec=dict(key=key,kind='snapshot',split=split,summary=snapshot_summary(s))
        done(key,src,[p],rec);records.append(rec);del s
    for number,meta in enumerate(models):
        key=c.pt.key(meta);dp=HERE/'jobs'/(key+'.done.json')
        if dp.exists():
            dd=c.read(dp);completed(dd,src);records.append(dd['record']);frozen.append(dd['record']['frozen']);continue
        c.stage('CHECKPOINT_BEGIN',checkpoint=key,completed_jobs=len(records),total_jobs=29)
        model,info=c.build(meta['family'],Path(meta['path']));state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model=model.to('cuda');names,params,layout=c.pt.setup(model);outputs=[];summaries={}
        for split in ('FIT','CAL'):
            paths=c.load_saved(meta,split);c.stage('SAVED_SNAPSHOT_CREDIT',checkpoint=key,split=split,completed_jobs=len(records))
            s=c.snapshot(data,paths,split,witness);audit_snap(s)
            sp=HERE/'jobs'/(key+'_'+split+'.snapshot.npz');npz(sp,s);outputs.append(sp)
            c.stage('CHANGE_PARAMETER_COMPONENTS',checkpoint=key,split=split,completed_jobs=len(records))
            grad,means,quant,align=c.param_accumulate(model,params,data,split,paths,report=True)
            gp=HERE/'jobs'/(key+'_'+split+'.gradients.npz');npz(gp,{'gradient_'+k:v for k,v in grad.items()});outputs.append(gp)
            summaries[split]=dict(snapshot=snapshot_summary(s),parameter_pairs=c.gradient_summary(grad),component_means=means,quantile_support=quant,replay_alignment=align,scope='CHANGE_GROUP_LOCAL_PRIMARY_TOTAL_POPULATION_CONTRIBUTION_NOT_ADAMW')
            del s,grad,paths;gc.collect();guard.check()
        c.pt.unchanged(model,state)
        fa=dict(key=key,status='EXACT_ALL_WEIGHTS_BOOL_SCALER_BUFFERS_NO_UPDATE',old_requires_grad_false=True,newhead_parameter_count=45798,optimizer=False)
        js=HERE/'jobs'/(key+'.summary.json');c.write(js,dict(model=meta,info=info,parameter_names=names,layout=layout,summaries=summaries,frozen=fa));outputs.append(js)
        rec=dict(key=key,kind='checkpoint',model=meta,epoch=info['epoch'],summaries=summaries,frozen=fa)
        done(key,src,outputs,rec);records.append(rec);frozen.append(fa)
        del model,params,state;gc.collect();torch.cuda.empty_cache()
    for family in ('raw','centered_raw','centered_own'):
        for seed in (1,2,3):
            key=f'edges_{family}_seed{seed}';dp=HERE/'jobs'/(key+'.done.json')
            if dp.exists():
                dd=c.read(dp);completed(dd,src);records.append(dd['record']);continue
            outfiles=[];summaries={}
            for split in ('FIT','CAL'):
                native=load_job(HERE/'jobs'/('native_'+split+'.npz'))
                low=load_job(HERE/'jobs'/(f'{family}_seed{seed}_constrained_{split}.snapshot.npz'))
                high=load_job(HERE/'jobs'/(f'{family}_seed{seed}_final_{split}.snapshot.npz'))
                for contrast,left,right in [('native_to_constrained',native,low),('constrained_to_final',low,high),('native_to_final',native,high)]:
                    v=c.relations(left,right);p=HERE/'jobs'/(key+'_'+split+'_'+contrast+'.npz');npz(p,v);outfiles.append(p)
                    summaries[split+'_'+contrast]=c.relationship_summary(v)
                del native,low,high,v;gc.collect();guard.check()
            p=HERE/'jobs'/(key+'.summary.json');c.write(p,summaries);outfiles.append(p)
            rec=dict(key=key,kind='checkpoint_edges',family=family,seed=seed,summaries=summaries)
            done(key,src,outfiles,rec);records.append(rec);c.stage('CHECKPOINT_EDGES_COMPLETE',completed_jobs=len(records),total_jobs=29)
    c.stage('SAVED_OUTPUT_ARITHMETIC_AUDIT',completed_jobs=29)
    outputs=0
    for rec in records:
        dj=c.read(HERE/'jobs'/(rec['key']+'.done.json'));completed(dj,src);outputs+=len(dj['outputs'])
        if rec['kind']=='snapshot':
            v=load_job(next(p for p in dj['outputs'] if p.endswith('.npz')));audit_snap(v)
            a,b=c.d.SPLITS[rec['split']];audit_geometry(v,data,data['base'][a:b],rec['split']);del v
        if rec['kind']=='checkpoint':
            for split in ('FIT','CAL'):
                s=load_job(HERE/'jobs'/(rec['key']+'_'+split+'.snapshot.npz'));audit_snap(s)
                assert snapshot_summary(s)==rec['summaries'][split]['snapshot']
                paths=c.load_saved(rec['model'],split);audit_geometry(s,data,paths,split);del paths
                g=load_job(HERE/'jobs'/(rec['key']+'_'+split+'.gradients.npz'))
                assert c.gradient_summary({k:g['gradient_'+k] for k in c.TERMS+c.PHYS})==rec['summaries'][split]['parameter_pairs']
                del s,g
        if rec['kind']=='checkpoint_edges':
            for split in ('FIT','CAL'):
                native=load_job(HERE/'jobs'/('native_'+split+'.npz'))
                low=load_job(HERE/'jobs'/(f"{rec['family']}_seed{rec['seed']}_constrained_{split}.snapshot.npz"))
                high=load_job(HERE/'jobs'/(f"{rec['family']}_seed{rec['seed']}_final_{split}.snapshot.npz"))
                for contrast,left,right in [('native_to_constrained',native,low),('constrained_to_final',low,high),('native_to_final',native,high)]:
                    rebuilt=c.relations(left,right);saved=load_job(HERE/'jobs'/(rec['key']+'_'+split+'_'+contrast+'.npz'))
                    assert rebuilt.keys()==saved.keys()
                    for k in saved:assert equal(rebuilt[k],saved[k]),k
                    assert c.relationship_summary(saved)==rec['summaries'][split+'_'+contrast]
                del native,low,high,rebuilt,saved
        guard.check()
    c.verify(src,'FINAL_SOURCE_SHA');c.verify(old,'FINAL_OLD_SHA')
    c.write(HERE/'ARITHMETIC_AUDIT.json',dict(status='PASS',jobs=29,outputs=outputs,saved_candidate_metrics_and_deltas_credit_and_gradient_summaries_recomputed=True,not_all_gradients_replayed=True,source_count=len(src),old_count=len(old)))
    c.write(HERE/'FROZEN_AUDIT.json',dict(status='PASS',records=frozen))
    c.write(HERE/'RESULT.json',dict(status='PASS_DIAGNOSTIC_ONLY_LOSS_QUALITY_CREDIT_MOTION',records=records,models=18,jobs=29,outputs=outputs,unique_IDs=28791,FIT=25912,CAL=2879,source_count=len(src),old_count=len(old),memory=guard.values(),formal_trial=False,elapsed_seconds=time.perf_counter()-START,scope='EVAL_POPULATION_ORIGINALLOSS_SAVEDCHECKPOINTS_OUTPUTCREDIT_PARAMETER_LOCAL_MOTION_NO_NEWTRAINING'))
    guard.save('SUCCESS');c.write(HERE/'exit_status.json',dict(status='SUCCESS',at=__import__('datetime').datetime.now().astimezone().isoformat()));c.stage('COMPLETE',completed_jobs=29)
guard=c.guard
