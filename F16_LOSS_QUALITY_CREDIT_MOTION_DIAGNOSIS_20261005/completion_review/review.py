"""Independent saved-data arithmetic completion review. No torch/model/optimizer."""
from pathlib import Path
import os,sys,json,hashlib,time,gc
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import numpy as np
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;ROOT=EXP.parent
sys.path.insert(0,str(EXP))
from resource_guard import guard
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for v in iter(lambda:f.read(1048576),b''):h.update(v)
    return h.hexdigest()
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        assert sha(p)==h,('SHA',p)
        if i%100==0:guard.check()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def same(a,b):
    return np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b)
def pair(a,b):
    an=float(np.sqrt(np.sum(a*a)));bn=float(np.sqrt(np.sum(b*b)));dot=float(np.sum(a*b))
    cos=None if min(an,bn)<=1e-10 else dot/(an*bn)
    return dict(quality_norm=an,other_norm=bn,dot=dot,cos=cos,other_derivative_along_quality_descent=None if cos is None else -dot/an)
def masks(group):return [('overall',np.ones(len(group),bool))]+[(n,group==i) for i,n in enumerate(('stable','transition','change'))]
def snapshot(v):
    a=v['ADE20'];f=v['FDE20'];good=(a<=.2)&(f<=.4)
    assert same(v['J_joint'],np.maximum(a/.2,f/.4).min(-1))
    assert same(v['coverage'],good.any(-1));assert same(v['raw_good'],good.sum(-1))
    assert same(v['quality_winner'],(a+.5*f).argmin(-1))
    assert not (v['primary_credit20'][np.arange(20)[None,:]!=v['quality_winner'][:,None]]!=0).any()
    rows=[]
    for n,m in masks(v['group']):
        ok=v['witness_slot'][m]>=0;wdown=ok&(v['witness_delta_L'][m]<-1e-8)
        rows.append(dict(group=n,n=int(m.sum()),coverage=int(v['coverage'][m].sum()),L_total=float(v['L_total'][m].mean()),L_primary=float(v['L_primary'][m].mean()),minADE=float(v['minADE'][m].mean()),minFDE=float(v['minFDE'][m].mean()),J=float(v['J_joint'][m].mean()),
        winner_bad_with_good=int(v['good_present_winner_bad'][m].sum()),Jnearest_not_winner=int((v['J_slot'][m]!=v['quality_winner'][m]).sum()),Jnearest_not_winner_uncovered=int(((v['J_slot'][m]!=v['quality_winner'][m])&~v['coverage'][m]).sum()),
        witness_available=int(ok.sum()),witness_lossdown=int(wdown.sum()),witness_lossdown_no_primary=int((wdown&(v['witness_direct_primary_credit'][m]==0)).sum()),effectivegood=int(v['effective_good'][m].sum()),accel_mean=float(v['accel_window_mean'][m].mean()),jerk_mean=float(v['jerk_window_mean'][m].mean())))
    return rows
def edges(left,right,saved):
    assert same(left['sample_id'],right['sample_id']) and same(saved['sample_id'],left['sample_id'])
    names=['L_'+k for k in ('primary','diversity','residual','smoothness','protection','total')]+['minADE','minFDE','J_joint','Q_joint','ADE20','FDE20','J20','raw_good','effective_good','all_distinct']+[k+'_window_'+v for k in ('speed','accel','jerk') for v in ('mean','p95')]
    for k in names:assert same(saved['delta_'+k],right[k].astype(float)-left[k].astype(float)),k
    rows=[]
    for n,m in masks(left['group']):
        dl=saved['delta_L_total'][m];da=saved['delta_minADE'][m];df=saved['delta_minFDE'][m];dj=saved['delta_J_joint'][m]
        down=dl<-1e-8;bg=left['coverage'][m];ag=right['coverage'][m];resc=~bg&ag;lost=bg&~ag
        cls=np.zeros(len(dl),np.int8);cls[down&resc]=1;cls[down&lost]=2;cls[down&bg&ag]=3;cls[down&~bg&~ag&(dj<-1e-8)]=4;cls[down&~bg&~ag&(dj>=-1e-8)]=5
        assert same(saved['loss_coverage_class'][m],cls)
        ra=right['accel_window_mean'][m]/np.maximum(left['accel_window_mean'][m],1e-30)-1
        rj=right['jerk_window_mean'][m]/np.maximum(left['jerk_window_mean'][m],1e-30)-1
        rows.append(dict(group=n,n=len(dl),before=int(bg.sum()),after=int(ag.sum()),rescue=int(resc.sum()),lost=int(lost.sum()),rescue_lossdown=int((resc&down).sum()),lost_lossdown=int((lost&down).sum()),lossdown=int(down.sum()),lossdown_ADE_FDE_down=int((down&(da<-1e-8)&(df<-1e-8)).sum()),lossdown_ADE_FDE_down_no_coverage_rescue=int((down&(da<-1e-8)&(df<-1e-8)&~resc).sum()),
        lossdown_J_worse=int((down&(dj>1e-8)).sum()),lossdown_J_improves=int((down&(dj<-1e-8)).sum()),lossdown_no_J_gain=int((down&(dj>=-1e-8)).sum()),lossdown_accel_harm_gt1percent=int((down&(ra>.01)).sum()),lossdown_jerk_harm_gt1percent=int((down&(rj>.01)).sum()),
        accel_harm_pct=float((right['accel_window_mean'][m].mean()/left['accel_window_mean'][m].mean()-1)*100),jerk_harm_pct=float((right['jerk_window_mean'][m].mean()/left['jerk_window_mean'][m].mean()-1)*100),
        loss_mean_before=float(left['L_total'][m].mean()),loss_mean_after=float(right['L_total'][m].mean()),loss_gain_pct=float((1-right['L_total'][m].mean()/left['L_total'][m].mean())*100),
        ADE_gain_pct=float((1-right['minADE'][m].mean()/left['minADE'][m].mean())*100),FDE_gain_pct=float((1-right['minFDE'][m].mean()/left['minFDE'][m].mean())*100),
        pearson_L_J=float(np.corrcoef(dl,dj)[0,1]) if np.std(dl)>0 and np.std(dj)>0 else None))
    return rows
def main():
    start=time.perf_counter();pf=read(EXP/'preflight.json');old=read(EXP/'old_results_manifest.json')['files'];verify(pf['source_hashes']);verify(old)
    result=read(EXP/'RESULT.json');audit=read(EXP/'ARITHMETIC_AUDIT.json');frozen=read(EXP/'FROZEN_AUDIT.json')
    assert result['status']=='PASS_DIAGNOSTIC_ONLY_LOSS_QUALITY_CREDIT_MOTION' and audit['status']=='PASS' and frozen['status']=='PASS'
    assert len(frozen['records'])==18 and all(x['optimizer'] is False for x in frozen['records'])
    outputs={}
    for r in result['records']:
        done=read(EXP/'jobs'/(r['key']+'.done.json'))
        assert done['status']=='SUCCESS' and done['sources']==pf['source_hashes']
        outputs.update(done['outputs'])
    assert len(outputs)==155;verify(outputs)
    snapshots=[];pairs=[];comparisons=[];maximum=0
    for split in ('FIT','CAL'):
        v=load(EXP/'jobs'/('native_'+split+'.npz'));snapshots.append(dict(key='native',split=split,groups=snapshot(v)));del v
    for r in result['records']:
        if r['kind']!='checkpoint':continue
        for split in ('FIT','CAL'):
            v=load(EXP/'jobs'/(r['key']+'_'+split+'.snapshot.npz'))
            snapshots.append(dict(key=r['key'],split=split,epoch=r['epoch'],groups=snapshot(v)))
            g=load(EXP/'jobs'/(r['key']+'_'+split+'.gradients.npz'))
            qq={k:v for k,v in g.items()}
            norms={k:float(np.linalg.norm(v)) for k,v in qq.items()}
            difference=float(np.linalg.norm(g['gradient_total']-sum(g['gradient_'+k] for k in ('primary','diversity','residual','smoothness','protection'))))
            assert difference<=1e-6+2e-4*norms['gradient_total']
            for q in ('primary','total'):
                for other in ('diversity','residual','smoothness','protection','speed_p95','accel_mean','accel_p95','jerk_mean','jerk_p95'):
                    computed=pair(g['gradient_'+q],g['gradient_'+other]);saved=r['summaries'][split]['parameter_pairs'][q+'__'+other]
                    for k,x in computed.items():
                        if x is None:assert saved[k] is None
                        else:
                            error=abs(x-saved[k]);maximum=max(maximum,error);assert error<=1e-10+1e-10*abs(x)
                    pairs.append(dict(key=r['key'],split=split,epoch=r['epoch'],quality=q,other=other,**computed,norm_ratio_other_to_quality=norms['gradient_'+other]/max(norms['gradient_'+q],1e-30)))
            del v,g;gc.collect();guard.check()
    for r in result['records']:
        if r['kind']!='checkpoint_edges':continue
        for split in ('FIT','CAL'):
            native=load(EXP/'jobs'/('native_'+split+'.npz'))
            low=load(EXP/'jobs'/(f"{r['family']}_seed{r['seed']}_constrained_{split}.snapshot.npz"))
            high=load(EXP/'jobs'/(f"{r['family']}_seed{r['seed']}_final_{split}.snapshot.npz"))
            for name,left,right in [('native_to_constrained',native,low),('constrained_to_final',low,high),('native_to_final',native,high)]:
                edge=load(EXP/'jobs'/(r['key']+'_'+split+'_'+name+'.npz'))
                comparisons.append(dict(family=r['family'],seed=r['seed'],split=split,contrast=name,groups=edges(left,right,edge)))
                del edge;guard.check()
            del native,low,high;gc.collect()
    verify(outputs);verify(pf['source_hashes']);verify(old)
    final=dict(status='PASS_SAVED_DATA_ONLY_NO_MODEL_FORWARD_OPTIMIZER',max_vector_summary_error=maximum,jobs=29,outputs=155,sources=len(pf['source_hashes']),old_files=len(old),all_before_after_SHA=True,
    scope='Independent NumPy snapshot quality/credit/edge-column-class-counts/vector summaries; original full geometry clustering rederived by completed worker, NOT rerun here; no all-gradient secondbackward.',snapshots=snapshots,comparisons=comparisons,parameter_pairs=pairs,memory=guard.values(),elapsed_seconds=time.perf_counter()-start)
    (HERE/'REVIEW.json').write_text(json.dumps(final,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:final[k] for k in ('status','max_vector_summary_error','jobs','outputs','sources','old_files','memory','elapsed_seconds')}),flush=True)
if __name__=='__main__':main()

