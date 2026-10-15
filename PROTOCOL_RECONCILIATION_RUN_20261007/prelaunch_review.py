"""Final actual saved-output smoke and current-source lock extension. No optimizer."""
import sys,time
sys.dont_write_bytecode=True
import run as r
import numpy as np,torch
from scipy.cluster.hierarchy import linkage,fcluster
def independently_check(z,met):
    errors=np.sqrt(np.square(z['paths'].astype(np.float64)-z['gt'][:,None].astype(np.float64)).sum(-1));a=errors.mean(-1);f=errors[:,:,-1];good=(a<=.2)&(f<=.4);native=good[:,:20].any(1);maximum=0.
    for name,slots in [('native20',slice(0,20)),('full24',slice(0,24)),('learned4',slice(20,24))]:
        paths=z['paths'][:,slots].astype(np.float64);g=good[:,slots];cl=[]
        for p,gg in zip(paths,g):
            delta=np.sqrt(np.square(p[:,None]-p[None,:]).sum(-1));D=np.maximum(delta.mean(-1)/.1,delta[:,:,-1]/.2);lab=fcluster(linkage(D[np.triu_indices(len(p),1)],method='complete'),t=1.,criterion='distance');cl.append((len(np.unique(lab)),len(np.unique(lab[gg]))))
        cl=np.asarray(cl)
        for group,mask in [('overall',np.ones(len(paths),bool)),('stable',z['groups']==0),('transition',z['groups']==1),('change',z['groups']==2)]:
            if not mask.any():continue
            m=met[name][group];gg=g.any(1);assert m['good']==int(gg[mask].sum()) and m['n']==int(mask.sum());assert m['rescued']==int((gg&~native&mask).sum()) and m['lost']==int((~gg&native&mask).sum())
            expect={'coverage':gg[mask].mean(),'minADE':a[:,slots][mask].min(1).mean(),'minFDE':f[:,slots][mask].min(1).mean(),'J':np.maximum(a[:,slots]/.2,f[:,slots]/.4)[mask].min(1).mean(),'effective_distinct':cl[mask,0].mean(),'effective_good':cl[mask,1].mean()}
            for k,v in expect.items():maximum=max(maximum,abs(m[k]-float(v)))
            p=paths[mask];v=np.concatenate([p[:,:,:1],np.diff(p,axis=2)],axis=2)/.4;acc=np.diff(v,axis=2)/.4;jerk=np.diff(acc,axis=2)/.4;s=np.sqrt((v*v).sum(-1));aa=np.sqrt((acc*acc).sum(-1));jj=np.sqrt((jerk*jerk).sum(-1));physical=dict(speed_p95=np.quantile(s,.95),accel_mean=aa.mean(),accel_p95=np.quantile(aa,.95),jerk_mean=jj.mean(),jerk_p95=np.quantile(jj,.95))
            for k,value in physical.items():maximum=max(maximum,abs(m['motion'][k]-float(value)))
    assert maximum<1e-10,maximum;return maximum
def main():
    start=time.perf_counter();r.guard();pre=r.read(r.H/'PREFLIGHT.json');assert pre['status']=='PASS';r.verify(pre['files']);ns=r.setup();data=r.load_data(ns);records=[];directory=r.H/'prelaunch_review';directory.mkdir(exist_ok=True)
    for tag,split,_ in r.PLAN:
        sp=r.validate_split(data,split);st=r.stats(data,sp['grouped_train_index']);ix=sp['outer_eval_index'][:128];m=ns['BestOf4Generator']().to(r.DEVICE);ck=torch.load(r.H/'resume_smoke'/tag/'resumed.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['state_dict']);m.eval();m.requires_grad_(False);before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};p=r.generated(ns,m,data,ix,st);z=dict(sample_id=data['ids'][ix],groups=data['groups'][ix],gt=data['gt'][ix],paths=np.concatenate([data['native'][ix],p],1));file=directory/f'{tag}.npz';assert not file.exists();np.savez_compressed(file,**z);saved=r.npz(file);assert all(np.array_equal(v,saved[k]) for k,v in z.items());t=time.perf_counter();metrics=r.evaluate(saved);metric_seconds=time.perf_counter()-t;error=independently_check(saved,metrics);assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in m.state_dict().items());records.append(dict(grouping=tag,actual_GPU_frozen_smoke128=True,all_head_weights_buffers_EXACT=True,full_saved_quality_cluster_motion_independent_max_error=error,metrics_128_seconds=metric_seconds,output_sha=r.sha(file)));del m,ck,before
    files=dict(pre['files']);files[str(r.H/'prelaunch_review.py')]=r.sha(r.H/'prelaunch_review.py');r.verify(files);pre['files']=files;pre['source_sha']=r.digest(files);pre['final_source_extension']='only new prelaunch reviewer added; previous locked sources unchanged';r.write('PREFLIGHT.json',pre);r.write('source_manifest.json',dict(files=files,scope='inherited previous relevant replay SHA extended, not all historical files'));r.guard()
    r.write('prelaunch_review/REVIEW.json',dict(status='PASS',records=records,current_source_SHA_count=len(files),prior_science_unchanged=True,old_F16_decoder_scorer_not_instantiated_or_updated=True,no_optimizer=True,resource=r.RESOURCE,elapsed_seconds=time.perf_counter()-start,limitations='same-assistant computation, not external independent review; old weights protected by SHA and absence from optimizer, not an old-model GPU replay'))
    print('PRELAUNCH_PASS',records,flush=True)
if __name__=='__main__':main()
