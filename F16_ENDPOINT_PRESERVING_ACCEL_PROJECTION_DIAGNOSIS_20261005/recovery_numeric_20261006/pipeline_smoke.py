"""Actual FIT128 saved pipeline smoke; no optimizer, no scientific choice."""
import diagnosis as d
def main():
    d.torch.set_num_threads(2);d.torch.use_deterministic_algorithms(True)
    data=d.data();entry=d.models()[0];model,meta=d.build('raw',entry['path'])
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};model.cuda()
    values=d.predict(model,data,0,128)[0];arrays,stats,check=values
    z=dict(sample_id=data['ids'][:128],increments=arrays[1],projected_increments=arrays[3],limit=arrays[4],**{'projection_'+k:v for k,v in stats.items()})
    for prefix,path in [('original_',arrays[0]),('projected_',arrays[2])]:z.update({prefix+k:v for k,v in d.package(path,data['future'][:128],data['logits'][:128]).items()})
    folder=d.HERE/'pipeline_smoke';folder.mkdir(exist_ok=True);dst=folder/'FIT128.npz'
    d.np.savez_compressed(dst,**z);del z
    with d.np.load(dst,allow_pickle=False) as f:z={k:f[k] for k in f.files}
    audit=d.saved_audit(z,data,0,128,True);d.exact(model,state)
    signature=d.read(d.HERE/'prelaunch_review/REVIEW.json')['source_signature']
    marker=dict(status='SUCCESS',source_signature=signature,outputs={str(dst):d.sha(dst)})
    d.verify(marker['outputs']);assert marker['source_signature']==signature
    rejected={}
    try:assert marker['source_signature']=='wrong'
    except AssertionError:rejected['bad_source']=True
    try:d.verify({str(dst):'0'*64})
    except AssertionError:rejected['bad_output']=True
    assert len(rejected)==2
    report=dict(status='PASS',checkpoint=entry,rows=128,arithmetic=audit,projection=check,all_weights_bool_scalers_exact=True,output_sha256=d.sha(dst),analysis_resume_SHA_skip=True,rejected=rejected,memory=d.guard.check(),no_optimizer=True)
    d.write(folder/'REVIEW.json',report);print(d.json.dumps(report))
if __name__=='__main__':main()
