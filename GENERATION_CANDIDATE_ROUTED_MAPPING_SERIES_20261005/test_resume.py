"""Actual2epoch continuous/interrupted parameter, optimizer and RNG checks."""
import run_head as h
import torch,numpy as np,json
def compare(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    else:assert a==b
if __name__=='__main__':
    _,_,original=h.n.prepare_backend(torch.device('cpu'));del original
    cache,cfg,fullfit,fullcal,values=h.prepared_train(torch.device('cuda:0'))
    target=h.p.classify_velocity(values[2][:,1:],values[3][:,-1]/12,h.r.m.LABEL_CFG)
    fit=np.concatenate([fullfit[target[fullfit]==i][:3] for i in range(7)])
    cal=np.concatenate([fullcal[cache['change_group'][fullcal]==i][-4:] for i in range(3)])
    assert not(set(fit)&set(cal)) and set(fit)<=set(fullfit) and set(cal)<=set(fullcal)
    original_settings=h.m.settings;original_save=h.m.save_atomic
    reports=[]
    for arm in h.c.ARMS:
        directory=h.HERE/'resume_smoke'/arm
        assert not directory.exists(),'Preserve previous smoke evidence, do not overwrite'
        directory.mkdir(parents=True)
        def small_settings(seed):
            settings=h.c.settings(seed,arm);settings.update(epochs=2,batch_size=4);return settings
        h.m.settings=small_settings
        h.m.MODELS=directory/'continuous';h.m.train(7,'balanced_protected',cache,values,fit,cal,torch.device('cuda:0'))
        h.m.MODELS=directory/'interrupted'
        def interrupted_save(value,path):
            original_save(value,path)
            if path.name=='resume.pth' and value['epoch']==1:raise InterruptedError('intentional epoch1 saved interruption')
        h.m.save_atomic=interrupted_save
        try:h.m.train(7,'balanced_protected',cache,values,fit,cal,torch.device('cuda:0'))
        except InterruptedError:(directory/'intentional_interruption.txt').write_text('Expected after epoch1 resume.pth, before epoch2; not production failure.')
        else:raise AssertionError('Interruption did not trigger')
        h.m.save_atomic=original_save
        h.m.train(7,'balanced_protected',cache,values,fit,cal,torch.device('cuda:0'))
        parent='seed_7/balanced_protected'
        a=torch.load(directory/'continuous'/parent/'final.pth',map_location='cpu',weights_only=False)
        b=torch.load(directory/'interrupted'/parent/'final.pth',map_location='cpu',weights_only=False)
        compare(a['state_dict'],b['state_dict']);compare(a['calibration'],b['calibration'])
        assert a['config']['set_context']==b['config']['set_context']==arm
        ra=torch.load(directory/'continuous'/parent/'resume.pth',map_location='cpu',weights_only=False)
        rb=torch.load(directory/'interrupted'/parent/'resume.pth',map_location='cpu',weights_only=False)
        for key in ('optimizer','torch_rng','cuda_rng','loader_rng','numpy_rng','python_rng','history','objective_diagnostics'):compare(ra[key],rb[key])
        for saved in (a,b):
            model=h.TimeResponseHead(saved['input_dim'],saved['config']);model.load_state_dict(saved['state_dict']);h.frozen_audit(model)
        # The real production runner skips a completed seed only after these
        # hashes pass. Exercise that exact verification logic here.
        for kind in ('continuous','interrupted'):
            manifest=json.loads((directory/kind/parent/'complete.json').read_text())
            h.verify(manifest['model_hashes'])
        reports.append(dict(arm=arm,status='PASS',params_buffers_optimizer_allRNG_history_calibration_exact=True,completed_seed_SHA_skip_verified=True))
        print('EXACT RESUME PASS '+arm,flush=True)
    h.m.settings=original_settings;h.m.save_atomic=original_save
    h.old_manifest()

    fixture=h.HERE/'resume_smoke'/'negative_output_fixture.bin'
    fixture.write_bytes(b'verified-output-v1');good_hash=h.p.sha256(fixture);h.verify({str(fixture):good_hash})
    fixture.write_bytes(b'corrupted-output-v2')
    try:h.verify({str(fixture):good_hash})
    except AssertionError:pass
    else:raise AssertionError('bad output accepted')
    try:h.verify({str(h.HERE/'protocol.json'):'0'*64})
    except AssertionError:pass
    else:raise AssertionError('bad scientific source accepted')
    h.m.write_json(h.HERE/'analysis_negative_checks.json',dict(status='PASS',actual_completed_seed_SHA_skip=True,bad_output_rejected=True,bad_source_rejected=True,fixture_preserved=True,not_training_RNG_claim=True))
    h.m.write_json(h.HERE/'resume_check.json',dict(status='PASS',reports=reports,old_files_unchanged=True))
