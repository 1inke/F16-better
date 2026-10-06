"""Actual two-epoch subset exact resume and throwaway throughput; no science sweep."""
import adapter as h
import numpy as np,torch,time,traceback,gc
def compare(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    else:assert a==b,(a,b)
def call_smoke(seed,cache,values,fit,cal,device):
    h._TRACE.clear();h._ROWS=[];resume=h.m.MODELS/f'seed_{seed}/balanced_protected/resume.pth'
    if resume.exists():h._ROWS=list(torch.load(resume,map_location='cpu',weights_only=False)['objective_diagnostics'])
    h._LOGGING=True
    try:h.original_train(seed,'balanced_protected',cache,values,fit,cal,device)
    finally:h._LOGGING=False;h._TRACE.clear()
def main():
    assert h.read(h.HERE/'preflight.json')['status']=='PASS_GPU_STATISTICS'
    h.verify(h.read(h.HERE/'preflight.json')['source_hashes'])
    cache,values,splits=h.load_data();Class=h.initialize_classes();device=torch.device('cuda:0');reports=[]
    target=h.p.classify_velocity(values[2][:,1:],values[3][:,-1]/12,h.LABEL_CFG)
    fullcal=splits['inner_cal_index'];cal=np.concatenate([fullcal[cache['change_group'][fullcal]==i][-4:] for i in range(3)])
    original_settings=h.m.settings;original_save=h.m.save_atomic;original_models=h.m.MODELS
    for arm in ('random','grouped'):
        h.stage('REAL_TWO_EPOCH_RESUME',arm=arm);fullfit=splits[arm+'_train_index']
        fit=np.concatenate([fullfit[target[fullfit]==i][:3] for i in range(7)]);assert len(fit)==21 and len(cal)==12
        directory=h.HERE/'resume_smoke'/arm;assert not directory.exists(),'Preserve previous evidence'
        def cfg(seed):
            value=h.settings(seed,arm);value.update(epochs=2,batch_size=4);return value
        h.m.settings=cfg;h.m.MODELS=directory/'continuous';call_smoke(7,cache,values,fit,cal,device)
        h.m.MODELS=directory/'interrupted'
        def interrupted(value,path):
            original_save(value,path)
            if path.name=='resume.pth' and value['epoch']==1:raise InterruptedError('Intentional saved epoch1 interruption')
        h.m.save_atomic=interrupted
        try:call_smoke(7,cache,values,fit,cal,device)
        except InterruptedError:h.write(directory/'intentional_interruption.json',dict(expected=True,after_saved_epoch=1))
        else:raise AssertionError('Expected interruption absent')
        h.m.save_atomic=original_save;call_smoke(7,cache,values,fit,cal,device)
        suffix='seed_7/balanced_protected';a=torch.load(directory/'continuous'/suffix/'final.pth',map_location='cpu',weights_only=False);b=torch.load(directory/'interrupted'/suffix/'final.pth',map_location='cpu',weights_only=False)
        for key in ('state_dict','calibration','config'):compare(a[key],b[key])
        ra=torch.load(directory/'continuous'/suffix/'resume.pth',map_location='cpu',weights_only=False);rb=torch.load(directory/'interrupted'/suffix/'resume.pth',map_location='cpu',weights_only=False)
        for key in ('optimizer','torch_rng','cuda_rng','loader_rng','numpy_rng','python_rng','history','objective_diagnostics'):compare(ra[key],rb[key])
        for saved in (a,b):
            model=Class(940,saved['config']);model.load_state_dict(saved['state_dict']);h.frozen_audit(model);del model
        for kind in ('continuous','interrupted'):h.verify(h.read(directory/kind/suffix/'complete.json')['model_hashes'])
        reports.append(dict(arm=arm,status='PASS',fit_subset=len(fit),inner_subset=len(cal),allparams_bool_scalers_optimizer_allRNG_history_trace_EXACT=True,completed_seed_SHA_verification=True))
        print('EXACT_RESUME_PASS',arm,flush=True)
    h.m.settings=original_settings;h.m.save_atomic=original_save;h.m.MODELS=original_models
    for bad in ({str(h.HERE/'protocol.json'):'0'*64},{str(h.HERE/'resume_smoke/random/continuous/seed_7/balanced_protected/final.pth'):'0'*64}):
        try:h.verify(bad)
        except AssertionError:pass
        else:raise AssertionError('Invalid source/output accepted')
    h.write(h.HERE/'resume_check.json',dict(status='PASS',reports=reports,badsource_output_rejected=True,not_full18909_twice=True))
    h.stage('THROWAWAY_BENCHMARK');bench=[]
    h.setup_weights(cache,splits['common_statistics_index'],values,h.settings(1),device)
    scale=torch.from_numpy(np.load(h.HERE/'common_statistics.npz')['protection_scales']).to(device)
    for arm in ('random','grouped'):
        idx=splits[arm+'_train_index'][:128];o,b,s,f=[torch.from_numpy(v[idx]).to(device) for v in values[:4]];groups=torch.from_numpy(cache['change_group'][idx]).to(device)
        h.p.seed_everything(99001);model=Class(940,h.settings(99001,arm)).to(device);cfg=h.settings(99001,arm)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);times=[]
        for step in range(36):
            torch.cuda.synchronize();start=time.perf_counter();out,res=model(o,b,s);loss=h.objective(out,b,f,res,s,groups,cfg,'balanced_protected',scale)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();torch.cuda.synchronize()
            if step>=4:times.append(time.perf_counter()-start)
        h.frozen_audit(model);inner=splits['inner_cal_index'];torch.cuda.synchronize();start=time.perf_counter();prediction=h.predict(model,*[v[inner] for v in values[:3]],device);torch.cuda.synchronize();forward_seconds=time.perf_counter()-start
        start=time.perf_counter();h.r.generation_metrics(prediction,values[1][inner],values[2][inner],values[3][inner]);metric_seconds=time.perf_counter()-start
        bench.append(dict(arm=arm,FIT128_seconds=float(np.mean(times)),inner_forward_seconds=forward_seconds,inner_metrics_seconds=metric_seconds,production_train_n=len(splits[arm+'_train_index']),no_saved_model=True))
        del model,opt,out,res,loss,prediction;gc.collect();torch.cuda.empty_cache()
    compute=sum((math_ceil(len(splits[row['arm']+'_train_index'])/128)*row['FIT128_seconds']+row['inner_forward_seconds']+row['inner_metrics_seconds'])*80*3 for row in bench)/60
    h.verify(h.read(h.SERIES/'old_results_manifest.json')['files']);source=h.source_hashes()
    h.write(h.HERE/'benchmark.json',dict(status='PASS',records=bench,estimated_compute_minutes=compute,resource=h.guard.check(),not_production_guarantee=True,no_benchmark_models_saved=True))
    audit=h.read(h.HERE/'preflight.json');audit.update(status='PASS',resume_status='PASS',source_hashes=source,final_resource=h.guard.check(),benchmark_compute_minutes=compute,production_started=False)
    h.write(h.HERE/'preflight.json',audit);h.guard.save('FINAL_PREFLIGHT_PASS');print('FINAL_PREFLIGHT_PASS',compute,flush=True)
def math_ceil(x):return int(np.ceil(x))
if __name__=='__main__':
    try:main()
    except Exception:
        d=h.HERE/'failures/resume_benchmark1';d.mkdir(parents=True,exist_ok=True);(d/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');h.write(d/'SOURCE.json',{str(p):h.sha(p) for p in h.HERE.glob('*.py')});raise
