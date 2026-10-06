"""Throughput under current GPU contention only; no benchmark checkpoint or outcome choice."""
import run_head as h
import torch,time,json,math
if __name__=='__main__':
    _,_,original=h.n.prepare_backend(torch.device('cpu'));del original
    device=torch.device('cuda:0');cache,cfg,fit,cal,values=h.prepared_train(device)
    scale=torch.tensor(json.loads((h.HERE/'preflight.json').read_text())['fit_protection_scales'],device=device)
    records=[]
    for arm in h.c.ARMS:
        h.p.seed_everything(77);model=h.TimeResponseHead(940,h.c.settings(77,arm)).to(device)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        times=[]
        for step in range(36):
            idx=fit[(step*128)%len(fit):(step*128)%len(fit)+128]
            o,b,s,gt=[torch.from_numpy(v[idx]).to(device) for v in values[:4]]
            groups=torch.tensor(cache['change_group'][idx],device=device)
            torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad()
            out,res=model(o,b,s);loss=h.m.objective(out,b,gt,res,s,groups,h.c.settings(77,arm),'balanced_protected',scale)
            loss.backward();opt.step();torch.cuda.synchronize()
            if step>=4:times.append(time.perf_counter()-start)
        start=time.perf_counter();pred=h.r.g.predict(model,*[v[cal] for v in values[:3]],device);torch.cuda.synchronize()
        elapsed=time.perf_counter()-start;assert pred.shape==(2879,20,12,2)
        h.frozen_audit(model)
        records.append(dict(arm=arm,step128_seconds=sum(times)/len(times),CAL2879_seconds=elapsed,measured_steps=32,warmup_steps=4))
        del model,opt
    compute_minutes=sum(x['step128_seconds']*math.ceil(len(fit)/128)*80*3+x['CAL2879_seconds']*80*3 for x in records)/60
    h.m.write_json(h.HERE/'benchmark.json',dict(status='PASS_THROUGHPUT_ONLY',records=records,compute_minutes=compute_minutes,not_production_guarantee=True,not_gpu_isolation_guaranteed=True,saved_model=False))
    h.guard.save('BENCHMARK_PASS');print(json.dumps(dict(records=records,compute_minutes=compute_minutes,memory=h.guard.check())),flush=True)
