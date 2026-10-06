"""FIT workload timing only; discard temporary models; no outcome selection."""
import run_head as h
import torch,time,json
if __name__=='__main__':
    device=torch.device('cuda:0');_,_,original=h.n.prepare_backend(torch.device('cpu'));del original
    cache=h.p.load_npz(h.p.TRAIN_CACHE);cfg=h.c.settings(1);fit,cal=h.p.internal_split(len(cache['sample_id']),.1)
    values=h.r.prepared(cache,False,fit=fit);h.r.setup_weights(cache,fit,values,cfg,device)
    audit=json.loads((h.HERE/'preflight.json').read_text());scale=torch.tensor(audit['fit_protection_scales'],device=device);records={}
    for arm in h.c.ARMS:
        h.p.seed_everything(7);model=h.PairwiseHead(940,h.c.settings(7,arm)).to(device).eval()
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        start=None
        for step in range(36):
            if step==4:torch.cuda.synchronize();start=time.perf_counter()
            ids=fit[step*128:(step+1)*128];o,b,s,gt=[torch.from_numpy(v[ids]).to(device) for v in values[:4]];g=torch.tensor(cache['change_group'][ids],device=device)
            optimizer.zero_grad();out,res=model(o,b,s);loss=h.m.objective(out,b,gt,res,s,g,h.c.settings(7,arm),'balanced_protected',scale);loss.backward();optimizer.step()
        torch.cuda.synchronize();seconds=(time.perf_counter()-start)/32;h.frozen_audit(model)
        records[arm]=seconds;print('TIMING '+arm+' '+str(seconds),flush=True)
    h.m.write_json(h.HERE/'benchmark.json',dict(status='PASS_TIMING_ONLY',seconds_per_128_fitstep=records,discarded_models=True,compute_minutes=sum(records.values())*203*80*3/60,source_hash=h.p.sha256(h.HERE/'benchmark.py')))
