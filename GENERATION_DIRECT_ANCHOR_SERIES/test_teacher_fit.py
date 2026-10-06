"""Matched seed7 engineering feasibility,20 fit-only native-anchor teacher epochs.
Not a supervised experiment, no GT optimizer target or dev/test access.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import run_head as h

def main():
    device=torch.device('cuda:0');_,_,source=h.n.prepare_backend(device);del source
    cfg=h.c.settings(7);cache=h.p.load_npz(h.p.TRAIN_CACHE);fit,cal=h.p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    values=h.r.prepared(cache,False,fit=fit);obs,base,speed=values[:3]
    root=h.HERE/'teacher_feasibility';root.mkdir(exist_ok=True)
    rows=[]
    for arm in h.c.ARMS:
        path=root/f'{arm}_resume.pth';h.p.seed_everything(7)
        model=h.DirectHead(obs.shape[1],h.c.settings(7,arm)).to(device)
        optimizer=torch.optim.AdamW(model.head.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        generator=torch.Generator().manual_seed(7)
        loader=DataLoader(TensorDataset(*[torch.from_numpy(v[fit]) for v in (obs,base,speed)]),batch_size=128,shuffle=True,generator=generator,num_workers=0)
        first=1;history=[]
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['arm']==arm and saved['epochs']==20
            model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);generator.set_state(saved['loader_rng'])
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);first=saved['epoch']+1;history=saved['history']
        for epoch in range(first,21):
            model.train();model.decoder.eval();losses=[]
            for o,b,s in loader:
                o,b,s=[v.to(device) for v in (o,b,s)]
                # Teacher anchors are only original observation-derived candidate points.
                anchors=model.head(model.inputs(o)).reshape(-1,20,3,2)*(s[:,0,None,None,None]+.03)
                loss=(anchors-b[:,:,[3,7,11]]).square().mean();assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.head.parameters(),5);optimizer.step();losses.append(float(loss))
            history.append(dict(epoch=epoch,fit_teacher_anchor_mse=float(np.mean(losses))))
            h.m.save_atomic(dict(arm=arm,epochs=20,epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),loader_rng=generator.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history),path)
            if epoch==1 or epoch%5==0:print('[TEACHER]',arm,epoch,history[-1],flush=True)
        h.frozen_audit(model)
        prediction=h.r.g.predict(model,obs[cal],base[cal],speed[cal],device)
        teacher_ade=float(np.linalg.norm(prediction-base[cal],axis=-1).mean())
        # Cal GT is diagnostic only after final epoch, never teacher optimization/selection.
        stats=h.r.generation_metrics(prediction,base[cal],speed[cal],values[3][cal])
        native=h.r.generation_metrics(base[cal],base[cal],speed[cal],values[3][cal])
        feasible,key,groups=h.m.select_key(stats,native,cache['change_group'][cal],cfg)
        row=dict(arm=arm,final_epoch=20,cal_teacher_ade=teacher_ade,cal_gt_feasible=bool(feasible),groups=groups,history=history)
        rows.append(row);print('[TEACHER FINAL]',arm,teacher_ade,feasible,flush=True)
    h.m.write_json(root/'summary.json',dict(status='COMPLETE',scope='seed7 engineering only;20 matched teacherfit epochs;fit nativeanchors targets/no futureGT;calGT diagnostic final only;no dev/test;not independent experiment or successclaim',rows=rows,
        initializer_quality_ready=all(r['cal_teacher_ade']<=.2 for r in rows)))
if __name__=='__main__':main()
