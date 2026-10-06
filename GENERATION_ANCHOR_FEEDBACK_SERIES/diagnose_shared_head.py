"""Fixed FIT/CAL group gradient diagnostic; no optimizer, dev or test access."""
import json
from pathlib import Path
import numpy as np
import torch
import run_head as x

def main():
    device=torch.device('cuda:0')
    _,_,original=x.n.prepare_backend(device);del original
    cache=x.p.load_npz(x.p.TRAIN_CACHE)
    cfg=x.c.settings(1,'zero')
    fit,cal=x.p.internal_split(len(cache['sample_id']),cfg['calibration_fraction'])
    values=x.r.prepared(cache,False,fit=fit)
    x.r.setup_weights(cache,fit,values,cfg,device)
    lock=json.loads((x.HERE/'results/model_lock.json').read_text())['models']
    rows=[];sampling={}
    for split,indices in [('fit',fit),('cal',cal)]:
        for seed in (1,2,3):
            path=x.HERE/f'results/models/zero/seed_{seed}/balanced_protected/constrained.pth'
            assert x.p.sha256(path)==lock[str(path)]
            model=x.FeedbackHead(values[0].shape[1],x.c.settings(seed,'zero')).to(device)
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict']);model.eval()
            parameters=list(model.head.parameters());grads=[];losses=[];counts=[]
            for group in (0,1,2):
                pool=indices[cache['change_group'][indices]==group]
                ids=np.random.default_rng(20261002+group).permutation(pool)[:128]
                sampling[f'{split}_{group}']=cache['sample_id'][ids].tolist()
                o,b,s,gt=[torch.from_numpy(v[ids]).to(device) for v in values[:4]]
                out,res=model(o,b,s)
                winner_cfg=dict(cfg,diversity_weight=0.,residual_weight=0.,smoothness_weight=0.)
                loss=x.r.objective(out,b,gt,res,s,winner_cfg,'native_balanced')
                grad=torch.autograd.grad(loss,parameters,allow_unused=True)
                vector=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(parameters,grad)])
                error=(out-gt[:,None]).norm(dim=-1)
                choice=(error.mean(-1)+cfg['fde_loss_weight']*error[:,:,-1]).argmin(-1)
                counts.append(torch.bincount(choice,minlength=20).cpu().tolist())
                grads.append(vector);losses.append(float(loss))
            cos={}
            for a,b in ((0,1),(0,2),(1,2)):
                denom=float(grads[a].norm()*grads[b].norm())
                cos[f'{a}_{b}']=float(grads[a]@grads[b])/denom if denom>1e-15 else None
            rows.append(dict(split=split,seed=seed,winner_losses=losses,gradient_norms=[float(g.norm()) for g in grads],group_gradient_cosines=cos,winner_slot_counts=counts))
    sources={str(Path(__file__).resolve()):x.p.sha256(Path(__file__)),str(x.p.TRAIN_CACHE):x.p.sha256(x.p.TRAIN_CACHE),**lock}
    result=dict(status='PASS',scope='Fixed128/group FIT/CAL only, 3 zero-arm constrained checkpoints, no optimizer/development/test; descriptive gradients not causality or training-history estimates',rows=rows,sample_ids=sampling,source_hashes=sources)
    x.m.write_json(x.HERE/'shared_head_diagnosis.json',result)
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
