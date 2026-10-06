"""Read-only original-objective gradient decomposition, FIT/CAL only."""
from pathlib import Path
import importlib.util,sys,json,copy
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('fixed_memorization_source',ROOT/'GENERATION_FIT_MEMORIZATION_DIAGNOSIS/run_diagnosis.py')
s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)
def grad(loss,parameters):
    values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if v is None else v).reshape(-1) for p,v in zip(parameters,values)])
def cosine(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-20))
def main():
    assert not (HERE/'RESULT.json').exists(),'Already complete'
    oldcount=s.manifest();model,cfg,data,scale,ids,prior=s.setup()
    initial=copy.deepcopy(model.state_dict())
    saved=torch.load(s.HERE/'resume.pth',map_location='cpu',weights_only=False)
    result=json.loads((s.HERE/'RESULT.json').read_text());assert saved['epoch']==120
    with np.load(s.HERE/'split_ids.npz') as splitids:
        for name in ids:assert np.array_equal(ids[name],splitids[name])
    rows=[]
    for stage,state in [('initial',initial),('final',saved['state_dict'])]:
        model.load_state_dict(state);s.check_frozen(model);model.eval()
        for split,batch in data.items():
            features,base,speed,future,groups=batch
            output,residual,raw,limit=s.cached_forward(model,features,base,speed)
            error=(output-future[:,None]).norm(dim=-1)
            joint=error.mean(-1)+cfg['fde_loss_weight']*error[:,:,-1]
            target=s.r.m.labels(speed[:,1:],future[:,None,-1]/12).squeeze(1)
            weights=s.r.WEIGHTS[target]
            winner=(weights*joint.min(-1).values).mean()
            unprotected=s.r.objective(output,base,future,residual,speed,cfg,'native_balanced')
            total=s.m.objective(output,base,future,residual,speed,groups,cfg,'balanced_protected',scale)
            terms=dict(winner=winner,regularizers=unprotected-winner,protection=total-unprotected,total=total)
            parameters=list(model.head.parameters());vectors={key:grad(value,parameters) for key,value in terms.items()}
            masks=[('overall',torch.ones(len(groups),dtype=torch.bool,device=groups.device))]+[(name,groups==i) for i,name in enumerate(('stable','transition','change'))]
            groupgrad={name:grad((weights[mask]*joint.min(-1).values[mask]).mean(),parameters) for name,mask in masks if mask.any()}
            occupancy={name:torch.bincount(joint.argmin(-1)[mask],minlength=20).cpu().tolist() for name,mask in masks if mask.any()}
            norms={key:float(value.norm()) for key,value in vectors.items()}
            row=dict(stage=stage,split=split,loss_terms={key:float(value) for key,value in terms.items()},gradient_norms=norms,term_cosines={f'{a}_vs_{b}':cosine(vectors[a],vectors[b]) for a,b in [('winner','regularizers'),('winner','protection'),('winner','total')]},group_winner_cosines={f'{a}_vs_{b}':cosine(groupgrad[a],groupgrad[b]) for a,b in [('stable','change'),('transition','change')]},winner_mode_occupancy=occupancy,raw_clip_fraction=float((raw.abs()>=limit).float().mean()),raw_limit_max_ratio=float((raw.abs()/limit).max()))
            rows.append(row)
            if stage=='final':
                recomputed=s.metrics(model,batch,cfg,scale);stored=result['history'][-1][split]
                assert abs(recomputed['objective']-stored['objective'])<1e-6
                for a,b in zip(recomputed['rows'],stored['rows']):
                    for key in a:
                        if isinstance(a[key],(float,int)):assert abs(a[key]-b[key])<1e-6,key
            print(json.dumps(row),flush=True)
    s.manifest()
    for path,digest in saved['source_hashes'].items():assert s.digest(path)==digest,path
    provenance={str(f):s.digest(f) for f in [Path(__file__),s.HERE/'run_diagnosis.py',s.HERE/'RESULT.json',s.HERE/'resume.pth',s.HERE/'split_ids.npz']}
    (HERE/'RESULT.json').write_text(json.dumps(dict(status='PASS_DESCRIPTIVE_ONLY',rows=rows,final_saved_metrics_independently_recomputed=True,original_weights_exact=True,old_files_unchanged=oldcount,optimizer_steps=0,dev_test_access=False,source_hashes=provenance,limitations='Single seed/fixed tinyFIT/CAL local gradients are not causal proof or production benefit; initial zero lastlayer hides front gradients by design. Group balanced weights from allFIT and protected objective unchanged.'),indent=2,allow_nan=False),encoding='utf-8')
if __name__=='__main__':main()
