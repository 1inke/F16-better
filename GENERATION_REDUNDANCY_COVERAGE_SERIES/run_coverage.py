"""K20 proposal slots freed only from tightly bounded same-family redundancy."""
from pathlib import Path
import importlib.util
import sys
import json
import argparse
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
source=HERE.parent/'GENERATION_FAMILY_WINNER_SERIES/run_family.py'
spec=importlib.util.spec_from_file_location('coverage_reporting_source',source)
f=importlib.util.module_from_spec(spec);sys.modules[spec.name]=f;spec.loader.exec_module(f)
m=f.m;r=f.r;p=f.p
m.HERE=HERE;m.RESULTS=HERE/'results';m.MODELS=m.RESULTS/'models';m.objective=f.original_objective
CONTROL=HERE.parent/'GENERATION_RESIDUAL_CAPACITY_SERIES';ARM='balanced_protected';POLICIES=f.POLICIES
ORIGINAL=r.m.Correction
FIT_NORMALIZER=None


def proposals(past):
    last=past[:,-1]-past[:,-2];smooth=(past[:,-1]-past[:,-4])/3
    velocity=torch.where((last.norm(dim=-1)>.03)[:,None],last,smooth)
    fallback=torch.zeros_like(velocity);fallback[:,0]=.05
    velocity=torch.where((velocity.norm(dim=-1)>.015)[:,None],velocity,fallback)
    progress=torch.arange(1,13,device=past.device,dtype=past.dtype)/12
    out=[torch.zeros((len(past),12,2),device=past.device,dtype=past.dtype)]
    for factors,angle in [(torch.linspace(.92,0,12,device=past.device),0),(torch.linspace(1.05,1.5,12,device=past.device),0),
                           (torch.ones(12,device=past.device),35),(torch.ones(12,device=past.device),-35),
                           (torch.ones(12,device=past.device),75),(torch.ones(12,device=past.device),-75),(torch.ones(12,device=past.device),180)]:
        theta=progress*(angle*np.pi/180);cs,sn=theta.cos(),theta.sin()
        x=velocity[:,0,None]*cs-velocity[:,1,None]*sn;y=velocity[:,0,None]*sn+velocity[:,1,None]*cs
        out.append((torch.stack((x,y),-1)*factors[None,:,None]).cumsum(1))
    return torch.stack(out,1)


def allocate(base,past,speed,cfg):
    """No GT/logits: each removed original retains a protected close witness."""
    batch=len(base);device=base.device;anchor=proposals(past)
    path=torch.linalg.vector_norm(base[:,:,None]-base[:,None,:],dim=-1).mean(-1)
    endpoint=torch.linalg.vector_norm(base[:,:,None,-1]-base[:,None,:,-1],dim=-1)
    family=r.m.labels(speed[:,1:],base[:,:,-1]/12)
    pair=(path<=cfg['path_redundancy_m'])&(endpoint<=cfg['endpoint_redundancy_m'])&(family[:,:,None]==family[:,None,:])
    pair=pair & torch.triu(torch.ones((20,20),device=device,dtype=torch.bool),diagonal=1)
    alive=torch.ones((batch,20),device=device,dtype=torch.bool);protected=torch.zeros_like(alive);used=torch.zeros((batch,8),device=device,dtype=torch.bool)
    result=base.clone();counts=torch.zeros(batch,device=device,dtype=torch.long);witness=torch.full((batch,20),-1,device=device,dtype=torch.long)
    rows=torch.arange(batch,device=device)
    for _ in range(cfg['max_replacements']):
        valid=pair & alive[:,:,None] & alive[:,None,:] & ~(protected[:,:,None]&protected[:,None,:])
        candidate=path.masked_fill(~valid,float('inf')).flatten(1);distance,index=candidate.min(-1)
        left=index//20;right=index%20
        drop=torch.where(protected[rows,right],left,right);keep=torch.where(drop==right,left,right)
        # Novelty considers retained native paths and any proposals already allocated.
        novelty=torch.linalg.vector_norm(anchor[:,:,None]-result[:,None],dim=-1).mean(-1)
        novelty[rows[:,None],torch.arange(8,device=device)[None],drop[:,None]]=float('inf')
        novelty=novelty.min(-1).values.masked_fill(used,-float('inf'))
        score,chosen=novelty.max(-1);active=torch.isfinite(distance)&(score>cfg['path_redundancy_m'])
        selected=rows[active];victim=drop[active];retained=keep[active];proposal=chosen[active]
        result[selected,victim]=anchor[selected,proposal];alive[selected,victim]=False;protected[selected,retained]=True;used[selected,proposal]=True
        witness[selected,victim]=retained;counts+=active.long()
    return result,counts,witness


class Coverage(ORIGINAL):
    def __init__(self,dimension,cfg):
        super().__init__(dimension,cfg);self.cfg=cfg
        self.register_buffer('normalizer_mean',torch.zeros((1,dimension)))
        self.register_buffer('normalizer_std',torch.ones((1,dimension)))
        self.register_buffer('proposal_enabled',torch.tensor(True))
        if FIT_NORMALIZER is not None:
            self.normalizer_mean.copy_(torch.from_numpy(FIT_NORMALIZER[0]));self.normalizer_std.copy_(torch.from_numpy(FIT_NORMALIZER[1]))

    def forward(self,observation,base,speed):
        if bool(self.proposal_enabled):
            past=(observation*self.normalizer_std+self.normalizer_mean)[:,:16].reshape(-1,8,2)
            with torch.no_grad():effective,_,_=allocate(base,past,speed,self.cfg)
        else:effective=base
        return super().forward(observation,effective,speed)


def factory(dimension,cfg):
    return Coverage(dimension,cfg) if cfg.get('geometry_augment',False) else ORIGINAL(dimension,cfg)


r.m.Correction=factory


def checkpoint(seed,arm,tag):
    root=CONTROL/'results/models' if arm=='control' else m.MODELS
    return root/f'seed_{seed}'/ARM/f'{tag}.pth'


def preflight(device):
    global FIT_NORMALIZER
    cfg=m.settings(1);assert cfg['test_access']=='forbidden' and cfg['seeds']==[1,2,3]
    prior=json.loads((CONTROL/'preflight.json').read_text())
    for path,digest in {**prior['source_hashes'],**prior['controls']}.items():assert p.sha256(Path(path))==digest,path
    lock=json.loads((CONTROL/'results/model_lock.json').read_text())['models']
    controls={str(checkpoint(seed,'control',tag)):lock[str(checkpoint(seed,'control',tag))] for seed in cfg['seeds'] for tag in POLICIES}
    for path,digest in controls.items():assert p.sha256(Path(path))==digest
    cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),cfg['calibration_fraction']);values=r.prepared(cache,False,fit=fit);FIT_NORMALIZER=values[4:6]
    assert not(set(cache['sample_id'][fit]) & set(cache['sample_id'][cal]))
    r.setup_weights(cache,fit,values,cfg,device)
    protection_scale=np.array([cache['ade'][fit].min(-1).mean(),cache['fde'][fit].min(-1).mean()],np.float32).clip(1e-6)
    assert np.array_equal(protection_scale,np.array(prior['fit_protection_scales'],np.float32))
    for seed in cfg['seeds']:
        saved=torch.load(checkpoint(seed,'control','constrained'),map_location='cpu',weights_only=False)
        for key,value in saved['config'].items():
            if key!='protocol':assert m.settings(seed)[key]==value,key
        assert np.array_equal(saved['mean'],values[4]) and np.array_equal(saved['std'],values[5])
    o,b,s,gt=[torch.from_numpy(v[:128]).to(device) for v in values[:4]];past=torch.from_numpy(cache['motion'][:128,:16].reshape(-1,8,2)).to(device)
    assert torch.allclose(proposals(past),torch.from_numpy(p.explicit_candidates(past.cpu().numpy())).to(device),atol=3e-6)
    effective,count,witness=allocate(b,past,s,cfg);assert effective.shape==b.shape and count.max()<=4 and torch.isfinite(effective).all()
    def check_witness(base,result,witness):
        sample,slot=torch.nonzero(witness>=0,as_tuple=True);keep=witness[sample,slot]
        assert torch.allclose(result[sample,keep],base[sample,keep],atol=1e-7)
        assert ((base[sample,slot]-base[sample,keep]).norm(dim=-1).mean(-1)<=.100001).all()
        assert ((base[sample,slot,-1]-base[sample,keep,-1]).norm(dim=-1)<=.200001).all()
    check_witness(b,effective,witness)
    toy=b.clone();toy[:,1]=toy[:,0];toy_result,toy_count,toy_witness=allocate(toy,past,s,cfg);check_witness(toy,toy_result,toy_witness)
    assert (toy_count>0).any()
    p.seed_everything(1);model=factory(o.shape[1],cfg).to(device);model.eval();out,res=model(o,b,s)
    assert torch.allclose(out,effective,atol=2e-5)  # Constructed baseline, not claimed native identity.
    groups=torch.tensor(cache['change_group'][:128],device=device);loss=m.objective(out,b,gt,res,s,groups,cfg,ARM,torch.tensor(protection_scale,device=device));loss.backward()
    assert torch.isfinite(loss) and model.network[-1].weight.grad.abs().sum()>0
    with torch.no_grad():model.proposal_enabled.fill_(False);native,_=model(o,b,s);assert torch.equal(native,b)
    # Untouched raw-cache/norm provenance plus internal-calibration proposal diagnostic.
    constructed=[];counts=[]
    for start in range(0,len(cal),256):
        ids=cal[start:start+256];base,sp=[torch.from_numpy(v[ids]).to(device) for v in values[1:3]]
        history=torch.from_numpy(cache['motion'][ids,:16].reshape(-1,8,2)).to(device);out,n,_=allocate(base,history,sp,cfg);constructed.append(out.cpu().numpy());counts.append(n.cpu().numpy())
    constructed=np.concatenate(constructed);count=np.concatenate(counts)
    metrics=r.generation_metrics(constructed,values[1][cal],values[2][cal],values[3][cal]);native=r.generation_metrics(values[1][cal],values[1][cal],values[2][cal],values[3][cal])
    m.write_json(HERE/'internal_calibration_diagnostic.json',dict(native=m.grouped(native,cache['change_group'][cal]),constructed=m.grouped(metrics,cache['change_group'][cal]),mean_replacements=float(count.mean()),replacement_fraction=float((count>0).mean()),counts=np.bincount(count,minlength=5)))
    sources={**prior['source_hashes'],str(CONTROL/'run_capacity.py'):p.sha256(CONTROL/'run_capacity.py'),str(source):p.sha256(source),str(Path(__file__)):p.sha256(Path(__file__)),str(HERE/'protocol.json'):p.sha256(HERE/'protocol.json')}
    m.write_json(HERE/'preflight.json',dict(status='PASS',fit_n=len(fit),cal_n=len(cal),new_trainings=3,fit_protection_scales=protection_scale.tolist(),source_hashes=sources,controls=controls,test_access=False))
    print('PREFLIGHT PASS: no logits/GT allocator, protected witnesses, proposal equivalence, bounded K20, native fallback, GPU/hash/config',flush=True)


def repair_fallback(seed):
    directory=checkpoint(seed,'family','constrained').parent
    path=directory/'constrained.pth';saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['epoch']==0:
        saved['state_dict']['proposal_enabled'].fill_(False);m.save_atomic(saved,path)
    manifest=json.loads((directory/'complete.json').read_text());manifest['model_hashes']={str(directory/f'{tag}.pth'):p.sha256(directory/f'{tag}.pth') for tag in POLICIES}
    m.write_json(directory/'complete.json',manifest)


def evaluate(device):
    cfg=m.settings(1);paths=[checkpoint(seed,arm,tag) for seed in cfg['seeds'] for arm in ('control','family') for tag in POLICIES]
    p.seed_everything(1);dim=FIT_NORMALIZER[0].shape[1];untrained=factory(dim,cfg)
    m.save_atomic(dict(config=cfg,input_dim=dim,mean=FIT_NORMALIZER[0],std=FIT_NORMALIZER[1],state_dict=untrained.state_dict()),m.RESULTS/'constructed_untrained.pth')
    paths.append(m.RESULTS/'constructed_untrained.pth');m.write_json(m.RESULTS/'all_model_lock.json',dict(models={str(path):p.sha256(path) for path in paths},dev_used_for_selection=False,test_access=False))
    f.checkpoint=checkpoint;f.evaluate(device)
    saved=torch.load(m.RESULTS/'constructed_untrained.pth',map_location='cpu',weights_only=False);cache=p.load_npz(p.DEV_CACHE);values=r.prepared(cache,False,mean=saved['mean'],std=saved['std'])
    model=factory(dim,cfg).to(device);model.load_state_dict(saved['state_dict']);prediction=r.g.predict(model,*values[:3],device)
    stats=r.generation_metrics(prediction,values[1],values[2],values[3]);np.savez_compressed(m.RESULTS/'constructed_untrained.npz',sample_id=cache['sample_id'],trajectory=prediction,**stats)
    path=m.RESULTS/'summary.json';summary=json.loads(path.read_text());summary['variant']='redundancy_coverage';summary['constructed_untrained']=m.grouped(stats,cache['change_group'])
    summary['scope']='ETH reused dev; K20; up to4 original slots replaced only with protected same-family witness within.1m ADE-distance/.2m endpoint; no logits/GT allocation; explicit history proposals plus learned correction, not end-to-end intent understanding; no family auxiliary/test/backbone/scorer training'
    summary['verdict']='MAJOR_DEV_SIGNAL_REQUIRES_INDEPENDENT_VALIDATION' if all(summary['gates'].values()) else 'REDUNDANCY_COVERAGE_GATE_FAIL'
    for row in summary['aggregate']:
        if row['arm']=='family':row['arm']='coverage'
    m.write_json(path,summary)
    lines=['# Redundancy protected proposal coverage','',summary['verdict'],'',summary['scope'],'','| Arm | Policy | Group | minADE | minFDE |','|---|---|---|---:|---:|']
    for row in summary['aggregate']:
        if row['group'] in ('overall','change'):lines.append(f'| {row["arm"]} | {row["policy"]} | {row["group"]} | {row["minade20"]:.6f} | {row["minfde20"]:.6f} |')
    for name,metrics in summary['constructed_untrained'].items():
        if name in ('overall','change'):lines.append(f'| constructed | untrained | {name} | {metrics["minade20"]:.6f} | {metrics["minfde20"]:.6f} |')
    (m.RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('COMPLETE '+summary['verdict'],flush=True)


def main():
    global FIT_NORMALIZER
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('preflight','run'),default='preflight');parser.add_argument('--device',default='cuda:0');args=parser.parse_args();device=torch.device(args.device)
    if args.action=='preflight':preflight(device);return
    audit=json.loads((HERE/'preflight.json').read_text())
    for path,digest in {**audit['source_hashes'],**audit['controls']}.items():assert p.sha256(Path(path))==digest,path
    if (m.RESULTS/'summary.json').exists():raise RuntimeError('Already complete')
    m.MODELS.mkdir(parents=True,exist_ok=True);cache=p.load_npz(p.TRAIN_CACHE);fit,cal=p.internal_split(len(cache['sample_id']),m.settings(1)['calibration_fraction']);values=r.prepared(cache,False,fit=fit);FIT_NORMALIZER=values[4:6]
    for seed in m.settings(1)['seeds']:
        manifest=checkpoint(seed,'family','constrained').parent/'complete.json'
        if manifest.exists():
            for path,digest in json.loads(manifest.read_text())['model_hashes'].items():assert p.sha256(Path(path))==digest
            repair_fallback(seed)
            print(f'[SKIP VERIFIED] seed{seed}',flush=True);continue
        print(f'[TRAIN] redundancy coverage seed{seed}',flush=True);m.train(seed,ARM,cache,values,fit,cal,device);repair_fallback(seed)
    evaluate(device);m.write_json(m.RESULTS/'exit_status.json',dict(status='SUCCESS',time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        m.RESULTS.mkdir(parents=True,exist_ok=True);(m.RESULTS/'failure.txt').write_text(traceback.format_exc());m.write_json(m.RESULTS/'exit_status.json',dict(status='FAILURE',time=time.time()));raise
