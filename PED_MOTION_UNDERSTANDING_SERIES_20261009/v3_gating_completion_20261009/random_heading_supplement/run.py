import os,sys,time,importlib.util,gc,hashlib,traceback
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
sp=importlib.util.spec_from_file_location('random_parent_forward',P/'model_diagnosis_20261009/forward_diagnostics/run.py');f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f);a=f.a;Q=a.read(V/'protocol.json');a.H=H;a.Q=Q;a.START=time.perf_counter();torch=f.torch
def runforward(model,obs,nv,angles):
    pp=[];ll=[]
    for b in range(0,len(obs),128):
        a.guard();past=torch.from_numpy(np.array(obs[b:b+128],np.float32,copy=True)).cuda();n=torch.from_numpy(np.array(nv[b:b+128],np.int64,copy=True)).cuda();u=f.unit(past,'original');fb=torch.linalg.vector_norm(past[:,0,-1]-past[:,0,-4],dim=1)<=.03;ang=torch.tensor(angles[b:b+128],dtype=past.dtype,device='cuda');u=torch.where(fb[:,None],torch.stack((torch.cos(ang),torch.sin(ang)),1),u);c,s=u[:,0],u[:,1];M=torch.stack((c,-s,s,c),-1).reshape(len(past),1,1,2,2);local=torch.matmul((past-past[:,:1,-1:]).unsqueeze(-2),M).squeeze(-2);center=local.mean(-2,keepdim=True)
        with torch.no_grad():pred=model(local-center,None,None,mode='testing',num_valid=n,graph_loc=local)[2];log=model.last_mode_logits[:,0].clone();r=pred[:,0]+center[:,0,None]-local[:,0,-1,None,None];world=torch.stack((r[...,0]*u[:,None,None,0]-r[...,1]*u[:,None,None,1],r[...,0]*u[:,None,None,1]+r[...,1]*u[:,None,None,0]),-1)
        pp.append(world.cpu().numpy().astype(float));ll.append(log.cpu().numpy())
    return np.concatenate(pp),np.concatenate(ll)
def main():
    a.guard()
    for p,s in a.read(P/'source_manifest.json')['files'].items():a.lock(p,s)
    for p in (Path(__file__),H/'DESIGN.md',V/'protocol.json',P/'model_diagnosis_20261009/forward_diagnostics/run.py'):a.lock(p)
    audits=[]
    for fold in Q['folds']:
        d=a.load(P/f'stage2_raw_interventions/data/{fold}_selected.npz');angles=np.array([int.from_bytes(hashlib.sha256(f'v3-random-fallback-20261009:{fold}:{int(ID)}'.encode()).digest()[:8],'big')/2**64*(2*np.pi) for ID in d['ids']])
        for seed in Q['seeds']:
            model,tr,cfg=f.s.build(fold,seed);snap=f.s.snapshot(model);n=min(128,len(d['ids']));p,l=f.forward(model,d['obs'][:n],d['valid'][:n],'original');old=a.load(V/f'G4_{fold}_seed{seed}.npz');ep=float(abs(p-old['post_paths'][:n]).max());el=float(abs(l-old['logits'][:n]).max());assert max(ep,el)<=2e-5;pp,ll=runforward(model,d['obs'],d['valid'],angles);fallback=np.linalg.norm(d['obs'][:,0,-1]-d['obs'][:,0,-4],axis=1)<=.03;assert np.array_equal(pp[~fallback],old['post_paths'][~fallback]) and np.array_equal(ll[~fallback],old['logits'][~fallback]);f.s.exact(model,snap);a.save(f'{fold}_seed{seed}.npz',sample_id=d['ids'],angles_radians=angles,paths=pp,logits=ll);audits.append(dict(fold=fold,seed=seed,all_weights_bool_statbuffers_exact=True,base_path_max=ep,base_logits_max=el,nonfallback_bitwise_exact=True));del model,snap;gc.collect();torch.cuda.empty_cache()
    for p,s in a.LOCKS.items():assert a.sha(p)==s
    a.write('input_manifest.json',dict(files=a.LOCKS));a.write('RESULT.json',dict(status='PASS_FROZEN_RANDOM_HEADING_ALL15',audit=audits,output_sha={str(p):a.sha(p) for p in H.glob('*.npz')},resource=a.RES,elapsed_seconds=time.perf_counter()-a.START,no_training=True));print('RANDOM15_PASS',round(time.perf_counter()-a.START,2))
if __name__=='__main__':
    try:main()
    except Exception:a.write('FAILURE.json',dict(status='STOP',traceback=traceback.format_exc()));raise
