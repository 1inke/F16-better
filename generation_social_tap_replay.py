"""FIT-only observed social replay alignment; no future tensor passed forward."""
from pathlib import Path
import sys,json
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
sys.path.insert(0,str(Path(__file__).resolve().parent/'GENERATION_TOKEN_SEQUENCE_SERIES'))
import run_head as x
device=torch.device('cuda:0')
cfg,trainer,model=x.n.prepare_backend(device)
cache=x.n.RAW_LOAD(x.p.TRAIN_CACHE);fit,cal=x.p.internal_split(len(cache['sample_id']),.1)
ids=fit[:128];sample_ids=cache['sample_id'][ids]
dataset=trainer.TrajectoryDataset(*x.n.f.fold_data('eth','train'),cfg)
loader=DataLoader(Subset(dataset,sample_ids.tolist()),batch_size=128,shuffle=False,num_workers=0)
seen=0;socialerror=0.;patherror=0.
model.eval()
with torch.inference_mode():
    for past,unused_future,num_valid,*_ in loader:
        past=past.to(device);local,_=trainer.normalize_to_focal_heading(past,past)
        center=local.mean(-2,keepdim=True)
        prediction=model(local-center,None,None,mode='testing',num_valid=num_valid.to(device).int(),graph_loc=local)[2]
        social=model.agent_aware_trans_encoder.last_agent_features[:,0].cpu().numpy()
        native=(prediction[:,0]+center[:,0,None]-local[:,0,-1,None,None]).cpu().numpy()
        batch_ids=ids[seen:seen+len(past)]
        socialerror=max(socialerror,float(abs(social-cache['social'][batch_ids]).max()))
        patherror=max(patherror,float(abs(native-cache['candidate'][batch_ids,:,:24].reshape(-1,20,12,2)).max()))
        seen+=len(past)
assert seen==128
out=dict(status='PASS' if max(socialerror,patherror)<2e-5 else 'FAIL_ALIGNMENT',fit_sample_ids=sample_ids.tolist(),max_social_error=socialerror,max_native_path_error=patherror,tolerance=2e-5,no_gt_no_dev=True,source_hashes={str(v):x.p.sha256(Path(v)) for v in [Path(__file__),x.p.TRAIN_CACHE,x.n.f.FRONT_MODELS,Path(x.n.f.__file__),x.n.f.CONFIG,x.n.f.fold_run('eth')/'best_selected.pth',x.n.f.fold_run('eth','h3')/'best_selected.pth']})
x.m.write_json(Path(__file__).resolve().parent/'GENERATION_SOCIAL_TAP_REPLAY.json',out)
print(json.dumps(out,indent=2))
assert out['status']=='PASS',out
