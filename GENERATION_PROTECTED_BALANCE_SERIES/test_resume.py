"""Small real-GPU interrupted/continuous optimizer+RNG equivalence test."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('resume_test_source',HERE/'run_protected.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def main():
    directory=HERE/'resume_smoke'
    if directory.exists():raise RuntimeError('Smoke artifacts already exist; do not overwrite')
    cache=m.p.load_npz(m.p.TRAIN_CACHE)
    fullfit,_=m.p.internal_split(len(cache['sample_id']),.1);values=m.r.prepared(cache,False,fit=fullfit)
    target=m.p.classify_velocity(values[2][:,1:],values[3][:,-1]/12,m.r.m.LABEL_CFG)
    fit=np.concatenate([np.flatnonzero(target==i)[:3] for i in range(7)])
    cal=np.concatenate([np.flatnonzero((cache['change_group']==i)&(np.arange(len(target))>10000))[-4:] for i in range(3)])
    assert not(set(fit)&set(cal))
    original_settings=m.settings
    def small_settings(seed):
        cfg=original_settings(seed);cfg.update(epochs=2,batch_size=4);return cfg
    m.settings=small_settings
    original_save=m.save_atomic
    m.MODELS=directory/'continuous'
    m.train(7,'balanced_protected',cache,values,fit,cal,torch.device('cuda:0'))
    m.MODELS=directory/'interrupted'
    def interrupted_save(value,path):
        original_save(value,path)
        if path.name=='resume.pth' and value['epoch']==1:raise InterruptedError('deliberate after epoch1 checkpoint')
    m.save_atomic=interrupted_save
    try:m.train(7,'balanced_protected',cache,values,fit,cal,torch.device('cuda:0'))
    except InterruptedError:pass
    else:raise AssertionError('Interruption not triggered')
    m.save_atomic=original_save
    m.train(7,'balanced_protected',cache,values,fit,cal,torch.device('cuda:0'))
    a=torch.load(directory/'continuous/seed_7/balanced_protected/final.pth',map_location='cpu',weights_only=False)
    b=torch.load(directory/'interrupted/seed_7/balanced_protected/final.pth',map_location='cpu',weights_only=False)
    maximum=max((a['state_dict'][k]-b['state_dict'][k]).abs().max().item() for k in a['state_dict'])
    assert maximum<1e-7,maximum
    assert a['calibration']==b['calibration']
    print(f'PASS resume exact history/optimizer/RNG reachability; maximum tensor difference={maximum}')

if __name__=='__main__':main()
