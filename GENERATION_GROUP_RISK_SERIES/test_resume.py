from pathlib import Path
import importlib.util
import sys
import torch
import run_group_risk as g
path=g.HERE.parent/'GENERATION_PROTECTED_BALANCE_SERIES/test_resume.py'
spec=importlib.util.spec_from_file_location('group_risk_resume_smoke',path)
smoke=importlib.util.module_from_spec(spec);sys.modules[spec.name]=smoke;spec.loader.exec_module(smoke)
smoke.HERE=Path(__file__).resolve().parent;smoke.m=g.m
if __name__=='__main__':
    cache=g.p.load_npz(g.p.TRAIN_CACHE);fit,_=g.p.internal_split(len(cache['sample_id']),.1);values=g.r.prepared(cache,False,fit=fit)
    g.scales_from_fit(cache,fit,values,g.m.settings(1),torch.device('cuda:0'));smoke.main()
