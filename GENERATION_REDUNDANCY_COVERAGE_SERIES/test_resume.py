from pathlib import Path
import importlib.util
import sys
import torch
import run_coverage as c
path=c.HERE.parent/'GENERATION_PROTECTED_BALANCE_SERIES/test_resume.py'
spec=importlib.util.spec_from_file_location('coverage_resume_smoke',path)
smoke=importlib.util.module_from_spec(spec);sys.modules[spec.name]=smoke;spec.loader.exec_module(smoke)
smoke.HERE=Path(__file__).resolve().parent;smoke.m=c.m
if __name__=='__main__':
    cache=c.p.load_npz(c.p.TRAIN_CACHE);fit,_=c.p.internal_split(len(cache['sample_id']),.1);values=c.r.prepared(cache,False,fit=fit);c.FIT_NORMALIZER=values[4:6]
    directory=c.HERE/'resume_smoke'
    if not directory.exists():
        try:smoke.main()
        except RuntimeError as error:
            if 'two bool tensors' not in str(error):raise
    # The reused test's subtraction checker predates non-floating model buffers.
    # Compare those exactly and floats separately, preserving completed smoke files.
    a=torch.load(directory/'continuous/seed_7/balanced_protected/final.pth',map_location='cpu',weights_only=False)
    b=torch.load(directory/'interrupted/seed_7/balanced_protected/final.pth',map_location='cpu',weights_only=False)
    assert a['calibration']==b['calibration'] and a['state_dict'].keys()==b['state_dict'].keys()
    maximum=0.
    for key,value in a['state_dict'].items():
        other=b['state_dict'][key]
        if value.is_floating_point():maximum=max(maximum,(value-other).abs().max().item())
        else:assert torch.equal(value,other),key
    assert maximum==0,maximum
    c.m.write_json(c.HERE/'resume_check.json',dict(status='PASS',maximum_tensor_difference=maximum,nonfloating_buffers_exact=True,calibration_exact=True))
    print('PASS enlarged allocator resume: float tensors difference0; bool buffers/history exact')
