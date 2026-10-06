import run_head as h
import torch,numpy as np
def compare(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    else:assert a==b
if __name__=='__main__':
    _,_,original=h.n.prepare_backend(torch.device('cpu'));del original
    smoke=h.n.module('neighbor_readout_resume',h.ROOT/'GENERATION_PROTECTED_BALANCE_SERIES/test_resume.py');smoke.m=h.m
    for arm in h.c.ARMS:
        smoke.HERE=h.HERE/arm/'exact_resume';smoke.HERE.mkdir(parents=True,exist_ok=True)
        h.m.settings=lambda seed:h.c.settings(seed,arm)
        directory=smoke.HERE/'resume_smoke'
        if not directory.exists():
            try:smoke.main()
            except RuntimeError as error:
                if 'Subtraction' not in str(error) or 'bool' not in str(error):raise
                print('Legacy bool formatter: exact comparison below',flush=True)
        a=torch.load(directory/'continuous/seed_7/balanced_protected/final.pth',map_location='cpu',weights_only=False)
        b=torch.load(directory/'interrupted/seed_7/balanced_protected/final.pth',map_location='cpu',weights_only=False)
        assert a['config']['set_context']==b['config']['set_context']==arm
        compare(a['state_dict'],b['state_dict']);compare(a['calibration'],b['calibration'])
        for saved in (a,b):
            model=h.PairwiseHead(saved['input_dim'],saved['config']);model.load_state_dict(saved['state_dict']);h.frozen_audit(model)
        ra=torch.load(directory/'continuous/seed_7/balanced_protected/resume.pth',map_location='cpu',weights_only=False)
        rb=torch.load(directory/'interrupted/seed_7/balanced_protected/resume.pth',map_location='cpu',weights_only=False)
        for key in ('optimizer','torch_rng','cuda_rng','loader_rng','numpy_rng','python_rng','history'):compare(ra[key],rb[key])
        print('PASS exact params/buffers/optimizer/allRNG/history/calibration',arm,flush=True)
    h.m.write_json(h.HERE/'resume_check.json',dict(status='PASS',both_arms_optimizer_rng_calibration_exact=True))
