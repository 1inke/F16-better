import run_geometry as h
import torch
if __name__=='__main__':
    _,_,model=h.n.prepare_backend(torch.device('cpu'));del model
    smoke=h.n.module('geometry_resume',h.HERE.parent/'GENERATION_PROTECTED_BALANCE_SERIES/test_resume.py')
    smoke.HERE=h.HERE;smoke.m=h.m;smoke.main()
    h.m.write_json(h.HERE/'resume_check.json',dict(status='PASS',geometry_optimizer_rng_calibration_exact=True))
