import run_modes as h
import torch
if __name__=='__main__':
    _,_,model=h.n.prepare_backend(torch.device('cpu'));del model
    smoke=h.n.module('conditional_resume',h.HERE.parent/'GENERATION_PROTECTED_BALANCE_SERIES/test_resume.py')
    smoke.m=h.m
    for arm in h.c.ARMS:
        smoke.HERE=h.HERE/arm;smoke.HERE.mkdir(exist_ok=True)
        h.m.settings=lambda seed:h.c.settings(seed,arm)
        smoke.main()
    h.m.write_json(h.HERE/'resume_check.json',dict(status='PASS',both_arms_optimizer_rng_calibration_exact=True))
