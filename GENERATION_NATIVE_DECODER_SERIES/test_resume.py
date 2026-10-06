from pathlib import Path
import run_native as n
import torch
if __name__=='__main__':
    _,_,model=n.prepare_backend(torch.device('cpu'));del model
    smoke=n.module('native_resume_smoke',n.ROOT/'GENERATION_PROTECTED_BALANCE_SERIES/test_resume.py')
    for arm in n.c.ARMS:
        smoke.HERE=n.HERE/arm;smoke.HERE.mkdir(exist_ok=True);smoke.m=n.m
        n.m.settings=lambda seed:n.c.settings(seed,arm)
        smoke.main()
    n.m.write_json(n.HERE/'resume_check.json',dict(status='PASS',both_arms=True,optimizer_rng_calibration_exact=True))
