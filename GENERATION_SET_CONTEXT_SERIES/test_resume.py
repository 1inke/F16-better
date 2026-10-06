from pathlib import Path
import importlib.util
import sys
import run_context as c
path=c.HERE.parent/'GENERATION_PROTECTED_BALANCE_SERIES/test_resume.py'
spec=importlib.util.spec_from_file_location('set_context_resume',path)
smoke=importlib.util.module_from_spec(spec);sys.modules[spec.name]=smoke;spec.loader.exec_module(smoke)
if __name__=='__main__':
    for arm in c.ARMS:
        smoke.HERE=c.HERE/arm;smoke.HERE.mkdir(exist_ok=True);smoke.m=c.m
        c.m.settings=lambda seed:c.settings(seed,arm)
        smoke.main()
    c.m.write_json(c.HERE/'resume_check.json',dict(status='PASS',both_arms=True,continuous_interrupted_exact=True))
