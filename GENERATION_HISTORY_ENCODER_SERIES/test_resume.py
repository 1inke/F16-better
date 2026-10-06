from pathlib import Path
import importlib.util
import sys
import run_history as h
path=h.HERE.parent/'GENERATION_PROTECTED_BALANCE_SERIES/test_resume.py'
spec=importlib.util.spec_from_file_location('history_resume',path)
smoke=importlib.util.module_from_spec(spec);sys.modules[spec.name]=smoke;spec.loader.exec_module(smoke)
if __name__=='__main__':
    h.initialize_fit()
    for arm in h.c.ARMS:
        smoke.HERE=h.HERE/arm;smoke.HERE.mkdir(exist_ok=True);smoke.m=h.m
        h.m.settings=lambda seed:h.c.settings(seed,arm)
        smoke.main()
    h.m.write_json(h.HERE/'resume_check.json',dict(status='PASS',both_encoders=True,continuous_interrupted_exact=True))
