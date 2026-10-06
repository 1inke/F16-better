"""Reuse the exact-recovery test with enlarged-bound settings and isolated files."""
from pathlib import Path
import importlib.util
import sys
import run_capacity as capacity

path=capacity.CONTROL/'test_resume.py'
spec=importlib.util.spec_from_file_location('capacity_resume_test',path)
smoke=importlib.util.module_from_spec(spec);sys.modules[spec.name]=smoke;spec.loader.exec_module(smoke)
smoke.HERE=Path(__file__).resolve().parent
smoke.m=capacity.m

if __name__=='__main__':smoke.main()
