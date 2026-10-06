"""Additive memory recovery. Parent scientific implementation stays immutable."""
from pathlib import Path
from datetime import datetime
import argparse, ctypes, gc, importlib.util, os, struct, sys, threading, time
sys.dont_write_bytecode = True
import numpy as np
import psutil
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
spec = importlib.util.spec_from_file_location('immutable_concordance_parent', PARENT/'diagnosis.py')
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
base.HERE = HERE  # All parent output paths now point to this additive recovery.
original_calculate = base.calculate
original_sources = base.sources
write, read, digest, stage = base.write, base.read, base.digest, base.stage
CHUNK = 256
GIB = 2**30
PROCESS_BUDGET = 6*GIB
SYSTEM_RESERVE = 6*GIB

class MemoryStatus(ctypes.Structure):
    _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong),
      ('total_phys',ctypes.c_ulonglong),('avail_phys',ctypes.c_ulonglong),
      ('total_page',ctypes.c_ulonglong),('avail_page',ctypes.c_ulonglong),
      ('total_virtual',ctypes.c_ulonglong),('avail_virtual',ctypes.c_ulonglong),
      ('avail_extended',ctypes.c_ulonglong)]

def memory():
    m = psutil.Process().memory_info()
    s = MemoryStatus(); s.length = ctypes.sizeof(s)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
        raise ctypes.WinError()
    return dict(rss_bytes=m.rss, private_bytes=m.private,
      lifetime_peak_rss_bytes=m.peak_wset, lifetime_peak_private_bytes=m.peak_pagefile,
      physical_available_bytes=s.avail_phys, physical_total_bytes=s.total_phys,
      system_commit_available_bytes=s.avail_page, pointer_bits=struct.calcsize('P')*8)

class Monitor:
    def __init__(self):
        self.stop = threading.Event(); self.maximum_rss=0; self.maximum_private=0
        self.minimum_available=2**64; self.minimum_commit=2**64; self.samples=0
        self.error=None; self.thread=None
    def sample(self):
        v=memory(); self.samples+=1
        self.maximum_rss=max(self.maximum_rss,v['rss_bytes'])
        self.maximum_private=max(self.maximum_private,v['private_bytes'])
        self.minimum_available=min(self.minimum_available,v['physical_available_bytes'])
        self.minimum_commit=min(self.minimum_commit,v['system_commit_available_bytes'])
        if max(v['rss_bytes'],v['private_bytes'])>PROCESS_BUDGET:
            self.error='Process exceeds 6GiB memory budget'
        if min(v['physical_available_bytes'],v['system_commit_available_bytes'])<SYSTEM_RESERVE:
            self.error='System physical/commit headroom below 6GiB reserve'
        return v
    def check(self):
        self.sample()
        if self.error: raise MemoryError(self.error+'; stop safely, preserve evidence')
    def begin(self):
        self.check()
        def watch():
            while not self.stop.wait(.05):
                try:self.sample()
                except BaseException as e:self.error=repr(e);break
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()
    def finish(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=1)
        self.check()
        return dict(peak_rss_bytes=self.maximum_rss,peak_private_bytes=self.maximum_private,
          minimum_system_available_bytes=self.minimum_available,
          minimum_system_commit_available_bytes=self.minimum_commit,samples=self.samples,
          sampling_interval_seconds=.05,process_budget_bytes=PROCESS_BUDGET,
          system_reserve_bytes=SYSTEM_RESERVE,final_memory=memory())

monitor = Monitor()

def calculate(x,p,chunk=CHUNK):
    """All event/distance/count operations are unchanged, only row batching changes."""
    n=len(x['ids']); assert n>0 and 0<chunk<=1024
    out=None
    for left in range(0,n,chunk):
        monitor.check();right=min(n,left+chunk)
        part={k:v[left:right] for k,v in x.items()}
        value=original_calculate(part,p)
        assert all(isinstance(v,np.ndarray) and len(v)==right-left for v in value.values())
        if out is None:
            out={k:np.empty((n,)+v.shape[1:],dtype=v.dtype) for k,v in value.items()}
        assert set(out)==set(value)
        for k,v in value.items():
            assert out[k].dtype==v.dtype and out[k].shape[1:]==v.shape[1:]
            out[k][left:right]=v
        del value,part
    return out

base.calculate=calculate

def snapshot():
    target=HERE/'old_results_manifest.json'
    if target.exists():return read(target)['files']
    files=dict(read(PARENT/'old_results_manifest.json')['files'])
    for p in PARENT.rglob('*'):
        if HERE in p.parents or not p.is_file():continue
        if '__pycache__' in p.parts or p.suffix in ('.lock','.pending','.tmp') or p.name=='runtime_state.json':continue
        files[str(p)]=digest(p)
    write(target,dict(files=files,inherited_count=21756,
      includes='All parent failed sources, logs, preflight, exit and failure provenance; recovery subtree excluded',
      excludes=['.memory','AUTONOMOUS_GENERATION_CYCLE.json','runtime_state.json','locks/temp/bytecode']))
    return files

def sources():
    locks=original_sources()
    for p in (PARENT/'diagnosis.py',PARENT/'DESIGN.md',PARENT/'protocol.json',
              PARENT/'preflight.json',HERE/'MEMORY_PREFLIGHT.json'):
        if p.exists():locks[str(p)]=digest(p)
    return locks

base.snapshot=snapshot; base.sources=sources

def unique_input_bytes(pops):
    roots={}
    for x in pops.values():
        for v in x.values():
            while isinstance(v.base,np.ndarray):v=v.base
            roots[id(v)]=v.nbytes
    return sum(roots.values())

def preflight():
    assert not (HERE/'preflight.json').exists(),'Preserve prior recovery preflight evidence'
    assert read(HERE/'resume_check.json')['status']=='PASS'
    p=read(HERE/'protocol.json');assert p==read(PARENT/'protocol.json'),'Scientific protocol changed'
    assert struct.calcsize('P')==8
    began=time.perf_counter(); monitor.begin()
    initial_memory=memory();old=snapshot()
    hash_s=base.verify({**old,**sources()},'RECOVERY_PREFLIGHT_ALL_SHA')
    load_start=time.perf_counter();pops=base.load_populations();load_s=time.perf_counter()-load_start
    ids=np.concatenate([pops[k]['ids'] for k in ('FIT_native','CAL_native','DEV_native')])
    assert len(np.unique(ids))==30307
    # Non-divisible chunk sizes and scene boundaries, not only first rows.
    fit=pops['FIT_native'];take=np.unique(np.concatenate([np.arange(400),np.arange(11000,11400),np.arange(25000,25233)]))
    sample={k:v[take] for k,v in fit.items()}
    reference=original_calculate(sample,p)
    tests=[]
    for chunk in (31,256):
        candidate=calculate(sample,p,chunk)
        for k in reference:
            assert reference[k].dtype==candidate[k].dtype
            assert np.array_equal(reference[k],candidate[k]),('Chunk changed scientific output',chunk,k)
        assert base.summaries(reference,p)==base.summaries(candidate,p)
        tests.append(dict(chunk_rows=chunk,rows=len(take),all_columns_exact=True,all_summaries_exact=True))
        del candidate
    del reference,sample;gc.collect()
    synthetic=base.synthetic(p)
    stage('FULL_SIZE_MEMORY_PREFLIGHT',rows=len(fit['ids']),chunk_rows=CHUNK,model_forward=False,optimizer=False)
    started=time.perf_counter();full=calculate(fit,p)
    base.summaries(full,p);full_audit=base.audit_job(fit,full,p)
    full_seconds=time.perf_counter()-started
    output_bytes=sum(v.nbytes for v in full.values()); input_bytes=unique_input_bytes(pops)
    # This is a resource/correctness stress test, not a selected scientific result.
    del full;gc.collect()
    dev=pops['DEV_native'];d=calculate(dev,p)
    good=(d['ade']<=.2)&(d['fde']<=.4);clear=d['GT_event'].astype(bool)
    cue=d['cue_any']&clear;picked=good[np.arange(len(good)),d['selected_slot']]
    assert len(dev['ids'])==1516 and int(clear.sum())==469 and int(good.any(1).sum())==733
    assert int(cue.sum())==274 and int((cue&~good.any(1)).sum())==228
    assert int((cue&good.any(1)&~picked).sum())==41
    del d,pops;gc.collect()
    final_hash_s=base.verify({**old,**sources()},'RECOVERY_PREFLIGHT_FINAL_ALL_SHA')
    resource=monitor.finish()
    # Before the monitor started NumPy/SciPy imports also consumed memory;
    # Windows process lifetime peaks cover these and any short sampling miss.
    resource['peak_rss_bytes']=max(resource['peak_rss_bytes'],resource['final_memory']['lifetime_peak_rss_bytes'])
    resource['peak_private_bytes']=max(resource['peak_private_bytes'],resource['final_memory']['lifetime_peak_private_bytes'])
    assert max(resource['peak_rss_bytes'],resource['peak_private_bytes'])<PROCESS_BUDGET
    full_array_bytes=25912*190*12*2*8;chunk_array_bytes=CHUNK*190*12*2*8
    stress=dict(status='PASS_FULL_SIZE_MEMORY_AND_EXACT_ARITHMETIC',initial_memory=initial_memory,
      resource=resource,input_unique_bytes=input_bytes,largest_job_output_bytes=output_bytes,
      FIT_rows=25912,all_jobs=9,scientific_unique_IDs=30307,analysis_row_instances=25912+2879+7*1516,
      original_single_pairwise_array_bytes=full_array_bytes,
      recovery_single_pairwise_array_bytes=chunk_array_bytes,chunk_rows=CHUNK,
      exact_comparisons=tests,full_FIT_arithmetic_audit=full_audit,
      full_FIT_compute_audit_seconds=full_seconds,input_load_seconds=load_s,
      measured_initial_hash_seconds=hash_s,measured_final_hash_seconds=final_hash_s,
      all_old_SHA_unchanged=len(old),scientific_protocol_exact=True,
      no_new_scientific_threshold_or_precision_change=True,optimizer=False,model_forward=False,
      old_failure_allocation_root_cause='Allocation exception established; complete OS-level cause not established',
      resource_test_outputs_discarded_no_new_scientific_claim=True)
    write(HERE/'MEMORY_PREFLIGHT.json',stress)
    estimate=full_seconds*(25912+2879+7*1516)/25912+hash_s+final_hash_s+load_s+90
    planning=max(6,int(np.ceil(estimate/60)))
    locks=sources();base.verify(locks,'RECOVERY_FINAL_SOURCE_LOCK')
    report=dict(status='PASS',source_hashes=locks,old_files=len(old),tests=synthetic,
      memory_preflight='PASS_FULL_SIZE_MEMORY_AND_EXACT_ARITHMETIC',chunk_rows=CHUNK,
      planning_minutes=planning,eta_minutes=[max(4,planning-2),planning+4],
      estimate_seconds=estimate,full_FIT_seconds=full_seconds,resource=resource,
      IDs_unique=30307,FIT=25912,CAL=2879,DEV=1516,DEV_handcue=274,
      optimizer=False,model_forward=False,training_resume='NOT_APPLICABLE; analysis job SHA-skip/source-mismatch tested',
      preflight_seconds=time.perf_counter()-began)
    write(HERE/'preflight.json',report)
    print({k:v for k,v in report.items() if k!='source_hashes'},flush=True)

def run():
    assert read(HERE/'MEMORY_PREFLIGHT.json')['status']=='PASS_FULL_SIZE_MEMORY_AND_EXACT_ARITHMETIC'
    global monitor
    monitor=Monitor();monitor.begin()
    try:base.run()
    finally:
        report=monitor.finish()
        write(HERE/'RESOURCE_USAGE.json',report)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume-test',action='store_true');args=parser.parse_args()
    if args.preflight:preflight()
    elif args.resume_test:base.resume_test()
    else:run()
