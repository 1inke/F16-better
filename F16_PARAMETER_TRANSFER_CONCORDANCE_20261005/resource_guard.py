"""Bounded host RAM/commit checks inside batches, not dialog polling."""
import ctypes,psutil,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
class Guard:
    def __init__(self):self.rss=0;self.private=0;self.phys=2**64;self.commit=2**64;self.calls=0
    def check(self):
        v=psutil.Process().memory_info();m=Mem();m.length=ctypes.sizeof(m)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        self.rss=max(self.rss,v.peak_wset);self.private=max(self.private,v.peak_pagefile)
        self.phys=min(self.phys,m.avail_phys);self.commit=min(self.commit,m.avail_page);self.calls+=1
        if max(v.rss,v.private)>6*2**30 or min(m.avail_phys,m.avail_page)<6*2**30:
            self.save('STOP_RESOURCE');raise MemoryError('6GiB process / physical+commit reserve: STOP human')
        return self.values()
    def values(self):return dict(peak_RSS_GiB=self.rss/2**30,peak_private_GiB=self.private/2**30,min_physical_GiB=self.phys/2**30,min_commit_GiB=self.commit/2**30,checks=self.calls)
    def save(self,status):
        (HERE/'RESOURCE_USAGE.json').write_text(json.dumps(dict(status=status,**self.values(),process_budget_GiB=6,system_reserve_GiB=6,time=time.time()),indent=2),encoding='utf-8')
guard=Guard();guard.check()
