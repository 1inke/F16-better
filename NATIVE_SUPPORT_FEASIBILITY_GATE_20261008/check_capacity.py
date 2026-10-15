"""Isolated saved-label prelaunch gate. No torch/models/GT/selector/optimizer."""
import sys,os,json,hashlib,ctypes,time
from pathlib import Path
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np,psutil
H=Path(__file__).resolve().parent;P=H.parent/'F16_NATIVE_SUPPORT_PROTECTED_INSERTION_DIAGNOSIS_20261008'
resource=dict(peak_private_GiB=0.,min_physical_GiB=100.,min_commit_GiB=100.,checks=0)
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
def write(name,x):(H/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def guard():
    m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));p=psutil.Process().memory_info();resource['checks']+=1
    resource['peak_private_GiB']=max(resource['peak_private_GiB'],p.peak_pagefile/2**30);resource['min_physical_GiB']=min(resource['min_physical_GiB'],m.avail_phys/2**30);resource['min_commit_GiB']=min(resource['min_commit_GiB'],m.avail_page/2**30)
    assert max(p.private,p.rss)<=6*2**30 and min(m.avail_phys,m.avail_page)>=6*2**30,'STOP_RESOURCE'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()
def verify(files):
    for i,(p,h) in enumerate(files.items()):
        if i%32==0:guard()
        assert sha(p)==h,('SOURCE_OUTPUT_SHA',p)
def main():
    start=time.perf_counter();guard();r=read(P/'RESULT.json');locks=read(P/'source_manifest.json')['files'];verify(locks);verify(r['output_sha']);out=[];seen={};unique={}
    for rec in r['records']:
        tag,seed,split=rec['grouping'],rec['seed'],rec['split'];guard()
        with np.load(P/'jobs'/f'{tag}_{seed}_{split}.npz',allow_pickle=False) as z:ids=z['sample_id'];labels=z['native_cluster_labels'];groups=z['groups']
        assert labels.shape==(len(ids),20) and len(set(ids.tolist()))==len(ids)
        c=np.array([len(set(row.tolist())) for row in labels]);capacity=20-c;assert np.all((capacity>=0)&(capacity<=19))
        key=(tag,split)
        if key in seen:assert all(np.array_equal(a,b) for a,b in zip(seen[key],(ids,c,groups)))
        else:seen[key]=(ids.copy(),c.copy(),groups.copy())
        for sid,cc,g in zip(ids,c,groups):
            k=int(sid);v=(int(cc),int(g))
            if k in unique:assert unique[k]==v
            else:unique[k]=v
        for name,mask in [('overall',np.ones(len(ids),bool)),('stable',groups==0),('transition',groups==1),('change',groups==2)]:
            a=capacity[mask];out.append(dict(grouping=tag,seed=seed,split=split,group=name,n=len(a),no_slot=int((a==0).sum()),at_least_one_slot=int((a>0).sum()),at_least_four_slots=int((a>=4).sum()),capacity_hist=np.bincount(a,minlength=20).tolist(),capacity_mean=float(a.mean())))
    verify(locks);verify(r['output_sha']);guard();vals=np.array(list(unique.values()));u=[]
    for name,mask in [('overall',np.ones(len(vals),bool)),('stable',vals[:,1]==0),('transition',vals[:,1]==1),('change',vals[:,1]==2)]:
        c=20-vals[mask,0];u.append(dict(group=name,n=len(c),no_slot=int((c==0).sum()),at_least_one_slot=int((c>0).sum()),capacity_hist=np.bincount(c,minlength=20).tolist()))
    result=dict(status='PASS_COUNTING_GATE_ONLY',records=out,unique_ID_records=u,three_seed_native_capacity_exact=True,source737_output82_initial_final_SHA_PASS=True,source_locked_count=len(locks),output_count=len(r['output_sha']),necessary_not_sufficient=True,no_GT_read=True,no_new_model_forward_backward_optimizer=True,no_production_worker=True,resource=resource,elapsed_seconds=time.perf_counter()-start)
    write('RESULT.json',result);print(json.dumps(dict(status=result['status'],unique=u,primary=[x for x in out if x['seed']==42 and x['split']=='outer' and x['group']=='change'],CAL=[x for x in out if x['seed']==42 and x['split']=='original_CAL' and x['group']=='change'][:1],resource=resource)),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write('FAILURE.json',dict(traceback=traceback.format_exc(),resource=resource));raise
