import common as c
import numerics as n
import run as r
import numpy as np,ast,time,traceback
def main():
    start=time.perf_counter();c.guard();files=r.inherited();c.write("source_manifest.json",dict(files=files,scope="related inherited sources and outputs; not all historical files"))
    t=time.perf_counter();c.verify(files);sha_seconds=time.perf_counter()-t
    reg,lab,ids,native,gt,chosen=c.load()
    assert native.shape[1:]==(20,12,2) and len(np.unique(ids))==len(ids) and np.isfinite(native).all() and np.isfinite(gt).all()
    # Read-only AST extraction: never execute the old module top-level.
    src=c.R/"PED_FOUR_DIRECTION_SERIES_20261011/experiment3_samepath_quality/run.py"
    tree=ast.parse(src.read_text(encoding="utf-8-sig"));node=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ("bases","decode_numpy")]
    assert len(node)==2
    env=dict(np=np,ARMS=("sum","joint"));exec(compile(ast.Module(body=node,type_ignores=[]),str(src),"exec"),env)
    rng=np.random.default_rng(20261012);maxmap=0.
    for i in chosen[:8]:
        for raw in (np.zeros((3,2),np.float32),rng.normal(size=(3,2)).astype(np.float32)):
            p,res,u,lam=n.decode(native[i,0],raw)
            original,rr,ll=env["decode_numpy"](native[i:i+1,:1],u[None,None],"sum")
            maxmap=max(maxmap,float(np.max(np.abs(p-original[0,0]))));assert maxmap<=2e-12
            assert np.array_equal(n.decode(native[i,0],np.zeros((3,2)))[0],native[i,0].astype(float))
    # Synthetic genuine quality witness and pointwise derivative reference.
    synthetic=np.zeros((20,12,2));synthetic[:,:,0]=np.arange(1,13)*.04
    z,path,_=n.solve(synthetic,synthetic[0],0,"shape",c.guard)
    assert z["quality_good"] and z["bound_literal"]
    badgt=np.ones((12,2))*100.;b,_,_=n.solve(synthetic,badgt,0,"shape",c.guard)
    assert not b["witness"] # failure means UNKNOWN, never certified infeasible
    zero=np.zeros((20,12,2));assert np.array_equal(n.motion(zero),np.zeros(5))
    todo=r.cases(lab,chosen);assert len(todo)>=64
    c.save("preflight_sample.npz",index=chosen,sample_id=ids[chosen],groups=reg["groups"][chosen],potential=lab["potential"][chosen])
    timings=[];smoke=[]
    for i,s in todo[:4]:
        t=time.perf_counter()
        for mode in ("shape","pointwise"):
            z,p=r.checked(native[i],gt[i],s,mode);smoke.append(dict(index=i,slot=s,**r.compact(z)))
        timings.append(time.perf_counter()-t)
    c.write("preflight_cases.json",smoke)
    p=native[chosen[:4]].astype(float);g=gt[chosen[:4]]
    a=r.quality_banks(p,g);b=r.quality_banks(p,g,True)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    # Literal gates remain unchanged. Reconstruction tolerance is not a science epsilon.
    c.verify(files);planning=2*max(timings)*len(todo)+2*sha_seconds+120
    result=dict(status="PASS" if planning<=1800 else "STOP_PLANNING_OVER30MIN",planning_minutes=planning/60,anchors=len(todo),solves=2*len(todo),benchmark_paired_anchor_seconds=timings,source_count=len(files),sha_seconds=sha_seconds,max_old_map_error=maxmap,all_sample_A_only=True,unique_tracks=int(len(np.unique(reg["track"][chosen]))),elapsed_seconds=time.perf_counter()-start,resource=c.RESOURCE)
    c.write("PREFLIGHT.json",result);print(result,flush=True)
if __name__=="__main__":
    try:main()
    except BaseException:
        c.write("PREFLIGHT_FAILURE_"+str(time.time_ns())+".json",dict(traceback=traceback.format_exc(),resource=c.RESOURCE));raise
