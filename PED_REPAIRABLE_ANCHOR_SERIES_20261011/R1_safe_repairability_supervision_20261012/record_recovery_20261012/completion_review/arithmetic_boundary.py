import common as c,numerics as n,numpy as np
c.guard()
d=c.npz(c.H/"sample_design.npz");s=c.npz(c.H/"solutions.npz");rows=[]
for k in range(len(s["paths"])):
 if k%64==0:c.guard()
 i=int(s["window"][k]);j=int(s["slot"][k]);a,q,bank=n.independent_verify(d["native"][i],d["gt"][i],j,s["raw32"][k]);p,_,_=n.validate(d["native"][i],d["gt"][i],j,s["raw32"][k])
 if a["witness"]!=p["witness"]:
  raw=s["raw32"][k].astype(float).reshape(3,2);u=np.array([np.tanh(z)/max(1.,float(np.sqrt((np.tanh(z)**2).sum()))) for z in raw]);node=u*n.NODE[:,None]
  r=np.stack([np.interp(n.T,[0,4,8,12],np.r_[0.,node[:,v]]) for v in range(2)],1);length=np.sqrt((r*r).sum(1));lam=min([1.]+[float(n.RAD[t]/length[t]) for t in range(12) if length[t]>0]);r*=lam
  rows.append(dict(case=k,window=i,sample_id=int(d["ids"][i]),slot=j,mode=str(s["mode"][k]),production=p,independent=a,independent_bound_excess=float((np.sqrt((r*r).sum(1))-n.RAD).max()),safe_intersection=bool(a["witness"] and p["witness"]),anchor_final_label=int(c.npz(c.H/"labels.npz")["labels"][i,j])))
assert len(rows)==1 and not rows[0]["safe_intersection"]
c.write("completion_review/ARITHMETIC_BOUNDARY.json",dict(status="PASS_CONSERVATIVE_INTERSECTION_UNCHANGED",records=rows,no_solver_fit_or_threshold_change=True))
print(c.read(c.H/"completion_review/ARITHMETIC_BOUNDARY.json"))
