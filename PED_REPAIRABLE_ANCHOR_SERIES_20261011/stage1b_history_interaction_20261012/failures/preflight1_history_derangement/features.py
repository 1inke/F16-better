"""GT-free observed-history/native candidate features. No groups/logits/IDs accepted."""
import numpy as np
def build(p2,past,native,arm):
    assert arm in ("additive","interaction")
    n=len(past);assert p2.shape==(n,96) and past.shape==(n,8,2) and native.shape==(n,20,12,2)
    assert np.isfinite(p2).all() and np.isfinite(past).all() and np.isfinite(native).all()
    p=past.astype(float);v=(p[:,-1]-p[:,-2])/.4;previous=(p[:,-2]-p[:,-3])/.4;dv=v-previous
    h=np.c_[v,dv];base=np.concatenate((np.broadcast_to(p2[:,None],(n,20,96)),native.reshape(n,20,24),np.broadcast_to(h[:,None],(n,20,4))),2).astype(np.float32)
    if arm=="additive":return base
    vel=np.diff(np.concatenate((np.zeros((n,20,1,2)),native.astype(float)),2),axis=2)/.4
    bands=vel.reshape(n,20,3,4,2).mean(3);parts=[]
    for u in (v,dv):
        for b in range(3):
            w=bands[:,:,b];parts.extend(((u[:,None]*w).sum(2),u[:,None,0]*w[:,:,1]-u[:,None,1]*w[:,:,0]))
    return np.concatenate((base,np.stack(parts,2).astype(np.float32)),2)
def independent(p2,past,native,arm):
    out=np.empty((len(past),20,124 if arm=="additive" else 136),np.float32)
    for i in range(len(past)):
        v=(past[i,-1].astype(float)-past[i,-2])/.4;dv=v-(past[i,-2].astype(float)-past[i,-3])/.4
        for j in range(20):
            row=list(p2[i])+list(native[i,j].ravel())+list(v)+list(dv)
            if arm=="interaction":
                for u in (v,dv):
                    for b in range(3):
                        velocities=[]
                        for t in range(b*4,b*4+4):
                            before=np.zeros(2) if t==0 else native[i,j,t-1].astype(float)
                            velocities.append((native[i,j,t].astype(float)-before)/.4)
                        w=np.mean(velocities,0);row.extend([u[0]*w[0]+u[1]*w[1],u[0]*w[1]-u[1]*w[0]])
            out[i,j]=row
    return out
def permutation(ids,tracks,index):
    import hashlib
    ordered=sorted(index.tolist(),key=lambda k:hashlib.sha256(("s1b-history-swap-v1|"+str(int(ids[k]))).encode()).digest())
    # Fixed first cyclic offset with all donors on a different track. No labels/outcome.
    for shift in range(1,len(ordered)):
        donor=np.roll(ordered,shift)
        if np.all(tracks[ordered]!=tracks[donor]):
            lookup=dict(zip(ordered,donor));return np.array([lookup[int(k)] for k in index])
    raise RuntimeError("STOP_NO_FIXED_DIFFERENT_TRACK_HISTORY_PERMUTATION")
