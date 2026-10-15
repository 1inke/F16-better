import common as c,importlib.util
import numpy as np
RAW=None
def raw():
 global RAW
 if RAW is not None:return RAW
 c.guard();d=c.inputs()[0];encoder,observation,cache,old_feature=c.RAW_PATHS
 def select(source,field):
  with np.load(source,allow_pickle=False) as z:
   ids=z["sample_id"];assert len(np.unique(ids))==len(ids);lu={int(v):i for i,v in enumerate(ids)};ix=np.array([lu[int(v)] for v in d["ids"]]);assert np.array_equal(ids[ix],d["ids"]);ans=z[field][ix].copy()
  c.guard();return ans
 past=select(encoder,"past");tokens=select(encoder,"tokens");p2=tokens.mean(1).astype(np.float32);obs=select(observation,"observation")
 assert past.dtype==np.float32 and past.shape==(384,8,2) and np.isfinite(past).all() and not np.count_nonzero(past[:,-1])
 assert np.array_equal(past.reshape(384,16),obs[:,:16]) and np.array_equal(past,obs[:,160:176].reshape(384,8,2))
 native=select(cache,"candidate")[...,:24].reshape(384,20,12,2).astype(np.float32)
 assert np.array_equal(native.reshape(384,20,24),d["X"][:,:,96:120]);assert np.array_equal(p2,d["X"][:,0,:96])
 spec=importlib.util.spec_from_file_location("pinned_S1b_purefeatures",old_feature);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 old=mod.build(p2,past,native,"interaction");ind=mod.independent(p2,past,native,"interaction")
 assert np.array_equal(old,ind) and np.array_equal(old,d["X"]),("OLD136_INPUT_MISMATCH",float(abs(old-d["X"]).max()))
 RAW=dict(past=past,native=native,p2=p2,sample_id=d["ids"]);return RAW
