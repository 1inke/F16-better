import common as c
import numpy as np

def attach(data,dictionary):
 labels=np.full(len(data['ids']),-1,np.int64);tr=data['split']['FIT']
 assert np.array_equal(dictionary['fit_ids'],data['ids'][tr]);labels[tr]=dictionary['labels'];data['fixed_labels']=labels
 return data

def features(data,dictionary):
 tr=data['split']['FIT'];cent=dictionary['centers'].reshape(20,24).astype(np.float32)
 return np.concatenate((data['p2'][tr],data['vel'][tr].astype(np.float32),cent[dictionary['labels']]),axis=1)

def stats(data,dictionary):
 x=features(data,dictionary).astype(float);mu=x.mean(0);sd=x.std(0);sd=np.where(sd<1e-6,1.,sd)
 return dict(mean=mu,std=sd,centers=dictionary['centers'],training_ids=dictionary['fit_ids'])

def chunk_error(data,dictionary,st):
 x=features(data,dictionary);errors=[]
 for chunk in (31,256):
  total=np.zeros(122);squares=np.zeros(122);n=0
  for first in range(0,len(x),chunk):
   a=x[first:first+chunk].astype(float);total+=a.sum(0);squares+=np.square(a).sum(0);n+=len(a)
  mu=total/n;sd=np.sqrt(np.maximum(0,squares/n-mu**2));sd=np.where(sd<1e-6,1.,sd)
  errors.extend((float(np.max(abs(mu-st['mean']))),float(np.max(abs(sd-st['std'])))))
 assert max(errors)<1e-11,('STAT_CHUNK_ERROR',errors)
 return max(errors)
