"""Training-only numeric transformations, subject-balanced set scaling."""
import hashlib
import numpy as np
from common import MODALITIES

def subset(bag, cap):
    if len(bag)<=cap:return bag
    # Content-hash sampling makes truncation independent of arbitrary row order.
    keys=[hashlib.blake2b(np.ascontiguousarray(row,dtype=np.float32).tobytes(),digest_size=8).digest() for row in bag]
    return bag[sorted(range(len(keys)),key=lambda i:keys[i])[:cap]]

class Robust:
    def fit(self,x):
        x=np.asarray(x,float); self.center=np.zeros(x.shape[1]);self.scale=np.ones(x.shape[1]);self.available=np.zeros(x.shape[1],bool)
        for j in range(x.shape[1]):
            v=x[np.isfinite(x[:,j]),j]
            if len(v):
                self.available[j]=True;self.center[j]=np.median(v)
                q=np.quantile(v,[.25,.75]);self.scale[j]=max(float(q[1]-q[0]),float(v.std())*.1,1e-6)
        return self
    def transform(self,x):
        x=np.asarray(x,float);mask=np.isfinite(x)&self.available[None,:]
        z=np.where(mask,(x-self.center)/self.scale,0.)
        return np.clip(z,-6,6).astype(np.float32),mask.astype(np.float32)

class BagPreprocessor:
    def fit(self,records,dims):
        self.dims=dims;self.scalers={}
        for m in MODALITIES:
            chunks=[subset(r[m],24) for r in records if len(r[m])]
            x=np.concatenate(chunks) if chunks else np.empty((0,dims[m]))
            self.scalers[m]=Robust().fit(x)
        return self
    def transform(self,records,cap=32):
        arrays={}; masks={}
        for m in MODALITIES:
            x=np.zeros((len(records),cap,2*self.dims[m]),np.float32);valid=np.zeros((len(records),cap),bool)
            for i,r in enumerate(records):
                bag=subset(r[m],cap); n=len(bag)
                if n:
                    z,observed=self.scalers[m].transform(bag)
                    x[i,:n]=np.c_[z,observed];valid[i,:n]=True
            arrays[m]=x;masks[m]=valid
        return arrays,masks

def summaries(records,dims,modalities=MODALITIES):
    rows=[]
    for r in records:
        row=[];present=[]
        for m in modalities:
            bag=r[m]; present.append(bool(len(bag)))
            if not len(bag):row.extend(np.full(2*dims[m],np.nan));continue
            key='_summary_'+m
            if key not in r:
                active=np.isfinite(bag).any(0);v=bag[:,active];result=np.full((dims[m],2),np.nan)
                if np.isfinite(v).all():
                    med=np.median(v,0);q=np.quantile(v,[.25,.75],axis=0)
                else:
                    med=np.nanmedian(v,0);q=np.nanquantile(v,[.25,.75],axis=0)
                result[active]=np.stack([med,q[1]-q[0]],1)
                # Input-wise statistics only. No cohort statistics or labels enter this cache.
                r[key]=result.ravel()
            row.extend(r[key])
        rows.append(np.r_[row,present])
    return np.asarray(rows,float)

def group_splits(indices,groups,k,seed):
    pools={}
    for i in sorted(indices):pools.setdefault(groups[i],[]).append(int(i))
    if len(pools)<k:raise ValueError('Too few independent groups')
    keys=sorted(pools);np.random.default_rng(seed).shuffle(keys);keys.sort(key=lambda x:-len(pools[x]))
    bins=[[] for _ in range(k)]
    for key in keys:bins[min(range(k),key=lambda j:(len(bins[j]),j))].extend(pools[key])
    result=[]
    for v in bins:
        a=sorted(set(indices)-set(v));b=sorted(v)
        assert not {groups[i] for i in a}&{groups[i] for i in b}
        result.append((a,b))
    return result

def target_stats(y):
    means=[];scales=[]
    for col in y.T:
        z=col[np.isfinite(col)];means.append(float(z.mean()) if len(z) else 0.);scales.append(max(float(z.std()),1e-3) if len(z) else 1.)
    return np.array(means),np.array(scales)

def normalized_mae(y,pred,scale,schema):
    values=[]
    for family in ['PSQI','SCL90','EPQ']:
        score=[];weights=[]
        for j,t in enumerate(schema):
            good=np.isfinite(y[:,j])&np.isfinite(pred[:,j])
            if t['family']==family and good.any():
                score.append(float(np.mean(abs(y[good,j]-pred[good,j]))/scale[j]));weights.append(t['weight'])
        if score:values.append(float(np.average(score,weights=weights)))
    return float(np.mean(values)) if values else float('inf')
