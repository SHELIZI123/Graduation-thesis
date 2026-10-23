"""Label-free, asynchronous subject bags. PPG and NNI are different views."""
import numpy as np
from preprocessing import Robust, subset
from heart import PPG_DIM, HEART_DIM

VIEWS = ('eeg', 'eye', 'ppg', 'nni')
CAPS = {'eeg':128, 'eye':16, 'ppg':8, 'nni':8}
FOCUS = ('psqi_total', 'scl_yy', 'scl_jl', 'epq_E', 'epq_N', 'epq_P')

def split_views(record):
    h=record['heart']
    if h.shape[1] != HEART_DIM: raise ValueError('Unsupported heart feature schema')
    return {'eeg':record['eeg'], 'eye':record['eye'],
            'ppg':h[h[:,-2]>.5,:PPG_DIM], 'nni':h[h[:,-1]>.5,PPG_DIM:-2]}

def dimensions(dims):
    return {'eeg':dims['eeg'],'eye':dims['eye'],'ppg':PPG_DIM,'nni':HEART_DIM-PPG_DIM-2}

def summarize_bag(bag, d, distribution=False):
    nstats=5 if distribution else 2
    out=np.full((d,nstats),np.nan)
    if len(bag):
        active=np.isfinite(bag).any(0)
        if active.any():
            x=bag[:,active]
            if distribution:
                q=np.nanquantile(x,[.1,.5,.9],axis=0)
                vals=np.vstack([q,np.nanmean(x,0),np.nanstd(x,0)])
            else:
                q=np.nanquantile(x,[.25,.5,.75],axis=0)
                vals=np.vstack([q[1],q[2]-q[0]])
            out[active]=vals.T
    return out.ravel()

def summary_matrix(records,dims,distribution=False,views=VIEWS):
    dd=dimensions(dims); rows=[]
    for r in records:
        key='_v171_distribution' if distribution else '_v171_summary'
        if key not in r:
            bags=split_views(r)
            r[key]={v:summarize_bag(bags[v],dd[v],distribution) for v in VIEWS}
        rows.append(np.concatenate([r[key][v] for v in views]))
    return np.asarray(rows,float)

class TypedPreprocessor:
    def fit(self,records,dims,cap=128):
        self.dims=dimensions(dims);self.original_dims=dims;self.cap=cap
        self.scalers={}
        for v in VIEWS:
            chunks=[subset(split_views(r)[v],24) for r in records if len(split_views(r)[v])]
            x=np.concatenate(chunks) if chunks else np.empty((0,self.dims[v]))
            self.scalers[v]=Robust().fit(x)
        self.summary_scaler=Robust().fit(summary_matrix(records,dims))
        return self
    def transform(self,records):
        xx={};mm={}
        for v in VIEWS:
            cap=min(CAPS[v],self.cap) if v=='eeg' else CAPS[v]
            x=np.zeros((len(records),cap,2*self.dims[v]),np.float32)
            mask=np.zeros((len(records),cap),bool)
            for i,r in enumerate(records):
                bag=subset(split_views(r)[v],cap);n=len(bag)
                if n:
                    z,observed=self.scalers[v].transform(bag)
                    x[i,:n]=np.c_[z,observed];mask[i,:n]=True
            xx[v]=x;mm[v]=mask
        z,observed=self.summary_scaler.transform(summary_matrix(records,self.original_dims))
        return xx,mm,np.c_[z,observed].astype(np.float32)

def focused_schema(schema, enabled=True):
    # Preserve all real labels. Only training/selection weights change.
    return [{**t,'weight':(1. if t['name'] in FOCUS else .05)} for t in schema] if enabled else schema

def focus_mae(y,p,scale,schema):
    values=[]
    for family in ('PSQI','SCL90','EPQ'):
        per=[]
        for j,t in enumerate(schema):
            if t['family']!=family or t['name'] not in FOCUS:continue
            ok=np.isfinite(y[:,j])&np.isfinite(p[:,j])
            if ok.any():per.append(float(np.mean(abs(y[ok,j]-p[ok,j]))/scale[j]))
        if per:values.append(float(np.mean(per)))
    return float(np.mean(values)) if values else float('inf')
