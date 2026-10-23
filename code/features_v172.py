"""Training-fold EEG geometry/PCA; no outcome or test-population fitting."""
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from common import load_data,load_signals,read,sha
from preprocessing import Robust
from features_v171 import summary_matrix,split_views

def load_inputs(data,extras,with_targets=True):
    if with_targets:meta,records,y=load_data(data)
    else:meta,records=load_signals(data);y=None
    em=read(Path(extras)/'manifest.json')
    if em['state']!='complete' or em['data_manifest_sha256']!=sha(Path(data)/'manifest.json'):
        raise ValueError('Incomplete/wrong EEG augmentation')
    for e,r in zip(meta['subjects'],records):
        a=em['subjects'][e['subject_id']];p=Path(extras)/a['file']
        if sha(p)!=a['sha256']:raise ValueError('Augmentation hash mismatch')
        with np.load(p,allow_pickle=False) as z:r.update(cov=z['cov'].copy(),embedding=z['embedding'].copy())
        r['subject_id']=e['subject_id'];r['group_id']=e['group_id']
        good=e['modality_available']['eeg']
        if r['cov'].shape!=(5,16,16) or r['embedding'].shape!=(400,):raise ValueError('Wrong EEG feature dimensions')
        if good and not (np.isfinite(r['cov']).all() and np.isfinite(r['embedding']).all()):raise ValueError('Missing observed EEG features')
        if not good and (np.isfinite(r['cov']).any() or np.isfinite(r['embedding']).any()):raise ValueError('Unexpected absent-EEG features')
    return meta,records,y

def symmetric(a):return (a+np.swapaxes(a,-1,-2))*.5

def matrix_function(a,fn):
    d,u=np.linalg.eigh(symmetric(a))
    floor=np.maximum(d.max(-1,keepdims=True)*1e-12,1e-20)
    if (d < -floor*10).any():raise ValueError('Matrix is not positive semidefinite')
    return symmetric((u*fn(np.maximum(d,floor))[...,None,:])@np.swapaxes(u,-1,-2))

def exp_symmetric(a):
    d,u=np.linalg.eigh(symmetric(a))
    return symmetric((u*np.exp(d)[...,None,:])@np.swapaxes(u,-1,-2))

def geometric_mean(c,tol=1e-6,max_iter=80):
    g=exp_symmetric(matrix_function(c,np.log).mean(0))
    history=[]
    for iteration in range(max_iter):
        root=matrix_function(g,np.sqrt);inv=matrix_function(g,lambda d:1/np.sqrt(d))
        delta=matrix_function(inv@c@inv,np.log).mean(0)
        error=float(np.linalg.norm(delta,'fro'));history.append(error)
        if error<tol:break
        g=symmetric(root@exp_symmetric(.5*delta)@root)
    if history[-1]>tol*10:raise RuntimeError('Riemannian mean did not converge')
    return g,dict(iterations=iteration+1,residual_norm=history[-1])

class EEGGeometry:
    def fit(self,records):
        c=np.stack([r['cov'] for r in records if np.isfinite(r['cov']).all()]) if any(np.isfinite(r['cov']).all() for r in records) else None
        self.references=[];self.convergence=[]
        if c is not None:
            for band in range(5):
                g,log=geometric_mean(c[:,band]);self.references.append(matrix_function(g,lambda x:1/np.sqrt(x)));self.convergence.append(log)
        self.n_fit=0 if c is None else len(c)
        return self
    def transform(self,records):
        result=np.full((len(records),680),np.nan)
        upper=np.triu_indices(16);weights=np.where(upper[0]==upper[1],1.,np.sqrt(2))
        if not self.references:return result
        for i,r in enumerate(records):
            if not np.isfinite(r['cov']).all():continue
            result[i]=np.concatenate([matrix_function(w@r['cov'][b]@w,np.log)[upper]*weights for b,w in enumerate(self.references)])
        return result

class ReducedBlock:
    def fit(self,x,components=32):
        good=np.isfinite(x).all(1);self.scaler=Robust().fit(x[good]);self.pca=None
        self.n_fit=int(good.sum());self.output_dim=components
        if good.sum()>1:
            z,_=self.scaler.transform(x[good]);k=min(components,len(z)-1,z.shape[1])
            self.pca=PCA(n_components=k,svd_solver='full',whiten=False).fit(z)
            self.post=Robust().fit(self.pca.transform(z));self.output_dim=k
        return self
    def transform(self,x):
        good=np.isfinite(x).all(1);out=np.zeros((len(x),self.output_dim),np.float32)
        if self.pca is not None and good.any():
            z,_=self.scaler.transform(x[good]);out[good]=self.post.transform(self.pca.transform(z))[0]
        return np.c_[out,good.astype(np.float32)]

class FeaturePrep:
    def fit(self,records,dims,kind='combined'):
        self.dims=dims;self.kind=kind;self.training_ids=[r['subject_id'] for r in records]
        self.blocks={};self.geometry=None
        self.scalers=[Robust().fit(self.raw(records,v)) for v in range(3)]
        if kind in ['tangent','combined']:
            self.geometry=EEGGeometry().fit(records)
            self.blocks['tangent']=ReducedBlock().fit(self.geometry.transform(records))
        if kind in ['frozen','combined']:
            self.blocks['frozen']=ReducedBlock().fit(np.stack([r['embedding'] for r in records]))
        return self
    def raw(self,records,v):
        views=[('eeg',),('eye',),('ppg','nni')][v]
        return summary_matrix(records,self.dims,views=views)
    def transform(self,records):
        xx=[]
        for v,s in enumerate(self.scalers):
            z,mask=s.transform(self.raw(records,v));xx.append(np.c_[z,mask])
        if 'tangent' in self.blocks:xx[0]=np.c_[xx[0],self.blocks['tangent'].transform(self.geometry.transform(records))]
        if 'frozen' in self.blocks:xx[0]=np.c_[xx[0],self.blocks['frozen'].transform(np.stack([r['embedding'] for r in records]))]
        presence=np.array([[len(r['eeg'])>0,len(r['eye'])>0,len(r['heart'])>0] for r in records],bool)
        return [a.astype(np.float32) for a in xx],presence
    def matrix(self,records):
        x,m=self.transform(records);return np.c_[*x,m].astype(np.float32)
