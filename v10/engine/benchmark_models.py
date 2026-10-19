"""Train-only regression transforms and an asynchronous two-modality kernel."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
import importlib.metadata
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics.pairwise import euclidean_distances
from baselines import SafeImputer
from assets import checked_weight

SPECS = {
    'power_ridge':('eeg_power','ridge'), 'cov_ridge':('eeg_cov','ridge'),
    'eye_svr':('eye_base','svr'), 'eye_tabpfn':('eye_base','tabpfn'),
    'cbramod_ridge':('cbramod','ridge'), 'cbramod_svr':('cbramod','svr'),
    'concat_ridge':('fusion','ridge'), 'kernel_fusion':('fusion','kernel'),
    'demographics_ridge':('demographics','ridge'),
    'eye_target_svr':('eye_target','svr'), 'eye_target_tabpfn':('eye_target','tabpfn'),
    'target_concat_ridge':('fusion_target','ridge'), 'target_kernel_fusion':('fusion_target','kernel'),
}
DEFAULT_MODELS = ['power_ridge','cov_ridge','eye_svr','cbramod_ridge','concat_ridge','kernel_fusion','demographics_ridge']


def candidates(kind):
    if kind == 'ridge': return [{'alpha':a} for a in [1.,10.,100.,1000.,10000.]]
    if kind == 'svr': return [{'C':c} for c in [.1,1.,10.]]
    if kind == 'tabpfn': return [{'n_estimators':8}]
    if kind == 'kernel': return [{'alpha':a,'weight':w} for a in [.1,1.,10.] for w in [0.,.25,.5,.75,1.]]
    raise ValueError(kind)


class BlendedKernel(BaseEstimator, RegressorMixin):
    """Each modal scaler and median RBF width are fitted on training people only."""
    def __init__(self, alpha=1., weight=.5, split=400):
        self.alpha=alpha; self.weight=weight; self.split=split

    def fit(self,x,y):
        if not 0 <= self.weight <= 1 or self.alpha <= 0: raise ValueError('Invalid kernel parameters')
        if x.shape[1] <= self.split: raise ValueError('Missing eye block')
        self.transforms_=[]; self.reference_=[]; self.widths_=[]
        kernels=[]
        for part in [x[:,:self.split],x[:,self.split:]]:
            transform=make_pipeline(SafeImputer(),StandardScaler())
            z=transform.fit_transform(part)
            d=euclidean_distances(z,z,squared=True)
            positive=d[np.triu_indices(len(z),1)];positive=positive[positive>1e-12]
            width=float(np.median(positive)) if len(positive) else 1.
            self.transforms_.append(transform);self.reference_.append(z);self.widths_.append(width)
            kernels.append(np.exp(-d/width))
        k=self.weight*kernels[0]+(1-self.weight)*kernels[1]
        self.dual_=np.linalg.solve(k+self.alpha*np.eye(len(k)),np.asarray(y))
        return self

    def predict(self,x):
        kernels=[]
        for part,tr,ref,width in zip([x[:,:self.split],x[:,self.split:]],self.transforms_,self.reference_,self.widths_):
            kernels.append(np.exp(-euclidean_distances(tr.transform(part),ref,squared=True)/width))
        return (self.weight*kernels[0]+(1-self.weight)*kernels[1])@self.dual_


def fit_bundle(kind,param,x,y,device='cpu',threads=2,seed=20261015,weights=None):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=2 or not np.isfinite(y).all() or np.isinf(x).any(): raise ValueError('Invalid training data')
    mean=float(y.mean());std=float(max(y.std(),1e-6))
    if kind in {'ridge','svr'}:
        reg=Ridge(**param) if kind=='ridge' else SVR(**param,epsilon=.1,gamma='scale')
        model=make_pipeline(SafeImputer(),StandardScaler(),reg)
    elif kind=='kernel': model=BlendedKernel(**param)
    elif kind=='tabpfn':
        import torch
        if importlib.metadata.version('tabpfn')!='2.0.9': raise RuntimeError('Install tabpfn==2.0.9 for fixed v2 reproduction')
        from tabpfn import TabPFNRegressor
        torch.set_num_threads(threads)
        model=TabPFNRegressor(**param,model_path=checked_weight('tabpfn',weights),device=device,
            random_state=seed,n_jobs=1,inference_precision=torch.float32,fit_mode='fit_preprocessors',memory_saving_mode=True)
    else: raise ValueError(kind)
    model.fit(x,(y-mean)/std)
    return {'model':model,'target_mean':mean,'target_std':std,'n_features':x.shape[1],
            'kind':kind,'parameter':param}


def predict_bundle(bundle,x):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=bundle['n_features'] or np.isinf(x).any(): raise ValueError('Feature mismatch')
    p=np.asarray(bundle['model'].predict(x),float).reshape(-1)*bundle['target_std']+bundle['target_mean']
    if not np.isfinite(p).all(): raise ValueError('Nonfinite predictions')
    return p
