"""Official TabPFN-v2 regression adapter, isolated from the required core runtime."""
from pathlib import Path
import os,importlib.metadata
import numpy as np
from sklearn.decomposition import PCA
from common import sha
from preprocessing import Robust,target_stats
from features_v171 import summary_matrix,split_views,VIEWS
from engine import bound

def checkpoint():
    p=Path(__file__).resolve().parents[1]/'weights/tabpfn-v2-regressor.ckpt'
    if not p.is_file():raise FileNotFoundError('Missing official TabPFN-v2 checkpoint: '+str(p))
    return p

def matrix(records,dims):
    presence=np.array([[bool(len(split_views(r)[v])) for v in VIEWS] for r in records],np.float32)
    return summary_matrix(records,dims),presence

def fit_context(records,y,dims,schema,config,device):
    raw,presence=matrix(records,dims);scaler=Robust().fit(raw);z,_=scaler.transform(raw);pca=None
    if config['pca']:
        pca=PCA(n_components=min(config['pca'],len(z)-1,z.shape[1]),svd_solver='full').fit(z)
        z=pca.transform(z)
    x=np.c_[z,presence].astype('float32')
    if x.shape[1]>500:raise ValueError('Nature 2025 checkpoint feature range exceeded')
    mean,scale=target_stats(y)
    return dict(kind='foundation_context',config=config,dims=dims,schema=schema,scaler=scaler,pca=pca,
                x=x,y=np.array(y,copy=True),mean=mean,scale=scale,device=device,checkpoint_sha256=sha(checkpoint()),
                package_version='2.0.9',pretrained_model='official TabPFN-v2 regressor')

def predict_context(a,records,device=None):
    os.environ['TABPFN_DISABLE_TELEMETRY']='1'
    if importlib.metadata.version('tabpfn')!=a['package_version']:
        raise RuntimeError('Install requirements.foundation.txt in the active environment')
    from tabpfn import TabPFNRegressor
    path=checkpoint()
    if sha(path)!=a['checkpoint_sha256']:raise ValueError('Pretrained checkpoint changed')
    raw,presence=matrix(records,a['dims']);z,_=a['scaler'].transform(raw)
    if a['pca'] is not None:z=a['pca'].transform(z)
    query=np.c_[z,presence].astype('float32')
    p=np.broadcast_to(a['mean'],(len(records),len(a['schema']))).copy()
    model=TabPFNRegressor(n_estimators=4,device=device or a['device'],model_path=path,
                         random_state=20261023,n_jobs=1,fit_mode='low_memory')
    # Only the six declared primary targets require new foundation contexts.
    # All other targets retain training means within this candidate, documented explicitly.
    from features_v171 import FOCUS
    for j,t in enumerate(a['schema']):
        if t['name'] not in FOCUS:continue
        good=np.isfinite(a['y'][:,j])
        if good.sum()<10 or np.std(a['y'][good,j])<1e-8:continue
        yy=(a['y'][good,j]-a['mean'][j])/a['scale'][j]
        model.fit(a['x'][good],yy)
        p[:,j]=model.predict(query,output_type=a['config']['output'])*a['scale'][j]+a['mean'][j]
    return bound(p,a['schema'])
