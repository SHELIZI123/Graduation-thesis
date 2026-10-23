"""Official TabPFN-v2, explicit FP32/math attention and auditable query tensors.

Numerical reproducibility improvements, not a promise of cross-platform equality.
"""
import os,hashlib,platform,sys,importlib.metadata,contextlib
import numpy as np
import torch
from sklearn.decomposition import PCA
from common import sha
from preprocessing import Robust,target_stats
from foundation_v171 import matrix,checkpoint
from features_v171 import FOCUS
from engine import bound

def digest(x):
    x=np.ascontiguousarray(x);return hashlib.sha256(str(x.shape).encode()+str(x.dtype).encode()+x.tobytes()).hexdigest()

def runtime():
    names=['numpy','scipy','scikit-learn','joblib','torch','tabpfn','einops','threadpoolctl']
    versions={}
    for n in names:
        try:versions[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:versions[n]=None
    return dict(python=sys.version,platform=platform.platform(),versions=versions,
                cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                precision='torch.float32',attention_backend='math',allow_tf32=False)

def configure_numerics():
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')

def project(pca,z):
    z=np.asarray(z,dtype=np.float64)
    return z if pca is None else (z-pca.mean_)@pca.components_.T

def query(a,records):
    raw,presence=matrix(records,a['dims']);z,_=a['scaler'].transform(raw)
    return np.c_[project(a['pca'],z),presence].astype(np.float32)

def fit_context(records,y,dims,schema,config,device):
    if importlib.metadata.version('tabpfn')!='2.0.9':raise RuntimeError('TabPFN 2.0.9 required')
    raw,presence=matrix(records,dims);scaler=Robust().fit(raw);z,_=scaler.transform(raw);z=z.astype(np.float64)
    pca=PCA(n_components=min(32,len(z)-1,z.shape[1]),svd_solver='full').fit(z)
    x=np.c_[project(pca,z),presence].astype(np.float32);mean,scale=target_stats(y)
    return dict(kind='v173_foundation',config=config,dims=dims,schema=schema,scaler=scaler,pca=pca,
                x=x,y=y.copy(),mean=mean,scale=scale,device=device,checkpoint_sha256=sha(checkpoint()),
                package_version='2.0.9',precision='float32',attention_backend='math',
                training_feature_sha256=digest(x),fit_runtime=runtime())

def predict_context(a,records,device='cpu',replay=False):
    os.environ['TABPFN_DISABLE_TELEMETRY']='1';configure_numerics()
    if importlib.metadata.version('tabpfn')!=a['package_version']:raise RuntimeError('TabPFN version differs')
    if sha(checkpoint())!=a['checkpoint_sha256'] or digest(a['x'])!=a['training_feature_sha256']:raise ValueError('Foundation context changed')
    from tabpfn import TabPFNRegressor
    from torch.nn.attention import sdpa_kernel,SDPBackend
    ids=[r['subject_id'] for r in records];computed=query(a,records);used=computed
    snapshot=a.get('prediction_query_snapshot');difference=0.
    if replay:
        if snapshot is None or ids!=snapshot['subject_ids']:raise ValueError('Replay query IDs differ')
        if digest(snapshot['matrix'])!=snapshot['sha256']:raise ValueError('Saved query tensor changed')
        difference=float(abs(computed-snapshot['matrix']).max());used=snapshot['matrix']
    p=np.broadcast_to(a['mean'],(len(records),len(a['schema']))).copy();precision_records=[]
    with sdpa_kernel(SDPBackend.MATH):
        for j,t in enumerate(a['schema']):
            if t['name'] not in FOCUS:continue
            good=np.isfinite(a['y'][:,j])
            if good.sum()<10 or np.std(a['y'][good,j])<1e-8:continue
            model=TabPFNRegressor(n_estimators=4,device=device,model_path=checkpoint(),random_state=20261023,
                                 n_jobs=1,fit_mode='low_memory',inference_precision=torch.float32)
            model.fit(a['x'][good],(a['y'][good,j]-a['mean'][j])/a['scale'][j])
            if model.use_autocast_ or model.forced_inference_dtype_!=torch.float32:raise RuntimeError('Requested FP32 not active')
            p[:,j]=model.predict(used,output_type='median')*a['scale'][j]+a['mean'][j]
            precision_records.append(dict(target=t['name'],autocast=False,dtype=str(model.forced_inference_dtype_)))
    diagnostic=dict(query_replay_uses_saved_features=replay,computed_query_sha256=digest(computed),used_query_sha256=digest(used),
                    query_recompute_max_difference=difference,precision=precision_records)
    if not replay:
        a['prediction_query_snapshot']=dict(subject_ids=ids,matrix=used.copy(),sha256=digest(used))
        a['prediction_runtime']=runtime()
    return bound(p,a['schema']),diagnostic
