import copy,random
import numpy as np
import torch
from torch import nn
from scipy.optimize import linprog
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics.pairwise import rbf_kernel
from preprocessing import Robust,target_stats
from engine import bound
from features_v171 import TypedPreprocessor,summary_matrix,split_views,VIEWS,focused_schema,focus_mae
from models_v171 import TypedMIL,TabEnsemble,family_losses,pcgrad_backward

CONFIGS_NEW=[dict(id='small',width=16,hidden=32,dropout=.3,lr=.0015,weight_decay=.05,members=8),
             dict(id='regularized',width=24,hidden=64,dropout=.45,lr=.001,weight_decay=.1,members=16)]
RECIPES={
 'mil_mean':dict(attention=False,skip=True,pcgrad=False,focus=True),
 'mil_mean_pcgrad':dict(attention=False,skip=True,pcgrad=True,focus=True),
 'mil_attention':dict(attention=True,skip=True,pcgrad=False,focus=True),
 'mil_attention_pcgrad':dict(attention=True,skip=True,pcgrad=True,focus=True),
 'mil_no_skip':dict(attention=True,skip=False,pcgrad=True,focus=True),
 'mil_original_weights':dict(attention=True,skip=True,pcgrad=True,focus=False),
 'mil_cap32':dict(attention=True,skip=True,pcgrad=True,focus=True,cap=32),
 'tab_ensemble':dict(tabular=True,focus=True,pcgrad=False),
}

def pack(data,device):
    x,m,s=data
    return ({k:torch.as_tensor(v,device=device) for k,v in x.items()},
            {k:torch.as_tensor(v,device=device) for k,v in m.items()},torch.as_tensor(s,device=device))

def build(prep,schema,config,recipe):
    sd=2*len(prep.summary_scaler.center)
    if recipe.get('tabular'):return TabEnsemble(sd,schema,**config)
    return TypedMIL(prep.dims,sd,schema,**config,**recipe)

def fit_new(records,y,dims,schema,config,name,seeds,epochs,device='cpu',validation=None,patience=30,cap=128):
    recipe=RECIPES[name];weights=focused_schema(schema,recipe['focus'])
    cap=recipe.get('cap',cap)
    prep=TypedPreprocessor().fit(records,dims,cap);tx,tm,ts=pack(prep.transform(records),device)
    mean,scale=target_stats(y);truth=torch.as_tensor(np.nan_to_num((y-mean)/scale,nan=0).astype('float32'),device=device)
    mask=torch.as_tensor(np.isfinite(y),device=device)
    if validation:
        vr,vy=validation;vx,vm,vs=pack(prep.transform(vr),device)
    states=[];logs=[];pp=[]
    for seed in seeds:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
        model=build(prep,schema,config,recipe).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        best=float('inf');best_epoch=0;best_state=None;stale=0;trace=[]
        for epoch in range(epochs+1):
            conflict=None
            if epoch:
                model.train();optimizer.zero_grad(set_to_none=True)
                pred=model(tx,tm,ts);terms=family_losses(pred,truth,mask,weights)
                loss=torch.stack(terms).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                if recipe.get('pcgrad'):conflict=pcgrad_backward(terms,model)
                else:loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            if validation:
                model.eval()
                with torch.no_grad():
                    p=model(vx,vm,vs)
                    if p.ndim==3:p=p.mean(1)
                    p=bound(p.cpu().numpy()*scale+mean,schema)
                score=focus_mae(vy,p,scale,schema)
                if not np.isfinite(score):raise FloatingPointError('Nonfinite validation score')
                if score<best-1e-5:
                    best=score;best_epoch=epoch;stale=0
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:stale+=1
                trace.append(dict(epoch=epoch,score=score,pcgrad_conflict_fraction=conflict))
                if epoch>=30 and stale>=patience:break
        if validation:model.load_state_dict(best_state)
        else:best_epoch=epochs
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        logs.append(dict(seed=seed,epochs=best_epoch,last_epoch=epoch,score=best if validation else None,
                         parameters=sum(p.numel() for p in model.parameters()),trace=trace))
        if validation:
            model.eval()
            with torch.no_grad():
                p=model(vx,vm,vs)
                if p.ndim==3:p=p.mean(1)
                pp.append(bound(p.cpu().numpy()*scale+mean,schema))
    return dict(kind='new_neural',prep=prep,schema=schema,mean=mean,scale=scale,config=config,
                recipe_name=name,states=states,logs=logs,cap=cap),np.mean(pp,0) if pp else None

def predict_new(artifact,records,device='cpu'):
    x,m,s=pack(artifact['prep'].transform(records),device);pp=[]
    for state in artifact['states']:
        model=build(artifact['prep'],artifact['schema'],artifact['config'],RECIPES[artifact['recipe_name']]).to(device)
        model.load_state_dict(state);model.eval()
        with torch.no_grad():
            p=model(x,m,s)
            if p.ndim==3:p=p.mean(1)
            pp.append(bound(p.cpu().numpy()*artifact['scale']+artifact['mean'],artifact['schema']))
    return np.mean(pp,0)

def classical_configs(foundation=False):
    result={
      'ridge_distribution':[dict(kind='ridge',alpha=a) for a in [.1,1.,10.]],
      'svr_distribution':[dict(kind='svr',C=c,epsilon=.1) for c in [.1,1.,10.]],
      'extra_trees':[dict(kind='trees',leaf=k) for k in [8,16]],
      'kernel_distribution':[dict(kind='kernel',weights=w,alpha=a) for w in
         [[.25,.25,.25,.25],[.5,.5,0,0],[0,0,.5,.5],[1,0,0,0]] for a in [.01,.1,1.]],
    }
    if foundation:result['tabpfn_v2']=[dict(kind='tabpfn',pca=k,output=o) for k in [64,0] for o in ['median','mean']]
    return result

def fit_classical(records,y,dims,schema,config,seed=20261023,device='cpu'):
    if config['kind']=='tabpfn':
        from foundation_v171 import fit_context
        return fit_context(records,y,dims,schema,config,device)
    raw=summary_matrix(records,dims,True);scaler=Robust().fit(raw);z,missing=scaler.transform(raw)
    x=np.c_[z,missing];mu,scale=target_stats(y);models=[];kind=config['kind']
    artifact=dict(kind='new_classical',method=kind,config=config,scaler=scaler,dims=dims,
                  mean=mu,scale=scale,schema=schema,models=models)
    if kind=='kernel':
        kernel=KernelViews().fit(records,dims);K=kernel.transform(records)
        K=sum(w*k for w,k in zip(config['weights'],K));artifact['kernel']=kernel
    if kind=='trees':
        # Identical observed-label masks can share a multi-output forest. This avoids
        # refitting hundreds of near-identical trees for correlated questionnaire items.
        groups={}
        for j in range(y.shape[1]):
            good=np.isfinite(y[:,j])
            if good.sum()>=2:groups.setdefault(good.tobytes(),[]).append(j)
        artifact['tree_groups']=[]
        for key,columns in groups.items():
            good=np.frombuffer(key,dtype=bool)
            yy=(y[good][:,columns]-mu[columns])/scale[columns]
            reg=ExtraTreesRegressor(n_estimators=192,min_samples_leaf=config['leaf'],max_features=.7,n_jobs=1,random_state=seed).fit(x[good],yy.ravel() if len(columns)==1 else yy)
            artifact['tree_groups'].append((columns,reg))
        return artifact
    for j in range(y.shape[1]):
        good=np.isfinite(y[:,j]);n=int(good.sum())
        if n<2:models.append(None);continue
        yy=(y[good,j]-mu[j])/scale[j]
        if kind=='ridge':model=Ridge(alpha=config['alpha']*n,solver='cholesky').fit(x[good],yy)
        elif kind=='svr':model=SVR(C=config['C'],epsilon=config['epsilon'],gamma=1/x.shape[1]).fit(x[good],yy)
        elif kind=='trees':model=ExtraTreesRegressor(n_estimators=192,min_samples_leaf=config['leaf'],max_features=.7,n_jobs=1,random_state=seed).fit(x[good],yy)
        else:
            # Kernel intercept/label scaling estimated only from training labels.
            L=K[np.ix_(good,good)]+config['alpha']*n*np.eye(n)
            model=(np.flatnonzero(good),np.linalg.solve(L,yy))
        models.append(model)
    return artifact

class KernelViews:
    def fit(self,records,dims):
        self.dims=dims;self.scalers={};self.train={};self.present={}
        for v in VIEWS:
            raw=summary_matrix(records,dims,True,[v]);sc=Robust().fit(raw);z,_=sc.transform(raw)
            self.scalers[v]=sc;self.train[v]=z;self.present[v]=np.array([bool(len(split_views(r)[v])) for r in records])
        return self
    def transform(self,records):
        kernels=[]
        for v in VIEWS:
            raw=summary_matrix(records,self.dims,True,[v]);z,_=self.scalers[v].transform(raw)
            available=np.array([bool(len(split_views(r)[v])) for r in records])
            K=rbf_kernel(z,self.train[v],gamma=1/max(z.shape[1],1))
            # Product with a presence outer product preserves positive semidefiniteness.
            kernels.append(K*available[:,None]*self.present[v][None])
        return kernels

def predict_classical(artifact,records,device=None):
    if artifact['kind']=='foundation_context':
        from foundation_v171 import predict_context
        return predict_context(artifact,records,device)
    z,mask=artifact['scaler'].transform(summary_matrix(records,artifact['dims'],True));x=np.c_[z,mask]
    p=np.broadcast_to(artifact['mean'],(len(records),len(artifact['schema']))).copy()
    if 'tree_groups' in artifact:
        for columns,model in artifact['tree_groups']:
            pred=np.asarray(model.predict(x)).reshape(len(records),len(columns))
            p[:,columns]=pred*artifact['scale'][columns]+artifact['mean'][columns]
        return bound(p,artifact['schema'])
    if artifact['method']=='kernel':
        K=sum(w*k for w,k in zip(artifact['config']['weights'],artifact['kernel'].transform(records)))
    for j,model in enumerate(artifact['models']):
        if model is None:continue
        if artifact['method']=='kernel':idx,coef=model;q=K[:,idx]@coef
        else:q=model.predict(x)
        p[:,j]=q*artifact['scale'][j]+artifact['mean'][j]
    return bound(p,artifact['schema'])

def convex_weights(y,predictions,scale,anchor,penalty=.02):
    """L1 stacking simplex fit solely on outer-training inner-OOF predictions.

    Fixed penalty on departure from the V17 Ridge anchor. No outer-label tuning.
    """
    names=list(predictions);P=np.stack([predictions[n] for n in names],-1)
    weights=np.zeros((y.shape[1],len(names)));prior=np.zeros(len(names));prior[names.index(anchor)]=1
    for j in range(y.shape[1]):
        ok=np.isfinite(y[:,j])&np.isfinite(P[:,j]).all(1);n=int(ok.sum());k=len(names)
        if n<20:weights[j]=prior;continue
        X=P[ok,j]/scale[j];t=y[ok,j]/scale[j]
        # Variables: k weights, n absolute residuals, k |w-prior| auxiliaries.
        obj=np.r_[np.zeros(k),np.ones(n)/n,np.full(k,penalty)]
        A=np.vstack([np.c_[X,-np.eye(n),np.zeros((n,k))],
                     np.c_[-X,-np.eye(n),np.zeros((n,k))],
                     np.c_[np.eye(k),np.zeros((k,n)),-np.eye(k)],
                     np.c_[-np.eye(k),np.zeros((k,n)),-np.eye(k)]])
        b=np.r_[t,-t,prior,-prior];eq=np.r_[np.ones(k),np.zeros(n+k)][None]
        result=linprog(obj,A_ub=A,b_ub=b,A_eq=eq,b_eq=[1.],bounds=(0,None),method='highs')
        weights[j]=result.x[:k] if result.success else prior
        weights[j]=np.maximum(weights[j],0);weights[j]/=weights[j].sum()
    return names,weights

def combine(predictions,names,weights):
    return np.einsum('ntk,tk->nt',np.stack([predictions[n] for n in names],-1),weights)
