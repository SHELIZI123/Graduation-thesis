import copy,random
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
from common import MODALITIES
from preprocessing import Robust,BagPreprocessor,summaries,target_stats,normalized_mae
from model import TriScale,masked_group_loss

CONFIGS=[{'id':'compact','width':16,'hidden':32,'dropout':.25,'lr':.003,'weight_decay':.03,'modality_dropout':.15},
         {'id':'regularized','width':24,'hidden':48,'dropout':.4,'lr':.002,'weight_decay':.1,'modality_dropout':.2}]
RIDGE_LAMBDAS=[.1,1.,10.]

def bound(p,schema):
    p=np.asarray(p,float).copy()
    for j,t in enumerate(schema):
        if t['low'] is not None:p[:,j]=np.maximum(p[:,j],t['low'])
        if t['high'] is not None:p[:,j]=np.minimum(p[:,j],t['high'])
    return p

def tensor_pack(pair,device):
    x,m=pair
    return {k:torch.as_tensor(v,device=device) for k,v in x.items()}, {k:torch.as_tensor(v,device=device) for k,v in m.items()}

def fit_neural(records,y,dims,schema,config,routing,seeds,epochs,device='cpu',validation=None,patience=20,cap=32):
    prep=BagPreprocessor().fit(records,dims); tx,tm=tensor_pack(prep.transform(records,cap),device)
    mu,scale=target_stats(y); standardized=(y-mu)/scale
    observed=torch.as_tensor(np.isfinite(y),device=device)
    truth=torch.as_tensor(np.nan_to_num(standardized,nan=0).astype(np.float32),device=device)
    if validation:
        vr,vy=validation;vx,vm=tensor_pack(prep.transform(vr,cap),device)
    states=[];logs=[];validation_predictions=[]
    for seed in seeds:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
        model=TriScale(dims,schema,**{k:v for k,v in config.items() if k in ['width','hidden','dropout','modality_dropout']},routing=routing).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
        best=float('inf');best_epoch=0;best_state=copy.deepcopy(model.state_dict());stale=0;trace=[]
        for epoch in range(epochs+1):
            if epoch:
                model.train();optimizer.zero_grad(set_to_none=True)
                pred=model(tx,tm);loss=masked_group_loss(pred,truth,observed,schema)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if validation:
                model.eval()
                with torch.no_grad():p=bound(model(vx,vm).cpu().numpy()*scale+mu,schema)
                value=normalized_mae(vy,p,scale,schema)
                if not np.isfinite(value):raise FloatingPointError('Nonfinite validation loss')
                trace.append({'epoch':epoch,'validation_normalized_mae':value})
                if value<best-1e-5:
                    best=value;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
                else:stale+=1
                if epoch>=20 and stale>=patience:break
        if validation:model.load_state_dict(best_state)
        else:best_epoch=epochs
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        logs.append({'seed':seed,'epochs':best_epoch,'last_epoch':epoch,'inner_score':best if validation else None,
                     'parameters':sum(p.numel() for p in model.parameters()),'trace':trace})
        if validation:
            model.eval()
            with torch.no_grad():validation_predictions.append(bound(model(vx,vm).cpu().numpy()*scale+mu,schema))
    artifact={'kind':'neural','preprocessor':prep,'mean':mu,'scale':scale,'dims':dims,'schema':schema,
              'config':config,'routing':routing,'states':states,'logs':logs,'cap':cap,'seeds':seeds}
    p=np.mean(validation_predictions,0) if validation_predictions else None
    return artifact,p

def predict_neural(artifact,records,device='cpu'):
    x,m=tensor_pack(artifact['preprocessor'].transform(records,artifact['cap']),device)
    pp=[]
    for state in artifact['states']:
        c=artifact['config'];model=TriScale(artifact['dims'],artifact['schema'],**{k:v for k,v in c.items() if k in ['width','hidden','dropout','modality_dropout']},routing=artifact['routing']).to(device)
        model.load_state_dict(state);model.eval()
        with torch.no_grad():pp.append(bound(model(x,m).cpu().numpy()*artifact['scale']+artifact['mean'],artifact['schema']))
    return np.mean(pp,0)

def baseline_matrix(records,dims,modalities,mask_only=False):
    if mask_only:return np.array([[bool(len(r[m])) for m in MODALITIES] for r in records],float)
    return summaries(records,dims,modalities)

def fit_ridge(records,y,dims,schema,penalty,modalities=MODALITIES,mask_only=False):
    raw=baseline_matrix(records,dims,modalities,mask_only);scaler=Robust().fit(raw)
    z,mask=scaler.transform(raw);x=np.c_[z,mask];mu,scale=target_stats(y)
    groups={}
    for j,col in enumerate(y.T):
        good=np.isfinite(col)
        if good.sum()>=2:groups.setdefault(good.tobytes(),[]).append(j)
    models=[]
    for key,columns in groups.items():
        good=np.frombuffer(key,dtype=bool);yy=(y[good][:,columns]-mu[columns])/scale[columns]
        reg=Ridge(alpha=penalty*int(good.sum()),solver='cholesky').fit(x[good],yy)
        models.append((columns,reg))
    return {'kind':'ridge','scaler':scaler,'models':models,'mean':mu,'scale':scale,'modalities':tuple(modalities),
            'dims':dims,'schema':schema,'penalty':penalty,'mask_only':mask_only}

def predict_ridge(artifact,records):
    raw=baseline_matrix(records,artifact['dims'],artifact['modalities'],artifact['mask_only'])
    z,mask=artifact['scaler'].transform(raw);x=np.c_[z,mask]
    p=np.broadcast_to(artifact['mean'],(len(records),len(artifact['schema']))).copy()
    for columns,model in artifact['models']:
        p[:,columns]=model.predict(x)*artifact['scale'][columns]+artifact['mean'][columns]
    return bound(p,artifact['schema'])

def predict(artifact,records,device='cpu'):
    if artifact['kind']=='neural':return predict_neural(artifact,records,device)
    return predict_ridge(artifact,records)
