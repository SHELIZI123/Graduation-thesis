"""Complete, fixed-epoch pipelines. Validation labels never enter fitting."""
import random
import numpy as np
import torch
from sklearn.linear_model import Ridge
from features_v172 import FeaturePrep
from preprocessing import target_stats,group_splits
from engine import fit_ridge,predict_ridge,bound
from foundation_v171 import fit_context,predict_context
from models_v172 import ResidualFusion,masked_loss,update_gradients

class FitCache:
    """Run-local cache keyed by exact ordered training IDs, never validation IDs."""
    def __init__(self,dims,schema):self.dims=dims;self.schema=schema;self.preps={};self.contexts={}
    def prep(self,records,kind):
        key=(tuple(r['subject_id'] for r in records),kind)
        if key not in self.preps:self.preps[key]=FeaturePrep().fit(records,self.dims,kind)
        return self.preps[key]
    def residual(self,records,y):
        key=tuple(r['subject_id'] for r in records)
        if key in self.contexts:return self.contexts[key]
        n=len(records);groups=[r['group_id'] for r in records]
        base=fit_ridge(records,y,self.dims,self.schema,1.)
        oof=np.full_like(y,np.nan);splits=[]
        for tr,va in group_splits(list(range(n)),groups,3,20261024):
            b=fit_ridge([records[i] for i in tr],y[tr],self.dims,self.schema,1.)
            oof[va]=predict_ridge(b,[records[i] for i in va])
            splits.append(dict(train_ids=[key[i] for i in tr],validation_ids=[key[i] for i in va]))
        if not np.isfinite(oof).all():raise ValueError('Incomplete residual cross-fitting')
        prep=self.prep(records,'combined');scale=target_stats(y)[1]
        result=dict(base=base,prep=prep,residual=y-oof,scale=scale,crossfit=splits,base_penalty=1.)
        self.contexts[key]=result;return result

def configs(name,smoke=False,foundation=False):
    if name.startswith('residual_'):
        return [dict(epochs=0,shrink=0.),*[dict(epochs=e,shrink=.25) for e in ([2] if smoke else [8,24,48])]]
    if name=='tabpfn_v2':return [dict(pca=32,output='median')]
    if name=='ridge_fixed':return [dict(penalty=1.)]
    return [dict(penalty=a) for a in ([1.] if smoke else [.1,1.,10.])]

def fit_candidate(name,records,y,dims,schema,config,seeds,device,cache=None):
    cache=cache or FitCache(dims,schema)
    if name in ['training_mean','training_median']:
        fn=np.mean if name=='training_mean' else np.median
        center=np.array([fn(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in y.T])
        return dict(kind='v172_constant',values=center,schema=schema)
    if name=='tabpfn_v2':return fit_context(records,y,dims,schema,config,device)
    if name.startswith('residual_'):
        if config['shrink']==0:
            a=fit_ridge(records,y,dims,schema,1.);a['explicit_residual_fallback']=True;return a
        return fit_residual(name,records,y,schema,config,seeds,device,cache)
    if name in ['ridge_summary','ridge_tangent','ridge_frozen']:
        kind=name.removeprefix('ridge_');prep=cache.prep(records,kind);x=prep.matrix(records)
        mean,scale=target_stats(y);models=[]
        groups={}
        for j,c in enumerate(y.T):
            good=np.isfinite(c)
            if good.sum()>=2:groups.setdefault(good.tobytes(),[]).append(j)
        for key,columns in groups.items():
            good=np.frombuffer(key,dtype=bool);reg=Ridge(alpha=config['penalty']*int(good.sum()),solver='cholesky')
            reg.fit(x[good],(y[good][:,columns]-mean[columns])/scale[columns]);models.append((columns,reg))
        return dict(kind='v172_aug_ridge',prep=prep,models=models,mean=mean,scale=scale,schema=schema)
    views={'ridge_eeg':('eeg',),'ridge_eye':('eye',),'ridge_heart':('heart',),
           'ridge_eeg_eye':('eeg','eye'),'ridge_trimodal':('eeg','eye','heart'),
           'ridge_mask_only':('eeg','eye','heart'),'ridge_fixed':('eeg','eye','heart')}
    return fit_ridge(records,y,dims,schema,config['penalty'],views[name],name=='ridge_mask_only')

def tensors(prep,records,device):
    x,m=prep.transform(records)
    return [torch.as_tensor(v,device=device) for v in x],torch.as_tensor(m,device=device)

def fit_residual(name,records,y,schema,config,seeds,device,cache):
    ctx=cache.residual(records,y);x,m=tensors(ctx['prep'],records,device)
    truth=torch.as_tensor(np.nan_to_num(ctx['residual']/ctx['scale'],nan=0).astype(np.float32),device=device)
    observed=torch.as_tensor(np.isfinite(y),device=device);dims=[a.shape[1] for a in x]
    states=[];logs=[];variant=name.removeprefix('residual_')
    if variant not in ['plain','aux','mmpareto']:raise ValueError('Unknown residual method')
    for seed in seeds:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if str(device).startswith('cuda'):torch.cuda.manual_seed_all(seed)
        rng=np.random.default_rng(seed);model=ResidualFusion(dims,schema).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
        history=[];steps=0;clip_count=0;grad_logs=[];first_encoder_norms=None
        for epoch in range(1,config['epochs']+1):
            model.train();order=rng.permutation(len(records));losses=[]
            for start in range(0,len(order),32):
                ix=order[start:start+32];optimizer.zero_grad(set_to_none=True)
                pred,aux=model([v[ix] for v in x],m[ix])
                fused=masked_loss(pred,truth[ix],observed[ix],schema)+.02*pred.square().mean()
                terms=[masked_loss(p,truth[ix],observed[ix]&m[ix,k:k+1],schema) for k,p in enumerate(aux)]
                if not torch.isfinite(fused):raise FloatingPointError('Nonfinite residual loss')
                gl=update_gradients(fused,terms,model,variant);grad_logs.extend(gl)
                if first_encoder_norms is None:
                    first_encoder_norms=[float(torch.sqrt(sum((p.grad.square().sum() for p in enc.parameters() if p.grad is not None),torch.tensor(0.,device=device)))) for enc in model.encoders]
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.));clip_count+=norm>10
                optimizer.step();steps+=1;losses.append(float(fused.detach()))
            if epoch in [1,config['epochs']]:history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
        model.eval()
        with torch.no_grad():p=model(x,m)[0].cpu().numpy()
        logs.append(dict(seed=seed,epochs=config['epochs'],optimizer_steps=steps,parameters=sum(p.numel() for p in model.parameters()),
                         first_encoder_grad_norm=first_encoder_norms,clip_fraction=clip_count/max(steps,1),history=history,
                         train_residual_prediction_sd=p.std(0).tolist(),
                         gradient_conflict_fraction=float(np.mean([g['cosine']<0 for g in grad_logs])) if grad_logs else None,
                         mean_gradient_rescale=float(np.mean([g['rescale'] for g in grad_logs])) if grad_logs else None))
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    return dict(kind='v172_residual',prep=ctx['prep'],base=ctx['base'],scale=ctx['scale'],schema=schema,
                variant=variant,config=config,dims=dims,states=states,logs=logs,
                crossfit=ctx['crossfit'],base_penalty=ctx['base_penalty'],no_outer_early_stopping=True)

def predict_candidate(a,records,device='cpu',components=False):
    kind=a['kind']
    if kind=='v172_constant':p=np.broadcast_to(a['values'],(len(records),len(a['values']))).copy()
    elif kind=='foundation_context':p=predict_context(a,records,device)
    elif kind=='v172_aug_ridge':
        x=a['prep'].matrix(records);p=np.broadcast_to(a['mean'],(len(records),len(a['schema']))).copy()
        for columns,reg in a['models']:p[:,columns]=reg.predict(x)*a['scale'][columns]+a['mean'][columns]
    elif kind=='v172_residual':
        x,m=tensors(a['prep'],records,device);pp=[]
        with torch.inference_mode():
            for state in a['states']:
                model=ResidualFusion(a['dims'],a['schema']).to(device);model.load_state_dict(state);model.eval()
                pp.append(model(x,m)[0].cpu().numpy())
        base=predict_ridge(a['base'],records);delta=np.mean(pp,0)*a['scale']*a['config']['shrink'];p=base+delta
        if components:return bound(p,a['schema']),dict(base_sd=base.std(0).tolist(),correction_sd=delta.std(0).tolist())
    else:p=predict_ridge(a,records)
    p=bound(p,a['schema'])
    if not np.isfinite(p).all():raise FloatingPointError('Invalid prediction')
    return (p,{}) if components else p
