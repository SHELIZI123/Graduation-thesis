"""Nested-tuned anchor, matched representation controls, auditable residuals."""
import numpy as np
from sklearn.linear_model import Ridge
from common import sha
from preprocessing import target_stats,group_splits
from engine import fit_ridge,predict_ridge,bound
from features_v171 import focus_mae
from features_v172 import FeaturePrep
from features_v173 import EEGOnlyPrep,BalancedPrep
from engine_v172 import FitCache as OldCache,fit_candidate as old_fit,predict_candidate as old_predict,fit_residual
from engine_v171 import fit_classical,predict_classical
from foundation_v173 import fit_context,predict_context

class FitCache(OldCache):
    def __init__(self,dims,schema):
        super().__init__(dims,schema);self.anchors={};self.extra_preps={}
    def anchor(self,records,y):
        key=tuple(r['subject_id'] for r in records)
        if key in self.anchors:return self.anchors[key]
        groups=[r['group_id'] for r in records];inner=group_splits(list(range(len(records))),groups,3,20261024)
        search=[]
        for penalty in [1.,10.,100.]:
            scores=[]
            for tr,va in inner:
                model=fit_ridge([records[i] for i in tr],y[tr],self.dims,self.schema,penalty)
                p=predict_ridge(model,[records[i] for i in va])
                scores.append(focus_mae(y[va],p,target_stats(y[tr])[1],self.schema))
            search.append(dict(penalty=penalty,score=float(np.mean(scores)),fold_scores=scores))
        chosen=min(search,key=lambda s:(s['score'],-s['penalty']))
        model=fit_ridge(records,y,self.dims,self.schema,chosen['penalty'])
        model['anchor_selection']=dict(config=chosen['penalty'],search=search,fit_ids=list(key),
            folds=[dict(train_ids=[key[i] for i in tr],validation_ids=[key[i] for i in va]) for tr,va in inner])
        self.anchors[key]=model;return model
    def residual(self,records,y):
        key=tuple(r['subject_id'] for r in records)
        if key in self.contexts:return self.contexts[key]
        groups=[r['group_id'] for r in records];oof=np.full_like(y,np.nan);splits=[]
        for tr,va in group_splits(list(range(len(records))),groups,3,20261024):
            b=self.anchor([records[i] for i in tr],y[tr]);oof[va]=predict_ridge(b,[records[i] for i in va])
            splits.append(dict(train_ids=[key[i] for i in tr],validation_ids=[key[i] for i in va],
                               selected_penalty=b['penalty'],anchor_selection=b['anchor_selection']))
        if not np.isfinite(oof).all():raise ValueError('Incomplete residual cross-fit')
        base=self.anchor(records,y)
        ctx=dict(base=base,prep=self.prep(records,'combined'),residual=y-oof,scale=target_stats(y)[1],
                 crossfit=splits,base_penalty=base['penalty'])
        self.contexts[key]=ctx;return ctx
    def feature_prep(self,records,name):
        key=(tuple(r['subject_id'] for r in records),name)
        if key not in self.extra_preps:
            if name.startswith('eeg_'):p=EEGOnlyPrep().fit(records,self.dims,name[4:])
            else:p=BalancedPrep().fit(records,self.dims,name=='ridge_block_augmented')
            self.extra_preps[key]=p
        return self.extra_preps[key]

def configs(name,smoke=False):
    if name.startswith('residual_'):return [dict(epochs=0,shrink=0.),*[dict(epochs=e,shrink=.25) for e in ([2] if smoke else [2,8,24])]]
    if name in ['ridge_anchor','tabpfn_v2_fp32']:return [{}]
    if name.startswith('ridge_block_'):penalties=[.01,.1,1.]
    else:penalties=[.1,1.,10.]
    return [dict(penalty=p) for p in ([1.] if smoke else penalties)]

def fit_candidate(name,records,y,dims,schema,config,seeds,device,cache=None):
    cache=cache or FitCache(dims,schema)
    if name=='ridge_anchor':return dict(cache.anchor(records,y))
    if name=='tabpfn_v2_fp32':return fit_context(records,y,dims,schema,config,device)
    if name=='ridge_distribution':return fit_classical(records,y,dims,schema,dict(kind='ridge',alpha=config['penalty']),device=device)
    if name.startswith('residual_'):
        if config['shrink']==0:
            a=dict(cache.anchor(records,y));a['explicit_residual_fallback']=True;return a
        a=fit_residual(name,records,y,schema,config,seeds,device,cache)
        a['anchor_selection']=a['base']['anchor_selection'];return a
    if name.startswith('eeg_') or name.startswith('ridge_block_'):
        keep=np.array([bool(len(r['eeg'])) for r in records]) if name.startswith('eeg_') else np.ones(len(records),bool)
        rr=[r for r,k in zip(records,keep) if k];yy=y[keep]
        if not rr:raise ValueError('No EEG training subjects')
        prep=cache.feature_prep(rr,name);x=prep.matrix(rr);mean,scale=target_stats(yy);models=[];groups={}
        for j,c in enumerate(yy.T):
            good=np.isfinite(c)
            if good.sum()>=2:groups.setdefault(good.tobytes(),[]).append(j)
        for key,columns in groups.items():
            good=np.frombuffer(key,dtype=bool);reg=Ridge(alpha=config['penalty']*int(good.sum()),solver='cholesky')
            reg.fit(x[good],(yy[good][:,columns]-mean[columns])/scale[columns]);models.append((columns,reg))
        return dict(kind='v173_feature_ridge',prep=prep,mean=mean,scale=scale,models=models,schema=schema,
                    eeg_only=name.startswith('eeg_'),effective_fit_ids=[r['subject_id'] for r in rr])
    return old_fit(name,records,y,dims,schema,config,seeds,device)

def predict_candidate(a,records,device='cpu',components=False,replay=False):
    kind=a['kind'];diagnostic={}
    if kind=='v173_foundation':p,diagnostic=predict_context(a,records,device,replay)
    elif kind=='new_classical':p=predict_classical(a,records,device)
    elif kind=='v173_bank':
        bank={k:predict_candidate(v,records,device) for k,v in a['models'].items()}
        p=np.stack([bank[str(k)][:,j] for j,k in enumerate(a['selection'])],1)
    elif kind=='v173_feature_ridge':
        x=a['prep'].matrix(records);p=np.broadcast_to(a['mean'],(len(records),len(a['schema']))).copy()
        for columns,reg in a['models']:p[:,columns]=reg.predict(x)*a['scale'][columns]+a['mean'][columns]
        if a['eeg_only']:
            absent=np.array([not len(r['eeg']) for r in records]);p[absent]=a['mean']
    else:
        p,diagnostic=old_predict(a,records,device,components=True)
        if 'anchor_selection' in a:diagnostic['anchor_penalty']=a['penalty'] if kind=='ridge' else a['base_penalty']
    p=bound(p,a['schema'])
    if not np.isfinite(p).all():raise FloatingPointError('Invalid prediction')
    return (p,diagnostic) if components else p
