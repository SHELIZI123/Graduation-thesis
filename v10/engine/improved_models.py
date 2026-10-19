"""Fixed small-sample models. All fitted transforms see training people only."""
import copy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from baselines import SafeImputer

# Disjoint, predeclared electrode regions; these are not task or diagnosis labels.
REGIONS = {
    'frontal': [0,1,2,3,10,11], 'central': [4,5],
    'temporal': [12,13,14,15], 'parietal': [6,7], 'occipital': [8,9],
}
BLOCKS = {'power': (0,50), 'eye': (50,70), 'regional': (70,2070), 'demographics': (2070,2072)}
MODELS = ['power_pca','regional_cbramod','eye_regularized','late_fusion','guarded_fusion','demographics_control']


def compact_power(vector):
    v=np.asarray(vector,float)
    if v.shape!=(160,) or not np.isfinite(v).all(): raise ValueError('Expected finite 160D spectral features')
    absolute,relative=v[:80].reshape(16,5),v[80:].reshape(16,5)
    return np.concatenate([np.r_[absolute[g].mean(0),relative[g].mean(0)] for g in REGIONS.values()])


def base_candidates():
    out=[{'id':'constant_'+k,'kind':'constant','statistic':k} for k in ['mean','median']]
    for block in ['power','regional']:
        for n in [8,16]:
            for a in [10.,100.,1000.]:
                out.append({'id':f'{block}_p{n}_a{int(a)}','kind':'pca_ridge','block':block,'n_components':n,'alpha':a})
    for c in [.1,1.]:
        out.append({'id':f'eye_c{c}','kind':'svr','block':'eye','C':c})
    for a in [1.,10.,100.,1000.]:
        out.append({'id':f'demographics_a{int(a)}','kind':'ridge','block':'demographics','alpha':a})
    return out


def candidate_catalog():
    base=base_candidates();out=list(base)
    for e in [s for s in base if s.get('block') in {'power','regional'}]:
        for eye in [s for s in base if s.get('block')=='eye']:
            for w in [.25,.5,.75]:
                out.append({'id':f"mix_{e['id']}_{eye['id']}_w{w}",'kind':'blend',
                    'eeg':copy.deepcopy(e),'eye':copy.deepcopy(eye),'eeg_weight':w})
    return out


def complexity(spec):
    if spec['kind']=='constant': return (0,0,0)
    if spec['kind']=='blend': return (4,-spec['eeg']['alpha'],spec['eeg']['n_components'])
    block=spec['block']
    if block=='eye':return (1,spec['C'],0)
    return ({'power':2,'regional':3,'demographics':1}[block],-spec['alpha'],spec.get('n_components',0))


def choose(records, allowed, conservative=True):
    pool=[r for r in records if allowed(r['spec'])]
    if not pool:raise ValueError('No eligible candidates')
    best=min(pool,key=lambda r:(r['inner_mae'],r['spec']['id']))
    # Three correlated validation folds: a heuristic for conservatism, NOT a confidence interval.
    se=float(np.std(best['fold_mae'],ddof=1)/np.sqrt(len(best['fold_mae']))) if len(best['fold_mae'])>1 else 0.
    eligible=[r for r in pool if r['inner_mae'] <= best['inner_mae']+(se if conservative else 0.)+1e-12]
    selected=min(eligible,key=lambda r:(complexity(r['spec']),r['inner_mae'],r['spec']['id']))
    return {'spec':copy.deepcopy(selected['spec']),'inner_mae':selected['inner_mae'],
        'fold_mae':selected['fold_mae'],'best_inner_mae':best['inner_mae'],
        'best_fold_se_heuristic':se,'rule':'one_standard_error' if conservative else 'minimum_inner_mae',
        'eligible_count':len(eligible),'candidate_count':len(pool)}


def choose_all(records):
    physio=lambda s:s.get('block')!='demographics'
    return {
        'power_pca':choose(records,lambda s:s.get('block')=='power'),
        'regional_cbramod':choose(records,lambda s:s.get('block')=='regional'),
        'eye_regularized':choose(records,lambda s:s.get('block')=='eye'),
        # Single-modality endpoints are included; no forced multimodal benefit.
        'late_fusion':choose(records,lambda s:physio(s) and s['kind']!='constant',False),
        'guarded_fusion':choose(records,physio,True),
        'demographics_control':choose(records,lambda s:s.get('block')=='demographics'),
    }


class V8Regressor(BaseEstimator, RegressorMixin):
    def __init__(self,spec): self.spec=spec

    def fit(self,x,y):
        x=np.asarray(x,float);y=np.asarray(y,float);s=self.spec
        if x.ndim!=2 or x.shape[1]!=2072 or len(y)!=len(x) or not np.isfinite(y).all() or np.isinf(x).any():
            raise ValueError('Invalid V8 training matrix')
        self.mean_=float(y.mean());self.scale_=float(max(y.std(),1e-6))
        if s['kind']=='constant':
            self.value_=float(np.mean(y) if s['statistic']=='mean' else np.median(y));return self
        if s['kind']=='blend':
            self.eeg_=V8Regressor(s['eeg']).fit(x,y);self.eye_=V8Regressor(s['eye']).fit(x,y);return self
        start,end=BLOCKS[s['block']];self.columns_=slice(start,end)
        steps=[SafeImputer(),StandardScaler()]
        if s['kind']=='pca_ridge':
            n=min(s['n_components'],x.shape[0]-1,end-start)
            if n<1:raise ValueError('PCA requires at least two training people')
            steps += [PCA(n_components=n,whiten=True,svd_solver='full'),Ridge(alpha=s['alpha'])]
        elif s['kind']=='ridge':steps += [Ridge(alpha=s['alpha'])]
        elif s['kind']=='svr':steps += [SVR(C=s['C'],epsilon=.1,gamma='scale')]
        else:raise ValueError('Unknown V8 model')
        self.pipeline_=make_pipeline(*steps)
        self.pipeline_.fit(x[:,self.columns_],(y-self.mean_)/self.scale_)
        return self

    def predict(self,x):
        x=np.asarray(x,float);s=self.spec
        if x.ndim!=2 or x.shape[1]!=2072:raise ValueError('Feature dimension mismatch')
        if s['kind']=='constant':return np.full(len(x),self.value_)
        if s['kind']=='blend':return s['eeg_weight']*self.eeg_.predict(x)+(1-s['eeg_weight'])*self.eye_.predict(x)
        return self.pipeline_.predict(x[:,self.columns_])*self.scale_+self.mean_


def fit_v8_bundle(spec,x,y):
    model=V8Regressor(spec).fit(x,y)
    # Common bundle interface; V8Regressor already returns original score units.
    return {'model':model,'target_mean':0.,'target_std':1.,'n_features':2072,'kind':'v8','parameter':spec}


def inner_records(x,y,fold,index):
    """Each candidate is evaluated only on the supplied inner partitions.

    A blend uses held-out component predictions, never fitted training predictions.
    All component configurations and blend weights are jointly compared here.
    """
    catalog=candidate_catalog();losses={s['id']:[] for s in catalog}
    for inner in fold['inner']:
        a=[index[s] for s in inner['train']];b=[index[s] for s in inner['validation']]
        predictions={}
        for s in catalog:
            if s['kind']=='blend':
                pred=s['eeg_weight']*predictions[s['eeg']['id']]+(1-s['eeg_weight'])*predictions[s['eye']['id']]
            else:pred=V8Regressor(s).fit(x[a],y[a]).predict(x[b])
            predictions[s['id']]=pred
            losses[s['id']].append(float(np.mean(abs(pred-y[b]))))
    return [{'spec':s,'fold_mae':losses[s['id']],'inner_mae':float(np.mean(losses[s['id']]))} for s in catalog]
