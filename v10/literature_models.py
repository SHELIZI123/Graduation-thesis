"""Paper-inspired V10 estimators. Every fitted transform is training-only."""
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.special import stdtr
from scipy.optimize import minimize
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import Ridge
from v9_models import Selector, EEG_BLOCKS, stack


class ExactSetKernel:
    """Gaussian on empirical mean embeddings, using ALL instances, not RFF.

    Inner kernel includes diagonal terms to remain PSD. No unbiased diagonal
    subtraction. Long bags do not give a person a larger regression weight.
    """
    def fit(self, bags):
        sample=np.concatenate([b[np.linspace(0,len(b)-1,20).round().astype(int)] for b in bags])
        self.scaler=RobustScaler(quantile_range=(10,90)).fit(sample)
        sample=np.clip(self.scaler.transform(sample),-5,5)
        sample=sample[np.linspace(0,len(sample)-1,min(800,len(sample))).round().astype(int)]
        distances=pdist(sample,'sqeuclidean');positive=distances[distances>1e-10]
        self.width=max(float(np.median(positive)) if len(positive) else 1.,1e-8)
        self.training=[self.standardize(b) for b in bags]
        self.train_lengths=np.array([len(b) for b in bags])
        self.train_starts=np.r_[0,np.cumsum(self.train_lengths)[:-1]]
        self.train_points=np.concatenate(self.training)
        g=self.mean_gram(self.training)
        g=.5*(g+g.T);self.self_gram=np.diag(g).copy()
        d=np.maximum(self.self_gram[:,None]+self.self_gram[None,:]-2*g,0)
        positive=d[np.triu_indices(len(bags),1)];positive=positive[positive>1e-12]
        self.outer_width=max(float(np.median(positive)) if len(positive) else 1.,1e-8)
        self.training_kernel=np.exp(-d/self.outer_width)
        return self

    def standardize(self, bag):return np.clip(self.scaler.transform(bag),-5,5)

    def mean_gram(self, bags):
        rows=[]
        for b in bags:
            k=np.exp(-cdist(b,self.train_points,'sqeuclidean')/self.width)
            rows.append(np.add.reduceat(k,self.train_starts,axis=1).mean(0)/self.train_lengths)
        return np.stack(rows)

    def transform(self, bags):
        bags=[self.standardize(b) for b in bags]
        g=self.mean_gram(bags)
        self_g=np.array([np.exp(-cdist(b,b,'sqeuclidean')/self.width).mean() for b in bags])
        d=np.maximum(self_g[:,None]+self.self_gram[None,:]-2*g,0)
        return np.exp(-d/self.outer_width)


def connection_matrix(records):
    # 5 bands x 15 sensor-region pair averages; plus 120 off-diagonal
    # log-covariance entries. These are NOT source-space DMN connections.
    upper=np.triu_indices(16);off=np.flatnonzero(upper[0]!=upper[1])
    return np.stack([np.r_[r['eeg_connectivity'][:75],r['eeg_connectivity'][150:][off]] for r in records])


class CPMSummary:
    def fit(self, records, y, rule):
        x=connection_matrix(records);self.scaler=StandardScaler().fit(x)
        z=np.clip(self.scaler.transform(x),-5,5);center=z-z.mean(0);yc=y-y.mean()
        r=center.T@yc/np.maximum(np.linalg.norm(center,axis=0)*np.linalg.norm(yc),1e-12)
        good=np.std(z,0)>1e-8;r[~good]=0
        t=abs(r)*np.sqrt((len(y)-2)/np.maximum(1-r*r,1e-12));p=2*stdtr(len(y)-2,-t)
        self.q=np.minimum.accumulate((p[np.argsort(p)]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
        q=np.empty_like(p);q[np.argsort(p)]=np.minimum(self.q,1);self.q=q
        if rule=='fdr01':selected=(q<.01)&good
        elif rule=='top16':
            selected=np.zeros(len(r),bool);order=np.argsort(-abs(r),kind='stable');selected[order[:16]]=True;selected &=good
        else:raise ValueError(rule)
        groups=np.r_[np.repeat(np.arange(5),15),np.repeat(5,120)]
        self.masks=[selected&(groups==g)&(r*sign>0) for g in range(6) for sign in [1,-1]]
        self.selected_count=int(selected.sum());self.correlations=r;self.rule=rule
        a=self.raw(records);self.summary_scaler=StandardScaler().fit(a)
        return self

    def raw(self, records):
        z=np.clip(self.scaler.transform(connection_matrix(records)),-5,5)
        # Mean instead of sum is equivalent up to train-only rescaling.
        return np.stack([z[:,m].mean(1) if m.any() else np.zeros(len(z)) for m in self.masks],axis=1)

    def transform(self, records):return self.summary_scaler.transform(self.raw(records))


class Context:
    def fit(self, records, y):
        self.mean=float(np.mean(y));self.scale=max(float(np.std(y)),1e-6);self.y=(y-self.mean)/self.scale
        self.eeg=ExactSetKernel().fit([r['eeg_bag'] for r in records])
        self.eye=ExactSetKernel().fit([r['eye_bag'] for r in records])
        self.kernels={'set_eeg':self.eeg.training_kernel,'set_eye':self.eye.training_kernel,
                      'set_fusion':.5*(self.eeg.training_kernel+self.eye.training_kernel)}
        self.sparse={m:Selector(8).fit(stack(records,b),y) for m,b in [('eeg',EEG_BLOCKS),('eye',['eye_invariant'])]}
        self.cpm={r:CPMSummary().fit(records,y,r) for r in ['fdr01','top16']}
        self.query=self.prepare(records,training=True)
        return self

    def prepare(self, records, training=False):
        if training:kernels=self.kernels
        else:
            e=self.eeg.transform([r['eeg_bag'] for r in records]);o=self.eye.transform([r['eye_bag'] for r in records])
            kernels={'set_eeg':e,'set_eye':o,'set_fusion':.5*(e+o)}
        return {'kernels':kernels,'sparse':{m:self.sparse[m].transform(stack(records,b)) for m,b in [('eeg',EEG_BLOCKS),('eye',['eye_invariant'])]},
                'cpm':{r:v.transform(records) for r,v in self.cpm.items()},'n':len(records)}


def catalog():
    result=[]
    for family in ['set_eeg','set_eye','set_fusion']:
        for lam in [.001,.01,.1]:result.append({'id':f'{family}_l{lam}','family':family,'lambda':lam})
    for rule in ['fdr01','top16']:
        for alpha in [1.,100.]:result.append({'id':f'cpm_{rule}_a{alpha}','family':'cpm_'+rule,'rule':rule,'alpha':alpha})
    # Identical fixed small-sample controls; no new outer-based hyperparameter search.
    for m in ['eeg','eye']:result.append({'id':f'control_{m}_k8_a100','family':'control_'+m,'alpha':100.})
    return result


class Regressor:
    def fit(self, context, spec):
        self.context=context;self.spec=spec.copy();f=spec['family']
        if f.startswith('set_'):
            k=context.kernels[f];n=len(k);self.train_col_mean=k.mean(0);self.train_mean=float(k.mean())
            centered=k-k.mean(1)[:,None]-self.train_col_mean[None,:]+self.train_mean
            self.coef=np.linalg.solve(centered+n*spec['lambda']*np.eye(n),context.y)
        else:
            x=context.query['cpm'][spec['rule']] if f.startswith('cpm_') else context.query['sparse'][f.split('_')[1]]
            self.linear=Ridge(alpha=spec['alpha']).fit(x,context.y)
        return self

    def predict_prepared(self, q):
        f=self.spec['family'];c=self.context
        if f.startswith('set_'):
            k=q['kernels'][f];k=k-k.mean(1)[:,None]-self.train_col_mean[None,:]+self.train_mean
            p=k@self.coef
        else:
            x=q['cpm'][self.spec['rule']] if f.startswith('cpm_') else q['sparse'][f.split('_')[1]]
            p=self.linear.predict(x)
        return p*c.scale+c.mean

    def predict(self, records):return self.predict_prepared(self.context.prepare(records))


BASE_IDS=['control_eeg_k8_a100','control_eye_k8_a100','set_eeg_l0.01','set_eye_l0.01','cpm_top16_a100.0']


def fit_weights(predictions, y):
    """Fixed regularized convex stacking, using only cross-fitted predictions.

    Column 0 is the fold-training mean. Penalty on all other weights is a
    deliberately conservative extension of Super Learner, not its theorem.
    """
    scale=max(float(np.std(y)),1e-6);p=predictions/scale;t=y/scale
    def loss(w):return np.mean((p@w-t)**2)+.02*np.sum(w[1:]**2)
    def jac(w):
        g=2*p.T@(p@w-t)/len(t);g[1:]+=.04*w[1:];return g
    init=np.zeros(p.shape[1]);init[0]=1
    res=minimize(loss,init,jac=jac,method='SLSQP',bounds=[(0,1)]*len(init),
                 constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)},
                 options={'ftol':1e-12,'maxiter':1000})
    if not res.success or abs(res.x.sum()-1)>1e-6:raise RuntimeError('Stack optimization failed: '+res.message)
    return res.x


class StackedRegressor:
    def fit(self, context, weights):
        self.context=context;self.weights=np.asarray(weights).copy()
        lookup={s['id']:s for s in catalog()};self.models=[Regressor().fit(context,lookup[k]) for k in BASE_IDS]
        return self
    def predict_prepared(self,q):
        p=np.column_stack([np.full(q['n'],self.context.mean)]+[m.predict_prepared(q) for m in self.models])
        return p@self.weights
    def predict(self,records):return self.predict_prepared(self.context.prepare(records))
