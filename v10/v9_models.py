"""Small-sample V9 models: fold-only selection, distribution kernels and target coupling."""
import copy
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

EEG_BLOCKS=['eeg_spectral','eeg_dynamics','eeg_connectivity','eeg_foundation']

def stack(records,blocks):return np.stack([np.concatenate([r[k] for k in blocks]) for r in records])

class Selector:
    def __init__(self,k=24):self.k=k
    def fit(self,x,y):
        self.scaler=RobustScaler(quantile_range=(10,90)).fit(x)
        z=np.clip(self.scaler.transform(x),-5,5)
        sy=np.std(y);good=np.std(z,axis=0)>1e-8
        centered=z-z.mean(0)
        scores=np.abs(centered.T@(y-y.mean()))/np.maximum(np.linalg.norm(centered,axis=0)*np.linalg.norm(y-y.mean()),1e-12)
        scores[~good]=-1
        self.order=np.argsort(-scores,kind='stable')[:min(self.k,int(good.sum()))]
        if not len(self.order):raise ValueError('All training features constant')
        return self
    def transform(self,x):return np.clip(self.scaler.transform(x),-5,5)[:,self.order]

class BagEmbedding:
    """Equal-person scaler + fixed Fourier map + distribution mean/SD. Fit on train only."""
    def fit(self,bags):
        # Exactly 20 deterministic positions/person. Repeats only weight the scaler;
        # they are never counted as independently labeled observations.
        sample=np.concatenate([b[np.linspace(0,len(b)-1,20).round().astype(int)] for b in bags])
        self.scaler=RobustScaler(quantile_range=(10,90)).fit(sample)
        d=sample.shape[1];rng=np.random.default_rng(20261016)
        self.weight=rng.normal(size=(d,64))/np.sqrt(d);self.bias=rng.uniform(0,2*np.pi,64)
        return self
    def transform(self,bags):
        result=[]
        for bag in bags:
            z=np.clip(self.scaler.transform(bag),-5,5)
            phi=np.sqrt(2/64)*np.cos(z@self.weight+self.bias)
            result.append(np.r_[phi.mean(0),phi.std(0)])
        return np.stack(result)

class RBFView:
    def fit(self,x):
        self.scaler=RobustScaler(quantile_range=(10,90)).fit(x)
        self.training=np.clip(self.scaler.transform(x),-5,5)
        d=cdist(self.training,self.training,'sqeuclidean')
        v=d[np.triu_indices(len(x),1)];self.width=max(float(np.median(v[v>1e-10])) if (v>1e-10).any() else 1.,1e-8)
        return self
    def transform(self,x):
        z=np.clip(self.scaler.transform(x),-5,5)
        return np.exp(-cdist(z,self.training,'sqeuclidean')/self.width)

class KernelContext:
    def fit(self,records,y,primary):
        self.primary=primary;self.mean=y.mean(0);self.scale=np.maximum(y.std(0),1e-6)
        self.y=(y-self.mean)/self.scale
        self.sparse={};self.sparse_train={}
        for modality,blocks in [('eeg',EEG_BLOCKS),('eye',['eye_invariant'])]:
            x=stack(records,blocks)
            for k in [8,24]:
                key=(modality,k);self.sparse[key]=Selector(k).fit(x,y[:,primary])
                self.sparse_train[key]=self.sparse[key].transform(x)
        self.statistics_selector=Selector(24).fit(stack(records,EEG_BLOCKS[:3]),y[:,primary])
        self.eye_selector=Selector(16).fit(stack(records,['eye_invariant']),y[:,primary])
        foundation=stack(records,['eeg_foundation'])
        self.foundation_scaler=StandardScaler().fit(foundation)
        self.pca=PCA(n_components=min(8,len(records)-1),whiten=True,svd_solver='full').fit(self.foundation_scaler.transform(foundation))
        self.eeg_bag=BagEmbedding().fit([r['eeg_bag'] for r in records])
        self.eye_bag=BagEmbedding().fit([r['eye_bag'] for r in records])
        vectors=self.vectors(records)
        self.views={name:RBFView().fit(x) for name,x in vectors.items()}
        self.eeg_half_scaler=StandardScaler().fit(np.stack([.5*(r['eeg_half_a']+r['eeg_half_b']) for r in records]))
        self.eye_half_scaler=StandardScaler().fit(np.stack([.5*(r['eye_half_a']+r['eye_half_b']) for r in records]))
        self.training_gate=self.gates(records)
        self.training_kernels=self.kernels(records)
        return self

    def vectors(self,records):
        return {'statistics':self.statistics_selector.transform(stack(records,EEG_BLOCKS[:3])),
            'foundation':self.pca.transform(self.foundation_scaler.transform(stack(records,['eeg_foundation']))),
            'eeg_distribution':self.eeg_bag.transform([r['eeg_bag'] for r in records]),
            'eye_statistics':self.eye_selector.transform(stack(records,['eye_invariant'])),
            'eye_distribution':self.eye_bag.transform([r['eye_bag'] for r in records]),
            'eye_raw':stack(records,['eye_raw'])}

    def gates(self,records):
        def agreement(a,b,scaler):
            diff=(np.stack(a)-np.stack(b))/np.maximum(scaler.scale_,1e-6)
            return np.exp(-np.minimum(np.mean(np.clip(diff,-5,5)**2,axis=1),5))
        qe=agreement([r['eeg_half_a'] for r in records],[r['eeg_half_b'] for r in records],self.eeg_half_scaler)
        qo=agreement([r['eye_half_a'] for r in records],[r['eye_half_b'] for r in records],self.eye_half_scaler)
        counts=np.stack([r['counts'] for r in records]);support=np.array([float(r['eye_repeat_support']) for r in records])
        qe=np.maximum(qe*counts[:,0]/(counts[:,0]+10),.05)
        qo=np.maximum(qo*support*counts[:,2]/(counts[:,2]+2),.05)
        return np.clip(qe/(qe+qo),.1,.9)

    def kernels(self,records):
        v=self.vectors(records);k={name:self.views[name].transform(x) for name,x in v.items()}
        eeg=(k['statistics']+k['foundation']+k['eeg_distribution'])/3
        eye=(k['eye_statistics']+k['eye_distribution'])/2
        gate=self.gates(records);tr=self.training_gate
        interaction=.5
        diag=lambda g:np.sqrt(g*g+(1-g)**2+interaction)
        gated=(gate[:,None]*tr[None,:]*eeg+(1-gate[:,None])*(1-tr[None,:])*eye+interaction*eeg*eye)/(diag(gate)[:,None]*diag(tr)[None,:])
        return {'eeg_kernel':eeg,'eye_kernel':eye,'eye_raw_kernel':k['eye_raw'],
            'fusion_additive':.5*(eeg+eye),
            'fusion_interaction':(eeg+eye+interaction*eeg*eye)/(2+interaction),
            'fusion_gated':gated,'fusion_multitask':gated}

    def prepare_query(self,records):
        sparse={}
        for key,selector in self.sparse.items():
            blocks=EEG_BLOCKS if key[0]=='eeg' else ['eye_invariant']
            sparse[key]=selector.transform(stack(records,blocks))
        return {'kernels':self.kernels(records),'sparse':sparse}

def catalog():
    out=[]
    for family in ['eeg_sparse','eye_sparse']:
        for k in [8,24]:
            for alpha in [10.,100.]:out.append({'id':f'{family}_k{k}_a{alpha}', 'family':family,'k':k,'alpha':alpha})
    for family in ['eeg_kernel','eye_kernel','eye_raw_kernel','fusion_additive','fusion_interaction','fusion_gated','fusion_multitask']:
        for alpha in [.1,1.,10.]:
            out.append({'id':f'{family}_a{alpha}', 'family':family,'alpha':alpha,
                        'coupling':.75 if family=='fusion_multitask' else 0.})
    out.append({'id':'fusion_attention_e70_s2','family':'fusion_attention','epochs':70,'seeds':[17,29],
                'hidden':8,'dropout':.5,'weight_decay':.05,'learning_rate':.001})
    return out

FAMILIES=list(dict.fromkeys(s['family'] for s in catalog()))

class Head:
    def fit(self,context,spec):
        self.context=context;self.spec=copy.deepcopy(spec);family=spec['family'];self.primary=context.primary
        if family.endswith('_sparse'):
            key=(family.split('_')[0],spec['k'])
            self.linear=Ridge(alpha=spec['alpha']).fit(context.sparse_train[key],context.y)
            return self
        k=context.training_kernels[family];beta=spec.get('coupling',0)
        b=(1-beta)*np.eye(context.y.shape[1])+beta*np.cov(context.y,rowvar=False,bias=True)
        vals,vecs=np.linalg.eigh(np.atleast_2d(b));vals=np.maximum(vals,.05)
        rotated=context.y@vecs
        # Separable output kernel: shrink target contrasts more than shared components.
        coefficients=np.stack([np.linalg.solve(k+spec['alpha']/v*np.eye(len(k)),rotated[:,j]) for j,v in enumerate(vals)],axis=1)
        self.coefficient=coefficients@vecs.T
        return self
    def predict(self,records):
        return self.predict_prepared(self.context.prepare_query(records))
    def predict_prepared(self,query):
        family=self.spec['family'];c=self.context
        if family.endswith('_sparse'):
            modality=family.split('_')[0]
            x=query['sparse'][(modality,self.spec['k'])]
            pred=self.linear.predict(x)
        else:pred=query['kernels'][family]@self.coefficient
        return pred*c.scale+c.mean

def choose(records):
    best=min(records,key=lambda r:(np.mean(r['fold_mae']),r['spec']['id']))
    se=float(np.std(best['fold_mae'],ddof=1)/np.sqrt(len(best['fold_mae'])))
    eligible=[r for r in records if np.mean(r['fold_mae'])<=np.mean(best['fold_mae'])+se+1e-12]
    selected=min(eligible,key=lambda r:(-r['spec'].get('alpha',0),r['spec'].get('k',0),r['spec']['id']))
    return {'spec':selected['spec'],'inner_mae':float(np.mean(selected['fold_mae'])),
        'fold_mae':selected['fold_mae'],'best_inner_mae':float(np.mean(best['fold_mae'])),
        'one_se_heuristic':se,'rule':'one-SE heuristic within family; not a confidence interval'}
