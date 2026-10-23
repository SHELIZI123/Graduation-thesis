"""Matched EEG-only representations and fixed-energy multimodal blocks."""
import numpy as np
from features_v172 import EEGGeometry,ReducedBlock,FeaturePrep,load_inputs
from features_v171 import summary_matrix

class EEGOnlyPrep:
    def fit(self,records,dims,kind):
        if any(not len(r['eeg']) for r in records):raise ValueError('EEG-only fitting requires EEG')
        self.dims=dims;self.kind=kind;self.geometry=None
        self.training_ids=[r['subject_id'] for r in records]
        if kind=='tangent':self.geometry=EEGGeometry().fit(records)
        self.reduction=ReducedBlock().fit(self.raw(records),components=32)
        return self
    def raw(self,records):
        if self.kind=='tangent':return self.geometry.transform(records)
        if self.kind=='frozen':return np.stack([r['embedding'] for r in records])
        if self.kind=='summary':return summary_matrix(records,self.dims,views=('eeg',))
        raise ValueError('Unknown representation')
    def matrix(self,records):return self.reduction.transform(self.raw(records))

class BalancedPrep:
    """Prespecified EEG/eye/heart energy; all centers/scales fitted on training only."""
    def fit(self,records,dims,augmented=False):
        self.training_ids=[r['subject_id'] for r in records];self.augmented=augmented
        self.prep=FeaturePrep().fit(records,dims,'combined' if augmented else 'summary')
        raw=self.blocks(records);self.centers={};self.scales={}
        self.weights={'eeg':.5,'tangent':.25,'frozen':.25,'eye':1.,'heart':1.} if augmented else {'eeg':1.,'eye':1.,'heart':1.}
        for k,x in raw.items():
            center=x.mean(0);energy=float(np.mean(np.sum((x-center)**2,1)))
            self.centers[k]=center;self.scales[k]=max(np.sqrt(energy),1e-6)
        return self
    def blocks(self,records):
        x,present=self.prep.transform(records)
        # Original EEG statistical vector precedes the new EEG PCA blocks.
        width=2*self.prep.scalers[0].center.size
        out={'eeg':x[0][:,:width],'eye':x[1],'heart':x[2]}
        if self.augmented:
            k=self.prep.blocks['tangent'].output_dim+1
            out['tangent']=x[0][:,width:width+k];out['frozen']=x[0][:,width+k:]
        return out
    def matrix(self,records):
        raw=self.blocks(records)
        # Each physiological modality has total prespecified energy 1.
        return np.concatenate([(x-self.centers[k])/self.scales[k]*np.sqrt(self.weights[k]) for k,x in raw.items()],1).astype(np.float64)
