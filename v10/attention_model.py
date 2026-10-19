"""Small hierarchical bag attention, frozen engineered inputs, patient-level masked loss."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from v9_models import BagEmbedding,stack

class Pool(nn.Module):
    def __init__(self,d,h,dropout):
        super().__init__();self.project=nn.Sequential(nn.Linear(d,h),nn.LayerNorm(h),nn.GELU())
        self.attention=nn.Sequential(nn.Linear(h,h),nn.Tanh(),nn.Linear(h,1,bias=False))
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,mask):
        z=self.project(x);score=self.attention(z).squeeze(-1).masked_fill(~mask,-1e9)
        w=torch.softmax(score,dim=1)
        return (self.dropout(z)*w.unsqueeze(-1)).sum(1),w

class SmallFusion(nn.Module):
    def __init__(self,ed,od,fd,targets,h=8,dropout=.5):
        super().__init__();self.eeg=Pool(ed,h,dropout);self.eye=Pool(od,h,dropout)
        self.eeg_join=nn.Sequential(nn.Linear(h+fd,h),nn.GELU())
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(3*h,h),nn.GELU(),nn.Linear(h,targets))
    def forward(self,eeg,em,eye,om,foundation):
        e,ew=self.eeg(eeg,em);o,ow=self.eye(eye,om)
        e=self.eeg_join(torch.cat([e,foundation],dim=1))
        return self.head(torch.cat([e,o,e*o],dim=1)),ew,ow

def padded(bags,scaler):
    maximum=max(len(b) for b in bags);d=bags[0].shape[1]
    x=np.zeros((len(bags),maximum,d),np.float32);mask=np.zeros((len(bags),maximum),bool)
    for i,b in enumerate(bags):
        x[i,:len(b)]=np.clip(scaler.transform(b),-5,5);mask[i,:len(b)]=True
    return x,mask

class AttentionRegressor:
    def fit(self,records,y,spec,device='cpu',threads=2):
        torch.set_num_threads(threads)
        self.spec=spec;self.targets=y.shape[1]
        self.mean=np.nanmean(y,axis=0);self.scale=np.maximum(np.nanstd(y,axis=0),1e-6)
        if not np.isfinite(self.mean).all():raise ValueError('A target has no observed training labels')
        mask=np.isfinite(y);target=np.nan_to_num((y-self.mean)/self.scale)
        self.eeg_scaler=BagEmbedding().fit([r['eeg_bag'] for r in records]).scaler
        self.eye_scaler=BagEmbedding().fit([r['eye_bag'] for r in records]).scaler
        foundation=stack(records,['eeg_foundation']);self.foundation_scaler=StandardScaler().fit(foundation)
        self.pca=PCA(n_components=min(8,len(records)-1),whiten=True,svd_solver='full').fit(self.foundation_scaler.transform(foundation))
        tensors=self.tensors(records,device);truth=torch.tensor(target,dtype=torch.float32,device=device)
        observed=torch.tensor(mask,dtype=torch.bool,device=device);self.networks=[];self.losses=[]
        for seed in spec['seeds']:
            torch.manual_seed(seed)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
            net=SmallFusion(tensors[0].shape[-1],tensors[2].shape[-1],tensors[4].shape[-1],self.targets,
                            spec['hidden'],spec['dropout']).to(device)
            optimizer=torch.optim.AdamW(net.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
            net.train();history=[]
            for epoch in range(spec['epochs']):
                optimizer.zero_grad();pred,ew,ow=net(*tensors)
                losses=torch.nn.functional.smooth_l1_loss(pred,truth,reduction='none',beta=1.)
                supervised=(losses*observed).sum()/observed.sum()
                # Gentle anti-collapse penalty; no validation labels or epoch selection.
                penalty=(ew.square().sum(1)*tensors[1].sum(1)).mean()+(ow.square().sum(1)*tensors[3].sum(1)).mean()
                loss=supervised+.005*penalty
                if not torch.isfinite(loss):raise ValueError('Nonfinite attention training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                optimizer.step();history.append(float(loss.detach().cpu()))
            self.networks.append(net.cpu().eval());self.losses.append(history)
        self.parameter_count=sum(p.numel() for p in self.networks[0].parameters())
        return self
    def tensors(self,records,device):
        e,em=padded([r['eeg_bag'] for r in records],self.eeg_scaler)
        o,om=padded([r['eye_bag'] for r in records],self.eye_scaler)
        f=np.clip(self.pca.transform(self.foundation_scaler.transform(stack(records,['eeg_foundation']))),-5,5).astype('float32')
        return tuple(torch.as_tensor(x,device=device) for x in [e,em,o,om,f])
    def predict(self,records):
        tensors=self.tensors(records,'cpu');pred=[]
        with torch.inference_mode():
            for net in self.networks:pred.append(net(*tensors)[0].numpy())
        return np.mean(pred,axis=0)*self.scale+self.mean
