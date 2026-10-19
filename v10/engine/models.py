"""EEGNet-inspired encoder, dilated temporal eye encoder and hierarchical MIL.

This is an original lightweight adaptation, NOT an exact EEGNet/E2Mo reproduction.
Eye and EEG views are pooled independently; no temporal alignment is assumed.
"""
import torch
from torch import nn


class AttentionPool(nn.Module):
    def __init__(self,dim,attention=True):
        super().__init__();self.attention=attention
        self.score=nn.Sequential(nn.Linear(dim,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x,mask=None):
        score=self.score(x).squeeze(-1) if self.attention else torch.zeros_like(x[...,0])
        if mask is not None:score=score.masked_fill(~mask,-1e4)
        weight=score.softmax(-1)
        return (x*weight.unsqueeze(-1)).sum(-2)


class EEGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,8,(1,63),padding=(0,31),bias=False),nn.GroupNorm(4,8),
            nn.Conv2d(8,16,(16,1),groups=8,bias=False),nn.GroupNorm(4,16),nn.ELU(),nn.AvgPool2d((1,4)),
            nn.Conv2d(16,16,(1,15),padding=(0,7),groups=16,bias=False),nn.Conv2d(16,16,1),
            nn.GroupNorm(4,16),nn.ELU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten())
        self.power=nn.Sequential(nn.Linear(160,16),nn.ELU())
        self.project=nn.Linear(32,32)
    def forward(self,x,power):
        return self.project(torch.cat([self.net(x.unsqueeze(1)),self.power(power)],-1))


class EyeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(4,16,7,padding=3),nn.GroupNorm(4,16),nn.ELU(),
            nn.Conv1d(16,16,5,padding=4,dilation=2),nn.GroupNorm(4,16),nn.ELU(),
            nn.Conv1d(16,16,5,padding=8,dilation=4),nn.ELU(),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.stats=nn.Sequential(nn.Linear(10,16),nn.ELU())
        self.project=nn.Linear(32,32)
    def forward(self,x,stats):
        return self.project(torch.cat([self.net(x),self.stats(stats)],-1))


class SubjectModel(nn.Module):
    def __init__(self,modality='fusion',outputs=2,pooling='attention',fusion='gate',dropout=.3):
        super().__init__();self.modality=modality;self.fusion=fusion
        self.eeg=EEGEncoder() if modality in {'eeg','fusion'} else None
        self.eye=EyeEncoder() if modality in {'eye','fusion'} else None
        self.eeg_pool=AttentionPool(32,pooling=='attention')
        self.eye_pool=AttentionPool(32,pooling=='attention')
        self.measurement_pool=AttentionPool(32,pooling=='attention')
        self.gate=nn.Linear(64,2)
        width=64 if modality=='fusion' and fusion=='concat' else 32
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Dropout(dropout),nn.Linear(width,outputs))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,b):
        zs=[]
        if self.eeg is not None:
            x=b['eeg'];n,k=x.shape[:2]
            h=self.eeg(x.flatten(0,1),b['power'].flatten(0,1)).reshape(n,k,32)
            zs.append(self.eeg_pool(h))
        if self.eye is not None:
            x=b['eye'];n,m,k=x.shape[:3]
            h=self.eye(x.flatten(0,2),b['eye_stats'].flatten(0,2)).reshape(n,m,k,32)
            # First windows within each measurement, then measurements within person.
            h=self.eye_pool(h)
            zs.append(self.measurement_pool(h,b['measurement_mask']))
        if len(zs)==1:z=zs[0]
        elif self.fusion=='concat':z=torch.cat(zs,-1)
        else:
            w=self.gate(torch.cat(zs,-1)).softmax(-1)
            z=w[:,:1]*zs[0]+w[:,1:]*zs[1]
        return self.head(z)
