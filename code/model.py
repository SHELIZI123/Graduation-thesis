"""DeepSets + HeMIS-style moments + compact PLE/MMoE multi-questionnaire heads.

This is a new adaptation, not the original papers' datasets or exact reproduction.
"""
import torch
from torch import nn
from common import MODALITIES,FAMILIES

class SetEncoder(nn.Module):
    def __init__(self,features,width,dropout):
        super().__init__()
        self.phi=nn.Sequential(nn.Linear(features,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.rho=nn.Sequential(nn.Linear(width*2,width),nn.LayerNorm(width),nn.SiLU(),nn.Dropout(dropout))
    def forward(self,x,mask):
        w=mask.to(x.dtype).unsqueeze(-1);n=w.sum(1).clamp_min(1)
        h=self.phi(x);mu=(h*w).sum(1)/n
        var=((h-mu[:,None])**2*w).sum(1)/n
        z=self.rho(torch.cat([mu,var],-1))
        return z*mask.any(1).to(x.dtype).unsqueeze(-1)

def expert(d,p):return nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Dropout(p),nn.Linear(d,d),nn.SiLU())

class CGC(nn.Module):
    def __init__(self,d,p,last):
        super().__init__();self.last=last
        self.shared=nn.ModuleList([expert(d,p) for _ in range(2)])
        self.private=nn.ModuleList([expert(d,p) for _ in FAMILIES])
        self.gates=nn.ModuleList([nn.Linear(d,3) for _ in FAMILIES])
        if not last:self.shared_gate=nn.Linear(d,5)
    def forward(self,states):
        common=[f(states[-1]) for f in self.shared]
        private=[f(states[i]) for i,f in enumerate(self.private)]
        outputs=[]
        for i,g in enumerate(self.gates):
            e=torch.stack([private[i],*common],1);w=g(states[i]).softmax(-1)
            outputs.append((e*w[:,:,None]).sum(1))
        if not self.last:
            e=torch.stack([*private,*common],1);w=self.shared_gate(states[-1]).softmax(-1)
            outputs.append((e*w[:,:,None]).sum(1))
        return outputs

class TriScale(nn.Module):
    def __init__(self,dims,schema,width=24,hidden=48,dropout=.25,routing='ple',modality_dropout=.15):
        super().__init__();self.routing=routing;self.modality_dropout=modality_dropout
        self.schema=schema
        self.encoders=nn.ModuleDict({m:SetEncoder(2*dims[m],width,dropout) for m in MODALITIES})
        self.fusion=nn.Sequential(nn.Linear(5*width+3,hidden),nn.LayerNorm(hidden),nn.SiLU(),nn.Dropout(dropout))
        if routing=='ple':self.layers=nn.ModuleList([CGC(hidden,dropout,False),CGC(hidden,dropout,True)])
        elif routing=='mmoe':
            self.shared=nn.ModuleList([expert(hidden,dropout) for _ in range(3)])
            self.gates=nn.ModuleList([nn.Linear(hidden,3) for _ in FAMILIES])
        elif routing=='shared':self.shared=expert(hidden,dropout)
        else:raise ValueError(routing)
        self.indices=[[j for j,t in enumerate(schema) if t['family']==family] for family in FAMILIES]
        self.heads=nn.ModuleList([nn.Linear(hidden,len(ix)) for ix in self.indices])
        # Epoch zero exactly predicts training means. Validation may legitimately retain it.
        for head in self.heads:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    def forward(self,x,masks,return_embeddings=False):
        z=torch.stack([self.encoders[m](x[m],masks[m]) for m in MODALITIES],1)
        present=torch.stack([masks[m].any(1) for m in MODALITIES],1)
        if not present.any(1).all():raise ValueError('Each subject needs at least one real signal modality')
        if self.training and self.modality_dropout:
            kept=present&(torch.rand_like(present.float())>=self.modality_dropout)
            empty=~kept.any(1); first=present.float().argmax(1)
            kept[empty,first[empty]]=True;present=kept
        w=present.to(z.dtype).unsqueeze(-1);n=w.sum(1)
        mu=(z*w).sum(1)/n.clamp_min(1)
        # Sample variance matches HeMIS; single-modality variance is zero.
        var=((z-mu[:,None])**2*w).sum(1)/(n-1).clamp_min(1)
        var=torch.where(n>1,var,torch.zeros_like(var))
        h=self.fusion(torch.cat([(z*w).flatten(1),mu,var,present.float()],-1))
        if self.routing=='ple':
            states=[h]*4
            for layer in self.layers:states=layer(states)
        elif self.routing=='mmoe':
            e=torch.stack([f(h) for f in self.shared],1)
            states=[(e*g(h).softmax(-1)[:,:,None]).sum(1) for g in self.gates]
        else:states=[self.shared(h)]*3
        pred=torch.zeros((len(h),len(self.schema)),device=h.device,dtype=h.dtype)
        for ix,head,state in zip(self.indices,self.heads,states):pred[:,ix]=head(state)
        return (pred,z,present) if return_embeddings else pred

def masked_group_loss(pred,truth,observed,schema):
    safe=torch.where(observed,truth,torch.zeros_like(truth))
    loss=nn.functional.smooth_l1_loss(pred,safe,reduction='none')
    counts=observed.sum(0);per_target=(loss*observed).sum(0)/counts.clamp_min(1)
    terms=[]
    for family in FAMILIES:
        weights=torch.tensor([t['weight'] if t['family']==family else 0. for t in schema],device=pred.device)
        weights=weights*(counts>0)
        if weights.sum()>0:terms.append((per_target*weights).sum()/weights.sum())
    if not terms:return pred.sum()*0
    return torch.stack(terms).mean()
